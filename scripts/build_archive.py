"""Replay the raw recorder log deterministically into a snapshot archive.

    python scripts/build_archive.py [--raw data/raw] [--out data/archive]

Every ``segment-*.jsonl.zst`` in ``--raw`` is replayed in order through the same
:class:`~exchange_engine.archive.SnapshotGate` the recorder runs live, emitting at
most one snapshot per product every 10 ms and only when something in the top 50
levels changed. Trades are de-duplicated by exchange trade id (resubscribes
replay recent trades). Building twice from the same log yields byte-identical
arrays; ``tests/test_archive.py`` checks that with the content hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exchange_engine.archive import (  # noqa: E402
    PRODUCTS,
    TICK_SIZES,
    ArchiveWriter,
    SnapshotGate,
    iso_to_ns,
    iter_raw_log,
    to_ticks,
    validate,
)

BUILDER_VERSION = "1.0"
TRADE_DEDUP_WINDOW = 50_000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(raw_dir: Path, out_dir: Path, products: List[str], max_snapshots: int = 0,
          quiet: bool = False) -> dict:
    """Build the archive and return its sidecar."""
    segments = sorted(p for p in Path(raw_dir).glob("segment-*.jsonl.zst"))
    if not segments:
        raise SystemExit(f"no raw segments found in {raw_dir}")
    index = {p: i for i, p in enumerate(products)}
    ticks = {p: TICK_SIZES[p] for p in products}
    gates: Dict[str, SnapshotGate] = {p: SnapshotGate() for p in products}
    seen: Dict[str, "OrderedDict[str, None]"] = {p: OrderedDict() for p in products}
    writer = ArchiveWriter(out_dir, source="real", products=products, tick_sizes=ticks)
    counters = {"lines": 0, "frames": 0, "gaps": 0, "connects": 0, "l2_events": 0,
                "trades_seen": 0, "trades_deduped": 0, "tick_rejects": 0}
    t0 = time.perf_counter()
    done = False
    for line in iter_raw_log(segments):
        counters["lines"] += 1
        recv_ns = int(line["recv_ns"])
        if "msg" not in line:
            if "gap" in line:
                counters["gaps"] += 1
            elif "connect" in line:
                counters["connects"] += 1
                for gate in gates.values():
                    gate.ready = False
            continue
        counters["frames"] += 1
        msg = line["msg"]
        channel = msg.get("channel")
        if channel == "l2_data":
            for event in msg.get("events", []):
                pid = event.get("product_id")
                gate = gates.get(pid)
                if gate is None:
                    continue
                counters["l2_events"] += 1
                if gate.apply_event(event, recv_ns):
                    bids, asks = gate.last_top  # type: ignore[misc]
                    tick = ticks[pid]
                    try:
                        b = [(to_ticks(px, tick), float(qty)) for px, qty in bids]
                        a = [(to_ticks(px, tick), float(qty)) for px, qty in asks]
                    except ValueError:
                        counters["tick_rejects"] += 1
                        continue
                    writer.add_snapshot(recv_ns, index[pid], b, a)
                    if max_snapshots and writer.n_snapshots >= max_snapshots:
                        done = True
                        break
            if done:
                break
        elif channel == "market_trades":
            for event in msg.get("events", []):
                for trade in event.get("trades", []):
                    pid = trade.get("product_id")
                    if pid not in index:
                        continue
                    counters["trades_seen"] += 1
                    tid = str(trade.get("trade_id", ""))
                    bucket = seen[pid]
                    if tid in bucket:
                        counters["trades_deduped"] += 1
                        continue
                    bucket[tid] = None
                    if len(bucket) > TRADE_DEDUP_WINDOW:
                        bucket.popitem(last=False)
                    try:
                        px = to_ticks(Decimal(str(trade["price"])), ticks[pid])
                    except (ValueError, KeyError):
                        counters["tick_rejects"] += 1
                        continue
                    try:
                        exch_ns = iso_to_ns(str(trade.get("time", "")))
                    except ValueError:
                        exch_ns = 0
                    writer.add_trade(recv_ns, exch_ns, index[pid], px,
                                     float(Decimal(str(trade["size"]))), str(trade.get("side", "")))
        if not quiet and counters["lines"] % 200_000 == 0:
            print(f"  {counters['lines']:,} lines, {writer.n_snapshots:,} snapshots, "
                  f"{writer.n_trades:,} trades ({time.perf_counter() - t0:.0f}s)", flush=True)
    extra = {
        "builder_version": BUILDER_VERSION,
        "gap_count": counters["gaps"],
        "connect_count": counters["connects"],
        "raw_lines": counters["lines"],
        "raw_frames": counters["frames"],
        "l2_events": counters["l2_events"],
        "trades_seen": counters["trades_seen"],
        "trades_deduped": counters["trades_deduped"],
        "tick_rejects": counters["tick_rejects"],
        "crossed_skipped": {p: g.crossed for p, g in gates.items()},
        "raw_segments": [{"file": s.name, "bytes": s.stat().st_size, "sha256": _sha256(s)}
                         for s in segments],
        "max_snapshots_cap": max_snapshots or None,
    }
    sidecar = writer.finish(extra)
    report = validate(out_dir)
    sidecar["validation"] = report
    if not quiet:
        print(json.dumps({k: sidecar[k] for k in (
            "source", "n_snapshots", "n_trades", "per_product_snapshots", "capture_seconds",
            "gap_count", "content_hash")}, indent=2))
        print(f"built {out_dir} in {time.perf_counter() - t0:.1f}s; validation: "
              f"{'ok' if not report['problems'] else report['problems']}")
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the snapshot archive from the raw log")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/archive")
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS))
    parser.add_argument("--max-snapshots", type=int, default=0, help="stop after this many (0 = all)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    build(Path(args.raw), Path(args.out), args.products, args.max_snapshots, args.quiet)


if __name__ == "__main__":
    main()
