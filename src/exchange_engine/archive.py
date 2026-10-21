"""Snapshot archive: tick encoding, snapshot gating, and the binary SoA format.

The GPU analytics layer never touches live data. Instead the recorder writes a raw
message log, and ``scripts/build_archive.py`` replays it through
:class:`SnapshotGate` to produce a deterministic archive of top-50 book snapshots.

Archive layout (one directory, little-endian, level-major / structure-of-arrays):

``bid_px.i32``  int32 ticks, shape ``(LEVELS, n)`` so ``bid_px[level * n + i]`` is
                level ``level`` of snapshot ``i``; level 0 is the best bid.
``bid_sz.f32``  float32 sizes, same layout. ``ask_px.i32`` / ``ask_sz.f32`` likewise
                (level 0 is the best ask). Missing levels are ``px = 0, sz = 0``.
``snap_ts.i64`` int64 local receive timestamp in nanoseconds per snapshot.
``snap_product.u8`` uint8 product index into ``products`` in the sidecar.
``trade_*``     structure-of-arrays trade file (``ts.i64``, ``exch_ts.i64``,
                ``product.u8``, ``px.i32``, ``sz.f32``, ``side.i8``).
``archive.json`` sidecar with counts, tick sizes, capture window, gap count,
                ``source`` ("real" or "synthetic") and ``content_hash``.

Prices are stored as integer tick counts rather than float32: near $100,000 a
float32 has a spacing of 2**-7 = $0.0078, most of the $0.01 tick, so a float32
spread would be mostly rounding error (``tests/test_archive.py`` demonstrates it).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .feed import ProductBook

LEVELS = 50
MIN_SNAPSHOT_INTERVAL_NS = 10_000_000  # at most one snapshot per product per 10 ms
SCHEMA_VERSION = 1
PRODUCTS: Tuple[str, ...] = ("BTC-USD", "ETH-USD")
TICK_SIZES: Dict[str, Decimal] = {"BTC-USD": Decimal("0.01"), "ETH-USD": Decimal("0.01")}

SNAPSHOT_ARRAYS: Dict[str, str] = {
    # name -> numpy dtype string (little endian)
    "snap_ts": "<i8",
    "snap_product": "u1",
    "bid_px": "<i4",
    "bid_sz": "<f4",
    "ask_px": "<i4",
    "ask_sz": "<f4",
}
TRADE_ARRAYS: Dict[str, str] = {
    "trade_ts": "<i8",
    "trade_exch_ts": "<i8",
    "trade_product": "u1",
    "trade_px": "<i4",
    "trade_sz": "<f4",
    "trade_side": "i1",
}
LEVEL_MAJOR = ("bid_px", "bid_sz", "ask_px", "ask_sz")
FILE_SUFFIX = {"<i8": "i64", "u1": "u8", "<i4": "i32", "<f4": "f32", "i1": "i8"}
ALL_ARRAYS: Dict[str, str] = {**SNAPSHOT_ARRAYS, **TRADE_ARRAYS}


def array_filename(name: str) -> str:
    """Return the on-disk file name for an archive array."""
    return f"{name}.{FILE_SUFFIX[ALL_ARRAYS[name]]}"


def to_ticks(price: Decimal, tick: Decimal) -> int:
    """Convert an exact decimal price to an integer tick count.

    Raises:
        ValueError: If ``price`` is not an exact multiple of ``tick``.
    """
    quotient = price / tick
    ticks = int(quotient)
    if quotient != ticks:
        raise ValueError(f"price {price} is not a multiple of tick {tick}")
    return ticks


def iso_to_ns(stamp: str) -> int:
    """Parse an ISO-8601 UTC timestamp with up to 9 fractional digits to epoch ns."""
    from datetime import datetime, timezone

    s = stamp.strip()
    if s.endswith("Z"):
        s = s[:-1]
    if "+" in s[10:]:
        s = s[: 10 + s[10:].index("+")]
    frac_ns = 0
    if "." in s:
        s, frac = s.split(".", 1)
        frac = (frac + "000000000")[:9]
        frac_ns = int(frac)
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) * 1_000_000_000 + frac_ns


class SnapshotGate:
    """Decide, deterministically, when a product's book becomes a snapshot.

    A snapshot is emitted when (a) a level2 snapshot has been seen for the
    product, (b) at least :data:`MIN_SNAPSHOT_INTERVAL_NS` elapsed on the local
    receive clock since the previous emission, and (c) something in the top
    :data:`LEVELS` levels of either side differs from the previously emitted
    snapshot. Crossed tops are counted and skipped. The same class drives the
    recorder's live count and the archive builder, so the two agree exactly.
    """

    def __init__(
        self, levels: int = LEVELS, min_interval_ns: int = MIN_SNAPSHOT_INTERVAL_NS
    ) -> None:
        self.levels = levels
        self.min_interval_ns = min_interval_ns
        self.book = ProductBook()
        self.ready = False
        self.last_emit_ns: Optional[int] = None
        self.last_top: Optional[Tuple[tuple, tuple]] = None
        self.count = 0
        self.crossed = 0

    def top(self) -> Tuple[Tuple[Tuple[Decimal, Decimal], ...], Tuple[Tuple[Decimal, Decimal], ...]]:
        """Return ``(bids, asks)`` as tuples of ``(price, qty)``, best first."""
        bids: List[Tuple[Decimal, Decimal]] = []
        for key, qty in self.book.bids.items():
            if len(bids) >= self.levels:
                break
            bids.append((-key, qty))
        asks: List[Tuple[Decimal, Decimal]] = []
        for key, qty in self.book.asks.items():
            if len(asks) >= self.levels:
                break
            asks.append((key, qty))
        return tuple(bids), tuple(asks)

    def apply_event(self, event: dict, recv_ns: int) -> bool:
        """Apply one ``l2_data`` event; return ``True`` if a snapshot should be emitted."""
        if event.get("type") == "snapshot":
            self.book.clear()
            self.ready = True
        for update in event.get("updates", []):
            self.book.set_level(
                update["side"], Decimal(update["price_level"]), Decimal(update["new_quantity"])
            )
        if not self.ready:
            return False
        if self.last_emit_ns is not None and recv_ns - self.last_emit_ns < self.min_interval_ns:
            return False
        top = self.top()
        if top == self.last_top:
            return False
        bids, asks = top
        if bids and asks and bids[0][0] >= asks[0][0]:
            self.crossed += 1
            return False
        self.last_top = top
        self.last_emit_ns = recv_ns
        self.count += 1
        return True


@dataclass
class ArchiveInfo:
    """Parsed ``archive.json`` sidecar."""

    path: Path
    source: str
    n_snapshots: int
    n_trades: int
    levels: int
    products: List[str]
    tick_sizes: Dict[str, str]
    content_hash: str
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, directory: Path) -> "ArchiveInfo":
        directory = Path(directory)
        meta = json.loads((directory / "archive.json").read_text(encoding="utf-8"))
        return cls(
            path=directory,
            source=meta["source"],
            n_snapshots=int(meta["n_snapshots"]),
            n_trades=int(meta["n_trades"]),
            levels=int(meta["levels"]),
            products=list(meta["products"]),
            tick_sizes=dict(meta["tick_sizes"]),
            content_hash=str(meta["content_hash"]),
            extra=meta,
        )


def _np():
    import numpy as np  # local import keeps the core package importable without numpy

    return np


def load_arrays(
    directory: Path, names: Optional[Iterable[str]] = None, mmap: bool = False
) -> dict:
    """Load archive arrays into numpy; level-major arrays come back as ``(levels, n)``."""
    np = _np()
    directory = Path(directory)
    info = ArchiveInfo.load(directory)
    out: dict = {}
    wanted = list(names) if names is not None else list(ALL_ARRAYS)
    for name in wanted:
        dtype = ALL_ARRAYS[name]
        path = directory / array_filename(name)
        if mmap:
            arr = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        else:
            arr = np.fromfile(path, dtype=np.dtype(dtype))
        if name in LEVEL_MAJOR:
            arr = arr.reshape(info.levels, info.n_snapshots)
        out[name] = arr
    out["info"] = info
    return out


def content_hash(directory: Path, names: Optional[Sequence[str]] = None) -> str:
    """SHA-256 over every array file, in a fixed order, so two builds can be compared."""
    directory = Path(directory)
    h = hashlib.sha256()
    for name in names or list(ALL_ARRAYS):
        path = directory / array_filename(name)
        h.update(name.encode())
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()


class ArchiveWriter:
    """Incrementally write an archive directory in the level-major format.

    Snapshots are appended in snapshot-major chunks to temporary files and
    transposed to level-major on :meth:`finish`, so memory stays bounded.
    """

    def __init__(
        self,
        directory: Path,
        source: str,
        products: Sequence[str] = PRODUCTS,
        tick_sizes: Optional[Dict[str, Decimal]] = None,
        levels: int = LEVELS,
    ) -> None:
        if source not in ("real", "synthetic"):
            raise ValueError("source must be 'real' or 'synthetic'")
        self.np = _np()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.products = list(products)
        self.tick_sizes = {p: (tick_sizes or TICK_SIZES)[p] for p in self.products}
        self.levels = levels
        self._tmp = {name: open(self.directory / f".{name}.tmp", "wb") for name in SNAPSHOT_ARRAYS}
        self._trade_fh = {
            name: open(self.directory / array_filename(name), "wb") for name in TRADE_ARRAYS
        }
        self.n_snapshots = 0
        self.n_trades = 0
        self.per_product: Dict[str, int] = {p: 0 for p in self.products}
        self.first_ts: Optional[int] = None
        self.last_ts: Optional[int] = None
        self._buf: Dict[str, list] = {name: [] for name in SNAPSHOT_ARRAYS}
        self._tbuf: Dict[str, list] = {name: [] for name in TRADE_ARRAYS}
        self.chunk = 16384

    def add_snapshot(
        self,
        recv_ns: int,
        product_idx: int,
        bids: Sequence[Tuple[int, float]],
        asks: Sequence[Tuple[int, float]],
    ) -> None:
        """Append one snapshot (``bids``/``asks`` are ``(ticks, size)`` best first)."""
        L = self.levels
        bpx = [0] * L
        bsz = [0.0] * L
        apx = [0] * L
        asz = [0.0] * L
        for i, (px, sz) in enumerate(bids[:L]):
            bpx[i] = px
            bsz[i] = sz
        for i, (px, sz) in enumerate(asks[:L]):
            apx[i] = px
            asz[i] = sz
        self._buf["snap_ts"].append(recv_ns)
        self._buf["snap_product"].append(product_idx)
        self._buf["bid_px"].append(bpx)
        self._buf["bid_sz"].append(bsz)
        self._buf["ask_px"].append(apx)
        self._buf["ask_sz"].append(asz)
        self.n_snapshots += 1
        self.per_product[self.products[product_idx]] += 1
        if self.first_ts is None:
            self.first_ts = recv_ns
        self.last_ts = recv_ns
        if len(self._buf["snap_ts"]) >= self.chunk:
            self._flush_snapshots()

    def add_trade(
        self,
        recv_ns: int,
        exch_ns: int,
        product_idx: int,
        px_ticks: int,
        size: float,
        maker_side: str,
    ) -> None:
        """Append one trade; ``maker_side`` is Coinbase's ``BUY``/``SELL`` maker side."""
        self._tbuf["trade_ts"].append(recv_ns)
        self._tbuf["trade_exch_ts"].append(exch_ns)
        self._tbuf["trade_product"].append(product_idx)
        self._tbuf["trade_px"].append(px_ticks)
        self._tbuf["trade_sz"].append(size)
        self._tbuf["trade_side"].append(1 if maker_side.upper() == "BUY" else -1)
        self.n_trades += 1
        if len(self._tbuf["trade_ts"]) >= self.chunk:
            self._flush_trades()

    def add_snapshot_batch(self, snap_ts, snap_product, bid_px, bid_sz, ask_px, ask_sz) -> None:
        """Append a block of snapshots given snapshot-major ``(n, levels)`` arrays."""
        np = self.np
        self._flush_snapshots()
        n = int(len(snap_ts))
        if n == 0:
            return
        arrays = {
            "snap_ts": np.asarray(snap_ts, dtype=np.dtype("<i8")),
            "snap_product": np.asarray(snap_product, dtype=np.dtype("u1")),
            "bid_px": np.asarray(bid_px, dtype=np.dtype("<i4")),
            "bid_sz": np.asarray(bid_sz, dtype=np.dtype("<f4")),
            "ask_px": np.asarray(ask_px, dtype=np.dtype("<i4")),
            "ask_sz": np.asarray(ask_sz, dtype=np.dtype("<f4")),
        }
        for name in LEVEL_MAJOR:
            if arrays[name].shape != (n, self.levels):
                raise ValueError(f"{name} must have shape ({n}, {self.levels})")
        for name, arr in arrays.items():
            np.ascontiguousarray(arr).tofile(self._tmp[name])
        prods = arrays["snap_product"]
        for idx, product in enumerate(self.products):
            self.per_product[product] += int(np.count_nonzero(prods == idx))
        self.n_snapshots += n
        if self.first_ts is None:
            self.first_ts = int(arrays["snap_ts"][0])
        self.last_ts = int(arrays["snap_ts"][-1])

    def add_trade_batch(self, trade_ts, trade_exch_ts, trade_product, trade_px, trade_sz, trade_side) -> None:
        """Append a block of trades given one-dimensional arrays."""
        np = self.np
        self._flush_trades()
        cols = {
            "trade_ts": trade_ts, "trade_exch_ts": trade_exch_ts, "trade_product": trade_product,
            "trade_px": trade_px, "trade_sz": trade_sz, "trade_side": trade_side,
        }
        n = int(len(trade_ts))
        for name, dtype in TRADE_ARRAYS.items():
            np.ascontiguousarray(np.asarray(cols[name], dtype=np.dtype(dtype))).tofile(self._trade_fh[name])
        self.n_trades += n

    def _flush_snapshots(self) -> None:
        np = self.np
        for name, dtype in SNAPSHOT_ARRAYS.items():
            if not self._buf[name]:
                continue
            np.asarray(self._buf[name], dtype=np.dtype(dtype)).tofile(self._tmp[name])
            self._buf[name] = []

    def _flush_trades(self) -> None:
        np = self.np
        for name, dtype in TRADE_ARRAYS.items():
            if not self._tbuf[name]:
                continue
            np.asarray(self._tbuf[name], dtype=np.dtype(dtype)).tofile(self._trade_fh[name])
            self._tbuf[name] = []

    def finish(self, extra: Optional[dict] = None) -> dict:
        """Transpose to level-major, write the sidecar and return it."""
        np = self.np
        self._flush_snapshots()
        self._flush_trades()
        for fh in self._tmp.values():
            fh.close()
        for fh in self._trade_fh.values():
            fh.close()
        n = self.n_snapshots
        for name, dtype in SNAPSHOT_ARRAYS.items():
            tmp = self.directory / f".{name}.tmp"
            arr = np.fromfile(tmp, dtype=np.dtype(dtype))
            if name in LEVEL_MAJOR:
                arr = np.ascontiguousarray(arr.reshape(n, self.levels).T)
            arr.tofile(self.directory / array_filename(name))
            del arr
            os.remove(tmp)
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "source": self.source,
            "layout": "level_major",
            "levels": self.levels,
            "n_snapshots": n,
            "n_trades": self.n_trades,
            "products": self.products,
            "tick_sizes": {p: str(t) for p, t in self.tick_sizes.items()},
            "per_product_snapshots": self.per_product,
            "capture_start_ns": self.first_ts,
            "capture_end_ns": self.last_ts,
            "capture_seconds": (
                None if self.first_ts is None else (self.last_ts - self.first_ts) / 1e9
            ),
            "arrays": {
                name: {
                    "file": array_filename(name),
                    "dtype": dtype,
                    "shape": [self.levels, n] if name in LEVEL_MAJOR else [n],
                }
                for name, dtype in SNAPSHOT_ARRAYS.items()
            },
            "trade_arrays": {
                name: {"file": array_filename(name), "dtype": dtype, "shape": [self.n_trades]}
                for name, dtype in TRADE_ARRAYS.items()
            },
            "bytes": sum(
                (self.directory / array_filename(n_)).stat().st_size for n_ in ALL_ARRAYS
            ),
        }
        if extra:
            sidecar.update(extra)
        sidecar["content_hash"] = content_hash(self.directory)
        (self.directory / "archive.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return sidecar


def validate(directory: Path) -> dict:
    """Check archive invariants; return a report dict and raise on violation."""
    np = _np()
    data = load_arrays(directory, mmap=True)
    info: ArchiveInfo = data["info"]
    bid_px, ask_px = data["bid_px"], data["ask_px"]
    bid_sz, ask_sz = data["bid_sz"], data["ask_sz"]
    ts, prod = data["snap_ts"], data["snap_product"]
    report: dict = {"n_snapshots": info.n_snapshots, "n_trades": info.n_trades, "problems": []}
    if info.n_snapshots:
        both = (bid_sz[0] > 0) & (ask_sz[0] > 0)
        crossed = int(np.count_nonzero(bid_px[0][both] >= ask_px[0][both]))
        if crossed:
            report["problems"].append(f"{crossed} crossed books")
        for idx in range(len(info.products)):
            t = ts[prod == idx]
            if t.size > 1 and not bool(np.all(np.diff(t) > 0)):
                report["problems"].append(f"timestamps not strictly increasing for product {idx}")
        for lvl in range(1, info.levels):
            m = bid_sz[lvl] > 0
            if m.any() and not bool(np.all(bid_px[lvl][m] < bid_px[lvl - 1][m])):
                report["problems"].append(f"bid level {lvl} not strictly below level {lvl - 1}")
                break
        for lvl in range(1, info.levels):
            m = ask_sz[lvl] > 0
            if m.any() and not bool(np.all(ask_px[lvl][m] > ask_px[lvl - 1][m])):
                report["problems"].append(f"ask level {lvl} not strictly above level {lvl - 1}")
                break
    if content_hash(directory) != info.content_hash:
        report["problems"].append("content_hash mismatch")
    if report["problems"]:
        raise ValueError("archive validation failed: " + "; ".join(report["problems"]))
    return report


def iter_raw_log(paths: Iterable[Path]) -> Iterator[dict]:
    """Yield decoded JSON lines from zstd-compressed raw log segments, in order.

    A segment cut short by a crash is read up to its last complete frame; the
    truncated tail is skipped.
    """
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    for path in paths:
        with open(path, "rb") as fh:
            reader = dctx.stream_reader(fh, read_across_frames=True)
            pending = b""
            while True:
                try:
                    chunk = reader.read(1 << 20)
                except zstd.ZstdError:
                    break
                if not chunk:
                    break
                pending += chunk
                *lines, pending = pending.split(b"\n")
                for line in lines:
                    if line:
                        yield json.loads(line)
