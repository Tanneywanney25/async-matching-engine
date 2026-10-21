"""Raw Coinbase market-data recorder.

Subscribes to ``level2`` and ``market_trades`` for the configured products through
the existing :class:`~exchange_engine.feed.CoinbaseFeed` (which already handles
the per-channel subscribe frames, ``l2_data`` absolute quantities, heartbeats and
exponential-backoff reconnects) and appends every raw frame, untouched, to an
append-only zstd-compressed JSON-lines segment in ``data/raw/``::

    {"recv_ns": 1760000000000000000, "msg": {...original frame...}}

Control lines (no ``msg`` key) mark connections and sequence gaps::

    {"recv_ns": ..., "connect": 3}
    {"recv_ns": ..., "gap": {"expected": 120, "got": 125}}

Coinbase's ``sequence_num`` is per connection and increments by one for every
frame on every channel. On a gap the recorder records it, resubscribes to
``level2`` for a fresh snapshot, and counts the gap in the sidecar.

A sidecar ``data/raw/recorder_state.json`` is rewritten every few seconds with
counts, the exact snapshot count the archive builder will produce (the same
:class:`~exchange_engine.archive.SnapshotGate` runs live), and liveness. The
process stops gracefully when ``data/recorder.stop`` appears, when the deadline
passes, or when the snapshot target (plus margin) is reached.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from .archive import LEVELS, MIN_SNAPSHOT_INTERVAL_NS, PRODUCTS, SnapshotGate
from .feed import CoinbaseFeed

logger = logging.getLogger("exchange_engine.recorder")

STALE_SECONDS = 30.0  # force a reconnect if the socket goes silent this long
FLUSH_SECONDS = 1.0
SIDECAR_SECONDS = 5.0


class RawLogWriter:
    """Append-only, zstd-compressed JSON-lines writer with frame-level flushes.

    Every :meth:`flush` ends the current zstd frame, so a crash loses at most the
    data written since the last flush and the file stays readable.
    """

    def __init__(self, raw_dir: Path, level: int = 3) -> None:
        import zstandard as zstd

        raw_dir.mkdir(parents=True, exist_ok=True)
        self.path = raw_dir / f"segment-{time.time_ns()}.jsonl.zst"
        self._fh = open(self.path, "ab")
        self._writer = zstd.ZstdCompressor(level=level).stream_writer(self._fh, closefd=False)
        self._flush_mode = zstd.FLUSH_FRAME
        self.lines = 0
        self.bytes_in = 0
        self._dirty = False

    def write_line(self, obj_text: str) -> None:
        data = (obj_text + "\n").encode("utf-8")
        self._writer.write(data)
        self.lines += 1
        self.bytes_in += len(data)
        self._dirty = True

    def write_frame(self, recv_ns: int, raw: str) -> None:
        """Record one raw frame verbatim (``raw`` is already a JSON object)."""
        self.write_line(f'{{"recv_ns":{recv_ns},"msg":{raw}}}')

    def write_control(self, recv_ns: int, key: str, value) -> None:
        self.write_line(json.dumps({"recv_ns": recv_ns, key: value}, separators=(",", ":")))

    def flush(self) -> None:
        if self._dirty:
            self._writer.flush(self._flush_mode)
            self._fh.flush()
            self._dirty = False

    def close(self) -> None:
        self.flush()
        self._writer.close()
        self._fh.close()

    @property
    def bytes_out(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0


class RecordingFeed(CoinbaseFeed):
    """A :class:`CoinbaseFeed` that records raw frames and tracks sequence numbers."""

    def __init__(self, recorder: "Recorder", product_ids: List[str]) -> None:
        super().__init__(product_ids)
        self.recorder = recorder
        self._ws = None
        self._expected_seq: Optional[int] = None

    async def _subscribe(self, ws) -> None:  # type: ignore[override]
        self._ws = ws
        self._expected_seq = None
        self.recorder.on_connect()
        await super()._subscribe(ws)

    async def resubscribe_level2(self) -> None:
        """Ask for a fresh level2 snapshot on the live connection after a gap."""
        if self._ws is None:
            return
        for msg_type in ("unsubscribe", "subscribe"):
            await self._ws.send(
                json.dumps({"type": msg_type, "product_ids": self.product_ids, "channel": "level2"})
            )

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 - best effort
                pass

    async def _handle_message(self, raw: str) -> None:  # type: ignore[override]
        recv_ns = time.time_ns()
        self.recorder.on_frame(recv_ns, raw)
        msg = json.loads(raw)
        seq = msg.get("sequence_num")
        if seq is not None:
            if self._expected_seq is not None and seq != self._expected_seq:
                self.recorder.on_gap(recv_ns, self._expected_seq, seq)
                await self.resubscribe_level2()
            self._expected_seq = seq + 1
        self.recorder.on_message(recv_ns, msg)


class Recorder:
    """Owns the raw log, the live snapshot gates and the sidecar."""

    def __init__(
        self,
        data_dir: Path,
        product_ids: List[str] = list(PRODUCTS),
        target_snapshots: int = 1_200_000,
        margin: float = 0.05,
        max_hours: float = 10.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.product_ids = list(product_ids)
        self.target_snapshots = target_snapshots
        self.margin = margin
        self.max_hours = max_hours
        self.stop_file = self.data_dir / "recorder.stop"
        self.pid_file = self.data_dir / "recorder.pid"
        self.sidecar_path = self.raw_dir / "recorder_state.json"
        self.writer: Optional[RawLogWriter] = None
        self.feed = RecordingFeed(self, self.product_ids)
        self.gates: Dict[str, SnapshotGate] = {p: SnapshotGate() for p in self.product_ids}
        self.started_ns = time.time_ns()
        self.last_frame_ns = 0
        self.frames = 0
        self.l2_messages: Dict[str, int] = {p: 0 for p in self.product_ids}
        self.l2_snapshots: Dict[str, int] = {p: 0 for p in self.product_ids}
        self.trades: Dict[str, int] = {p: 0 for p in self.product_ids}
        self.heartbeats = 0
        self.gaps: List[dict] = []
        self.connects = 0
        self.stop_reason: Optional[str] = None
        self.segments: List[str] = []
        self._prior = self._load_prior_state()

    # -- prior state (restarts append) --------------------------------------

    def _load_prior_state(self) -> dict:
        try:
            prior = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return prior if isinstance(prior, dict) else {}

    @property
    def first_start_ns(self) -> int:
        return int(self._prior.get("first_start_ns") or self.started_ns)

    def prior_snapshot_count(self) -> int:
        return int(self._prior.get("snapshots_total_prior_runs", 0)) + int(
            self._prior.get("snapshots_this_run", 0)
        )

    # -- feed callbacks ------------------------------------------------------

    def on_connect(self) -> None:
        self.connects += 1
        for gate in self.gates.values():
            gate.ready = False  # a fresh connection re-snapshots every product
        if self.writer is not None:
            self.writer.write_control(time.time_ns(), "connect", self.connects)
        logger.info("connected (%d this run)", self.connects)

    def on_frame(self, recv_ns: int, raw: str) -> None:
        self.frames += 1
        self.last_frame_ns = recv_ns
        if self.writer is not None:
            self.writer.write_frame(recv_ns, raw)

    def on_gap(self, recv_ns: int, expected: int, got: int) -> None:
        self.gaps.append({"recv_ns": recv_ns, "expected": expected, "got": got})
        if self.writer is not None:
            self.writer.write_control(recv_ns, "gap", {"expected": expected, "got": got})
        logger.warning("sequence gap: expected %d got %d; resubscribing level2", expected, got)

    def on_message(self, recv_ns: int, msg: dict) -> None:
        channel = msg.get("channel")
        if channel == "l2_data":
            for event in msg.get("events", []):
                pid = event.get("product_id")
                gate = self.gates.get(pid)
                if gate is None:
                    continue
                self.l2_messages[pid] += 1
                if event.get("type") == "snapshot":
                    self.l2_snapshots[pid] += 1
                gate.apply_event(event, recv_ns)
        elif channel == "market_trades":
            for event in msg.get("events", []):
                for trade in event.get("trades", []):
                    pid = trade.get("product_id")
                    if pid in self.trades:
                        self.trades[pid] += 1
        elif channel == "heartbeats":
            self.heartbeats += 1

    # -- state ---------------------------------------------------------------

    def snapshots_this_run(self) -> int:
        return sum(g.count for g in self.gates.values())

    def snapshots_total(self) -> int:
        return int(self._prior.get("snapshots_total_prior_runs", 0)) + int(
            self._prior.get("snapshots_this_run", 0)
        ) + self.snapshots_this_run()

    def state(self, alive: bool) -> dict:
        now = time.time_ns()
        this_run = self.snapshots_this_run()
        return {
            "alive": alive,
            "pid": os.getpid(),
            "stop_reason": self.stop_reason,
            "first_start_ns": self.first_start_ns,
            "started_ns": self.started_ns,
            "updated_ns": now,
            "uptime_seconds": (now - self.started_ns) / 1e9,
            "hours_since_first_start": (now - self.first_start_ns) / 3.6e12,
            "deadline_hours": self.max_hours,
            "target_snapshots": self.target_snapshots,
            "target_with_margin": int(self.target_snapshots * (1 + self.margin)),
            "snapshots_total_prior_runs": int(self._prior.get("snapshots_total_prior_runs", 0))
            + int(self._prior.get("snapshots_this_run", 0)),
            "snapshots_this_run": this_run,
            "snapshots_total": self.snapshots_total(),
            "snapshots_per_product_this_run": {p: g.count for p, g in self.gates.items()},
            "crossed_skipped_this_run": {p: g.crossed for p, g in self.gates.items()},
            "snapshot_rule": {"levels": LEVELS, "min_interval_ms": MIN_SNAPSHOT_INTERVAL_NS / 1e6},
            "frames_this_run": self.frames,
            "l2_messages_this_run": self.l2_messages,
            "l2_snapshots_this_run": self.l2_snapshots,
            "trades_this_run": self.trades,
            "heartbeats_this_run": self.heartbeats,
            "gaps_this_run": len(self.gaps),
            "gaps_total": int(self._prior.get("gaps_total", 0)) - int(self._prior.get("gaps_this_run", 0))
            + int(self._prior.get("gaps_this_run", 0)) + len(self.gaps)
            if self._prior else len(self.gaps),
            "gap_events_this_run": self.gaps[-20:],
            "connects_this_run": self.connects,
            "last_frame_ns": self.last_frame_ns,
            "seconds_since_last_frame": (now - self.last_frame_ns) / 1e9 if self.last_frame_ns else None,
            "segment": str(self.writer.path.name) if self.writer else None,
            "segments": self.segments,
            "bytes_compressed_this_run": self.writer.bytes_out if self.writer else 0,
            "bytes_raw_this_run": self.writer.bytes_in if self.writer else 0,
            "products": self.product_ids,
        }

    def write_sidecar(self, alive: bool) -> None:
        tmp = self.sidecar_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state(alive), indent=2), encoding="utf-8")
        os.replace(tmp, self.sidecar_path)

    def should_stop(self) -> Optional[str]:
        if self.stop_file.exists():
            return "stop file"
        if (time.time_ns() - self.first_start_ns) / 3.6e12 >= self.max_hours:
            return f"deadline of {self.max_hours} hours reached"
        if self.snapshots_total() >= self.target_snapshots * (1 + self.margin):
            return "snapshot target plus margin reached"
        return None

    # -- main loop -----------------------------------------------------------

    async def run(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.writer = RawLogWriter(self.raw_dir)
        self.segments = list(self._prior.get("segments", [])) + [self.writer.path.name]
        self.pid_file.write_text(str(os.getpid()), encoding="utf-8")
        if self.stop_file.exists():
            self.stop_file.unlink()
        self.write_sidecar(alive=True)
        logger.info(
            "recording %s to %s (target %d snapshots +%.0f%%, deadline %.1f h)",
            self.product_ids, self.writer.path, self.target_snapshots, self.margin * 100,
            self.max_hours,
        )
        feed_task = asyncio.create_task(self.feed.run())
        last_flush = last_sidecar = time.monotonic()
        try:
            while True:
                await asyncio.sleep(0.25)
                now = time.monotonic()
                if now - last_flush >= FLUSH_SECONDS:
                    self.writer.flush()
                    last_flush = now
                if now - last_sidecar >= SIDECAR_SECONDS:
                    self.write_sidecar(alive=True)
                    last_sidecar = now
                    if self.last_frame_ns and (time.time_ns() - self.last_frame_ns) / 1e9 > STALE_SECONDS:
                        logger.warning("no frames for %.0fs; forcing reconnect", STALE_SECONDS)
                        self.last_frame_ns = time.time_ns()
                        await self.feed.close()
                reason = self.should_stop()
                if reason:
                    self.stop_reason = reason
                    logger.info("stopping: %s", reason)
                    break
                if feed_task.done():
                    exc = feed_task.exception()
                    self.stop_reason = f"feed task exited: {exc!r}"
                    logger.error(self.stop_reason)
                    break
        finally:
            await self.feed.stop()
            feed_task.cancel()
            try:
                await feed_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self.writer.close()
            self.write_sidecar(alive=False)
            logger.info("recorder stopped: %s", self.stop_reason)
