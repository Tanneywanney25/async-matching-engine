"""scripts/benchmark_native.py replays exactly scripts/benchmark.py's stream (CPU only, no build needed)."""

from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import benchmark  # noqa: E402
import benchmark_native as bn  # noqa: E402


def test_stream_is_benchmark_py_order_for_order(monkeypatch) -> None:
    seen = []

    class Recording(benchmark.OrderBook):
        def add_order(self, order):
            seen.append((order.side, order.price, order.quantity))
            return super().add_order(order)

    monkeypatch.setattr(benchmark, "OrderBook", Recording)
    benchmark.benchmark_inserts(500, 64_000.0, 42)
    stream = bn.synthetic_stream(500, 50, 64_000.0, 42)
    assert list(zip(stream["sides"], stream["prices"], stream["qtys"])) == seen
    assert len(stream["cancels"]) == 50 and len(set(stream["cancels"])) == 50


def test_percentiles_match_benchmark_py() -> None:
    xs = sorted(random.Random(3).random() for _ in range(1001))
    for pct in (50, 95, 99):
        assert bn.percentile(xs, pct) == benchmark._percentile(xs, pct)


def test_stream_file_layout(tmp_path: Path) -> None:
    stream = bn.synthetic_stream(10, 3, 64_000.0, 42)
    path = tmp_path / "s.bin"
    bn.write_stream(path, stream)
    data = path.read_bytes()
    magic, (version, _pad, n_adds, n_cancels) = data[:8], struct.unpack("<IIQQ", data[8:32])
    assert magic == b"EESTREAM" and version == 1 and (n_adds, n_cancels) == (10, 3)
    oid, ticks, lots, side = struct.unpack("<qqqB7x", data[32:64])
    assert oid == 1 and ticks * bn.TICK == stream["prices"][0] and lots * bn.LOT == stream["qtys"][0]
    assert side == (0 if stream["sides"][0].value == "bid" else 1)
    assert len(data) == 32 + 32 * 10 + 8 * 3
