"""Tests for the snapshot archive: tick encoding, gating, format, determinism."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

import build_archive  # scripts/ is on sys.path via conftest
import synth_archive
from exchange_engine.archive import (
    LEVELS,
    ArchiveInfo,
    SnapshotGate,
    array_filename,
    content_hash,
    iso_to_ns,
    iter_raw_log,
    load_arrays,
    to_ticks,
    validate,
)
from exchange_engine.recorder import RawLogWriter

TICK = Decimal("0.01")


def test_float32_prices_cannot_represent_a_one_cent_spread_near_100k() -> None:
    """float32 spacing at $100,000 is 2**-7 = $0.0078, most of the $0.01 tick."""
    bid, ask = Decimal("100000.00"), Decimal("100000.01")
    f32_spread = float(np.float32(ask) - np.float32(bid))
    assert f32_spread != pytest.approx(0.01, rel=1e-3)
    assert abs(f32_spread - 0.01) / 0.01 > 0.2  # more than 20% rounding error
    assert np.spacing(np.float32(100000.0)) == 2.0**-7
    # The same prices as int32 ticks are exact.
    assert to_ticks(ask, TICK) - to_ticks(bid, TICK) == 1
    assert to_ticks(Decimal("110234.56"), TICK) == 11023456
    with pytest.raises(ValueError):
        to_ticks(Decimal("100000.005"), TICK)


def test_iso_to_ns_handles_nine_fractional_digits() -> None:
    assert iso_to_ns("2023-02-09T20:32:50.714964855Z") == 1675974770714964855
    assert iso_to_ns("2023-02-09T20:32:50Z") == 1675974770000000000
    assert iso_to_ns("2023-02-09T20:32:50.5Z") == 1675974770500000000


def _snapshot_event(product: str = "BTC-USD") -> dict:
    return {
        "type": "snapshot",
        "product_id": product,
        "updates": [
            {"side": "bid", "price_level": "100.00", "new_quantity": "1"},
            {"side": "bid", "price_level": "99.99", "new_quantity": "2"},
            {"side": "offer", "price_level": "100.01", "new_quantity": "3"},
            {"side": "offer", "price_level": "100.02", "new_quantity": "4"},
        ],
    }


def _update(side: str, price: str, qty: str) -> dict:
    return {"type": "update", "updates": [{"side": side, "price_level": price, "new_quantity": qty}]}


def test_snapshot_gate_rules() -> None:
    gate = SnapshotGate()
    t0 = 1_000_000_000_000
    # Updates before the first snapshot are ignored.
    assert gate.apply_event(_update("bid", "100.00", "9"), t0) is False
    assert gate.apply_event(_snapshot_event(), t0) is True
    # A change inside the 10 ms window is deferred, not dropped.
    assert gate.apply_event(_update("bid", "99.98", "5"), t0 + 5_000_000) is False
    assert gate.apply_event(_update("bid", "50.00", "1"), t0 + 20_000_000) is True
    # Nothing changed in the top 50: no snapshot.
    assert gate.apply_event({"type": "update", "updates": []}, t0 + 40_000_000) is False
    # A crossed top of book is skipped and counted.
    assert gate.apply_event(_update("bid", "100.05", "1"), t0 + 60_000_000) is False
    assert gate.crossed == 1
    assert gate.count == 2


def _write_raw_log(raw_dir: Path) -> Path:
    writer = RawLogWriter(raw_dir)
    t = 1_700_000_000_000_000_000
    writer.write_control(t, "connect", 1)
    writer.write_frame(t + 1, json.dumps({"channel": "subscriptions", "sequence_num": 0}))
    writer.write_frame(
        t + 2, json.dumps({"channel": "l2_data", "sequence_num": 1, "events": [_snapshot_event()]})
    )
    writer.write_frame(
        t + 3,
        json.dumps({"channel": "l2_data", "sequence_num": 2, "events": [_snapshot_event("ETH-USD")]}),
    )
    for i in range(1, 40):
        step = t + 20_000_000 * i
        price = f"{99.99 - 0.01 * (i % 5):.2f}"
        writer.write_frame(
            step,
            json.dumps(
                {
                    "channel": "l2_data",
                    "sequence_num": 2 + i,
                    "events": [dict(_update("bid", price, str(i % 3)), product_id="BTC-USD")],
                }
            ),
        )
    trades = [
        {"trade_id": "t1", "product_id": "BTC-USD", "price": "100.00", "size": "0.5",
         "side": "BUY", "time": "2023-11-14T22:13:20.123456789Z"},
        {"trade_id": "t2", "product_id": "ETH-USD", "price": "100.01", "size": "0.25",
         "side": "SELL", "time": "2023-11-14T22:13:21Z"},
        {"trade_id": "t1", "product_id": "BTC-USD", "price": "100.00", "size": "0.5",
         "side": "BUY", "time": "2023-11-14T22:13:20.123456789Z"},  # duplicate (resubscribe)
    ]
    writer.write_frame(
        t + 900_000_000,
        json.dumps({"channel": "market_trades", "sequence_num": 50,
                    "events": [{"type": "update", "trades": trades}]}),
    )
    writer.write_control(t + 950_000_000, "gap", {"expected": 51, "got": 53})
    writer.flush()
    writer.close()
    return writer.path


def test_build_archive_is_deterministic_and_validates(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_log(raw_dir)
    lines = list(iter_raw_log(sorted(raw_dir.glob("segment-*.jsonl.zst"))))
    assert lines[0]["connect"] == 1 and "gap" in lines[-1]

    a = build_archive.build(raw_dir, tmp_path / "a", ["BTC-USD", "ETH-USD"], quiet=True)
    b = build_archive.build(raw_dir, tmp_path / "b", ["BTC-USD", "ETH-USD"], quiet=True)
    assert a["content_hash"] == b["content_hash"]
    assert a["content_hash"] == content_hash(tmp_path / "a") == content_hash(tmp_path / "b")
    for name in ("bid_px", "bid_sz", "ask_px", "ask_sz", "snap_ts", "snap_product", "trade_px"):
        fa, fb = tmp_path / "a" / array_filename(name), tmp_path / "b" / array_filename(name)
        assert fa.read_bytes() == fb.read_bytes()
    assert a["source"] == "real"
    assert a["gap_count"] == 1 and a["connect_count"] == 1
    assert a["n_trades"] == 2 and a["trades_deduped"] == 1
    assert a["per_product_snapshots"]["ETH-USD"] == 1
    assert a["n_snapshots"] > 10
    assert validate(tmp_path / "a")["problems"] == []

    data = load_arrays(tmp_path / "a")
    info: ArchiveInfo = data["info"]
    n = info.n_snapshots
    # Level-major: element [level, i] lives at level * n + i in the flat file.
    flat = np.fromfile(tmp_path / "a" / array_filename("bid_px"), dtype="<i4")
    assert flat[1 * n + 0] == data["bid_px"][1, 0]
    assert data["bid_px"].shape == (LEVELS, n)
    assert data["bid_px"].dtype == np.int32 and data["bid_sz"].dtype == np.float32
    assert data["snap_ts"].dtype == np.int64 and data["snap_product"].dtype == np.uint8
    first_btc = int(np.flatnonzero(data["snap_product"] == 0)[0])
    assert data["bid_px"][0, first_btc] == 10000 and data["ask_px"][0, first_btc] == 10001
    assert data["bid_sz"][2, first_btc] == 0 and data["bid_px"][2, first_btc] == 0  # padding
    assert data["trade_side"].tolist() == [1, -1]
    assert data["trade_exch_ts"][0] == iso_to_ns("2023-11-14T22:13:20.123456789Z")


def test_synthetic_archive_has_identical_schema(tmp_path: Path) -> None:
    sidecar = synth_archive.generate(tmp_path / "synth", 3000, 500, seed=3,
                                     products=["BTC-USD", "ETH-USD"])
    assert sidecar["source"] == "synthetic"
    info = ArchiveInfo.load(tmp_path / "synth")
    assert info.n_snapshots == 3000 and info.n_trades == 500
    data = load_arrays(tmp_path / "synth")
    assert data["bid_px"].shape == (LEVELS, 3000)
    assert set(sidecar["arrays"]) == {"snap_ts", "snap_product", "bid_px", "bid_sz", "ask_px", "ask_sz"}
    assert validate(tmp_path / "synth")["problems"] == []
    # No crossed books and strictly ordered levels, by construction and by validation.
    assert np.all(data["bid_px"][0] < data["ask_px"][0])
    assert np.all(np.diff(data["snap_ts"]) > 0)


def test_validate_rejects_crossed_book(tmp_path: Path) -> None:
    synth_archive.generate(tmp_path / "bad", 200, 10, seed=5, products=["BTC-USD", "ETH-USD"])
    path = tmp_path / "bad" / array_filename("bid_px")
    arr = np.fromfile(path, dtype="<i4")
    arr[0] = arr[0] + 10_000  # cross the first snapshot's best bid over its best ask
    arr.tofile(path)
    with pytest.raises(ValueError, match="crossed"):
        validate(tmp_path / "bad")
