"""GPU state sampling on two-card machines and the kernel launch recorder (CPU only)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bench import common

TWO_CARDS = "0, 1590, 5000, 52, 61.2, 1590\n1, 300, 405, 40, 9.9, 1590\n"


def test_parse_gpu_query_keeps_cards_apart() -> None:
    cards = common.parse_gpu_query(TWO_CARDS)
    assert cards[0]["sm_clock_mhz"] == 1590.0 and cards[1]["sm_clock_mhz"] == 300.0
    assert cards[1]["power_w"] == 9.9
    with pytest.raises(ValueError):
        common.parse_gpu_query("1590, 5000, 52")  # the old, index-less query


def test_gpu_state_on_two_cards(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, TWO_CARDS, ""))
    one = common.gpu_state()
    assert one["sm_clock_mhz"] == 1590.0 and list(one["per_gpu"]) == ["0"] and "error" not in one
    both = common.gpu_state((0, 1))
    assert set(both["per_gpu"]) == {"0", "1"} and both["per_gpu"]["1"]["sm_clock_mhz"] == 300.0
    second = common.gpu_state((1,))
    assert second["sm_clock_mhz"] == 300.0  # the first requested card sits at the top level


def _state(sm0: float, sm1: float) -> dict:
    per = {"0": {"sm_clock_mhz": sm0, "mem_clock_mhz": 5000.0}, "1": {"sm_clock_mhz": sm1, "mem_clock_mhz": 5000.0}}
    return {**per["0"], "per_gpu": per}


def test_summarize_flags_a_drop_on_either_card() -> None:
    ok = common.summarize([1.0] * 30, _state(1590, 1590), _state(1500, 1545))
    assert not ok["throttle_flag"] and ok["clock_drop_pct"] == pytest.approx(5.66, abs=0.01)
    bad = common.summarize([1.0] * 30, _state(1590, 1590), _state(1590, 1200))
    assert bad["throttle_flag"] and bad["clock_drop_pct_per_gpu"]["1"] == pytest.approx(24.53, abs=0.01)


def test_summarize_keeps_temperature_power_and_memory_clock_with_the_sm_clocks() -> None:
    before = {**_state(1590, 1590), "temperature_c": 41.0, "power_w": 30.5}
    after = {**_state(1500, 1545), "temperature_c": 55.0, "power_w": 69.0}
    m = common.summarize([1.0] * 30, before, after)
    assert (m["temperature_before_c"], m["temperature_after_c"]) == (41.0, 55.0)
    assert (m["power_before_w"], m["power_after_w"]) == (30.5, 69.0)
    assert m["mem_clock_before_mhz"] == 5000.0 and "clocks_missing" not in m
    only_after = common.summarize([1.0] * 30, None, after)  # one missing state must not raise
    assert only_after["clocks_missing"] and only_after["temperature_after_c"] == 55.0


def test_summarize_says_when_clocks_were_not_sampled() -> None:
    err = {"error": "ValueError('too many values to unpack (expected 5)')", "t": 0.0}
    m = common.summarize([1.0] * 30, err, err)
    assert "throttle_flag" not in m and "too many values" in m["clocks_missing"]


class FakeKernel:
    num_regs = 32

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, grid, block, args, **kwargs):
        self.calls.append((grid, block, args, kwargs))


def test_recording_kernel_counts_configurations_and_forwards(monkeypatch, tmp_path: Path) -> None:
    rec = common.LaunchRecorder()
    monkeypatch.setattr(common, "LAUNCHES", rec)
    fake = FakeKernel()
    k = common.RecordingKernel(fake, "k1", ("a.cu",), ("-Icuda",))
    k((40,), (256,), (1, 2))
    k((40,), (256,), (3, 4))
    k(80, 128, (5,), shared_mem=1024)
    assert len(fake.calls) == 3 and k.num_regs == 32 and k.raw is fake
    cfgs = {(tuple(c["block"]), c["shared_mem"]): c["launches"] for c in rec.snapshot()["k1"]["configs"]}
    assert cfgs == {((256, 1, 1), 0): 2, ((128, 1, 1), 1024): 1}
    path = rec.dump(str(tmp_path))
    child = json.loads(path.read_text(encoding="utf-8"))
    merged = common.merge_launch_logs([rec.snapshot(), child])
    assert {c["launches"] for c in merged["k1"]["configs"]} == {4, 2}


def test_dump_without_a_directory_writes_nothing(monkeypatch) -> None:
    monkeypatch.delenv("BENCH_LAUNCH_LOG_DIR", raising=False)
    rec = common.LaunchRecorder()
    rec.note("k", (), (), 1, 32, 0)
    assert rec.dump() is None
