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


MONITOR_LOG = """timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate,clocks_event_reasons.active
2026/10/22 00:40:00.100, 0, 585, 5000, 28.3, 48, P0, 0x0000000000000001
2026/10/22 00:40:00.200, 0, 1590, 5000, 61.0, 49, P0, 0x0000000000000000
2026/10/22 00:40:00.300, 0, 1440, 5000, 69.9, 50, P0, 0x0000000000000004
2026/10/22 00:40:00.400, 0, 1395, 5000, 70.1, 51, P0, 0x0000000000000004
WARNING: a stray line
2026/10/22 00:40:00.5"""


def test_monitor_log_counts_reasons_and_ignores_idle_and_broken_lines(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    path.write_text(MONITOR_LOG, encoding="utf-8")
    rows = common.parse_monitor(MONITOR_LOG)
    assert len(rows) == 4 and rows[0]["reasons"] == ["gpu_idle"] and rows[2]["reasons"] == ["sw_power_cap"]
    summary = common.summarize_monitor(path)
    assert summary["samples"] == 4 and summary["busy_samples"] == 3
    assert (summary["sm_clock_min_mhz"], summary["sm_clock_max_mhz"]) == (1395.0, 1590.0)  # the idle sample is excluded
    assert summary["reason_counts"] == {"gpu_idle": 1, "sw_power_cap": 2} and summary["power_max_w"] == 70.1
    assert common.summarize_monitor(tmp_path / "missing.csv")["samples"] == 0


def test_interleaved_runs_the_hook_after_warmup_and_stamps_every_timed_call() -> None:
    calls = []
    stamps = []
    variants = {"a": lambda: calls.append("a") or 1.0, "b": lambda: calls.append("b") or 2.0}
    out = common.run_interleaved(variants, 3, 2, after_warmup=lambda: calls.append("hook"), stamps=stamps)
    assert calls.index("hook") == 4 and calls.count("hook") == 1  # two warmup rounds of two variants first
    assert out == {"a": [1.0] * 3, "b": [2.0] * 3}
    assert len(stamps) == 6 and [s["variant"] for s in stamps[:4]] == ["a", "b", "b", "a"]
    assert all(s["t1"] >= s["t0"] for s in stamps)


def test_block_samples_clocks_after_warmup_and_survives_a_missing_gpu(tmp_path: Path, monkeypatch) -> None:
    states = iter([{"sm_clock_mhz": 585.0, "max_sm_clock_mhz": 1590.0}, {"sm_clock_mhz": 1590.0, "max_sm_clock_mhz": 1590.0},
                   {"sm_clock_mhz": 1395.0, "max_sm_clock_mhz": 1590.0}])
    monkeypatch.setattr(common, "gpu_state", lambda devices=(0,): next(states))
    bench = common.Bench(tmp_path, "synthetic", n_iters=3, warmup=1)
    res = bench.block("blk", {"v": lambda: 1.0})["v"]
    assert res["sm_clock_before_warmup_mhz"] == 585.0 and res["sm_clock_before_mhz"] == 1590.0
    assert res["throttle_flag"] and res["clock_drop_pct"] == pytest.approx(12.26, abs=0.01)
    raw = json.loads((tmp_path / "raw" / "blk__v.json").read_text(encoding="utf-8"))
    assert raw["before_warmup"]["sm_clock_mhz"] == 585.0
    monkeypatch.setattr(common, "gpu_state", lambda devices=(0,): {"error": "no nvidia-smi", "t": 0.0})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")))
    res = bench.block("cpu", {"v": lambda: 1.0}, monitor=True)["v"]
    assert res["clocks_missing"] and "error" in res["monitor"]
