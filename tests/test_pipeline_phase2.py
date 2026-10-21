"""Pure parts of the two-card pipeline and the int16 checks (CPU only)."""

from __future__ import annotations

import numpy as np

from bench import bench_pipeline as bp

TOPO = """\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tPHB\t0-3\t0\t\tN/A
GPU1\tPHB\t X \t0-3\t0\t\tN/A

Legend:

  X    = Self
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
"""


def test_parse_topology_reads_links_and_affinity() -> None:
    t = bp.parse_topology(TOPO)
    assert t["gpus"] == ["GPU0", "GPU1"]
    assert t["links"] == {"GPU0-GPU1": "PHB", "GPU1-GPU0": "PHB"}
    assert t["cpu_affinity"] == {"GPU0": "0-3", "GPU1": "0-3"}
    assert t["numa_affinity"] == {"GPU0": "0", "GPU1": "0"}
    assert bp.parse_topology("") == {}


def _outputs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name, dtype in bp.OUTPUT_FIELDS:
        out[name] = (rng.integers(0, 50, 100) if np.dtype(dtype).kind == "i" else rng.standard_normal(100)).astype(dtype)
    out["imbalance10"][3] = np.nan
    return out


def test_identical_outputs_is_bitwise_and_nan_safe() -> None:
    a = _outputs()
    b = {k: v.copy() for k, v in a.items()}
    assert bp.identical_outputs(a, b)["identical"]
    b["microprice_offset"][7] = np.nextafter(b["microprice_offset"][7], np.float32(1e9))
    r = bp.identical_outputs(a, b)
    assert not r["identical"] and r["per_output"]["microprice_offset"] is False and r["per_output"]["imbalance10"]


def test_env_flags_default_on_and_switch_off(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_INT16", raising=False)
    assert bp._env_flag("PIPELINE_INT16", default=True) is True
    monkeypatch.setenv("PIPELINE_INT16", "0")
    assert bp._env_flag("PIPELINE_INT16", default=True) is False
    monkeypatch.setenv("PIPELINE_INT16", "1")
    assert bp._env_flag("PIPELINE_INT16") is True


def test_two_card_split_covers_every_snapshot_once() -> None:
    n = 863_153
    halves = bp.chunk_bounds(n, 2)
    assert halves[0][0] == 0 and halves[-1][1] == n and halves[0][1] == halves[1][0]
    per_device = [bp.chunk_bounds(e - s, 8) for s, e in halves]
    assert sum(e - s for part in per_device for s, e in part) == n


def test_parse_topology_ignores_the_ansi_underline_on_kaggle() -> None:
    real = "\t\x1b[4mGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\nGPU0\t X \tPHB\t0-3\t0\t\tN/A\nGPU1\tPHB\t X \t0-3\t0\t\tN/A\n"
    t = bp.parse_topology(real)
    assert t["links"] == {"GPU0-GPU1": "PHB", "GPU1-GPU0": "PHB"} and t["cpu_affinity"]["GPU0"] == "0-3"
