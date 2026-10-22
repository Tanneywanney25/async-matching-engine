"""Renderer, provenance check and bullet check, end to end on a temporary results tree.

The fixture copies the committed smoke run (synthetic, quick) and relabels it as a full real
T4 run inside the temporary directory only, so the renderer has something to render; nothing
here touches the repository's documents or results.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import check_bullets
import check_numbers
import render_docs

REPO = Path(__file__).resolve().parent.parent
SMOKE = REPO / "results" / "bench_quick-20261021-185806"


def _relabel(run_dir: Path, source: str = "real", quick: bool = False) -> None:
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    met = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    man["source"] = source
    met["source"] = source
    met.setdefault("run_all", {})["quick"] = quick
    (run_dir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(met), encoding="utf-8")


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    if not SMOKE.exists():
        pytest.skip("smoke results not present")
    results = tmp_path / "results"
    run = results / "fixture-run"
    shutil.copytree(SMOKE, run)
    _relabel(run)
    for name in ("README.md", "CHANGELOG.md", "DECISIONS.md"):
        shutil.copy(REPO / name, tmp_path / name)
    return tmp_path


def _render(tree: Path) -> int:
    return render_docs.main(["--results", str(tree / "results"), "--out-dir", str(tree)])


def test_render_then_checks_pass(tree: Path, monkeypatch) -> None:
    monkeypatch.setattr(render_docs, "REPO", tree)
    assert _render(tree) == 0
    for name in ("BENCHMARKS.md", "BULLETS.md", "DESIGN.md", "INTERVIEW_NOTES.md", "README.md", "CHANGELOG.md"):
        assert (tree / name).exists() and (tree / name).stat().st_size > 200, name
    readme = (tree / "README.md").read_text(encoding="utf-8")
    assert readme.count(render_docs.MARK_START) == 1 and "## License" in readme
    data = json.loads((tree / "public" / "data" / "gpu_lab.json").read_text(encoding="utf-8"))
    for key in ("ladder", "roofline", "pipeline", "precision", "scans", "slippage", "ofi", "benchmarks", "engine", "provenance"):
        assert key in data, key
    assert check_numbers.main(["--root", str(tree), "--results", str(tree / "results")]) == 0
    assert check_bullets.main(["--file", str(tree / "BULLETS.md")]) == 0
    interview = (tree / "INTERVIEW_NOTES.md").read_text(encoding="utf-8")
    assert interview.count("\n### ") == 15


def test_stray_number_fails_the_provenance_check(tree: Path, monkeypatch) -> None:
    monkeypatch.setattr(render_docs, "REPO", tree)
    assert _render(tree) == 0
    path = tree / "BENCHMARKS.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nA made-up figure: 123.456 ms.\n", encoding="utf-8")
    assert check_numbers.main(["--root", str(tree), "--results", str(tree / "results")]) == 1


def test_synthetic_or_quick_input_is_refused(tree: Path) -> None:
    run = tree / "results" / "fixture-run"
    _relabel(run, source="synthetic")
    assert _render(tree) == 2
    _relabel(run, source="real", quick=True)
    assert _render(tree) == 2


def test_numbers_in_code_links_and_dates_are_ignored() -> None:
    text = "See `run-20261021` and [raw](results/a1/raw/9.json) on 2026-10-21T07:49:23Z: 4.2 times."
    assert check_numbers.tokens(text) == ["4.2"]


def test_bullet_rules() -> None:
    ok = ("Built CUDA C++ kernels on an NVIDIA T4 for spread, depth and order imbalance over 863,153 recorded book "
          "snapshots, 45.2 times faster than single-threaded C++ per kernel and 1.6 times faster than NumPy end to end.")
    lines = check_bullets.split_two_lines(ok)
    assert lines and not check_bullets.problems(lines)
    assert check_bullets.problems(["GPU kernels " + "x" * 90, "y" * 100])  # abbreviation and lengths
    assert check_bullets.problems(lines, alternate=True)  # a three-item list is not allowed in alternates
    assert check_bullets.split_two_lines("too short") is None


def _stats(p50: float, p99: float) -> dict:
    return {"p50_us": p50, "p99_us": p99, "p999_us": p99 * 2, "mean_us": p50, "n": 10}


def test_recorded_book_stream_renders_beside_the_synthetic_one(tree: Path, monkeypatch) -> None:
    """A hand-made native_engine section (fixture values, not measurements) renders both tables."""
    monkeypatch.setattr(render_docs, "REPO", tree)
    met_path = tree / "results" / "fixture-run" / "metrics.json"
    met = json.loads(met_path.read_text(encoding="utf-8"))
    paths = ("native", "binding", "binding_decimal", "python_engine")
    synthetic = {p: {"add_order": _stats(0.1 * (i + 1), 0.4 * (i + 1)), "cancel_order": _stats(0.1, 0.3)} for i, p in enumerate(paths)}
    realistic = {p: {k: _stats(0.07 * (i + 1), 0.31 * (i + 1)) for k in ("limit", "market", "cancel")} for i, p in enumerate(paths)}
    realistic.update({"stream": "derived from the first 9.5 minutes of a fixture book", "ops": 1234, "consistent": True,
                      "op_counts": {"limit": 600, "market": 34, "cancel": 600}, "raw_file": "raw/realistic.json",
                      **{f"p99_{k}_python_engine_over_native": 4.0 for k in ("limit", "market", "cancel")}})
    met["sections"]["native_engine"] = {**synthetic, "stream": "a fixture stream", "consistent": True,
                                        "p99_add_python_engine_over_native": 4.0, "raw_file": "raw/synthetic.json",
                                        "realistic": realistic}
    met_path.write_text(json.dumps(met), encoding="utf-8")
    assert _render(tree) == 0
    bench = (tree / "BENCHMARKS.md").read_text(encoding="utf-8")
    synth_at = bench.index("## C++20 order book core (CPU)")
    real_at = bench.index("### The same paths on a stream derived from the recorded book")
    assert synth_at < real_at
    assert "1,234 operations: 600 limit orders, 34 market orders, 600 cancels" in bench
    assert "| Python engine (the reference implementation) | 0.280 | 1.240 |" in bench
    assert check_numbers.main(["--root", str(tree), "--results", str(tree / "results")]) == 0
    lab = json.loads((tree / "public" / "data" / "gpu_lab.json").read_text(encoding="utf-8"))
    real_lab = lab["native_engine"]["realistic"]
    assert [p["key"] for p in real_lab["paths"]] == list(paths) and real_lab["ops"] == 1234


def test_triton_rung_is_reported_beside_the_cuda_rungs_but_never_best(tree: Path, monkeypatch) -> None:
    """A Triton rung made from the fixture's own v2 rung (scaled, not measured) renders with its ratios."""
    monkeypatch.setattr(render_docs, "REPO", tree)
    met_path = tree / "results" / "fixture-run" / "metrics.json"
    met = json.loads(met_path.read_text(encoding="utf-8"))
    lad = met["sections"]["ladder"]
    v2 = next(rung for rung in lad["rungs"] if rung["name"] == "v2_soa_fused")
    tri = json.loads(json.dumps(v2))
    tri["name"], tri["extra"] = "triton_fused", True
    tri["kernel_ms"]["median_ms"] = v2["kernel_ms"]["median_ms"] * 1.25
    lad["rungs"].append(tri)
    lad["triton"] = {"attempted": True, "ok": True, "version": "9.9.9", "identical_to_v2_all": True,
                     "sweep": [{"block_size": 512, "num_warps": 4, "median_ms": 1.5, "iqr_ms": 0.01}]}
    met_path.write_text(json.dumps(met), encoding="utf-8")
    assert _render(tree) == 0
    bench = (tree / "BENCHMARKS.md").read_text(encoding="utf-8")
    assert "### Extra rung: the fused kernel in Triton" in bench
    assert "runs in 1.25 times the time of the CUDA v2 kernel" in bench
    assert "| Triton, fused like v2 (extra rung, best of a small sweep) |" in bench
    assert "are bit-for-bit identical to v2's" in bench
    assert check_numbers.main(["--root", str(tree), "--results", str(tree / "results")]) == 0
    # with the queued block present, the comparison uses GPU-only time and names the dispatch cost
    names = [rung["name"] for rung in lad["rungs"]]
    q_ms = {name: {"median_ms": 0.1, "iqr_ms": 0.001} for name in names}
    q_ms["triton_fused"] = {"median_ms": 0.105, "iqr_ms": 0.001}
    lad["queued"] = {"spin_us": 300.0, "kernel_ms": q_ms, "host_enqueue_max_us": 41.5, "dispatch_hidden": True,
                     "host_enqueue_us": {name: {"median_us": 6.25, "max_us": 9.0} for name in names}}
    lad["queued"]["host_enqueue_us"]["triton_fused"] = {"median_us": 33.75, "max_us": 41.5}
    met_path.write_text(json.dumps(met), encoding="utf-8")
    assert _render(tree) == 0
    bench = (tree / "BENCHMARKS.md").read_text(encoding="utf-8")
    assert "### GPU time with the launches queued ahead" in bench
    assert "behind a 300 µs busy-wait kernel" in bench and "The slowest enqueue took 41.50 µs, well inside the wait." in bench
    assert "with its launch queued ahead, the best one runs in 1.05 times the GPU time of the CUDA v2 kernel" in bench
    assert "the ratios are 1.25 and" in bench and "a median 33.75 µs on the host, against 6.25 µs for v2" in bench
    assert check_numbers.main(["--root", str(tree), "--results", str(tree / "results")]) == 0
    lad.pop("queued")
    lad["triton"] = {"attempted": True, "ok": False, "reason": "triton is not installed"}
    lad["rungs"].pop()
    met_path.write_text(json.dumps(met), encoding="utf-8")
    assert _render(tree) == 0
    assert "Not measured in this run: `triton is not installed`." in (tree / "BENCHMARKS.md").read_text(encoding="utf-8")
