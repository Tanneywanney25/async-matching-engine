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
    assert "no limit order matched on arrival" not in bench  # the fixture did not count them
    # a ratio below one reads as slower, and limit orders that matched are reported, not denied
    met["sections"]["native_engine"]["p99_add_python_engine_over_binding_decimal"] = 0.5
    met["sections"]["native_engine"]["realistic"].update({"limit_orders_matched": 3, "limit_fills": 4})
    met_path.write_text(json.dumps(met), encoding="utf-8")
    assert _render(tree) == 0
    bench = (tree / "BENCHMARKS.md").read_text(encoding="utf-8")
    assert "2.0 times slower with Decimal orders" in bench
    assert "3 limit orders crossed a stale level outside a snapshot's reconciled range and matched on arrival (4 fills)" in bench
    assert check_numbers.main(["--root", str(tree), "--results", str(tree / "results")]) == 0


def test_second_card_verdict_follows_the_measurement_in_every_document(tree: Path, monkeypatch) -> None:
    """Fixture values (not measurements): copies double on two cards but the pipeline does not speed up."""
    monkeypatch.setattr(render_docs, "REPO", tree)
    met_path = tree / "results" / "fixture-run" / "metrics.json"
    met = json.loads(met_path.read_text(encoding="utf-8"))
    pipe = met["sections"].setdefault("pipeline", {})
    pipe["transfer_share_pct"] = 80.0
    pipe["two_gpu"] = {"attempted": True, "ok": True, "chunks_per_device": 8, "speedup": 1.02, "scaling_efficiency_pct": 51.0,
                       "one_gpu_ms": {"median_ms": 56.0, "iqr_ms": 0.5}, "two_gpu_ms": {"median_ms": 55.0, "iqr_ms": 0.5},
                       "h2d_gbs_one_gpu": 12.0, "h2d_gbs_two_gpu_aggregate": 23.0, "h2d_aggregate_vs_one_gpu": 23.0 / 12.0,
                       "topology": {"links": {"GPU0-GPU1": "SYS"}}, "pcie": ["0, 00000000:00:04.0, 3, 16"],
                       "pcie_current_after_copies": ["0, 3, 16"]}
    met_path.write_text(json.dumps(met), encoding="utf-8")
    assert _render(tree) == 0
    bench = (tree / "BENCHMARKS.md").read_text(encoding="utf-8")
    assert "yet the whole pipeline's speedup stays well short of that" in bench
    assert "rated `Gen3 x16` and running at `Gen3 x16` just after the copies" in bench
    design = (tree / "DESIGN.md").read_text(encoding="utf-8")
    assert "## Two cards\n" in design and "one bottleneck" not in design and "Measured, it nearly does" in design
    interview = (tree / "INTERVIEW_NOTES.md").read_text(encoding="utf-8")
    assert "What did the second T4 buy?" in interview and "not speed up" not in interview
    assert "through the CPU interconnect between NUMA nodes" in interview
    assert check_numbers.main(["--root", str(tree), "--results", str(tree / "results")]) == 0


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
    assert "with its launch queued ahead, the best one takes 1.05 times the GPU time of the CUDA v2 kernel" in bench
    assert "the ratios are 1.25 and" in bench and "a median 33.75 µs on the host, against 6.25 µs for v2" in bench
    assert check_numbers.main(["--root", str(tree), "--results", str(tree / "results")]) == 0
    lad.pop("queued")
    lad["triton"] = {"attempted": True, "ok": False, "reason": "triton is not installed"}
    lad["rungs"].pop()
    met_path.write_text(json.dumps(met), encoding="utf-8")
    assert _render(tree) == 0
    assert "was not timed in this run: `triton is not installed`." in (tree / "BENCHMARKS.md").read_text(encoding="utf-8")
    lad["triton"] = {"attempted": True, "ok": False, "version": "9.9.9",
                     "reason": "no launch configuration passed verification against the NumPy reference",
                     "verification_by_config": {"b512_w4": {"verification": {"verified": False}},
                                                "b1024_w4": {"verification": {"verified": False}}}}
    met_path.write_text(json.dumps(met), encoding="utf-8")
    assert _render(tree) == 0
    bench = (tree / "BENCHMARKS.md").read_text(encoding="utf-8")
    assert "Triton `9.9.9` was not timed in this run" in bench
    assert "failed the tolerance check against the NumPy reference: `b512_w4`, `b1024_w4`." in bench


def test_went_wrong_reads_tags_only_where_entries_carry_them() -> None:
    log = "\n".join([
        "- 2026-10-21 10:00 [went-wrong] [biggest] Old big problem.",
        "- 2026-10-21 11:00 [went-wrong] Smaller problem.",
        "- 2026-10-21 12:00 A note that mentions the `[biggest]` and [went-wrong] tags in prose.",
        "- 2026-10-21 13:00 [went-wrong][biggest] New big problem.",
    ])
    assert render_docs.went_wrong(log) == ["New big problem.", "Old big problem.", "Smaller problem."]


def test_an_earlier_run_with_a_different_second_card_verdict_is_named(tree: Path, monkeypatch) -> None:
    """Two fixture runs (values made up): the newer one scales on two cards, the older one did not."""
    monkeypatch.setattr(render_docs, "REPO", tree)
    newer = tree / "results" / "fixture-run"
    older = tree / "results" / "fixture-older"
    shutil.copytree(newer, older)

    def set_two_gpu(run_dir: Path, aggregate: float, speedup: float, started: float) -> None:
        met = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        met["sections"].setdefault("pipeline", {})["two_gpu"] = {
            "attempted": True, "ok": True, "speedup": speedup, "one_gpu_ms": {"median_ms": 56.0, "iqr_ms": 0.5},
            "two_gpu_ms": {"median_ms": 56.0 / speedup, "iqr_ms": 0.5}, "h2d_gbs_one_gpu": 12.0,
            "h2d_gbs_two_gpu_aggregate": aggregate, "h2d_aggregate_vs_one_gpu": aggregate / 12.0,
            "topology": {"links": {"GPU0-GPU1": "PHB"}}}
        (run_dir / "metrics.json").write_text(json.dumps(met), encoding="utf-8")
        man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        man["started_unix"] = started
        (run_dir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")

    set_two_gpu(older, 12.0, 1.0, 1000.0)
    set_two_gpu(newer, 24.0, 2.0, 2000.0)
    assert _render(tree) == 0
    bench = (tree / "BENCHMARKS.md").read_text(encoding="utf-8")
    # the fixtures carry no PCIe lines, so the topology is not claimed to match
    assert "An earlier full run in another Kaggle session, `fixture-older`, found that two cards copied no faster" in bench
    assert "these two runs do not show why" in bench
    assert "showed no gain, so the result varies between sessions" in (tree / "INTERVIEW_NOTES.md").read_text(encoding="utf-8")
    assert "`fixture-older`" in (tree / "DESIGN.md").read_text(encoding="utf-8")
    assert check_numbers.main(["--root", str(tree), "--results", str(tree / "results")]) == 0


def test_same_topology_needs_matching_links_and_pcie_lines() -> None:
    a = {"topology": {"links": {"GPU0-GPU1": "PHB"}}, "pcie": ["0, 00000000:00:04.0, 3, 16"]}
    assert render_docs.same_topology(a, json.loads(json.dumps(a)))
    assert not render_docs.same_topology(a, {**a, "topology": {"links": {"GPU0-GPU1": "SYS"}}})
    assert not render_docs.same_topology(a, {**a, "pcie": ["0, 00000000:00:05.0, 3, 16"]})
    assert not render_docs.same_topology({"topology": a["topology"]}, {"topology": a["topology"]})  # nothing to compare
