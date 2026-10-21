"""CPU tests for Phase 7: the SVG timeline renderer, the chunk math, the int16 encoding.

The renderer is exercised on a fake Nsight Systems SQLite export (the two CUPTI
tables plus StringIds and NVTX_EVENTS, with the column names of a real
``nsys export --type sqlite``), the chunk boundaries and buffer layouts are
checked arithmetically, and the int16 offset encode/decode is round-tripped in
NumPy including the overflow fallback. One GPU-marked test runs the real
pipeline on a tiny synthetic archive when a card is present.
"""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from bench import bench_pipeline as bp
from bench import timeline_svg as tsvg

MS = 1_000_000  # ns per ms


# --------------------------------------------------------------------------- #
# Fake nsys SQLite export
# --------------------------------------------------------------------------- #


def _fake_sqlite(path: Path, memcpy=(), kernels=(), nvtx=(), with_memcpy_table: bool = True,
                 strings=((1, "pipeline_features_i32"), (2, "overlapped_pass"))) -> Path:
    """Build a database with the tables and columns ``nsys export --type sqlite`` writes."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE StringIds (id INTEGER NOT NULL PRIMARY KEY, value TEXT NOT NULL)")
    conn.executemany("INSERT INTO StringIds VALUES (?, ?)", list(strings))
    if with_memcpy_table:
        conn.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER NOT NULL, "end" INTEGER NOT NULL, '
                     "deviceId INTEGER, contextId INTEGER, streamId INTEGER NOT NULL, correlationId INTEGER, "
                     "bytes INTEGER NOT NULL, copyKind INTEGER NOT NULL)")
        conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, 0, 1, ?, 0, ?, ?)",
                         [(s, e, stream, nbytes, kind) for s, e, kind, stream, nbytes in memcpy])
    conn.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER NOT NULL, "end" INTEGER NOT NULL, '
                 "deviceId INTEGER, contextId INTEGER, streamId INTEGER NOT NULL, correlationId INTEGER, "
                 "demangledName INTEGER NOT NULL, shortName INTEGER NOT NULL, gridX INTEGER, blockX INTEGER)")
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, 1, ?, 0, ?, ?, 1, 256)",
                     [(s, e, stream, name_id, name_id) for s, e, stream, name_id in kernels])
    if nvtx:
        conn.execute('CREATE TABLE NVTX_EVENTS (start INTEGER NOT NULL, "end" INTEGER, eventType INTEGER NOT NULL, '
                     "rangeId INTEGER, category INTEGER, color INTEGER, text TEXT, globalTid INTEGER, "
                     "endGlobalTid INTEGER, textId INTEGER, domainId INTEGER)")
        conn.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, 59, NULL, NULL, NULL, ?, 1, NULL, ?, 0)",
                         [(s, e, text, text_id) for s, e, text, text_id in nvtx])
    conn.commit()
    conn.close()
    return path


# Hand-placed pipeline: H2D back to back, kernels and D2H nested inside the next copy.
#   h2d    [0,10] [10,20] [20,30]
#   kernel        [10,15] [20,25]
#   d2h               [15,17] [25,27]
# Two or more lanes active during [10,15] + [15,17] + [20,25] + [25,27] = 14 ms of 30.
H2D = [(0 * MS, 10 * MS, 1, 7, 65_000_000), (10 * MS, 20 * MS, 1, 7, 65_000_000), (20 * MS, 30 * MS, 1, 7, 65_000_000)]
KERNELS = [(10 * MS, 15 * MS, 8, 1), (20 * MS, 25 * MS, 8, 1)]
D2H = [(15 * MS, 17 * MS, 2, 9, 1_600_000), (25 * MS, 27 * MS, 2, 9, 1_600_000)]
EXPECTED_OVERLAP = 14 / 30


def test_overlap_fraction_from_hand_placed_intervals() -> None:
    lanes = {"h2d": [(0, 10), (10, 20), (20, 30)], "kernel": [(10, 15), (20, 25)], "d2h": [(15, 17), (25, 27)]}
    assert tsvg.overlap_fraction(lanes) == pytest.approx(EXPECTED_OVERLAP)
    # a single busy lane never overlaps; three lanes fully stacked overlap all the time
    assert tsvg.overlap_fraction({"h2d": [(0, 10)], "kernel": [], "d2h": []}) == 0.0
    assert tsvg.overlap_fraction({"h2d": [(0, 10)], "kernel": [(0, 10)], "d2h": [(0, 10)]}) == pytest.approx(1.0)
    # intervals inside one lane are merged first: two kernels back to back are not "two lanes"
    assert tsvg.overlap_fraction({"kernel": [(0, 5), (5, 10)], "h2d": [], "d2h": []}) == 0.0
    # touching intervals in different lanes share no time
    assert tsvg.overlap_fraction({"h2d": [(0, 5)], "kernel": [(5, 10)], "d2h": []}) == 0.0
    assert tsvg.merge_intervals([(5, 9), (0, 3), (2, 6)]) == [(0, 9)]
    assert tsvg.busy_ns([(0, 3), (2, 6), (10, 11)]) == 7


def test_svg_has_three_lanes_and_one_bar_per_operation(tmp_path: Path) -> None:
    extra_d2d = [(12 * MS, 13 * MS, 8, 7, 1000)]  # device-to-device copy: not one of the three lanes
    db = _fake_sqlite(tmp_path / "trace.sqlite", memcpy=H2D + D2H + extra_d2d, kernels=KERNELS)
    svg_path = tmp_path / "timeline.svg"
    stats = tsvg.render(db, svg_path, nvtx_range=None)
    svg = svg_path.read_text(encoding="utf-8")

    assert svg.lstrip().startswith("<svg") and 'viewBox="0 0 ' in svg and 'width="100%"' in svg
    assert svg.count('class="lane"') == 3
    for key, short, _long, colour in tsvg.LANES:
        assert f'data-lane="{key}"' in svg
        assert f">{short}<" in svg
        assert colour in svg
    assert svg.count('class="bar" data-lane="h2d"') == 3
    assert svg.count('class="bar" data-lane="kernel"') == 2
    assert svg.count('class="bar" data-lane="d2h"') == 2
    assert stats["bars"] == {"h2d": 3, "kernel": 2, "d2h": 2}
    assert stats["other_memcpy"] == 1
    assert stats["overlap_fraction"] == pytest.approx(EXPECTED_OVERLAP)
    assert f"{EXPECTED_OVERLAP * 100:.1f}%" in svg  # the title line carries the measured overlap
    assert "time (ms)" in svg and ">30<" in svg     # axis in ms, last tick at the 30 ms window end
    assert stats["window_ms"] == pytest.approx(30.0)
    assert stats["busy_ms"] == pytest.approx({"h2d": 30.0, "kernel": 10.0, "d2h": 4.0})
    assert stats["bytes"] == {"h2d": 3 * 65_000_000, "d2h": 2 * 1_600_000}
    assert stats["kernels"] == {"pipeline_features_i32": 2}
    assert "<script" not in svg


def test_nvtx_range_restricts_the_drawing_to_the_measured_pass(tmp_path: Path) -> None:
    # a warmup copy and kernel 100 ms earlier must not be drawn when the range is requested
    warm_copy = [(-100 * MS, -90 * MS, 1, 7, 65_000_000)]
    warm_kernel = [(-90 * MS, -85 * MS, 8, 1)]
    nvtx = [(-101 * MS, -80 * MS, "warmup", None), (-1 * MS, 31 * MS, None, 2)]  # registered string via textId
    db = _fake_sqlite(tmp_path / "trace.sqlite", memcpy=warm_copy + H2D + D2H, kernels=warm_kernel + KERNELS, nvtx=nvtx)
    stats = tsvg.render(db, tmp_path / "t.svg", nvtx_range="overlapped_pass")
    assert stats["nvtx_range_found"] is True
    assert stats["bars"] == {"h2d": 3, "kernel": 2, "d2h": 2}
    assert stats["overlap_fraction"] == pytest.approx(EXPECTED_OVERLAP)
    whole = tsvg.render(db, tmp_path / "all.svg", nvtx_range=None)
    assert whole["bars"] == {"h2d": 4, "kernel": 3, "d2h": 2}
    missing = tsvg.render(db, tmp_path / "m.svg", nvtx_range="no_such_range")
    assert missing["nvtx_range_found"] is False and missing["bars"] == whole["bars"]


def test_renderer_tolerates_missing_tables_and_empty_traces(tmp_path: Path) -> None:
    db = _fake_sqlite(tmp_path / "k.sqlite", kernels=KERNELS, with_memcpy_table=False)
    stats = tsvg.render(db, tmp_path / "k.svg")
    svg = (tmp_path / "k.svg").read_text(encoding="utf-8")
    assert svg.count('class="lane"') == 3 and stats["bars"] == {"h2d": 0, "kernel": 2, "d2h": 0}
    empty = _fake_sqlite(tmp_path / "e.sqlite", with_memcpy_table=False)
    stats = tsvg.render(empty, tmp_path / "e.svg")
    assert stats["bars"] == {"h2d": 0, "kernel": 0, "d2h": 0} and stats["overlap_fraction"] == 0.0
    assert "no GPU operations" in (tmp_path / "e.svg").read_text(encoding="utf-8")


def test_timeline_cli(tmp_path: Path, capsys) -> None:
    db = _fake_sqlite(tmp_path / "trace.sqlite", memcpy=H2D + D2H, kernels=KERNELS)
    tsvg.main(["--sqlite", str(db), "--out", str(tmp_path / "cli.svg"), "--nvtx-range", "", "--title", "My title"])
    assert "My title" in (tmp_path / "cli.svg").read_text(encoding="utf-8")
    assert '"overlap_fraction"' in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Chunk math and layouts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n,k", [(1_300_000, 16), (20_000, 16), (17, 16), (16, 16), (5, 16), (1, 16), (100, 1), (7, 3)])
def test_chunk_bounds_cover_the_archive_once_in_order(n: int, k: int) -> None:
    bounds = bp.chunk_bounds(n, k)
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    assert all(e > s for s, e in bounds)                                   # no empty chunk
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))  # contiguous
    assert len(bounds) == min(n, k)
    lengths = [e - s for s, e in bounds]
    assert max(lengths) - min(lengths) <= 1                               # balanced
    if n >= k:
        assert bounds == [(i * n // k, (i + 1) * n // k) for i in range(k)]  # the spec's formula


def test_chunk_bounds_edge_cases() -> None:
    assert bp.chunk_bounds(0, 16) == []
    assert bp.chunk_bounds(10, 0) == []


def test_chunk_layouts_are_aligned_and_sized() -> None:
    levels, length = 50, 81_251  # odd length: the padding must still align every segment
    lay = bp.layout_int32(levels, length)
    assert all(off % bp.ALIGN == 0 for off in lay["offsets"].values())
    assert lay["block_bytes"] % bp.ALIGN == 0
    assert lay["payload_bytes"] == 16 * levels * length == 800 * length
    assert lay["block_bytes"] >= lay["payload_bytes"] and lay["block_bytes"] - lay["payload_bytes"] < 4 * bp.ALIGN
    assert lay["offsets"]["ask_sz"] + 4 * levels * length <= lay["block_bytes"]
    lay16 = bp.layout_int16(levels, length)
    assert all(off % bp.ALIGN == 0 for off in lay16["offsets"].values())
    assert lay16["payload_bytes"] == 608 * length
    assert lay16["offsets"]["ask_sz"] + 4 * levels * length <= lay16["block_bytes"]
    # the headline saving of the stretch: 800 -> 608 bytes per snapshot, 24%
    assert (lay["payload_bytes"] - lay16["payload_bytes"]) / lay["payload_bytes"] == pytest.approx(0.24)
    out = bp.layout_output(length)
    assert out["block_bytes"] == 20 * length
    assert out["offsets"]["microprice_offset"] == 16 * length
    # a 1.3M archive in 16 chunks keeps the triple buffers far below 3 GB
    bounds = bp.chunk_bounds(1_300_000, 16)
    biggest = max(bp.layout_int32(levels, e - s)["block_bytes"] for s, e in bounds)
    assert 3 * (biggest + bp.layout_output(max(e - s for s, e in bounds))["block_bytes"]) < 3 * 1024 ** 3


# --------------------------------------------------------------------------- #
# int16 offset encoding
# --------------------------------------------------------------------------- #


def _random_book(rng, levels=50, n=300, best=11_000_000, side=-1):
    gaps = rng.integers(1, 4, size=(levels, n))
    gaps[0] = 0
    px = best + side * np.cumsum(gaps, axis=0)
    missing = rng.random((levels, n)) < 0.05
    missing[0] = False
    px = np.where(missing, 0, px).astype(np.int32)
    return px


def test_int16_offsets_round_trip_including_missing_levels() -> None:
    rng = np.random.default_rng(3)
    for side in (-1, +1):  # bids go down from the best, asks go up
        px = _random_book(rng, side=side)
        best, off, overflow = bp.encode_int16_offsets(px)
        assert not overflow
        assert off.dtype == np.int16 and best.dtype == np.int32
        assert np.array_equal(best, px[0])
        assert np.all(off[0] == 0)
        assert np.all(off[px == 0] == bp.INT16_MISSING)
        assert np.array_equal(bp.decode_int16_offsets(best, off), px)
    # an empty side (best price 0): every level is the sentinel and decodes to 0
    empty = np.zeros((50, 4), dtype=np.int32)
    best, off, overflow = bp.encode_int16_offsets(empty)
    assert not overflow and np.all(off == bp.INT16_MISSING)
    assert np.array_equal(bp.decode_int16_offsets(best, off), empty)


def test_int16_offsets_overflow_is_detected_not_wrapped() -> None:
    px = _random_book(np.random.default_rng(4), side=-1)
    ok_best, ok_off, overflow = bp.encode_int16_offsets(px)
    assert not overflow
    # exactly the largest representable magnitude is fine, one more tick is not
    px_edge = px.copy()
    px_edge[49, 0] = px_edge[0, 0] - 32767
    _b, off_edge, overflow = bp.encode_int16_offsets(px_edge)
    assert not overflow and off_edge[49, 0] == -32767
    assert bp.decode_int16_offsets(_b, off_edge)[49, 0] == px_edge[49, 0]
    px_over = px.copy()
    px_over[49, 0] = px_over[0, 0] - 32768  # would collide with the missing-level sentinel
    _b, _o, overflow = bp.encode_int16_offsets(px_over)
    assert overflow
    px_wide = px.copy()
    px_wide[20, 5] = px_wide[0, 5] - 40_000  # a $400 hole in the book
    assert bp.encode_int16_offsets(px_wide)[2] is True


# --------------------------------------------------------------------------- #
# NumPy reference of the fused kernel
# --------------------------------------------------------------------------- #


def test_reference_features_hand_book_and_metrics_py_convention() -> None:
    from exchange_engine.metrics import book_imbalance

    levels = 50
    bid_px = np.zeros((levels, 3), dtype=np.int32)
    bid_sz = np.zeros((levels, 3), dtype=np.float32)
    ask_px = np.zeros((levels, 3), dtype=np.int32)
    ask_sz = np.zeros((levels, 3), dtype=np.float32)
    # snapshot 0: bid 100 x 2, ask 103 x 1 -> spread 3, mid 101.5, microprice 102, offset +0.5
    bid_px[0, 0], bid_sz[0, 0], ask_px[0, 0], ask_sz[0, 0] = 100, 2.0, 103, 1.0
    bid_px[1, 0], bid_sz[1, 0] = 99, 5.0
    ask_px[1, 0], ask_sz[1, 0] = 104, 1.0
    # snapshot 1: twelve levels a side so the top-10 imbalance differs from the full-depth one
    for lvl in range(12):
        bid_px[lvl, 1], bid_sz[lvl, 1] = 1000 - lvl, 1.0
        ask_px[lvl, 1], ask_sz[lvl, 1] = 1002 + lvl, 3.0
    # snapshot 2: empty ask side -> spread 0, offset 0, imbalance +1
    bid_px[0, 2], bid_sz[0, 2] = 500, 4.0
    ref = bp.reference_features(bid_px, bid_sz, ask_px, ask_sz)
    assert ref["spread_ticks"].tolist() == [3, 2, 0]
    assert ref["depth_bid"].tolist() == pytest.approx([7.0, 12.0, 4.0])
    assert ref["depth_ask"].tolist() == pytest.approx([2.0, 36.0, 0.0])
    assert ref["imbalance10"].tolist() == pytest.approx([(7 - 2) / 9, (10 - 30) / 40, 1.0])
    assert ref["microprice_offset"].tolist() == pytest.approx([0.5, 2 * 0.5 * (1 - 3) / 4, 0.0])
    # metrics.py convention, exactly
    for i in range(3):
        bids = [(Decimal(int(bid_px[l, i])), Decimal(str(float(bid_sz[l, i])))) for l in range(levels) if bid_sz[l, i] > 0]
        asks = [(Decimal(int(ask_px[l, i])), Decimal(str(float(ask_sz[l, i])))) for l in range(levels) if ask_sz[l, i] > 0]
        assert float(book_imbalance(bids, asks, levels=10)) == pytest.approx(ref["imbalance10"][i], abs=1e-12)
    # compare_features: exact ticks, float tolerance
    exact = bp.compare_features({k: v.copy() for k, v in ref.items()}, ref)
    assert exact["verified"] and exact["max_abs_err"]["spread_ticks"] == 0
    wrong = {k: v.copy() for k, v in ref.items()}
    wrong["spread_ticks"] = wrong["spread_ticks"] + 1
    assert not bp.compare_features(wrong, ref)["verified"]
    close = {k: v.copy() for k, v in ref.items()}
    close["depth_bid"] = (close["depth_bid"] * (1 + 2e-5)).astype(np.float32)
    assert bp.compare_features(close, ref)["verified"]


def test_reference_features_on_synthetic_archive_matches_metrics_py(tmp_path: Path) -> None:
    import synth_archive  # scripts/ is on sys.path via conftest
    from bench.common import load_archive
    from exchange_engine.archive import PRODUCTS
    from exchange_engine.metrics import book_imbalance

    synth_archive.generate(tmp_path / "arch", 64, 10, 11, list(PRODUCTS))
    arch = load_archive(tmp_path / "arch")
    sub = bp.subset_archive(arch, 40)
    assert sub["bid_px"].shape == (50, 40)
    ref = bp.reference_features(sub["bid_px"], sub["bid_sz"], sub["ask_px"], sub["ask_sz"])
    assert np.all(ref["spread_ticks"] > 0)
    for i in (0, 17, 39):
        bids = [(int(sub["bid_px"][l, i]), float(sub["bid_sz"][l, i])) for l in range(50)]
        asks = [(int(sub["ask_px"][l, i]), float(sub["ask_sz"][l, i])) for l in range(50)]
        assert float(book_imbalance(bids, asks, levels=10)) == pytest.approx(ref["imbalance10"][i], abs=1e-6)
        mid = (bids[0][0] + asks[0][0]) / 2
        micro = (bids[0][0] * asks[0][1] + asks[0][0] * bids[0][1]) / (bids[0][1] + asks[0][1])
        assert ref["microprice_offset"][i] == pytest.approx(micro - mid, abs=1e-9)


def test_consequence_text_states_the_design_implication() -> None:
    text = bp.consequence_text(100.0, 4.0, 2.5, 10.4, 96.2, 6.1)
    assert "Transfers dominate" in text and "96%" in text and "stay resident" in text and "one residency" in text
    other = bp.consequence_text(10.0, 100.0, 2.0, 10.0, 10.7, 9.5)
    assert "Compute dominates" in other


def test_bench_pipeline_imports_without_cupy_and_has_entry_points() -> None:
    assert callable(bp.run) and callable(bp.main) and callable(bp.nsys_child)
    import inspect

    params = list(inspect.signature(bp.run).parameters)
    assert params[:3] == ["bench", "arch", "quick"]


# --------------------------------------------------------------------------- #
# GPU (auto-skipped without nvidia-smi)
# --------------------------------------------------------------------------- #


@pytest.mark.gpu
def test_pipeline_on_gpu_matches_reference_for_both_encodings(tmp_path: Path) -> None:
    import synth_archive
    from bench.common import load_archive
    from exchange_engine.archive import PRODUCTS

    synth_archive.generate(tmp_path / "arch", 3000, 100, 7, list(PRODUCTS))
    arch = load_archive(tmp_path / "arch")
    staging = bp.HostStaging(arch, chunks=5)
    pipe = bp.DevicePipeline(staging)
    ref = bp.reference_features(arch["bid_px"], arch["bid_sz"], arch["ask_px"], arch["ask_sz"])
    for kind in ("serial", "overlapped"):
        res = bp.verify_pass(pipe, ref, kind)
        assert res["verified"], res
        assert res["bad_books_device"] == 0
    info = staging.build_int16(arch)
    assert info["overflow_fallbacks"] == 0
    pipe.ensure_input_capacity(info["max_block_bytes"])
    res = bp.verify_pass(pipe, ref, "overlapped", encoding="int16")
    assert res["verified"] and res["bad_books_device"] == 0, res
    pipe.run_pass("overlapped", instrument=True)
    sums = pipe.stage_sums_ms()
    assert all(v > 0 for v in sums.values())
