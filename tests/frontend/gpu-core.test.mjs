// node:test coverage for the GPU Lab's pure logic (public/js/gpu-core.mjs).
// The inputs here are hand-made test fixtures, not measurements.
import test from "node:test";
import assert from "node:assert/strict";

import {
  DASH, benchmarkRows, binColumns, decadeTicks, engineTiles, fmtGbs, fmtInt, fmtMs, fmtPct, get, heatColor,
  heatDomain, heatT, heatmapLayout, ladderLayout, logScale, ofiRows, ofiSelections, pipelineSegments,
  precisionRows, rawLink, rooflineLayout, scanRows, scatterLayout, sectionStatus, speedupTiles,
} from "../../public/js/gpu-core.mjs";

const ladder = {
  rungs: [
    { name: "v0_aos", label: "v0", kernel_median_ms: 4.0, kernel_iqr_ms: 0.1, e2e_median_ms: 40, pct_of_ceiling: 10, block_size: 256 },
    { name: "v3_soa_fused_vec4", label: "v3", kernel_median_ms: 0.4, kernel_iqr_ms: 0.01, e2e_median_ms: 30, pct_of_ceiling: 90, block_size: 256 },
  ],
  baselines: { cpp_single_ms: 20, numpy_ms: 60 },
  speedups: { kernel_vs_cpp_single: 50, e2e_vs_numpy: 2 },
  ceiling_gbs: 240,
};

test("get tolerates missing paths and returns fallbacks", () => {
  assert.equal(get({ a: { b: [1, 2] } }, "a.b.1"), 2);
  assert.equal(get({}, "x.y", "fb"), "fb");
  assert.equal(get(null, "x", 3), 3);
});

test("formatters handle numbers and missing values", () => {
  assert.equal(fmtInt(863153), "863,153");
  assert.equal(fmtInt(null), DASH);
  assert.match(fmtMs(0.4), /0\.4/);
  assert.match(fmtGbs(221.7), /221\.7/);
  assert.match(fmtPct(92.14), /92\.1/);
});

test("log scale maps decades to equal distances", () => {
  const x = logScale(0.1, 1000, 0, 400);
  assert.equal(x(0.1), 0);
  assert.equal(x(1000), 400);
  assert.ok(Math.abs(x(10) - x(1) - (x(1) - x(0.1))) < 1e-9);
  assert.deepEqual(decadeTicks(0.1, 100), [0.1, 1, 10, 100]);
});

test("ladder layout: one bar per rung, baselines as reference lines, bars ordered by time", () => {
  const L = ladderLayout(ladder, { width: 640 });
  assert.equal(L.bars.length, 2);
  assert.equal(L.refs.length, 2);
  assert.ok(L.bars[0].barX1 > L.bars[1].barX1, "slower rung draws a longer bar");
  assert.ok(L.domain[0] <= 0.4 && L.domain[1] >= 60);
  const tiles = speedupTiles(ladder);
  assert.deepEqual(tiles.map((t) => t.key), ["kernel_vs_cpp_single", "e2e_vs_numpy"]);
});

test("roofline layout flags memory-bound kernels below the ridge point", () => {
  const R = rooflineLayout({ peak_gflops: 8100, bw_ceiling_gbs: 240, bw_theoretical_gbs: 320,
    kernels: [{ name: "v3", intensity_flop_per_byte: 0.3, gflops: 60 }] });
  assert.equal(R.points.length, 1);
  assert.equal(R.points[0].memoryBound, true);
  assert.ok(Math.abs(R.ridge.measured - 8100 / 240) < 1e-9);
  assert.equal(R.lines.length, 3);
  assert.ok(R.points[0].pctOfRoof > 80 && R.points[0].pctOfRoof <= 100);
});

test("heat colour scale is clamped and log-aware", () => {
  assert.equal(heatT(null, { min: 0, max: 1 }), null);
  const d = heatDomain([0.1, 1, 100]);
  assert.equal(d.log, true);
  assert.match(heatColor(0), /^hsl|^rgb|^#/);
  assert.notEqual(heatColor(0), heatColor(1));
});

test("columns are binned to at most maxCols with medians", () => {
  const rows = Array.from({ length: 10 }, (_, i) => [i, i * 2]);
  const b = binColumns(rows, 4);
  assert.ok(b.cols <= 4);
  assert.equal(b.values.length, b.cols);
  assert.equal(b.spans.flat().length >= b.cols, true);
});

test("heatmap layout produces one cell per (column, size)", () => {
  const side = { minutes: [1, 2, 3], bps: [[0.1, 0.5], [0.2, 0.6], [null, 0.7]] };
  const H = heatmapLayout(side, [1000, 10000], { width: 300 });
  assert.equal(H.cells.length, H.cols * H.rows);
  assert.equal(H.rows, 2);
  assert.ok(H.legend.min <= H.legend.max);
});

test("scatter layout clips the fitted line to the plot", () => {
  const S = scatterLayout({ x: [-2, -1, 0, 1, 2], y: [-1, -0.4, 0, 0.6, 1] }, { alpha: 0, beta: 0.5, r2_oos: 0.9, n_test: 5 });
  assert.equal(S.n, 5);
  assert.ok(S.line);
  assert.ok(S.line.x0 >= S.plotLeft - 1e-9 && S.line.x1 <= S.plotRight + 1e-9);
  assert.equal(S.r2, 0.9);
});

test("pipeline segments sum to 100 percent", () => {
  const p = pipelineSegments({ h2d_ms: 30, kernel_ms: 1, d2h_ms: 2, transfer_share_pct: 97, consequence: "keep it resident" });
  assert.equal(p.segments.length, 3);
  assert.ok(Math.abs(p.segments.reduce((s, x) => s + x.pct, 0) - 100) < 1e-9);
  assert.equal(p.consequence, "keep it resident");
});

test("tables mark the production precision variant and the winning scan", () => {
  const rows = precisionRows({ production_choice: "float64", variants: [{ name: "float32_naive", max_rel_err: 1e-3 }, { name: "float64", max_rel_err: 1e-15 }] });
  assert.deepEqual(rows.map((r) => r.production), [false, true]);
  const scans = scanRows({ winner: "cub_device_scan", implementations: [{ name: "blelloch_shared", median_ms: 1 }, { name: "cub_device_scan", median_ms: 0.5 }] });
  assert.deepEqual(scans.map((r) => r.winner), [false, true]);
});

test("order flow imbalance rows and selections are sorted by horizon", () => {
  const ofi = { per_product: { "BTC-USD": { h10s: { r2_oos: 0.1 }, h1s: { r2_oos: 0.2 } } } };
  assert.deepEqual(ofiSelections(ofi).horizons, ["h1s", "h10s"]);
  assert.equal(ofiRows(ofi)[0].horizonKey, "h1s");
});

test("raw links point at the gpu-analytics branch and benchmark rows carry them", () => {
  assert.equal(rawLink("results/run 1/raw/a.json"), "https://github.com/Tanneywanney25/async-matching-engine/blob/gpu-analytics/results/run%201/raw/a.json");
  assert.equal(rawLink(42), null);
  const rows = benchmarkRows([{ section: "ladder", name: "x", value: 0.4, unit: "ms", run_id: "r", raw: "results/r/raw/x.json" }]);
  assert.match(rows[0].href, /results\/r\/raw\/x\.json$/);
});

test("section status and engine tiles tolerate a sparse payload", () => {
  const s = sectionStatus({ ladder, engine: { p99_us: 40.1, orders: 100000 } });
  assert.equal(s.ladder, true);
  assert.equal(s.slippage, false);
  assert.equal(s.engine, true);
  assert.equal(engineTiles({ p99_us: 40.1, orders: 100000 }).length, 2);
  assert.equal(sectionStatus({}).ladder, false);
});
