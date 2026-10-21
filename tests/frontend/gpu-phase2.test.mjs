// node:test coverage for the GPU Lab logic added in 0.3.0 (public/js/gpu-core.mjs).
// The inputs here are hand-made test fixtures, not measurements.
import test from "node:test";
import assert from "node:assert/strict";

import {
  DASH, RAW_BASE, benchmarkRows, fmtUsFine, limitLabel, logFrac, nativeRows, nativeSummary, rawLink, resourceRows,
  resourceSummary, scanRows, sectionStatus, transferBars, transferNotes,
} from "../../public/js/gpu-core.mjs";

const resources = {
  kernels: [
    { kernel: "wide", file: "a.cu", registers: 32, spill_stores: 0, spill_loads: 0, static_smem_bytes: 0, block: 256,
      blocks_per_sm: 4, occupancy: 1.0, limited_by: ["warps"] },
    { kernel: "heavy", file: "b.cu", registers: 66, spill_stores: 0, spill_loads: 0, static_smem_bytes: 0, block: 256,
      blocks_per_sm: 3, occupancy: 0.75, limited_by: ["registers"] },
    { kernel: "cub scan_kernel", file: "libscan_cub.so", registers: 49, spill_stores: null, spill_loads: null,
      static_smem_bytes: 7184, block: 128, blocks_per_sm: 8, occupancy: 1.0, limited_by: ["shared_memory", "warps"] },
  ],
};

test("raw links follow the base the data names, and fall back to the 0.2.0 branch", () => {
  const main = "https://github.com/Tanneywanney25/async-matching-engine/blob/main/";
  assert.equal(rawLink("results/r/raw/x.json", main), `${main}results/r/raw/x.json`);
  assert.equal(rawLink("results/r/raw/x.json", main.slice(0, -1)), `${main}results/r/raw/x.json`);
  assert.equal(rawLink("results/r/raw/x.json", "javascript:alert(1)"), `${RAW_BASE}results/r/raw/x.json`);
  const rows = benchmarkRows([{ section: "s", name: "n", value: 1, unit: "ms", run_id: "r", raw: "results/r/raw/x.json" }], main);
  assert.ok(rows[0].href.startsWith(main));
});

test("resource rows put the lowest occupancy first and keep missing spills as a dash", () => {
  const rows = resourceRows(resources);
  assert.deepEqual(rows.map((r) => r.kernel), ["heavy", "cub scan_kernel", "wide"]);
  assert.equal(rows[0].occupancy, "75%");
  assert.equal(rows[0].limitedBy, "registers");
  assert.equal(rows[0].full, false);
  assert.equal(rows[1].spills, DASH);
  assert.equal(rows[1].limitedBy, "shared memory + warp slots");
  assert.equal(rows[1].smem, "7,184");
  assert.equal(rows[2].occupancyFrac, 1);
});

test("resource summary counts full-occupancy kernels and names the lowest", () => {
  assert.equal(resourceSummary(resources),
    "2 of 3 kernels reach full theoretical occupancy; lowest: heavy at 75% (registers); no kernel spills registers.");
  const allFull = { kernels: [resources.kernels[0]] };
  assert.match(resourceSummary(allFull), /every launched kernel is at full theoretical occupancy/);
  assert.equal(resourceSummary({}), "");
});

test("resource helpers tolerate sparse payloads", () => {
  assert.deepEqual(resourceRows({ kernels: [null, { registers: 3 }] }), []);
  assert.equal(limitLabel([]), DASH);
  assert.equal(sectionStatus({ resources }).resources, true);
  assert.equal(sectionStatus({}).resources, false);
});

const transfer = {
  int16: { bytes_saved_pct: 24.0, fallbacks: 0, chunks: 16, e2e_int32_ms: 56.0, e2e_int16_ms: 43.0, identical: true },
  two_gpu: { one_gpu_ms: 56.4, two_gpu_ms: 56.2, h2d_gbs_one_gpu: 12.4, h2d_gbs_two_gpu: 12.3, link: "PHB", identical: true },
};

test("transfer bars scale each pair to its larger value", () => {
  const bars = transferBars(transfer);
  assert.equal(bars.length, 4);
  assert.deepEqual(bars.map((b) => b.label), ["int32 prices", "int16 offsets", "one card", "two cards"]);
  assert.equal(bars[0].frac, 1);
  assert.ok(Math.abs(bars[1].frac - 43 / 56) < 1e-12);
  assert.equal(bars[2].frac, 1);
  assert.deepEqual(transferBars({ int16: { e2e_int32_ms: 5 } }), []);
});

test("transfer notes say what moved and whether outputs stayed identical", () => {
  const notes = transferNotes(transfer);
  assert.match(notes[0], /24\.0% fewer bytes; 0 of 16 chunks fell back to int32; outputs identical bit for bit/);
  assert.match(notes[1], /12\.4 GB\/s on one card and 12\.3 GB\/s on two together \(topology PHB\)/);
  assert.match(transferNotes({ int16: { ...transfer.int16, identical: false } })[0], /NOT identical/);
});

const native = {
  paths: [
    { key: "native", label: "C++ core, native", add: { p50_us: 0.083, p99_us: 0.417, p999_us: 0.627 }, cancel: { p99_us: 0.4 } },
    { key: "python_engine", label: "Python engine", add: { p50_us: 6.3, p99_us: 29.1, p999_us: 56.2 }, cancel: { p99_us: 6.0 } },
  ],
  fuzz: { streams: 10000, operations: 548317, divergences: 0 },
  allocations: 0, alloc_operations: 200000, speedup_p99_vs_python: 69.9,
};

test("native rows format sub-microsecond latencies and place them on a log axis", () => {
  const rows = nativeRows(native);
  assert.equal(rows[0].addP99, "0.417 µs");
  assert.equal(rows[1].addP99, "29.1 µs");
  assert.ok(rows[0].native && !rows[1].native);
  assert.deepEqual(rows.map((r) => r.short), ["C++ native", "Python engine"]);
  assert.ok(rows[0].p99Frac > 0 && rows[0].p99Frac < rows[1].p99Frac && rows[1].p99Frac < 1);
  assert.equal(fmtUsFine(1.194), "1.19 µs");
  assert.equal(logFrac(0.03), 0);
  assert.equal(logFrac(1000), 1);
  assert.equal(logFrac(-1), 0);
});

test("native summary and section status", () => {
  assert.equal(nativeSummary(native),
    "10,000 fuzzed streams against the reference book, 0 divergences; 0 heap allocations over 200,000 operations; " +
    "p99 add 69.9× faster than the Python engine.");
  assert.equal(nativeSummary(null), "");
  assert.equal(sectionStatus({ native_engine: native, transfer }).native, true);
  assert.equal(sectionStatus({ native_engine: native, transfer }).transfer, true);
  assert.equal(sectionStatus({}).native, false);
});

test("scan rows carry the share of the read ceiling", () => {
  const rows = scanRows({ winner: "cub", implementations: [{ name: "decoupled_lookback", median_ms: 1.79, gbs: 150, pct_of_read_ceiling: 57.2 }] });
  assert.equal(rows[0].pctCeiling, "57.2%");
  assert.equal(scanRows({ implementations: [{ name: "x" }] })[0].pctCeiling, DASH);
});
