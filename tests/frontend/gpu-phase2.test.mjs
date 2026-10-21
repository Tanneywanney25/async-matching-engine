// node:test coverage for the GPU Lab logic added in 0.3.0 (public/js/gpu-core.mjs).
// The inputs here are hand-made test fixtures, not measurements.
import test from "node:test";
import assert from "node:assert/strict";

import { DASH, RAW_BASE, benchmarkRows, limitLabel, rawLink, resourceRows, resourceSummary, sectionStatus } from "../../public/js/gpu-core.mjs";

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
