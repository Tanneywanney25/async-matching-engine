// Pure, DOM-free logic for the GPU Lab section of the dashboard.
//
// public/js/gpu-lab.mjs is the DOM glue; everything here is deterministic and
// side-effect free so it can be unit-tested with `node --test`
// (tests/frontend/gpu-core.test.mjs). The input is public/data/gpu_lab.json,
// written by scripts/render_docs.py from committed T4 results (shape in
// bench/CONTRACT.md). Every function tolerates a missing or partial section:
// nothing here throws on absent data, it returns empty layouts and dashes.

export const RAW_BASE = "https://github.com/Tanneywanney25/async-matching-engine/blob/gpu-analytics/";
export const DASH = "—";
export const SIDES = ["buy", "sell"];

// --- Safe access ------------------------------------------------------------

/** Read a dotted (or array) path, returning `fallback` for any missing link. */
export function get(obj, path, fallback = undefined) {
  const parts = Array.isArray(path) ? path : String(path).split(".");
  let cur = obj;
  for (const key of parts) {
    if (cur == null || typeof cur !== "object") return fallback;
    cur = cur[key];
  }
  return cur == null ? fallback : cur;
}

export function isNum(v) {
  return typeof v === "number" && Number.isFinite(v);
}

export function num(v, fallback = null) {
  return isNum(v) ? v : fallback;
}

export function list(v) {
  return Array.isArray(v) ? v : [];
}

export function keys(v) {
  return v && typeof v === "object" && !Array.isArray(v) ? Object.keys(v) : [];
}

// --- Formatting -------------------------------------------------------------
// Locale-independent on purpose: node:test runs these under whatever ICU the
// machine has, and the dashboard should read the same everywhere.

export function groupThousands(digits) {
  return String(digits).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/** Fixed-point number with thousands separators; dash when not finite. */
export function fmtNum(v, digits = 2) {
  if (!isNum(v)) return DASH;
  const fixed = Math.abs(v).toFixed(digits);
  const [whole, frac] = fixed.split(".");
  const sign = v < 0 && Number(fixed) !== 0 ? "-" : "";
  return `${sign}${groupThousands(whole)}${frac ? `.${frac}` : ""}`;
}

export function fmtInt(v) {
  return isNum(v) ? fmtNum(Math.round(v), 0) : DASH;
}

/** Milliseconds with precision adapted to magnitude: 0.097 ms, 5.70 ms, 31.2 ms, 1,642 ms. */
export function fmtMs(v) {
  if (!isNum(v)) return DASH;
  const a = Math.abs(v);
  return `${fmtNum(v, a < 1 ? 3 : a < 10 ? 2 : a < 100 ? 1 : 0)} ms`;
}

export function fmtUs(v) {
  if (!isNum(v)) return DASH;
  return `${fmtNum(v, Math.abs(v) < 100 ? 1 : 0)} µs`;
}

export function fmtGbs(v) {
  return isNum(v) ? `${fmtNum(v, 1)} GB/s` : DASH;
}

/** Percentage points in, "71.3%" out (the JSON already carries percentages). */
export function fmtPct(v, digits = 1) {
  return isNum(v) ? `${fmtNum(v, digits)}%` : DASH;
}

/** Speedup factor: 72.4×, 313×. */
export function fmtX(v) {
  if (!isNum(v)) return DASH;
  return `${Math.abs(v) >= 100 ? fmtInt(v) : fmtNum(v, 1)}×`;
}

/** Relative error in compact scientific notation: 3.20e-7. */
export function fmtRelErr(v) {
  if (!isNum(v)) return DASH;
  if (v === 0) return "0";
  const [mantissa, exponent] = v.toExponential(2).split("e");
  return `${mantissa}e${Number(exponent)}`;
}

/** `digits` significant digits without trailing zeros. */
export function sig(v, digits = 3) {
  if (!isNum(v)) return DASH;
  if (v === 0) return "0";
  return String(Number(v.toPrecision(digits)));
}

/** SI-prefixed magnitude: 13.1G, 2.49G, 812. */
export function fmtSi(v, digits = 3) {
  if (!isNum(v)) return DASH;
  const a = Math.abs(v);
  for (const [suffix, factor] of [["T", 1e12], ["G", 1e9], ["M", 1e6], ["K", 1e3]]) {
    if (a >= factor) return `${sig(v / factor, digits)}${suffix}`;
  }
  return sig(v, digits);
}

/** Throughput: 13.1G/s. */
export function fmtRate(v, unit = "/s") {
  return isNum(v) ? `${fmtSi(v)}${unit}` : DASH;
}

/** Order-size labels: $1K, $100K, $3.2M, $10M. */
export function fmtUsdCompact(v) {
  if (!isNum(v)) return DASH;
  const a = Math.abs(v);
  for (const [suffix, factor] of [["M", 1e6], ["K", 1e3]]) {
    if (a >= factor) {
      const q = v / factor;
      const rounded = Number(q.toFixed(1));
      return `$${Number.isInteger(rounded) ? String(Math.round(q)) : q.toFixed(1)}${suffix}`;
    }
  }
  return `$${fmtNum(v, 0)}`;
}

export function fmtBps(v) {
  if (!isNum(v)) return DASH;
  const a = Math.abs(v);
  return `${fmtNum(v, a < 1 ? 3 : a < 10 ? 2 : a < 100 ? 1 : 0)} bps`;
}

/** Axis tick label: compact above 1,000, up to three significant digits below. */
export function fmtTick(v) {
  if (!isNum(v)) return "";
  if (Math.abs(v) >= 1000) return fmtSi(v);
  return sig(v, 3);
}

export function pad2(n) {
  return String(n).padStart(2, "0");
}

/**
 * Minute label for the slippage surface. Epoch minutes (anything above 1e6,
 * i.e. the renderer kept absolute time) become a UTC clock time; small values
 * are offsets from the start of the capture.
 */
export function fmtMinute(m) {
  if (!isNum(m)) return DASH;
  if (m > 1e6) {
    const d = new Date(m * 60000);
    return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`;
  }
  const hours = Math.floor(m / 60);
  const mins = Math.round(m % 60);
  return hours > 0 ? `${hours}h${pad2(mins)}` : `${mins} min`;
}

export function minuteAxisTitle(minutes) {
  const first = list(minutes).find(isNum);
  return isNum(first) && first > 1e6 ? "time of day (UTC), one column per minute bin" : "minutes since capture start";
}

export function shortSha(sha) {
  if (typeof sha !== "string" || !sha) return DASH;
  return sha.length >= 7 ? sha.slice(0, 7) : sha;
}

/** "h1s" -> "1 s", "h10s" -> "10 s". */
export function horizonLabel(key) {
  const m = /^h(\d+(?:\.\d+)?)s$/.exec(String(key ?? ""));
  return m ? `${m[1]} s` : String(key ?? DASH);
}

export function truncate(text, max) {
  const s = String(text ?? "");
  return s.length > max ? `${s.slice(0, Math.max(1, max - 1))}…` : s;
}

// --- Scales and statistics --------------------------------------------------

export function log10(v) {
  return Math.log(v) / Math.LN10;
}

export function clamp01(t) {
  return Math.min(1, Math.max(0, t));
}

export function linearScale(d0, d1, r0, r1) {
  const span = d1 - d0 || 1;
  const f = (v) => r0 + ((v - d0) / span) * (r1 - r0);
  f.invert = (p) => d0 + ((p - r0) / (r1 - r0 || 1)) * span;
  f.domain = [d0, d1];
  f.range = [r0, r1];
  return f;
}

/** Base-10 logarithmic scale; non-positive or degenerate domains are repaired. */
export function logScale(d0, d1, r0, r1) {
  const lo = d0 > 0 ? d0 : 1;
  let hi = d1 > 0 ? d1 : lo * 10;
  if (hi === lo) hi = lo * 10;
  const l0 = log10(lo);
  const l1 = log10(hi);
  const f = (v) => r0 + ((log10(v) - l0) / (l1 - l0)) * (r1 - r0);
  f.invert = (p) => 10 ** (l0 + ((p - r0) / (r1 - r0)) * (l1 - l0));
  f.domain = [lo, hi];
  f.range = [r0, r1];
  return f;
}

/** Powers of ten inside [d0, d1]. */
export function decadeTicks(d0, d1) {
  if (!(d0 > 0) || !(d1 > 0)) return [];
  const lo = Math.min(d0, d1);
  const hi = Math.max(d0, d1);
  const out = [];
  for (let e = Math.ceil(log10(lo) - 1e-9); e <= Math.floor(log10(hi) + 1e-9); e++) {
    out.push(Number((10 ** e).toPrecision(12)));
  }
  return out;
}

export function niceStep(rough) {
  if (!(rough > 0)) return 1;
  const exp = Math.floor(log10(rough));
  const base = rough / 10 ** exp;
  const nice = base < 1.5 ? 1 : base < 3 ? 2 : base < 7 ? 5 : 10;
  return Number((nice * 10 ** exp).toPrecision(12));
}

/** Round tick values covering [min, max] with about `count` steps. */
export function niceTicks(min, max, count = 5) {
  if (!isNum(min) || !isNum(max)) return [];
  let lo = Math.min(min, max);
  let hi = Math.max(min, max);
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const step = niceStep((hi - lo) / Math.max(1, count));
  const out = [];
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) {
    out.push(Number((k * step).toPrecision(12)));
  }
  return out;
}

export function padDomain(min, max, frac = 0.05) {
  if (!isNum(min) || !isNum(max)) return [0, 1];
  if (min === max) {
    const d = Math.abs(min) * 0.1 || 1;
    return [min - d, max + d];
  }
  const pad = (max - min) * frac;
  return [min - pad, max + pad];
}

export function extent(values) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of list(values)) {
    if (!isNum(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : null;
}

export function median(values) {
  const xs = [];
  for (const v of list(values)) if (isNum(v)) xs.push(v);
  if (!xs.length) return null;
  xs.sort((a, b) => a - b);
  const mid = xs.length >> 1;
  return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
}

/**
 * Push overlapping labels apart. `items` are {px, py}; the result is one label
 * y per item, nudged down by `minGap` when a neighbour close in x sits above.
 */
export function spreadLabels(items, { minGap = 12, nearX = 60 } = {}) {
  const order = items.map((p, i) => ({ i, px: p.px, py: p.py })).sort((a, b) => a.py - b.py || a.px - b.px);
  const out = new Array(items.length);
  const placed = [];
  for (const it of order) {
    let y = it.py;
    for (const q of placed) {
      if (Math.abs(q.px - it.px) < nearX && y - q.y < minGap) y = q.y + minGap;
    }
    placed.push({ px: it.px, y });
    out[it.i] = y;
  }
  return out;
}

/** Liang-Barsky clip of a segment to `rect` {x0, y0, x1, y1}; null when outside. */
export function clipSegment(x0, y0, x1, y1, rect) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const p = [-dx, dx, -dy, dy];
  const q = [x0 - rect.x0, rect.x1 - x0, y0 - rect.y0, rect.y1 - y0];
  let t0 = 0;
  let t1 = 1;
  for (let k = 0; k < 4; k++) {
    if (p[k] === 0) {
      if (q[k] < 0) return null;
      continue;
    }
    const r = q[k] / p[k];
    if (p[k] < 0) {
      if (r > t1) return null;
      if (r > t0) t0 = r;
    } else {
      if (r < t0) return null;
      if (r < t1) t1 = r;
    }
  }
  return [x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy];
}

// --- Optimization ladder -----------------------------------------------------

export const BASELINES = [
  { key: "cpp_single_ms", label: "C++ single-thread", cls: "ref-cpp" },
  { key: "cpp_omp_ms", label: "C++ OpenMP", cls: "ref-omp" },
  { key: "numpy_ms", label: "NumPy", cls: "ref-numpy" },
  { key: "cupy_kernel_ms", label: "CuPy kernels", cls: "ref-cupy" },
];

export function rungLabel(rung) {
  const label = get(rung, "label");
  if (typeof label === "string" && label) return label;
  const name = get(rung, "name");
  return typeof name === "string" ? name.replace(/_/g, " ") : DASH;
}

/**
 * Horizontal bars (one row per rung, label above the bar) on a log time axis,
 * so the GPU kernels and the CPU/library baselines, orders of magnitude apart,
 * share one honest axis where equal distances are equal ratios.
 */
export function ladderLayout(ladder, opts = {}) {
  const { width = 640, rowHeight = 44, left = 8, right = 16, top = 52, bottom = 36, barHeight = 14 } = opts;
  const rungs = list(get(ladder, "rungs")).filter((r) => r && isNum(r.kernel_median_ms) && r.kernel_median_ms > 0);
  const baselines = get(ladder, "baselines", {}) || {};
  const refs = BASELINES.filter((b) => isNum(baselines[b.key]) && baselines[b.key] > 0).map((b) => ({ ...b, ms: baselines[b.key] }));

  const values = [
    ...rungs.map((r) => r.kernel_median_ms),
    ...rungs.map((r) => r.e2e_median_ms).filter((v) => isNum(v) && v > 0),
    ...refs.map((r) => r.ms),
  ];
  const vmin = values.length ? Math.min(...values) : 1;
  const vmax = values.length ? Math.max(...values) : 1000;
  const dmin = 10 ** Math.floor(log10(vmin));
  const dmax = 10 ** Math.ceil(log10(vmax * 1.25));
  const plotLeft = left;
  const plotRight = Math.max(left + 10, width - right);
  const x = logScale(dmin, dmax, plotLeft, plotRight);
  const height = top + rungs.length * rowHeight + bottom;
  const plotBottom = height - bottom;
  const maxChars = Math.max(24, Math.floor((plotRight - plotLeft) / 6.4));

  const bars = rungs.map((r, i) => {
    const y = top + i * rowHeight;
    const x1 = x(r.kernel_median_ms);
    const e2e = isNum(r.e2e_median_ms) && r.e2e_median_ms > 0 ? x(r.e2e_median_ms) : null;
    const pct = num(r.pct_of_ceiling);
    const valueText = `${fmtMs(r.kernel_median_ms)} kernel${pct != null ? ` · ${fmtPct(pct, 0)} of ceiling` : ""}`;
    const inside = x1 + 150 > plotRight;
    return {
      name: String(r.name ?? ""),
      label: rungLabel(r),
      rowLabel: truncate(rungLabel(r), maxChars),
      kernelMs: r.kernel_median_ms,
      iqrMs: num(r.kernel_iqr_ms),
      e2eMs: num(r.e2e_median_ms),
      pctOfCeiling: pct,
      pctOfTheoretical: num(r.pct_of_theoretical),
      bwGbs: num(r.effective_bw_gbs),
      blockSize: num(r.block_size),
      labelY: y + 11,
      barY: y + 18,
      barX0: plotLeft,
      barX1: x1,
      barH: barHeight,
      e2eX: e2e,
      valueText,
      valueX: inside ? x1 - 6 : x1 + 6,
      valueAnchor: inside ? "end" : "start",
      valueInside: inside,
    };
  });

  const refLines = refs.map((r, i) => {
    const rx = x(r.ms);
    const flip = rx + 130 > plotRight;
    return {
      ...r,
      x: rx,
      y0: top - 4,
      y1: plotBottom,
      labelX: flip ? rx - 4 : rx + 4,
      labelY: top - 8 - (i % 3) * 12,
      labelAnchor: flip ? "end" : "start",
      text: `${r.label} ${fmtMs(r.ms)}`,
    };
  });

  return {
    width,
    height,
    plotLeft,
    plotRight,
    plotTop: top,
    plotBottom,
    domain: [dmin, dmax],
    bars,
    refs: refLines,
    ticks: decadeTicks(dmin, dmax).map((v) => ({ v, x: x(v), label: fmtTick(v) })),
    ceilingGbs: num(get(ladder, "ceiling_gbs")),
    theoreticalGbs: num(get(ladder, "theoretical_gbs")),
    axisLabel: "median time, ms (log scale)",
  };
}

export function speedupTiles(ladder) {
  const s = get(ladder, "speedups", {}) || {};
  const defs = [
    ["kernel_vs_cpp_single", "Kernel vs C++ single-thread"],
    ["e2e_vs_numpy", "End to end vs NumPy"],
    ["kernel_vs_cpp_omp", "Kernel vs C++ OpenMP"],
    ["kernel_vs_cupy", "Kernel vs CuPy"],
  ];
  return defs.filter(([k]) => isNum(s[k])).map(([key, label]) => ({ key, label, value: fmtX(s[key]), raw: s[key] }));
}

export function ladderRows(ladder) {
  return list(get(ladder, "rungs")).filter(Boolean).map((r) => ({
    name: String(r.name ?? ""),
    label: rungLabel(r),
    kernel: fmtMs(r.kernel_median_ms),
    iqr: fmtMs(r.kernel_iqr_ms),
    e2e: fmtMs(r.e2e_median_ms),
    bw: fmtGbs(r.effective_bw_gbs),
    pctCeiling: fmtPct(r.pct_of_ceiling),
    pctTheo: fmtPct(r.pct_of_theoretical),
    block: fmtInt(r.block_size),
  }));
}

// --- Roofline ---------------------------------------------------------------

/**
 * Data-space points of y = min(bw * x, peak) over the x domain, clipped to the
 * bottom of the y domain. Without a peak the slope never bends.
 */
export function roofPolyline(bw, peak, dx, dy) {
  if (!(bw > 0)) return [];
  const [x0, x1] = dx;
  const [y0] = dy;
  let xs = x0;
  if (bw * xs < y0) xs = y0 / bw;
  if (xs > x1) return [];
  const ridge = isNum(peak) && peak > 0 ? peak / bw : Infinity;
  if (ridge <= xs) return [{ x: xs, y: peak }, { x: x1, y: peak }];
  const pts = [{ x: xs, y: bw * xs }];
  if (ridge < x1) pts.push({ x: ridge, y: peak }, { x: x1, y: peak });
  else pts.push({ x: x1, y: bw * x1 });
  return pts;
}

export function rooflineLayout(roofline, opts = {}) {
  const { width = 640, height = 360, left = 52, right = 16, top = 18, bottom = 44 } = opts;
  const peak = num(get(roofline, "peak_gflops"));
  const bwC = num(get(roofline, "bw_ceiling_gbs"));
  const bwT = num(get(roofline, "bw_theoretical_gbs"));
  const bws = [bwC, bwT].filter((b) => b > 0);
  const kernels = list(get(roofline, "kernels")).filter(
    (k) => k && isNum(k.intensity_flop_per_byte) && k.intensity_flop_per_byte > 0 && isNum(k.gflops) && k.gflops > 0,
  );
  const ridges = peak > 0 ? bws.map((b) => peak / b) : [];
  const xs = [...kernels.map((k) => k.intensity_flop_per_byte), ...ridges];
  const xmin = 10 ** Math.floor(log10(xs.length ? Math.min(...xs) : 0.01));
  const xmax = 10 ** Math.ceil(log10((xs.length ? Math.max(...xs) : 100) * 1.25));
  const lows = [...kernels.map((k) => k.gflops), ...bws.map((b) => b * xmin)];
  const ymin = 10 ** Math.floor(log10(lows.length ? Math.min(...lows) : 1));
  const highs = peak > 0 ? [peak] : [...kernels.map((k) => k.gflops), ...bws.map((b) => b * xmax)];
  const ymax = 10 ** Math.ceil(log10((highs.length ? Math.max(...highs) : 10000) * 1.05));
  const plotRight = Math.max(left + 10, width - right);
  const plotBottom = height - bottom;
  const x = logScale(xmin, xmax, left, plotRight);
  const y = logScale(ymin, ymax, plotBottom, top);

  const lines = [];
  if (bwC > 0) lines.push({ key: "ceiling", cls: "roof-measured", label: `measured bandwidth ceiling ${fmtGbs(bwC)}`, data: roofPolyline(bwC, peak, [xmin, xmax], [ymin, ymax]) });
  if (bwT > 0) lines.push({ key: "theoretical", cls: "roof-theoretical", label: `theoretical ${fmtGbs(bwT)}`, data: roofPolyline(bwT, peak, [xmin, xmax], [ymin, ymax]) });
  if (peak > 0) lines.push({ key: "peak", cls: "roof-peak", label: `T4 FP32 peak ${fmtNum(peak / 1000, 1)} TFLOP/s`, data: [{ x: xmin, y: peak }, { x: xmax, y: peak }] });
  for (const l of lines) l.points = l.data.map((p) => ({ px: x(p.x), py: y(p.y) }));

  const points = kernels.map((k) => {
    const roofAt = bwC > 0 ? Math.min(bwC * k.intensity_flop_per_byte, peak > 0 ? peak : Infinity) : null;
    return {
      name: String(k.name ?? ""),
      intensity: k.intensity_flop_per_byte,
      gflops: k.gflops,
      px: x(k.intensity_flop_per_byte),
      py: y(k.gflops),
      pctOfRoof: roofAt ? (k.gflops / roofAt) * 100 : null,
      memoryBound: isNum(peak) && bwC > 0 ? k.intensity_flop_per_byte < peak / bwC : null,
    };
  });
  const labelYs = spreadLabels(points);
  points.forEach((p, i) => {
    const roomRight = p.px + 8 + Math.min(140, p.name.length * 6.2) <= plotRight;
    p.labelY = labelYs[i] + 4;
    p.labelX = roomRight ? p.px + 8 : p.px - 8;
    p.labelAnchor = roomRight ? "start" : "end";
  });

  return {
    width,
    height,
    plotLeft: left,
    plotRight,
    plotTop: top,
    plotBottom,
    xDomain: [xmin, xmax],
    yDomain: [ymin, ymax],
    lines,
    points,
    xTicks: decadeTicks(xmin, xmax).map((v) => ({ v, x: x(v), label: fmtTick(v) })),
    yTicks: decadeTicks(ymin, ymax).map((v) => ({ v, y: y(v), label: fmtTick(v) })),
    ridge: { measured: peak > 0 && bwC > 0 ? peak / bwC : null, theoretical: peak > 0 && bwT > 0 ? peak / bwT : null },
    peak,
    bwCeiling: bwC,
    bwTheoretical: bwT,
  };
}

export function rooflineRows(roofline) {
  return rooflineLayout(roofline).points.map((p) => ({
    name: p.name,
    intensity: fmtNum(p.intensity, 3),
    gflops: fmtNum(p.gflops, 1),
    pctOfRoof: fmtPct(p.pctOfRoof),
    bound: p.memoryBound == null ? DASH : p.memoryBound ? "memory" : "compute",
  }));
}

// --- Slippage heatmap -------------------------------------------------------
// One hue (navy -> the theme's cyan) so magnitude reads as brightness on the
// dark panel; blank cells are sizes the book could not fill within 50 levels.

export const HEAT_LOW = [217, 60, 18];
export const HEAT_HIGH = [183, 85, 74];

export function heatColor(t) {
  const u = clamp01(isNum(t) ? t : 0);
  const mix = (a, b) => (a + (b - a) * u).toFixed(1);
  return `hsl(${mix(HEAT_LOW[0], HEAT_HIGH[0])} ${mix(HEAT_LOW[1], HEAT_HIGH[1])}% ${mix(HEAT_LOW[2], HEAT_HIGH[2])}%)`;
}

export function heatDomain(values, { logRatio = 20 } = {}) {
  const ex = extent(values);
  if (!ex) return { min: 0, max: 1, log: false, n: 0 };
  const [min, max] = ex;
  let n = 0;
  for (const v of list(values)) if (isNum(v)) n++;
  return { min, max, log: min > 0 && max / min > logRatio, n };
}

export function heatT(v, domain) {
  if (!isNum(v)) return null;
  const { min, max, log } = domain;
  if (!(max > min)) return 0.5;
  if (log && v > 0) return clamp01((log10(v) - log10(min)) / (log10(max) - log10(min)));
  return clamp01((v - min) / (max - min));
}

/** Bin `rows` (per minute, each an array over sizes) into at most `maxCols` columns of medians. */
export function binColumns(rows, maxCols) {
  const n = list(rows).length;
  if (!n) return { cols: 0, values: [], spans: [] };
  const cols = Math.max(1, Math.min(maxCols, n));
  const width = list(rows[0]).length;
  const values = [];
  const spans = [];
  for (let c = 0; c < cols; c++) {
    const i0 = Math.floor((c * n) / cols);
    const i1 = Math.max(Math.floor(((c + 1) * n) / cols), i0 + 1);
    spans.push([i0, i1]);
    const col = new Array(width);
    for (let s = 0; s < width; s++) {
      const vals = [];
      for (let i = i0; i < i1; i++) {
        const v = rows[i] ? rows[i][s] : null;
        if (isNum(v)) vals.push(v);
      }
      col[s] = vals.length ? median(vals) : null;
    }
    values.push(col);
  }
  return { cols, values, spans };
}

/** Index of the size nearest each power of ten within the size range. */
export function decadeIndices(sizes) {
  const xs = list(sizes).filter((s) => isNum(s) && s > 0);
  if (!xs.length) return [];
  const out = [];
  for (const d of decadeTicks(Math.min(...xs), Math.max(...xs))) {
    let best = 0;
    let bestDist = Infinity;
    xs.forEach((s, i) => {
      const dist = Math.abs(log10(s) - log10(d));
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    });
    out.push({ index: best, value: d, label: fmtUsdCompact(d) });
  }
  return out;
}

export function heatmapLayout(side, sizes, opts = {}) {
  const { width = 640, maxCols = 96, rowHeight = 4, left = 48, right = 10, top = 8, bottom = 32 } = opts;
  const minutes = list(get(side, "minutes"));
  const rows = list(get(side, "bps")).filter(Array.isArray);
  const sizeList = list(sizes).filter(isNum);
  const binned = binColumns(rows, maxCols);
  const nRows = sizeList.length || (rows[0] ? rows[0].length : 0);
  const plotW = Math.max(10, width - left - right);
  const cw = binned.cols ? plotW / binned.cols : plotW;
  const all = [];
  for (const col of binned.values) for (const v of col) if (isNum(v)) all.push(v);
  const domain = heatDomain(all);

  const cells = [];
  let nullCount = 0;
  for (let c = 0; c < binned.cols; c++) {
    for (let r = 0; r < nRows; r++) {
      const v = binned.values[c][r];
      const t = heatT(v, domain);
      if (t == null) nullCount++;
      cells.push({
        col: c,
        row: r,
        x: left + c * cw,
        y: top + (nRows - 1 - r) * rowHeight,
        w: cw,
        h: rowHeight,
        value: isNum(v) ? v : null,
        color: t == null ? null : heatColor(t),
        size: sizeList[r],
        minute: minutes[binned.spans[c][0]],
      });
    }
  }
  const plotBottom = top + nRows * rowHeight;
  const yTicks = decadeIndices(sizeList).map((d) => ({ ...d, y: top + (nRows - 1 - d.index) * rowHeight + rowHeight / 2 }));
  const xTicks = [];
  const tickCount = Math.min(6, binned.cols);
  for (let k = 0; k < tickCount; k++) {
    const c = tickCount === 1 ? 0 : Math.round((k * (binned.cols - 1)) / (tickCount - 1));
    xTicks.push({ col: c, x: left + c * cw + cw / 2, label: fmtMinute(minutes[binned.spans[c][0]]) });
  }
  const stops = Array.from({ length: 8 }, (_, i) => heatColor(i / 7));
  return {
    width,
    height: plotBottom + bottom,
    plotLeft: left,
    plotRight: left + plotW,
    plotTop: top,
    plotBottom,
    cols: binned.cols,
    rows: nRows,
    cells,
    yTicks,
    xTicks,
    xTitle: minuteAxisTitle(minutes),
    domain,
    nullCount,
    legend: { min: domain.min, max: domain.max, log: domain.log, stops, minLabel: fmtBps(domain.min), maxLabel: fmtBps(domain.max) },
  };
}

/** Table twin of the heatmap: min / median / max slippage at each decade size. */
export function heatmapSummary(side, sizes) {
  const rows = list(get(side, "bps")).filter(Array.isArray);
  const sizeList = list(sizes).filter(isNum);
  return decadeIndices(sizeList).map((d) => {
    const col = rows.map((r) => r[d.index]);
    const ex = extent(col);
    return {
      size: d.label,
      min: fmtBps(ex ? ex[0] : null),
      median: fmtBps(median(col)),
      max: fmtBps(ex ? ex[1] : null),
      fillable: ex != null,
    };
  });
}

export function slippageProducts(slippage) {
  return keys(get(slippage, "products")).sort();
}

export function slippageNote(slippage) {
  const warp = num(get(slippage, "orders_per_s_warp"));
  const naive = num(get(slippage, "orders_per_s_naive"));
  const speedup = num(get(slippage, "speedup"));
  if (warp == null && naive == null) return "";
  const parts = [];
  if (warp != null) parts.push(`warp-cooperative kernel ${fmtRate(warp)} simulated orders`);
  if (naive != null) parts.push(`naive one-thread-per-order ${fmtRate(naive)}`);
  let text = parts.join(", ");
  if (speedup != null) text += ` (${fmtX(speedup)} from removing the divergent walk)`;
  return `${text}.`;
}

// --- Order flow imbalance ---------------------------------------------------

export function scatterLayout(sample, fit, opts = {}) {
  const { width = 640, height = 300, left = 52, right = 14, top = 12, bottom = 40, maxPoints = 2000 } = opts;
  const xsRaw = list(get(sample, "x"));
  const ysRaw = list(get(sample, "y"));
  const n = Math.min(xsRaw.length, ysRaw.length, maxPoints);
  const raw = [];
  for (let i = 0; i < n; i++) if (isNum(xsRaw[i]) && isNum(ysRaw[i])) raw.push({ x: xsRaw[i], y: ysRaw[i] });
  const ex = extent(raw.map((p) => p.x)) || [-1, 1];
  const ey = extent(raw.map((p) => p.y)) || [-1, 1];
  const xd = padDomain(ex[0], ex[1]);
  const yd = padDomain(ey[0], ey[1]);
  const plotRight = Math.max(left + 10, width - right);
  const plotBottom = height - bottom;
  const x = linearScale(xd[0], xd[1], left, plotRight);
  const y = linearScale(yd[0], yd[1], plotBottom, top);
  const points = raw.map((p) => ({ ...p, px: x(p.x), py: y(p.y) }));

  const alpha = num(get(fit, "alpha"));
  const beta = num(get(fit, "beta"));
  let line = null;
  if (alpha != null && beta != null) {
    const seg = clipSegment(xd[0], alpha + beta * xd[0], xd[1], alpha + beta * xd[1], { x0: xd[0], y0: yd[0], x1: xd[1], y1: yd[1] });
    if (seg) line = { x0: x(seg[0]), y0: y(seg[1]), x1: x(seg[2]), y1: y(seg[3]), alpha, beta };
  }
  return {
    width,
    height,
    plotLeft: left,
    plotRight,
    plotTop: top,
    plotBottom,
    xDomain: xd,
    yDomain: yd,
    points,
    line,
    xTicks: niceTicks(xd[0], xd[1], 5).map((v) => ({ v, x: x(v), label: fmtTick(v) })),
    yTicks: niceTicks(yd[0], yd[1], 5).map((v) => ({ v, y: y(v), label: fmtTick(v) })),
    zero: { x: xd[0] <= 0 && 0 <= xd[1] ? x(0) : null, y: yd[0] <= 0 && 0 <= yd[1] ? y(0) : null },
    n: points.length,
    r2: num(get(fit, "r2_oos")),
    nTest: num(get(fit, "n_test")),
    alpha,
    beta,
  };
}

function horizonOrder(a, b) {
  const na = Number((/^h([\d.]+)s$/.exec(a) || [])[1]);
  const nb = Number((/^h([\d.]+)s$/.exec(b) || [])[1]);
  if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
  return String(a).localeCompare(String(b));
}

export function ofiSelections(ofi) {
  const per = get(ofi, "per_product", {}) || {};
  const products = keys(per).sort();
  const horizons = new Set();
  for (const p of products) for (const hz of keys(per[p])) horizons.add(hz);
  return { products, horizons: [...horizons].sort(horizonOrder) };
}

export function ofiRows(ofi) {
  const per = get(ofi, "per_product", {}) || {};
  const rows = [];
  for (const product of keys(per).sort()) {
    for (const hz of keys(per[product]).sort(horizonOrder)) {
      const r = per[product][hz] || {};
      rows.push({
        product,
        horizon: horizonLabel(hz),
        horizonKey: hz,
        r2: fmtNum(r.r2_oos, 4),
        beta: fmtNum(r.beta, 3),
        alpha: fmtNum(r.alpha, 4),
        nTest: fmtInt(r.n_test),
      });
    }
  }
  return rows;
}

// --- Tables -----------------------------------------------------------------

export function precisionRows(precision) {
  const choice = get(precision, "production_choice");
  return list(get(precision, "variants")).filter(Boolean).map((v) => ({
    name: String(v.name ?? ""),
    maxRelErr: fmtRelErr(v.max_rel_err),
    medianRelErr: fmtRelErr(v.median_rel_err),
    medianMs: fmtMs(v.median_ms),
    production: v.name === choice,
  }));
}

export function scanRows(scans) {
  const winner = get(scans, "winner");
  return list(get(scans, "implementations")).filter(Boolean).map((s) => ({
    name: String(s.name ?? ""),
    medianMs: fmtMs(s.median_ms),
    elementsPerS: fmtRate(s.elements_per_s),
    gbs: fmtGbs(s.gbs),
    winner: s.name === winner,
  }));
}

/** Link a results-relative raw path to the file on GitHub. `base` defaults to the 0.2.0 branch;
 * gpu_lab.json carries `raw_base` (the branch the results were merged into) and the page passes it. */
export function rawLink(raw, base = RAW_BASE) {
  if (typeof raw !== "string") return null;
  const cleaned = raw.trim().replace(/\\/g, "/").replace(/^(\.\/)+/, "").replace(/^\/+/, "");
  if (!cleaned) return null;
  const root = typeof base === "string" && /^https:\/\//.test(base) ? (base.endsWith("/") ? base : `${base}/`) : RAW_BASE;
  return root + cleaned.split("/").map(encodeURIComponent).join("/");
}

export function formatMetric(value, unit) {
  if (typeof value === "string") return value;
  if (!isNum(value)) return DASH;
  const u = String(unit ?? "").trim();
  switch (u.toLowerCase()) {
    case "ms":
      return fmtMs(value);
    case "us":
    case "µs":
    case "μs":
      return fmtUs(value);
    case "gb/s":
      return fmtGbs(value);
    case "%":
    case "pct":
    case "percent":
      return fmtPct(value);
    case "x":
    case "×":
      return fmtX(value);
    case "gflop/s":
    case "gflops":
      return `${fmtNum(value, 1)} GFLOP/s`;
    case "flop/byte":
      return `${fmtNum(value, 3)} FLOP/byte`;
    case "rel":
    case "rel_err":
      return fmtRelErr(value);
    case "bps":
      return fmtBps(value);
    case "usd":
    case "$":
      return fmtUsdCompact(value);
    case "bytes":
      return `${fmtSi(value)}B`;
    case "":
    case "count":
    case "n":
      return Number.isInteger(value) ? fmtInt(value) : fmtNum(value, 3);
    default:
      if (/\/s$/i.test(u)) return `${fmtSi(value)} ${u}`;
      return `${fmtNum(value, 3)} ${u}`;
  }
}

export function benchmarkRows(benchmarks, base = RAW_BASE) {
  return list(benchmarks).filter(Boolean).map((b) => ({
    section: String(b.section ?? ""),
    name: String(b.name ?? ""),
    metric: String(b.metric ?? ""),
    value: formatMetric(b.value, b.unit),
    unit: String(b.unit ?? ""),
    runId: b.run_id ? String(b.run_id) : DASH,
    raw: typeof b.raw === "string" ? b.raw : null,
    href: rawLink(b.raw, base),
  }));
}

export function pipelineSegments(pipeline) {
  const defs = [
    ["h2d_ms", "Host to device", "seg-h2d"],
    ["kernel_ms", "Kernels", "seg-kernel"],
    ["d2h_ms", "Device to host", "seg-d2h"],
  ];
  const segments = defs
    .filter(([k]) => isNum(get(pipeline, k)) && get(pipeline, k) >= 0)
    .map(([key, label, cls]) => ({ key, label, cls, ms: get(pipeline, key) }));
  const totalMs = segments.reduce((s, x) => s + x.ms, 0);
  let acc = 0;
  for (const s of segments) {
    s.pct = totalMs > 0 ? (s.ms / totalMs) * 100 : 0;
    s.x0 = acc;
    acc += s.pct;
    s.x1 = acc;
    s.text = `${s.label} ${fmtMs(s.ms)} (${fmtPct(s.pct, 0)})`;
  }
  const consequence = get(pipeline, "consequence");
  return {
    segments,
    totalMs,
    overlappedMs: num(get(pipeline, "overlapped_ms")),
    serialMs: num(get(pipeline, "serial_ms")),
    savedPct: num(get(pipeline, "overlap_saved_pct")),
    transferSharePct: num(get(pipeline, "transfer_share_pct")),
    consequence: typeof consequence === "string" ? consequence : "",
  };
}

export function engineTiles(engine) {
  const out = [];
  const p99 = num(get(engine, "p99_us"));
  const p50 = num(get(engine, "p50_us"));
  const orders = num(get(engine, "orders"));
  const cpu = get(engine, "cpu_model");
  if (p99 != null) out.push({ key: "p99", label: "Engine add_order p99", value: fmtUs(p99), sub: "matching path only, T4 job host CPU" });
  if (p50 != null) out.push({ key: "p50", label: "Engine add_order p50", value: fmtUs(p50), sub: "" });
  if (orders != null) out.push({ key: "orders", label: "Orders benchmarked", value: fmtInt(orders), sub: "" });
  if (typeof cpu === "string" && cpu) out.push({ key: "cpu", label: "Benchmark CPU", value: cpu, sub: "", small: true });
  return out;
}

export function provenanceRows(provenance) {
  return list(get(provenance, "runs")).filter(Boolean).map((r) => ({
    runId: r.run_id ? String(r.run_id) : DASH,
    sha: shortSha(r.git_sha),
    gpu: r.gpu_name ? String(r.gpu_name) : DASH,
    source: r.source ? String(r.source) : DASH,
    date: r.date ? String(r.date) : DASH,
  }));
}

/** Which panels have anything to show; the glue hides or labels the rest. */
export function sectionStatus(data) {
  return {
    ladder: list(get(data, "ladder.rungs")).length > 0,
    roofline: list(get(data, "roofline.kernels")).length > 0,
    timeline: typeof get(data, "timeline_svg") === "string" && get(data, "timeline_svg").length > 0,
    pipeline: pipelineSegments(get(data, "pipeline")).segments.length > 0,
    precision: list(get(data, "precision.variants")).length > 0,
    scans: list(get(data, "scans.implementations")).length > 0,
    slippage: slippageProducts(get(data, "slippage")).length > 0,
    ofi: ofiSelections(get(data, "ofi")).products.length > 0,
    benchmarks: list(get(data, "benchmarks")).length > 0,
    engine: isNum(get(data, "engine.p99_us")),
    resources: list(get(data, "resources.kernels")).length > 0,
  };
}

// --- Kernel resources (phase 2) -----------------------------------------------

export const LIMIT_LABELS = { registers: "registers", shared_memory: "shared memory", warps: "warp slots", blocks: "block cap" };

export function limitLabel(names) {
  const parts = list(names).map((n) => LIMIT_LABELS[n] ?? String(n));
  return parts.length ? parts.join(" + ") : DASH;
}

/** Rows for the resource table, lowest occupancy first (ties: more registers first). */
export function resourceRows(resources) {
  return list(get(resources, "kernels"))
    .filter((k) => k && typeof k.kernel === "string")
    .map((k) => {
      const occ = num(k.occupancy);
      const spills = isNum(k.spill_stores) || isNum(k.spill_loads) ? (num(k.spill_stores, 0) + num(k.spill_loads, 0)) : null;
      return {
        kernel: k.kernel,
        file: String(k.file ?? ""),
        registers: fmtInt(k.registers),
        spills: spills === null ? DASH : fmtInt(spills),
        smem: fmtInt(k.static_smem_bytes),
        block: fmtInt(k.block),
        blocksPerSm: fmtInt(k.blocks_per_sm),
        occupancy: occ === null ? DASH : fmtPct(occ * 100, 0),
        occupancyFrac: occ === null ? 0 : clamp01(occ),
        limitedBy: limitLabel(k.limited_by),
        full: occ !== null && occ >= 0.999,
        sortKey: [occ === null ? 2 : occ, -num(k.registers, 0)],
      };
    })
    .sort((a, b) => a.sortKey[0] - b.sortKey[0] || a.sortKey[1] - b.sortKey[1] || a.kernel.localeCompare(b.kernel));
}

/** One sentence: how many kernels reach full theoretical occupancy, and the lowest one. */
export function resourceSummary(resources) {
  const rows = resourceRows(resources);
  if (!rows.length) return "";
  const full = rows.filter((r) => r.full).length;
  const low = rows[0];
  const spilling = rows.filter((r) => r.spills !== DASH && r.spills !== "0").length;
  const lowest = low.full ? "every launched kernel is at full theoretical occupancy" :
    `lowest: ${low.kernel} at ${low.occupancy} (${low.limitedBy})`;
  return `${full} of ${rows.length} kernels reach full theoretical occupancy; ${lowest}; ` +
    `${spilling === 0 ? "no kernel spills registers" : `${spilling} spill registers`}.`;
}
