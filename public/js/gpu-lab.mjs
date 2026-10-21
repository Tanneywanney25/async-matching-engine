// GPU Lab: renders /data/gpu_lab.json (written by scripts/render_docs.py from real T4 results).
// All layout math lives in gpu-core.mjs (pure, unit-tested); this module only builds DOM/SVG.

import {
  DASH, benchmarkRows, engineTiles, fmtBps, fmtInt, fmtMs, fmtPct, fmtX, heatmapLayout, heatmapSummary,
  ladderLayout, ofiRows, ofiSelections, pipelineSegments, precisionRows, provenanceRows, rooflineLayout,
  scanRows, scatterLayout, sectionStatus, slippageNote, slippageProducts, speedupTiles, get,
} from "./gpu-core.mjs";

const DATA_URL = "/data/gpu_lab.json";
const $ = (id) => document.getElementById(id);

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function table(cols, rows, { rowClass } = {}) {
  const head = cols.map((c) => `<th scope="col"${c.num ? ' class="num"' : ""}>${esc(c.label)}</th>`).join("");
  const body = rows
    .map((r) => {
      const cls = rowClass ? rowClass(r) : "";
      const cells = cols.map((c) => `<td${c.num ? ' class="num"' : ""}>${c.html ? c.html(r) : esc(r[c.key])}</td>`).join("");
      return `<tr${cls ? ` class="${cls}"` : ""}>${cells}</tr>`;
    })
    .join("");
  return `<div class="table-wrap"><table class="gpu-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function svgOpen(w, h, label) {
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}" preserveAspectRatio="xMinYMin meet">`;
}

// --- panels -------------------------------------------------------------------

function renderLadder(data) {
  const ladder = get(data, "ladder");
  const L = ladderLayout(ladder, { width: 640 });
  let s = svgOpen(L.width, L.height, "Optimization ladder: median kernel time per rung on a log axis");
  for (const t of L.ticks) {
    s += `<line class="grid-line" x1="${t.x}" x2="${t.x}" y1="${L.plotTop - 4}" y2="${L.plotBottom}"/>`;
    s += `<text class="axis-label" x="${t.x}" y="${L.plotBottom + 16}" text-anchor="middle">${esc(t.label)}</text>`;
  }
  s += `<text class="axis-title" x="${(L.plotLeft + L.plotRight) / 2}" y="${L.height - 4}" text-anchor="middle">${esc(L.axisLabel)}</text>`;
  for (const r of L.refs) {
    s += `<line class="ref-line ${r.cls}" x1="${r.x}" x2="${r.x}" y1="${r.y0}" y2="${r.y1}"/>`;
    s += `<text class="ref-label ${r.cls}" x="${r.labelX}" y="${r.labelY}" text-anchor="${r.labelAnchor}">${esc(r.text)}</text>`;
  }
  for (const b of L.bars) {
    s += `<text class="bar-label" x="${b.barX0}" y="${b.labelY}">${esc(b.rowLabel)}</text>`;
    s += `<rect class="bar" x="${b.barX0}" y="${b.barY}" width="${Math.max(1, b.barX1 - b.barX0)}" height="${b.barH}" rx="3"><title>${esc(b.label)}: ${esc(b.valueText)}</title></rect>`;
    if (b.e2eX != null) s += `<line class="e2e-tick" x1="${b.e2eX}" x2="${b.e2eX}" y1="${b.barY - 3}" y2="${b.barY + b.barH + 3}"><title>end to end ${esc(fmtMs(b.e2eMs))}</title></line>`;
    s += `<text class="bar-value${b.valueInside ? " inside" : ""}" x="${b.valueX}" y="${b.barY + b.barH - 3}" text-anchor="${b.valueAnchor}">${esc(b.valueText)}</text>`;
  }
  s += "</svg>";
  $("gpu-ladder-chart").innerHTML = s;
  const tiles = speedupTiles(ladder);
  $("gpu-speedups").innerHTML = tiles.map((t) => `<div class="tile"><span>${esc(t.label)}</span><b>${esc(t.value)}</b></div>`).join("");
}

function renderRoofline(data) {
  const R = rooflineLayout(get(data, "roofline"), { width: 640, height: 340 });
  let s = svgOpen(R.width, R.height, "Roofline: achieved GFLOP/s against arithmetic intensity, log-log");
  for (const t of R.xTicks) s += `<line class="grid-line" x1="${t.x}" x2="${t.x}" y1="${R.plotTop}" y2="${R.plotBottom}"/><text class="axis-label" x="${t.x}" y="${R.plotBottom + 16}" text-anchor="middle">${esc(t.label)}</text>`;
  for (const t of R.yTicks) s += `<line class="grid-line" x1="${R.plotLeft}" x2="${R.plotRight}" y1="${t.y}" y2="${t.y}"/><text class="axis-label" x="${R.plotLeft - 6}" y="${t.y + 4}" text-anchor="end">${esc(t.label)}</text>`;
  for (const l of R.lines) {
    const pts = l.points.map((p) => `${p.px},${p.py}`).join(" ");
    s += `<polyline class="roof ${l.cls}" points="${pts}" fill="none"><title>${esc(l.label)}</title></polyline>`;
  }
  for (const p of R.points) {
    s += `<circle class="roof-point" cx="${p.px}" cy="${p.py}" r="5"><title>${esc(p.name)}: ${esc(p.gflops.toFixed(1))} GFLOP/s at ${esc(p.intensity.toFixed(3))} FLOP/byte</title></circle>`;
    s += `<text class="point-label" x="${p.labelX}" y="${p.labelY}" text-anchor="${p.labelAnchor}">${esc(p.name.split("_")[0])}</text>`;
  }
  s += `<text class="axis-title" x="${(R.plotLeft + R.plotRight) / 2}" y="${R.height - 4}" text-anchor="middle">arithmetic intensity, FLOP/byte</text>`;
  s += "</svg>";
  $("gpu-roofline-chart").innerHTML = s;
  $("gpu-roofline-legend").innerHTML = R.lines.map((l) => `<span class="legend-item ${l.cls}">${esc(l.label)}</span>`).join("");
}

function renderTimeline(data) {
  const p = pipelineSegments(get(data, "pipeline"));
  const src = get(data, "timeline_svg");
  if (src) $("gpu-timeline-img").src = src;
  const bar = p.segments.map((s) => `<span class="seg ${s.cls}" style="width:${s.pct.toFixed(2)}%" title="${esc(s.text)}"></span>`).join("");
  $("gpu-pipeline").innerHTML = `<div class="seg-bar" aria-hidden="true">${bar}</div>` +
    `<ul class="seg-legend">${p.segments.map((s) => `<li class="${s.cls}">${esc(s.text)}</li>`).join("")}</ul>` +
    `<p class="small">Overlapped ${esc(fmtMs(p.overlappedMs))} vs serial ${esc(fmtMs(p.serialMs))}; transfers ${esc(fmtPct(p.transferSharePct))} of stage time. ${esc(p.consequence)}</p>`;
}

function renderTables(data) {
  $("gpu-precision").innerHTML = table(
    [{ key: "name", label: "Variant" }, { key: "maxRelErr", label: "Max rel. error", num: true },
     { key: "medianRelErr", label: "Median rel. error", num: true }, { key: "medianMs", label: "Median", num: true }],
    precisionRows(get(data, "precision")), { rowClass: (r) => (r.production ? "chosen" : "") });
  $("gpu-scans").innerHTML = table(
    [{ key: "name", label: "Scan" }, { key: "medianMs", label: "Median", num: true },
     { key: "elementsPerS", label: "Elements/s", num: true }, { key: "gbs", label: "Bandwidth", num: true }],
    scanRows(get(data, "scans")), { rowClass: (r) => (r.winner ? "chosen" : "") });
  const bench = benchmarkRows(get(data, "benchmarks"));
  $("gpu-bench-count").textContent = fmtInt(bench.length);
  $("gpu-bench").innerHTML = table(
    [{ key: "section", label: "Section" }, { key: "name", label: "Measurement" }, { key: "value", label: "Median", num: true },
     { key: "runId", label: "Run" }, { key: "href", label: "Raw", html: (r) => (r.href ? `<a href="${esc(r.href)}" target="_blank" rel="noopener noreferrer">raw</a>` : DASH) }],
    bench);
  $("gpu-prov").innerHTML = table(
    [{ key: "runId", label: "Run" }, { key: "sha", label: "Commit" }, { key: "gpu", label: "GPU" }, { key: "source", label: "Source" }, { key: "date", label: "Date" }],
    provenanceRows(get(data, "provenance")));
  $("gpu-engine").innerHTML = engineTiles(get(data, "engine"))
    .map((t) => `<div class="tile${t.small ? " small-tile" : ""}"><span>${esc(t.label)}</span><b>${esc(t.value)}</b>${t.sub ? `<em>${esc(t.sub)}</em>` : ""}</div>`).join("");
}

function renderSlippage(data) {
  const slip = get(data, "slippage");
  const products = slippageProducts(slip);
  const prodSel = $("gpu-slip-product");
  const sideSel = $("gpu-slip-side");
  prodSel.innerHTML = products.map((p) => `<option>${esc(p)}</option>`).join("");
  const draw = () => {
    const side = get(slip, `products.${prodSel.value}.${sideSel.value}`);
    const H = heatmapLayout(side, get(slip, "sizes_usd"), { width: 640 });
    let s = svgOpen(H.width, H.height, `Median slippage by minute and order size, ${prodSel.value} ${sideSel.value}`);
    for (const c of H.cells) if (c.color) s += `<rect x="${c.x}" y="${c.y}" width="${c.w + 0.5}" height="${c.h + 0.5}" fill="${c.color}"><title>${esc(fmtBps(c.value))}</title></rect>`;
    for (const t of H.yTicks) s += `<text class="axis-label" x="${H.plotLeft - 6}" y="${t.y + 3}" text-anchor="end">${esc(t.label)}</text>`;
    for (const t of H.xTicks) s += `<text class="axis-label" x="${t.x}" y="${H.plotBottom + 14}" text-anchor="middle">${esc(t.label)}</text>`;
    s += `<text class="axis-title" x="${(H.plotLeft + H.plotRight) / 2}" y="${H.height - 4}" text-anchor="middle">${esc(H.xTitle)}</text></svg>`;
    $("gpu-slip-chart").innerHTML = s;
    const stops = H.legend.stops.map((c, i) => `${c} ${((i / (H.legend.stops.length - 1)) * 100).toFixed(0)}%`).join(",");
    $("gpu-slip-legend").innerHTML = `<span>${esc(H.legend.minLabel)}</span><span class="ramp" style="background:linear-gradient(90deg,${stops})"></span><span>${esc(H.legend.maxLabel)}</span>`;
    $("gpu-slip-summary").innerHTML = table(
      [{ key: "size", label: "Order size" }, { key: "min", label: "Min", num: true }, { key: "median", label: "Median", num: true }, { key: "max", label: "Max", num: true }],
      heatmapSummary(side, get(slip, "sizes_usd")));
  };
  prodSel.onchange = draw;
  sideSel.onchange = draw;
  if (products.length) draw();
  $("gpu-slip-note").textContent = slippageNote(slip);
}

function renderOfi(data) {
  const ofi = get(data, "ofi");
  const sel = ofiSelections(ofi);
  $("gpu-ofi-table").innerHTML = table(
    [{ key: "product", label: "Product" }, { key: "horizon", label: "Horizon" }, { key: "r2", label: "Out-of-sample R²", num: true },
     { key: "beta", label: "Beta", num: true }, { key: "nTest", label: "Test points", num: true }], ofiRows(ofi));
  const pSel = $("gpu-ofi-product");
  const hSel = $("gpu-ofi-horizon");
  pSel.innerHTML = sel.products.map((p) => `<option>${esc(p)}</option>`).join("");
  hSel.innerHTML = sel.horizons.map((h) => `<option value="${esc(h)}">${esc(h.replace(/^h|s$/g, ""))} s</option>`).join("");
  const draw = () => {
    const S = scatterLayout(get(ofi, `scatter.${pSel.value}.${hSel.value}`), get(ofi, `per_product.${pSel.value}.${hSel.value}`), { width: 640, height: 300 });
    let s = svgOpen(S.width, S.height, "Order flow imbalance against future mid change, test set");
    for (const t of S.xTicks) s += `<text class="axis-label" x="${t.x}" y="${S.plotBottom + 16}" text-anchor="middle">${esc(t.label)}</text>`;
    for (const t of S.yTicks) s += `<line class="grid-line" x1="${S.plotLeft}" x2="${S.plotRight}" y1="${t.y}" y2="${t.y}"/><text class="axis-label" x="${S.plotLeft - 6}" y="${t.y + 4}" text-anchor="end">${esc(t.label)}</text>`;
    s += S.points.map((p) => `<circle class="dot" cx="${p.px.toFixed(1)}" cy="${p.py.toFixed(1)}" r="1.6"/>`).join("");
    if (S.line) s += `<line class="fit-line" x1="${S.line.x0}" y1="${S.line.y0}" x2="${S.line.x1}" y2="${S.line.y1}"/>`;
    s += `<text class="axis-title" x="${(S.plotLeft + S.plotRight) / 2}" y="${S.height - 4}" text-anchor="middle">order flow imbalance over the horizon</text></svg>`;
    $("gpu-ofi-chart").innerHTML = s;
  };
  pSel.onchange = draw;
  hSel.onchange = draw;
  if (sel.products.length) draw();
}

export async function initGpuLab() {
  const status = $("gpu-status");
  if (!status) return;
  let data;
  try {
    const res = await fetch(DATA_URL, { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    status.textContent = "GPU Lab results are published after the next real T4 run.";
    $("gpu-body").hidden = true;
    return;
  }
  const show = sectionStatus(data);
  const steps = [
    ["ladder", renderLadder, "gpu-ladder"], ["roofline", renderRoofline, "gpu-roofline"], ["pipeline", renderTimeline, "gpu-timeline"],
    ["slippage", renderSlippage, "gpu-slippage"], ["ofi", renderOfi, "gpu-ofi"],
  ];
  for (const [key, fn, id] of steps) {
    if (!show[key]) { $(id).hidden = true; continue; }
    try { fn(data); } catch (err) { $(id).hidden = true; console.error(`[gpu-lab] ${key}`, err); }
  }
  try { renderTables(data); } catch (err) { console.error("[gpu-lab] tables", err); }
  const a = get(data, "archive", {}) || {};
  status.textContent = `${fmtInt(a.n_snapshots)} recorded snapshots and ${fmtInt(a.n_trades)} trades, measured on an NVIDIA T4. ` +
    `Generated ${String(get(data, "generated_at", "")).slice(0, 10)}.`;
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initGpuLab);
  else initGpuLab();
}
