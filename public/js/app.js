// exchange-engine browser dashboard (DOM + WebSocket wiring).
// Pure logic lives in ./core.mjs; deterministic sample data in ./demo-fixture.mjs.

import {
  ConnState,
  DEPTH,
  applyFills,
  bookImbalance,
  makerToAggressor,
  markToMarket,
  newSimState,
  parsePositiveDecimal,
  rollingVwap,
  simulateExecution,
  slippageBps,
  sortLevels,
  tradeFlowImbalance,
} from "./core.mjs";
import { demoFrames } from "./demo-fixture.mjs";

const WS_URL = "wss://advanced-trade-ws.coinbase.com";
const PRODUCTS = ["BTC-USD", "ETH-USD"];
const MAX_TRADES = 500;
const STALE_MS = 8000; // no data for this long -> stale
const DEMO_PROMPT_MS = 6000; // offer sample data if no snapshot by now

const UNIT = { "BTC-USD": "BTC", "ETH-USD": "ETH" };

// --- State ----------------------------------------------------------------

const state = {};
const sim = {};
for (const pid of PRODUCTS) {
  state[pid] = { bids: new Map(), asks: new Map(), trades: [], ready: false, lastMsgAt: 0, lastSnapshotAt: 0 };
  sim[pid] = newSimState();
}

let activeProduct = PRODUCTS[0];
let mode = "live"; // "live" | "demo"
let connState = ConnState.CONNECTING;
let lastHeartbeatAt = 0;
let lastHeartbeatTs = null;
let reconnectCount = 0;
let msgTimestamps = []; // for a rough msgs/sec estimate
let ws = null;
let backoff = 2000;
let demoOffered = false;

const simUi = { side: "buy", type: "limit" };

// --- Feed message handling ------------------------------------------------

function markMessage(productId) {
  const now = Date.now();
  msgTimestamps.push(now);
  if (msgTimestamps.length > 400) msgTimestamps = msgTimestamps.slice(-400);
  if (productId && state[productId]) state[productId].lastMsgAt = now;
}

function handleMessage(msg) {
  switch (msg.channel) {
    case "l2_data":
      return handleL2(msg);
    case "market_trades":
      return handleTrades(msg);
    case "heartbeats":
      lastHeartbeatAt = Date.now();
      lastHeartbeatTs = msg.timestamp || new Date().toISOString();
      markMessage(null);
      return;
    default:
      return;
  }
}

function handleL2(msg) {
  for (const event of msg.events || []) {
    const book = state[event.product_id];
    if (!book) continue;
    if (event.type === "snapshot") {
      book.bids.clear();
      book.asks.clear();
    }
    for (const u of event.updates || []) {
      const tree = u.side === "bid" ? book.bids : book.asks; // "offer" -> asks
      const price = parseFloat(u.price_level);
      const qty = parseFloat(u.new_quantity);
      if (qty === 0) tree.delete(price);
      else tree.set(price, qty);
    }
    // A product is ready only after a snapshot leaves at least one bid and ask.
    if (book.bids.size > 0 && book.asks.size > 0) book.ready = true;
    markMessage(event.product_id);
    if (event.type === "snapshot") book.lastSnapshotAt = Date.now();
    if (mode === "live" && connState !== ConnState.LIVE && book.ready) connState = ConnState.LIVE;
  }
}

function handleTrades(msg) {
  for (const event of msg.events || []) {
    for (const t of event.trades || []) {
      const book = state[t.product_id];
      if (!book) continue;
      const makerSide = (t.side || "").toUpperCase();
      book.trades.push({
        price: parseFloat(t.price),
        size: parseFloat(t.size),
        makerSide,
        aggressorSide: makerToAggressor(makerSide),
        time: t.time,
      });
      if (book.trades.length > MAX_TRADES) book.trades.shift();
      markMessage(t.product_id);
    }
  }
}

// --- Connection -----------------------------------------------------------

function subscribe(socket) {
  socket.send(JSON.stringify({ type: "subscribe", product_ids: PRODUCTS, channel: "level2" }));
  socket.send(JSON.stringify({ type: "subscribe", product_ids: PRODUCTS, channel: "market_trades" }));
  socket.send(JSON.stringify({ type: "subscribe", channel: "heartbeats" }));
}

function connect() {
  if (mode === "demo") return;
  connState = ConnState.CONNECTING;
  try {
    ws = new WebSocket(WS_URL);
  } catch {
    scheduleReconnect();
    return;
  }
  ws.onopen = () => {
    subscribe(ws);
    backoff = 2000;
    connState = ConnState.SUBSCRIBED;
  };
  ws.onmessage = (ev) => {
    try {
      handleMessage(JSON.parse(ev.data));
    } catch (err) {
      console.error("[feed] parse error", err);
    }
  };
  ws.onclose = () => {
    if (mode === "demo") return;
    connState = ConnState.RECONNECTING;
    reconnectCount += 1;
    scheduleReconnect();
  };
  ws.onerror = () => ws && ws.close();
}

function scheduleReconnect() {
  setTimeout(connect, backoff);
  backoff = Math.min(backoff * 2, 30000);
}

// --- Demo mode ------------------------------------------------------------

function enterDemo() {
  mode = "demo";
  connState = ConnState.DEMO;
  try {
    if (ws) ws.close();
  } catch {}
  for (const pid of PRODUCTS) {
    state[pid].bids.clear();
    state[pid].asks.clear();
    state[pid].trades = [];
    state[pid].ready = false;
  }
  for (const frame of demoFrames()) handleMessage(frame);
  hideDemoBanner();
}

function retryLive() {
  mode = "live";
  backoff = 2000;
  reconnectCount = 0;
  for (const pid of PRODUCTS) {
    state[pid].bids.clear();
    state[pid].asks.clear();
    state[pid].trades = [];
    state[pid].ready = false;
  }
  connect();
}

function maybeOfferDemo() {
  if (mode === "demo") return;
  const anyReady = PRODUCTS.some((p) => state[p].ready);
  if (anyReady) {
    hideDemoBanner(); // live data arrived — retract any earlier offer
    return;
  }
  if (!demoOffered && Date.now() - startedAt > DEMO_PROMPT_MS) {
    demoOffered = true;
    showDemoBanner();
  }
}

// --- Derived per-product status ------------------------------------------

function productStatus(pid) {
  const book = state[pid];
  if (mode === "demo") return ConnState.DEMO;
  if (!book.ready) return connState === ConnState.RECONNECTING ? ConnState.RECONNECTING : ConnState.CONNECTING;
  if (Date.now() - book.lastMsgAt > STALE_MS) return ConnState.STALE;
  return ConnState.LIVE;
}

function isTradable(pid) {
  const s = productStatus(pid);
  return (s === ConnState.LIVE || s === ConnState.DEMO) && state[pid].ready;
}

// --- Formatting -----------------------------------------------------------

const fmtPx = (v) => (v == null ? "—" : v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
const fmtQty = (v) => (v == null ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: 6 }));
const fmtUsd = (v) => (v == null ? "—" : `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
const fmtPct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const fmtBps = (v) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(1)} bps`);
const $ = (id) => document.getElementById(id);

// --- Rendering ------------------------------------------------------------

const STATUS_LABEL = {
  connecting: "Connecting…",
  subscribed: "Subscribed — awaiting snapshot",
  live: "Live",
  stale: "Stale — data paused",
  reconnecting: "Reconnecting…",
  demo: "Demo data",
};

function renderStatus() {
  const s = productStatus(activeProduct);
  const dot = $("conn-dot");
  dot.className = `dot ${s}`;
  $("conn-text").textContent = STATUS_LABEL[s] || s;
  $("mode-badge").hidden = mode !== "demo";
}

function renderProductNav() {
  const nav = $("product-nav");
  nav.innerHTML = "";
  for (const pid of PRODUCTS) {
    const btn = document.createElement("button");
    const isActive = pid === activeProduct;
    btn.className = `pill ${isActive ? "active" : ""}`;
    btn.setAttribute("aria-pressed", String(isActive));
    btn.type = "button";
    const st = productStatus(pid);
    btn.innerHTML = `<span class="pill-dot ${st}"></span>${pid}`;
    btn.onclick = () => {
      activeProduct = pid;
      $("book-product").textContent = pid;
      renderProductNav();
      renderPnl();
      renderHistory();
      renderSimGate();
    };
    nav.appendChild(btn);
  }
}

function levelRow(price, size, cumulative, maxTotal, side) {
  const pct = maxTotal > 0 ? (cumulative / maxTotal) * 100 : 0;
  const row = document.createElement("div");
  row.className = `row ${side}`;
  row.setAttribute("role", "row");
  row.innerHTML =
    `<div class="bar" style="width:${pct}%"></div>` +
    `<span class="px">${fmtPx(price)}</span>` +
    `<span>${fmtQty(size)}</span>` +
    `<span class="cum">${fmtQty(cumulative)}</span>`;
  return row;
}

function renderBook(book) {
  const bids = sortLevels(book.bids, "bids");
  const asks = sortLevels(book.asks, "asks");
  const maxTotal = Math.max(
    bids.reduce((s, l) => s + l.size, 0),
    asks.reduce((s, l) => s + l.size, 0),
    1e-9
  );
  const asksEl = $("asks");
  const bidsEl = $("bids");
  asksEl.innerHTML = "";
  bidsEl.innerHTML = "";

  let cum = 0;
  const askRows = asks.map((l) => { cum += l.size; return levelRow(l.price, l.size, cum, maxTotal, "ask"); });
  askRows.reverse().forEach((r) => asksEl.appendChild(r));

  cum = 0;
  for (const l of bids) { cum += l.size; bidsEl.appendChild(levelRow(l.price, l.size, cum, maxTotal, "bid")); }

  const bestBid = bids.length ? bids[0].price : null;
  const bestAsk = asks.length ? asks[0].price : null;
  const mid = bestBid != null && bestAsk != null ? (bestBid + bestAsk) / 2 : null;
  const spread = bestBid != null && bestAsk != null ? bestAsk - bestBid : null;
  $("mid-price").textContent = fmtPx(mid);
  $("spread").textContent = spread == null ? "spread —" : `spread ${fmtPx(spread)}`;
  return { bids, asks, mid, spread };
}

function setMetric(id, text, cls) {
  const el = $(id);
  el.textContent = text;
  if (cls !== undefined) el.style.color = cls;
}

function renderMetrics(book, view) {
  setMetric("m-mid", fmtPx(view.mid));
  setMetric("m-spread", fmtPx(view.spread));
  const imb = bookImbalance(view.bids, view.asks);
  setMetric("m-imb", fmtPct(imb), imb == null ? "" : imb >= 0 ? "var(--up)" : "var(--down)");
  setMetric("m-vwap", fmtPx(rollingVwap(book.trades)), "var(--connection-cyan)");
  const flow = tradeFlowImbalance(book.trades);
  setMetric("m-flow", fmtPct(flow), flow == null ? "" : flow >= 0 ? "var(--up)" : "var(--down)");
}

function renderTape(book) {
  const tape = $("tape");
  tape.innerHTML = "";
  for (const t of book.trades.slice(-40).reverse()) {
    const row = document.createElement("div");
    row.className = "row";
    const ts = (t.time || "").split("T")[1]?.slice(0, 12) || t.time || "";
    // Colour by AGGRESSOR: aggressive buys green, aggressive sells red.
    const cls = t.aggressorSide === "BUY" ? "up" : "down";
    row.innerHTML =
      `<span>${ts}</span><span class="px ${cls}">${fmtPx(t.price)}</span>` +
      `<span>${fmtQty(t.size)}</span><span class="${cls}">${t.aggressorSide}</span>`;
    tape.appendChild(row);
  }
}

function renderFeedHealth() {
  const sinceMsg = msgTimestamps.length ? (Date.now() - msgTimestamps[msgTimestamps.length - 1]) / 1000 : null;
  const sinceHb = lastHeartbeatAt ? (Date.now() - lastHeartbeatAt) / 1000 : null;
  const oneSecAgo = Date.now() - 1000;
  const rate = msgTimestamps.filter((t) => t >= oneSecAgo).length;
  $("fh-source").textContent = mode === "demo" ? "Sample data" : "Coinbase (live)";
  $("fh-msg").textContent = sinceMsg == null ? "—" : `${sinceMsg.toFixed(1)}s ago`;
  $("fh-hb").textContent = sinceHb == null ? "—" : `${sinceHb.toFixed(1)}s ago`;
  $("fh-rate").textContent = mode === "demo" ? "—" : `~${rate}/s`;
  $("fh-reconnects").textContent = String(reconnectCount);
}

// --- Execution simulator --------------------------------------------------

function renderSimGate() {
  const tradable = isTradable(activeProduct);
  $("sim-submit").disabled = !tradable;
  const note = $("sim-gate");
  if (tradable) {
    note.textContent = "";
    note.hidden = true;
  } else {
    note.hidden = false;
    note.textContent = mode === "demo"
      ? "Loading sample data…"
      : `Waiting for a ${activeProduct} order-book snapshot…`;
  }
  updateAutoPriceHint();
}

function autoPrice() {
  const book = state[activeProduct];
  const asks = sortLevels(book.asks, "asks");
  const bids = sortLevels(book.bids, "bids");
  return simUi.side === "buy" ? asks[0]?.price ?? null : bids[0]?.price ?? null;
}

function updateAutoPriceHint() {
  const hint = $("auto-hint");
  const priceInput = $("sim-price");
  if (simUi.type === "market") {
    hint.textContent = "Market: fills against best available levels";
    priceInput.disabled = true;
    return;
  }
  priceInput.disabled = false;
  if (priceInput.value.trim() === "") {
    const ref = autoPrice();
    const label = simUi.side === "buy" ? "best ask" : "best bid";
    hint.textContent = ref == null ? `Auto: waiting for ${label}` : `Auto: current ${label} ${fmtPx(ref)}`;
  } else {
    hint.textContent = "";
  }
}

function submitSimulation(e) {
  e.preventDefault();
  const log = $("sim-log");
  if (!isTradable(activeProduct)) {
    announce("Order rejected: market data not ready.");
    return;
  }
  const qty = parsePositiveDecimal($("sim-qty").value);
  if (qty == null) {
    announce("Order rejected: enter a positive numeric quantity.");
    return;
  }
  let price = null;
  if (simUi.type === "limit") {
    const raw = $("sim-price").value.trim();
    if (raw === "") {
      price = autoPrice();
      if (price == null) {
        announce("Order rejected: no reference price available.");
        return;
      }
    } else {
      price = parsePositiveDecimal(raw);
      if (price == null) {
        announce("Order rejected: enter a positive numeric price.");
        return;
      }
    }
  }

  const book = state[activeProduct];
  const opp = simUi.side === "buy" ? sortLevels(book.asks, "asks", 50) : sortLevels(book.bids, "bids", 50);
  const bestBid = sortLevels(book.bids, "bids")[0]?.price ?? null;
  const bestAsk = sortLevels(book.asks, "asks")[0]?.price ?? null;
  const preMid = bestBid != null && bestAsk != null ? (bestBid + bestAsk) / 2 : null;

  const result = simulateExecution({ side: simUi.side, type: simUi.type, price, levels: opp, quantity: qty });
  applyFills(sim[activeProduct], simUi.side, result.fills);
  const slip = slippageBps(result.avgPrice, preMid, simUi.side);

  const status = result.filledQty <= 0
    ? (simUi.type === "limit" ? "No immediate fill" : "No liquidity")
    : result.unfilledQty > 1e-9 ? "Partial" : "Filled";

  const entry = {
    t: new Date(),
    side: simUi.side,
    type: simUi.type,
    requested: qty,
    filled: result.filledQty,
    avg: result.avgPrice,
    status,
  };
  sim[activeProduct].history.unshift(entry);
  sim[activeProduct].history = sim[activeProduct].history.slice(0, 25);

  renderExecutionResult(result, slip, status);
  renderHistory();
  renderPnl();
  announce(
    `${simUi.side} ${simUi.type} ${fmtQty(qty)} ${UNIT[activeProduct]}: ${status}, ` +
      `filled ${fmtQty(result.filledQty)} at ${result.avgPrice == null ? "—" : fmtPx(result.avgPrice)}.`
  );
}

function renderExecutionResult(r, slip, status) {
  const unit = UNIT[activeProduct];
  $("er-status").textContent = status;
  $("er-status").className = `er-status ${status === "Filled" ? "up" : status === "Partial" ? "warn" : "muted"}`;
  $("er-requested").textContent = `${fmtQty(r.requestedQty)} ${unit}`;
  $("er-filled").textContent = `${fmtQty(r.filledQty)} ${unit}`;
  $("er-unfilled").textContent = `${fmtQty(r.unfilledQty)} ${unit}`;
  $("er-avg").textContent = fmtPx(r.avgPrice);
  $("er-notional").textContent = fmtUsd(r.notional);
  $("er-slip").textContent = fmtBps(slip);
  $("er-levels").textContent = String(r.levelsConsumed);
  $("exec-result").hidden = false;
}

function renderHistory() {
  const tbody = $("history");
  const rows = sim[activeProduct].history;
  tbody.innerHTML = "";
  if (!rows.length) {
    tbody.innerHTML = `<div class="hist-empty muted">No simulations yet for ${activeProduct}.</div>`;
    return;
  }
  for (const h of rows) {
    const row = document.createElement("div");
    row.className = "hist-row";
    const ts = h.t.toLocaleTimeString();
    const cls = h.side === "buy" ? "up" : "down";
    row.innerHTML =
      `<span>${ts}</span><span class="${cls}">${h.side.toUpperCase()}</span>` +
      `<span>${h.type}</span><span>${fmtQty(h.requested)}</span>` +
      `<span>${fmtQty(h.filled)}</span><span>${fmtPx(h.avg)}</span>` +
      `<span class="hist-status">${h.status}</span>`;
    tbody.appendChild(row);
  }
}

function renderPnl() {
  const book = state[activeProduct];
  const bestBid = sortLevels(book.bids, "bids")[0]?.price ?? null;
  const bestAsk = sortLevels(book.asks, "asks")[0]?.price ?? null;
  const mid = bestBid != null && bestAsk != null ? (bestBid + bestAsk) / 2 : null;
  const s = sim[activeProduct];
  const pnl = markToMarket(s, mid);
  $("pnl-pos").textContent = `${fmtQty(s.position)} ${UNIT[activeProduct]}`;
  $("pnl-cash").textContent = fmtUsd(s.cash);
  const pnlEl = $("pnl-pnl");
  pnlEl.textContent = fmtUsd(pnl);
  pnlEl.style.color = pnl >= 0 ? "var(--up)" : "var(--down)";
}

function announce(text) {
  const log = $("sim-log");
  const line = document.createElement("div");
  line.textContent = text;
  log.prepend(line);
  while (log.childElementCount > 12) log.lastChild.remove();
}

// --- Controls / accessibility --------------------------------------------

function initSegmented(groupId, key, onChange) {
  const buttons = document.querySelectorAll(`#${groupId} button`);
  buttons.forEach((b) => {
    b.onclick = () => {
      simUi[key] = b.dataset.value;
      buttons.forEach((x) => {
        const on = x === b;
        x.classList.toggle("active", on);
        x.setAttribute("aria-pressed", String(on));
      });
      onChange && onChange();
    };
  });
}

function initSimulator() {
  initSegmented("sim-side", "side", () => updateAutoPriceHint());
  initSegmented("sim-type", "type", () => updateAutoPriceHint());
  $("sim-price").addEventListener("input", updateAutoPriceHint);
  $("sim-form").addEventListener("submit", submitSimulation);
  $("reset-product").onclick = () => {
    sim[activeProduct] = newSimState();
    renderPnl();
    renderHistory();
    $("exec-result").hidden = true;
    announce(`Reset ${activeProduct} simulation.`);
  };
  $("demo-run").onclick = enterDemo;
  $("demo-retry").onclick = () => { hideDemoBanner(); retryLive(); };
}

function showDemoBanner() { $("demo-banner").hidden = false; }
function hideDemoBanner() { $("demo-banner").hidden = true; }

// --- Main loop ------------------------------------------------------------

function tick() {
  const book = state[activeProduct];
  const view = renderBook(book);
  renderMetrics(book, view);
  renderTape(book);
  renderStatus();
  renderProductNav();
  renderFeedHealth();
  renderSimGate();
  maybeOfferDemo();
}

const startedAt = Date.now();
renderProductNav();
initSimulator();
renderHistory();
renderPnl();
connect();
setInterval(tick, 500);
setInterval(renderPnl, 1000);
