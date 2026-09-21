// exchange-engine browser dashboard.
//
// Connects directly to the public Coinbase Advanced Trade WebSocket and mirrors
// the Python `exchange_engine` package's book/metrics logic client-side, so the
// page is fully functional as a static deployment with no backend.

const WS_URL = "wss://advanced-trade-ws.coinbase.com";
const PRODUCTS = ["BTC-USD", "ETH-USD"];
const MAX_TRADES = 500;
const DEPTH = 10;

// Per-product state. Bids/asks are Maps keyed by numeric price -> size.
const state = {};
for (const pid of PRODUCTS) {
  state[pid] = { bids: new Map(), asks: new Map(), trades: [] };
}

let activeProduct = PRODUCTS[0];
let lastHeartbeat = null;

/** Apply one level2 update; a quantity of 0 removes the level. */
function setLevel(book, side, price, qty) {
  const tree = side === "bid" ? book.bids : book.asks; // "offer" -> asks
  if (qty === 0) tree.delete(price);
  else tree.set(price, qty);
}

/** Route a parsed Coinbase frame to the correct handler. */
function handleMessage(msg) {
  switch (msg.channel) {
    case "l2_data":
      return handleL2(msg);
    case "market_trades":
      return handleTrades(msg);
    case "heartbeats":
      lastHeartbeat = msg.timestamp || new Date().toISOString();
      return;
    default:
      return; // subscriptions / control frames
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
      setLevel(book, u.side, parseFloat(u.price_level), parseFloat(u.new_quantity));
    }
  }
}

function handleTrades(msg) {
  for (const event of msg.events || []) {
    for (const t of event.trades || []) {
      const book = state[t.product_id];
      if (!book) continue;
      book.trades.push({
        price: parseFloat(t.price),
        size: parseFloat(t.size),
        side: (t.side || "").toUpperCase(),
        time: t.time,
      });
      if (book.trades.length > MAX_TRADES) book.trades.shift();
    }
  }
}

/** Sorted top-N levels for a side: bids descending, asks ascending. */
function sortedLevels(book, side, n = DEPTH) {
  const entries = [...(side === "bids" ? book.bids : book.asks).entries()];
  entries.sort((a, b) => (side === "bids" ? b[0] - a[0] : a[0] - b[0]));
  return entries.slice(0, n).map(([price, size]) => ({ price, size }));
}

// --- WebSocket connection with exponential backoff ------------------------

let ws = null;
let backoff = 2000;

function subscribe(socket) {
  socket.send(JSON.stringify({ type: "subscribe", product_ids: PRODUCTS, channel: "level2" }));
  socket.send(JSON.stringify({ type: "subscribe", product_ids: PRODUCTS, channel: "market_trades" }));
  socket.send(JSON.stringify({ type: "subscribe", channel: "heartbeats" }));
}

function connect() {
  setStatus("connecting", "connecting…");
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    subscribe(ws);
    backoff = 2000;
    setStatus("live", "live");
  };
  ws.onmessage = (ev) => {
    try {
      handleMessage(JSON.parse(ev.data));
    } catch (err) {
      console.error("[feed] parse error", err);
    }
  };
  ws.onclose = () => {
    setStatus("down", `reconnecting in ${backoff / 1000}s…`);
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 30000);
  };
  ws.onerror = () => ws && ws.close();
}

// --- Analytics (mirror of exchange_engine.metrics) ------------------------

function rollingVwap(trades, window = 100) {
  const recent = trades.slice(-window);
  let num = 0;
  let vol = 0;
  for (const t of recent) {
    num += t.price * t.size;
    vol += t.size;
  }
  return vol === 0 ? null : num / vol;
}

function bookImbalance(bids, asks, levels = DEPTH) {
  const bidQty = bids.slice(0, levels).reduce((s, l) => s + l.size, 0);
  const askQty = asks.slice(0, levels).reduce((s, l) => s + l.size, 0);
  const total = bidQty + askQty;
  return total === 0 ? null : (bidQty - askQty) / total;
}

function tradeFlowImbalance(trades, window = 50) {
  const recent = trades.slice(-window);
  let buy = 0;
  let sell = 0;
  for (const t of recent) {
    if (t.side === "SELL") buy += t.size; // aggressive buyer lifted the offer
    else sell += t.size;
  }
  const total = buy + sell;
  return total === 0 ? null : (buy - sell) / total;
}

// --- Formatting helpers ---------------------------------------------------

const fmtPx = (v) =>
  v == null ? "—" : v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtQty = (v) => (v == null ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: 6 }));
const fmtPct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

function setStatus(cls, text) {
  const dot = document.getElementById("conn-dot");
  const label = document.getElementById("conn-text");
  if (dot) dot.className = `dot ${cls === "live" ? "live" : cls === "down" ? "down" : ""}`;
  if (label) label.textContent = text;
}

// --- Rendering ------------------------------------------------------------

function renderProductNav() {
  const nav = document.getElementById("product-nav");
  nav.innerHTML = "";
  for (const pid of PRODUCTS) {
    const btn = document.createElement("button");
    btn.textContent = pid;
    btn.className = pid === activeProduct ? "active" : "";
    btn.onclick = () => {
      activeProduct = pid;
      renderProductNav();
      document.getElementById("book-product").textContent = pid;
    };
    nav.appendChild(btn);
  }
}

function levelRow(price, size, cumulative, maxTotal, cls) {
  const pct = maxTotal > 0 ? (cumulative / maxTotal) * 100 : 0;
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML =
    `<div class="bar" style="width:${pct}%"></div>` +
    `<span class="px">${fmtPx(price)}</span>` +
    `<span>${fmtQty(size)}</span>` +
    `<span class="${cls}">${fmtQty(cumulative)}</span>`;
  return row;
}

function renderBook(book) {
  const bids = sortedLevels(book, "bids");
  const asks = sortedLevels(book, "asks");
  const maxTotal = Math.max(
    bids.reduce((s, l) => s + l.size, 0),
    asks.reduce((s, l) => s + l.size, 0),
    0.00000001
  );

  const asksEl = document.getElementById("asks");
  const bidsEl = document.getElementById("bids");
  asksEl.innerHTML = "";
  bidsEl.innerHTML = "";

  let cum = 0;
  const askRows = asks.map((l) => {
    cum += l.size;
    return levelRow(l.price, l.size, cum, maxTotal, "muted");
  });
  askRows.reverse().forEach((r) => asksEl.appendChild(r)); // best ask nearest the spread

  cum = 0;
  for (const l of bids) {
    cum += l.size;
    bidsEl.appendChild(levelRow(l.price, l.size, cum, maxTotal, "muted"));
  }

  const bestBid = bids.length ? bids[0].price : null;
  const bestAsk = asks.length ? asks[0].price : null;
  const mid = bestBid != null && bestAsk != null ? (bestBid + bestAsk) / 2 : null;
  const spread = bestBid != null && bestAsk != null ? bestAsk - bestBid : null;
  document.getElementById("mid-price").textContent = fmtPx(mid);
  document.getElementById("spread").textContent = spread == null ? "spread —" : `spread ${fmtPx(spread)}`;
  return { bids, asks, mid, spread };
}

function renderMetrics(book, view) {
  document.getElementById("m-mid").textContent = fmtPx(view.mid);
  document.getElementById("m-spread").textContent = fmtPx(view.spread);
  const imb = bookImbalance(view.bids, view.asks);
  const imbEl = document.getElementById("m-imb");
  imbEl.textContent = fmtPct(imb);
  imbEl.style.color = imb == null ? "" : imb >= 0 ? "var(--green)" : "var(--red)";
  document.getElementById("m-vwap").textContent = fmtPx(rollingVwap(book.trades));
  const flow = tradeFlowImbalance(book.trades);
  const flowEl = document.getElementById("m-flow");
  flowEl.textContent = fmtPct(flow);
  flowEl.style.color = flow == null ? "" : flow >= 0 ? "var(--green)" : "var(--red)";
  document.getElementById("m-hb").textContent = lastHeartbeat
    ? lastHeartbeat.split("T")[1]?.slice(0, 12) || lastHeartbeat
    : "—";
}

function renderTape(book) {
  const tape = document.getElementById("tape");
  tape.innerHTML = "";
  for (const t of book.trades.slice(-40).reverse()) {
    const row = document.createElement("div");
    row.className = "row";
    const ts = (t.time || "").split("T")[1]?.slice(0, 12) || t.time || "";
    const cls = t.side === "BUY" ? "buy" : "sell";
    row.innerHTML =
      `<span>${ts}</span><span class="px ${cls}">${fmtPx(t.price)}</span>` +
      `<span>${fmtQty(t.size)}</span><span class="${cls}">${t.side}</span>`;
    tape.appendChild(row);
  }
}

function tick() {
  const book = state[activeProduct];
  const view = renderBook(book);
  renderMetrics(book, view);
  renderTape(book);
}

// --- Order simulator (matches against the live opposite side) -------------

const sim = { side: "buy", type: "limit", position: 0, cash: 0 };

function simulate(side, type, price, qty) {
  const book = state[activeProduct];
  const opposite = side === "buy" ? sortedLevels(book, "asks", 50) : sortedLevels(book, "bids", 50);
  let remaining = qty;
  const fills = [];
  for (const level of opposite) {
    if (remaining <= 0) break;
    if (type === "limit") {
      if (side === "buy" && level.price > price) break;
      if (side === "sell" && level.price < price) break;
    }
    const traded = Math.min(remaining, level.size);
    fills.push({ price: level.price, qty: traded });
    remaining -= traded;
  }
  for (const f of fills) {
    if (side === "buy") {
      sim.cash -= f.price * f.qty;
      sim.position += f.qty;
    } else {
      sim.cash += f.price * f.qty;
      sim.position -= f.qty;
    }
  }
  return fills;
}

function renderPnl() {
  const book = state[activeProduct];
  const asks = sortedLevels(book, "asks");
  const bids = sortedLevels(book, "bids");
  const mid = bids.length && asks.length ? (bids[0].price + asks[0].price) / 2 : null;
  const pnl = mid == null ? sim.cash : sim.cash + sim.position * mid;
  document.getElementById("pnl-pos").textContent = fmtQty(sim.position);
  document.getElementById("pnl-cash").textContent = fmtPx(sim.cash);
  const pnlEl = document.getElementById("pnl-pnl");
  pnlEl.textContent = fmtPx(pnl);
  pnlEl.style.color = pnl >= 0 ? "var(--green)" : "var(--red)";
}

function logSim(text) {
  const log = document.getElementById("sim-log");
  const line = document.createElement("div");
  line.textContent = text;
  log.prepend(line);
  while (log.childElementCount > 20) log.lastChild.remove();
}

function initSimulator() {
  document.querySelectorAll("#sim-side button").forEach(
    (b) =>
      (b.onclick = () => {
        sim.side = b.dataset.side;
        document.querySelectorAll("#sim-side button").forEach((x) => x.classList.toggle("active", x === b));
      })
  );
  document.querySelectorAll("#sim-type button").forEach(
    (b) =>
      (b.onclick = () => {
        sim.type = b.dataset.type;
        document.querySelectorAll("#sim-type button").forEach((x) => x.classList.toggle("active", x === b));
        document.getElementById("sim-price").disabled = sim.type === "market";
      })
  );

  document.getElementById("sim-form").onsubmit = (e) => {
    e.preventDefault();
    const qty = parseFloat(document.getElementById("sim-qty").value);
    if (!qty || qty <= 0) return logSim("! invalid quantity");
    let price = parseFloat(document.getElementById("sim-price").value);
    const book = state[activeProduct];
    if (sim.type === "limit" && !price) {
      const asks = sortedLevels(book, "asks");
      const bids = sortedLevels(book, "bids");
      price = sim.side === "buy" ? asks[0]?.price ?? 0 : bids[0]?.price ?? 0;
    }
    const fills = simulate(sim.side, sim.type, price, qty);
    const filled = fills.reduce((s, f) => s + f.qty, 0);
    const avg = filled ? fills.reduce((s, f) => s + f.price * f.qty, 0) / filled : 0;
    logSim(
      `${sim.side.toUpperCase()} ${sim.type} ${qty} → ${fmtQty(filled)} @ ${filled ? fmtPx(avg) : "—"} (${fills.length} fills)`
    );
    renderPnl();
  };
}

// --- Bootstrap ------------------------------------------------------------

renderProductNav();
initSimulator();
connect();
setInterval(tick, 500);
setInterval(renderPnl, 1000);
