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
