// Pure, DOM-free logic for the exchange-engine browser dashboard.
//
// Everything here is deterministic and side-effect free so it can be unit-tested
// with `node --test` (see tests/frontend/core.test.mjs). The Python package is
// the precision implementation (Decimal); these browser calculations use IEEE
// Number and are explicitly labelled as estimates in the UI.

export const DEPTH = 10;

// Connection lifecycle states surfaced to the visitor.
export const ConnState = Object.freeze({
  CONNECTING: "connecting",
  SUBSCRIBED: "subscribed",
  LIVE: "live",
  STALE: "stale",
  RECONNECTING: "reconnecting",
  DEMO: "demo",
});

/**
 * Strictly parse a positive decimal from user input.
 * Rejects empty, non-finite, partially numeric ("0.01abc"), zero and negative.
 * @returns {number|null} the parsed value, or null if invalid.
 */
export function parsePositiveDecimal(raw) {
  if (raw == null) return null;
  const s = String(raw).trim();
  // Only digits with a single optional decimal point; no signs, no exponents.
  if (!/^\d*\.?\d+$/.test(s)) return null;
  const n = Number(s);
  if (!Number.isFinite(n) || n <= 0) return null;
  if (n > 1e12) return null; // reject absurdly large values
  return n;
}

/**
 * Coinbase market_trades.side is the MAKER side. The aggressor (taker) is the
 * opposite: a resting SELL maker was lifted by an aggressive BUY, and vice versa.
 * @returns {"BUY"|"SELL"} the aggressor side.
 */
export function makerToAggressor(makerSide) {
  return String(makerSide).toUpperCase() === "SELL" ? "BUY" : "SELL";
}

/**
 * Return the top-N levels of a side as [{price,size}], best price first.
 * @param {Map<number,number>} map price -> size
 * @param {"bids"|"asks"} side
 */
export function sortLevels(map, side, n = DEPTH) {
  const entries = [...map.entries()];
  entries.sort((a, b) => (side === "bids" ? b[0] - a[0] : a[0] - b[0]));
  return entries.slice(0, n).map(([price, size]) => ({ price, size }));
}

/** Volume-weighted average price over the last `window` trades, or null. */
export function rollingVwap(trades, window = 100) {
  const recent = trades.slice(-window);
  let num = 0;
  let vol = 0;
  for (const t of recent) {
    num += t.price * t.size;
    vol += t.size;
  }
  return vol === 0 ? null : num / vol;
}

/** Top-N book imbalance in [-1, 1], or null when both sides are empty. */
export function bookImbalance(bids, asks, levels = DEPTH) {
  const bidQty = bids.slice(0, levels).reduce((s, l) => s + l.size, 0);
  const askQty = asks.slice(0, levels).reduce((s, l) => s + l.size, 0);
  const total = bidQty + askQty;
  return total === 0 ? null : (bidQty - askQty) / total;
}

/**
 * Aggressive-flow imbalance over the last `window` trades, in [-1, 1], or null.
 * Uses aggressor side: positive means aggressive buyers dominated.
 */
export function tradeFlowImbalance(trades, window = 50) {
  const recent = trades.slice(-window);
  let buy = 0;
  let sell = 0;
  for (const t of recent) {
    if (t.aggressorSide === "BUY") buy += t.size;
    else sell += t.size;
  }
  const total = buy + sell;
  return total === 0 ? null : (buy - sell) / total;
}

/**
 * Estimate immediate fills of a hypothetical order against displayed liquidity.
 * This does NOT rest unfilled quantity; it only reports what would fill now.
 *
 * @param {object} o
 * @param {"buy"|"sell"} o.side
 * @param {"limit"|"market"} o.type
 * @param {number|null} o.price limit price (ignored for market)
 * @param {{price:number,size:number}[]} o.levels opposite side, best price first
 * @param {number} o.quantity requested quantity
 * @returns {{fills:{price:number,size:number}[], requestedQty:number,
 *   filledQty:number, unfilledQty:number, avgPrice:number|null,
 *   notional:number, levelsConsumed:number}}
 */
export function simulateExecution({ side, type, price, levels, quantity }) {
  let remaining = quantity;
  const fills = [];
  for (const level of levels) {
    if (remaining <= 1e-15) break;
    if (type === "limit") {
      if (side === "buy" && level.price > price) break;
      if (side === "sell" && level.price < price) break;
    }
    const traded = Math.min(remaining, level.size);
    fills.push({ price: level.price, size: traded });
    remaining -= traded;
  }
  const filledQty = fills.reduce((s, f) => s + f.size, 0);
  const notional = fills.reduce((s, f) => s + f.price * f.size, 0);
  const avgPrice = filledQty > 0 ? notional / filledQty : null;
  return {
    fills,
    requestedQty: quantity,
    filledQty,
    unfilledQty: Math.max(0, quantity - filledQty),
    avgPrice,
    notional,
    levelsConsumed: fills.length,
  };
}

/**
 * Slippage of an average fill price versus the pre-trade mid, in basis points.
 * Positive = worse than mid (paid up on a buy / sold below mid on a sell).
 * @returns {number|null}
 */
export function slippageBps(avgPrice, midPrice, side) {
  if (avgPrice == null || midPrice == null || midPrice === 0) return null;
  const diff = side === "buy" ? avgPrice - midPrice : midPrice - avgPrice;
  return (diff / midPrice) * 10000;
}

/** Fresh per-product simulation state. */
export function newSimState() {
  return { position: 0, cash: 0, history: [] };
}

/** Apply a set of fills to a per-product {position, cash} state (mutates). */
export function applyFills(state, side, fills) {
  for (const f of fills) {
    if (side === "buy") {
      state.cash -= f.price * f.size;
      state.position += f.size;
    } else {
      state.cash += f.price * f.size;
      state.position -= f.size;
    }
  }
  return state;
}

/** Hypothetical mark-to-market P&L (fees excluded). */
export function markToMarket(state, midPrice) {
  return midPrice == null ? state.cash : state.cash + state.position * midPrice;
}
