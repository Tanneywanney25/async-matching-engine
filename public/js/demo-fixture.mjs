// Deterministic offline sample data, shaped exactly like Coinbase frames so it
// exercises the same parsing, rendering, metrics and simulator code paths as
// live data. Clearly labelled DEMO DATA in the UI; never presented as live.

function snapshot(productId, mid, tick) {
  const updates = [];
  for (let i = 1; i <= 12; i++) {
    updates.push({
      side: "bid",
      price_level: (mid - i * tick).toFixed(2),
      new_quantity: (0.5 + ((i * 7) % 5) * 0.35).toFixed(6),
    });
    updates.push({
      side: "offer",
      price_level: (mid + i * tick).toFixed(2),
      new_quantity: (0.5 + ((i * 5) % 5) * 0.4).toFixed(6),
    });
  }
  return {
    channel: "l2_data",
    timestamp: "1970-01-01T00:00:00Z",
    events: [{ type: "snapshot", product_id: productId, updates }],
  };
}

function trades(productId, mid, tick) {
  const rows = [];
  for (let i = 0; i < 18; i++) {
    const up = i % 2 === 0;
    rows.push({
      trade_id: `demo-${productId}-${i}`,
      product_id: productId,
      price: (mid + (up ? 1 : -1) * (i % 4) * tick).toFixed(2),
      size: (0.002 + (i % 6) * 0.011).toFixed(6),
      side: up ? "SELL" : "BUY", // maker side
      time: `1970-01-01T00:00:${String(i % 60).padStart(2, "0")}.000Z`,
    });
  }
  return {
    channel: "market_trades",
    timestamp: "1970-01-01T00:00:00Z",
    events: [{ type: "update", trades: rows }],
  };
}

/** Return the full set of demo frames to replay through the feed handlers. */
export function demoFrames() {
  return [
    snapshot("BTC-USD", 64000, 0.5),
    trades("BTC-USD", 64000, 0.5),
    snapshot("ETH-USD", 3400, 0.1),
    trades("ETH-USD", 3400, 0.1),
    { channel: "heartbeats", timestamp: "1970-01-01T00:00:00Z", events: [{ heartbeat_counter: 1 }] },
  ];
}
