// Pure-logic tests for the browser dashboard core (no DOM, no network).
// Run with: node --test tests/frontend/
import test from "node:test";
import assert from "node:assert/strict";

import {
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
} from "../../public/js/core.mjs";

test("parsePositiveDecimal accepts clean positive numbers", () => {
  assert.equal(parsePositiveDecimal("0.01"), 0.01);
  assert.equal(parsePositiveDecimal(" 12.5 "), 12.5);
  assert.equal(parsePositiveDecimal("64000"), 64000);
});

test("parsePositiveDecimal rejects invalid input", () => {
  for (const bad of ["", "   ", "0", "-5", "0.01abc", "abc", "1e5", "1.2.3", "NaN", "1000000000000.1"]) {
    assert.equal(parsePositiveDecimal(bad), null, `should reject ${JSON.stringify(bad)}`);
  }
});

test("makerToAggressor inverts the maker side", () => {
  assert.equal(makerToAggressor("SELL"), "BUY");
  assert.equal(makerToAggressor("BUY"), "SELL");
  assert.equal(makerToAggressor("sell"), "BUY");
});

test("sortLevels orders bids desc and asks asc", () => {
  const m = new Map([[100, 1], [102, 2], [101, 3]]);
  assert.deepEqual(sortLevels(m, "bids").map((l) => l.price), [102, 101, 100]);
  assert.deepEqual(sortLevels(m, "asks").map((l) => l.price), [100, 101, 102]);
});

test("simulateExecution market order walks multiple levels", () => {
  const levels = [{ price: 100, size: 1 }, { price: 101, size: 1 }, { price: 102, size: 1 }];
  const r = simulateExecution({ side: "buy", type: "market", price: null, levels, quantity: 2.5 });
  assert.equal(r.filledQty, 2.5);
  assert.equal(r.levelsConsumed, 3);
  assert.equal(r.unfilledQty, 0);
  // avg = (100*1 + 101*1 + 102*0.5) / 2.5
  assert.ok(Math.abs(r.avgPrice - (100 + 101 + 51) / 2.5) < 1e-9);
  assert.ok(Math.abs(r.notional - (100 + 101 + 51)) < 1e-9);
});

test("simulateExecution limit stops at the limit price", () => {
  const levels = [{ price: 100, size: 1 }, { price: 105, size: 5 }];
  const r = simulateExecution({ side: "buy", type: "limit", price: 100, levels, quantity: 3 });
  assert.equal(r.filledQty, 1); // only the 100 level is acceptable
  assert.equal(r.unfilledQty, 2);
  assert.equal(r.levelsConsumed, 1);
});

test("simulateExecution reports no fill when nothing crosses", () => {
  const levels = [{ price: 105, size: 5 }];
  const r = simulateExecution({ side: "buy", type: "limit", price: 100, levels, quantity: 1 });
  assert.equal(r.filledQty, 0);
  assert.equal(r.avgPrice, null);
});

test("slippageBps signs cost correctly", () => {
  assert.ok(slippageBps(101, 100, "buy") > 0); // paid up
  assert.ok(slippageBps(99, 100, "sell") > 0); // sold below mid = worse
  assert.equal(slippageBps(null, 100, "buy"), null);
});

test("per-product sim state is isolated", () => {
  const btc = newSimState();
  const eth = newSimState();
  applyFills(btc, "buy", [{ price: 64000, size: 0.5 }]);
  assert.equal(btc.position, 0.5);
  assert.equal(btc.cash, -32000);
  assert.equal(eth.position, 0); // ETH untouched
  assert.equal(eth.cash, 0);
  // BTC P&L uses BTC mark, never ETH's
  assert.equal(markToMarket(btc, 65000), -32000 + 0.5 * 65000);
  assert.equal(markToMarket(eth, 3400), 0);
});

test("bookImbalance is zero for a symmetric book", () => {
  const bids = [{ price: 100, size: 5 }];
  const asks = [{ price: 101, size: 5 }];
  assert.equal(bookImbalance(bids, asks), 0);
});

test("tradeFlowImbalance uses aggressor side", () => {
  const trades = [
    { price: 10, size: 3, aggressorSide: "BUY" },
    { price: 10, size: 1, aggressorSide: "SELL" },
  ];
  assert.equal(tradeFlowImbalance(trades), 0.5); // (3-1)/4
});

test("rollingVwap matches known values", () => {
  const trades = [{ price: 10, size: 2 }, { price: 20, size: 3 }];
  assert.equal(rollingVwap(trades), 16); // (20+60)/5
  assert.equal(rollingVwap([]), null);
});
