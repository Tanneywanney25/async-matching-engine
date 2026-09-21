# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-09-21

### Added
- `models` — `Side`, `OrderType`, `OrderStatus` enums and `Order`, `Fill`,
  `BookLevel`, `Trade` dataclasses, all using `decimal.Decimal`.
- `orderbook` — price-time-priority limit order book on `SortedDict`/`deque`
  with `add_order`, `cancel_order`, `match_market_order`, and depth/imbalance
  views; O(log n) operations.
- `feed` — `CoinbaseFeed` async consumer of the public Coinbase Advanced Trade
  WebSocket (level2, market_trades, heartbeats) with a book mirror, trade
  buffer, heartbeat gap detection, and exponential-backoff reconnection.
- `metrics` — rolling VWAP, book imbalance, and trade-flow imbalance.
- `engine` — `MatchingEngine` simulator with P&L accounting, a FastAPI service
  (`/order`, `/book`, `/trades`, `/metrics`), and a stdin CLI.
- `dashboard` — Rich terminal UI with a live order book, metrics, and trade tape.
- `scripts/benchmark.py` — 100K-order latency benchmark (p50/p95/p99).
- `scripts/replay.py` — offline synthetic feed replay.
- Static Vercel web dashboard (`public/`) and serverless order simulator (`api/`).
- Test suite covering the order book, metrics, engine, feed, and models.
