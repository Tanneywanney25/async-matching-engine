# exchange-engine

A real-time limit order book and matching engine that consumes **live Coinbase
Advanced Trade** market data — no API key or authentication required. It maintains
an in-memory price-time-priority book, computes microstructure analytics (VWAP,
spread, order-book imbalance, trade-flow imbalance), and ships with a Rich
terminal dashboard, a FastAPI service, a latency benchmark, and a static
**Vercel-deployable web dashboard**.

## Architecture

```
                         wss://advanced-trade-ws.coinbase.com
                                (public, no auth)
                                        │
                       level2 · market_trades · heartbeats
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │      feed.py             │  CoinbaseFeed
                          │  • book mirror (SortedDict)│  parses & routes frames
                          │  • recent trades (deque)  │
                          └───────────┬──────────────┘
                                      │ events / accessors
             ┌────────────────────────┼─────────────────────────┐
             ▼                        ▼                         ▼
   ┌──────────────────┐   ┌────────────────────┐    ┌────────────────────┐
   │  metrics.py      │   │   engine.py        │    │   dashboard.py     │
   │  VWAP / imbalance│   │  MatchingEngine +  │    │  Rich live terminal│
   │  / trade flow    │   │  FastAPI + CLI     │    │  UI (500ms refresh)│
   └──────────────────┘   └─────────┬──────────┘    └────────────────────┘
                                    │ owns its own
                                    ▼
                          ┌──────────────────────────┐
                          │     orderbook.py         │  OrderBook
                          │  price-time priority,     │  add / cancel / match
                          │  O(log n) via SortedDict  │  market orders
                          └──────────────────────────┘

   public/  ── static browser dashboard: connects to Coinbase WS directly,
               mirrors the same book & metrics logic (deploys on Vercel).
   api/     ── optional Vercel Python serverless order simulator.
```

The **feed's book mirror** reflects the real market. The **engine's OrderBook**
is a *separate* instance that simulates what your synthetic orders would do.

## Project layout

```
exchange-engine/
├── src/exchange_engine/
│   ├── models.py        # Enums + dataclasses (Decimal everywhere)
│   ├── orderbook.py     # Limit order book + matching (SortedDict, deque)
│   ├── feed.py          # Coinbase WebSocket consumer (raw websockets)
│   ├── engine.py        # MatchingEngine simulator + FastAPI + CLI
│   ├── dashboard.py     # Rich terminal UI
│   └── metrics.py       # VWAP, imbalance, trade-flow analytics
├── tests/               # pytest suite (orderbook, metrics, engine, feed)
├── scripts/benchmark.py # 100K-order latency benchmark
├── public/              # Static Vercel web dashboard (HTML/CSS/JS)
├── api/index.py         # Vercel Python serverless order simulator
└── vercel.json
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run the terminal dashboard

```bash
python -m exchange_engine.dashboard              # BTC-USD
python -m exchange_engine.dashboard --product ETH-USD
```

Live-updating (≈500 ms) order book, metrics panel, and trade tape.

## Run the API server

```bash
uvicorn exchange_engine.engine:app --port 8000
```

| Method | Path       | Description                                             |
|--------|------------|---------------------------------------------------------|
| POST   | `/order`   | Submit a synthetic order to the internal book; returns fills + P&L. |
| GET    | `/book`    | Top-N bid/ask levels from the **live feed mirror**.     |
| GET    | `/trades`  | Recent trades from the feed.                            |
| GET    | `/metrics` | Spread, mid price, VWAP, imbalance.                     |

```bash
curl -X POST localhost:8000/order \
  -H 'content-type: application/json' \
  -d '{"side":"buy","price":"64000","quantity":"0.01","order_type":"limit"}'
```

### CLI order entry

```bash
python -m exchange_engine.engine
> BUY 64000 0.01 LIMIT
> SELL MARKET 0.5
```

## Run the tests

```bash
pytest
```

## Run the benchmark

```bash
python scripts/benchmark.py                      # 100K orders, 10K cancels
```

Reports p50/p95/p99 insert+match and cancel latency in microseconds. Target:
median `add_order` under 50 µs (typically ~5–6 µs on a modern laptop).

## Web dashboard (Vercel)

The `public/` directory is a fully static dashboard that connects **directly**
to Coinbase's public WebSocket from the browser and mirrors the same book,
metrics and matching-simulator logic — so it needs no backend.

```bash
npm i -g vercel
vercel deploy                                    # or: vercel --prod
```

Because the endpoint is public and auth-free, the deployed site streams live
data with zero configuration. `api/index.py` is an optional Python serverless
function exposing the stateless order simulator at `/api/order`.

## Data source notes

- Endpoint: `wss://advanced-trade-ws.coinbase.com` (public channels, no JWT/key).
- One subscribe message **per channel**; `heartbeats` keeps the socket alive.
- level2 responses arrive on channel `l2_data`; sides are `bid`/`offer`
  (`offer` → ask); `new_quantity` is **absolute** (`0` removes the level).
- All prices/quantities are parsed to `decimal.Decimal` — never `float`.

## License

MIT — see [LICENSE](LICENSE).
