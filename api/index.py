"""Vercel Python serverless function: the matching-engine order simulator.

Exposes a lightweight ASGI ``app`` that reuses the exact
:class:`~exchange_engine.engine.MatchingEngine` from the package. Only the
stateless simulator endpoints are served here; the live order book, trades and
metrics on the dashboard are streamed client-side straight from Coinbase, so no
long-running WebSocket process is needed on the serverless side.

Note: serverless instances are ephemeral, so P&L accumulates only within a warm
instance. Use ``POST /api/reset`` to clear it.
"""

from __future__ import annotations

import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict

# Make the package importable from the repository's src/ layout at runtime.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from exchange_engine.engine import MatchingEngine, _parse_side  # noqa: E402
from exchange_engine.models import OrderType  # noqa: E402

app = FastAPI(title="exchange-engine (serverless)", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

engine = MatchingEngine("BTC-USD")


def _health() -> Dict[str, object]:
    return {"service": "exchange-engine", "mode": "serverless", "pnl": engine.pnl_summary()}


def _order(payload: Dict) -> Dict[str, object]:
    try:
        side = _parse_side(str(payload["side"]))
        order_type = OrderType(str(payload.get("order_type", "limit")).lower())
        quantity = Decimal(str(payload["quantity"]))
        price = payload.get("price")
        price_dec = Decimal(str(price)) if price is not None else None
    except (KeyError, ValueError, InvalidOperation) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    fills = engine.submit_order(side, quantity, price_dec, order_type)
    return {
        "fills": [
            {"price": str(f.price), "quantity": str(f.quantity), "side": f.side.value}
            for f in fills
        ],
        "pnl": engine.pnl_summary(),
    }


def _reset() -> Dict[str, str]:
    global engine
    engine = MatchingEngine("BTC-USD")
    return {"status": "reset"}


@app.get("/")
@app.get("/health")
@app.get("/api")
@app.get("/api/health")
async def health() -> Dict[str, object]:
    """Health check and current P&L snapshot."""
    return _health()


@app.post("/order")
@app.post("/api/order")
async def order_endpoint(payload: Dict = Body(...)) -> Dict[str, object]:
    """Submit a synthetic order to the stateless serverless simulator."""
    return _order(payload)


@app.post("/reset")
@app.post("/api/reset")
async def reset_endpoint() -> Dict[str, str]:
    """Reset the warm-instance P&L state."""
    return _reset()
