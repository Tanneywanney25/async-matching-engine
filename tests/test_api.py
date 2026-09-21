"""API tests for the FastAPI app using Starlette's TestClient.

The client is used *without* the ``with`` context manager so the lifespan (which
would start the live Coinbase feed) never runs — these tests are deterministic
and never touch the network. Endpoints that read the feed mirror therefore see
an empty book, which is exactly what we assert against for serialization shape.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from exchange_engine.engine import app

client = TestClient(app)


def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "BTC-USD" in body["products"]


def test_order_valid_limit() -> None:
    resp = client.post(
        "/order",
        json={"side": "sell", "price": "101", "quantity": "1", "order_type": "limit"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "fills" in body and "pnl" in body


def test_order_market_after_seeding() -> None:
    client.post("/order", json={"side": "bid", "price": "100", "quantity": "2"})
    resp = client.post("/order", json={"side": "ask", "quantity": "1", "order_type": "market"})
    assert resp.status_code == 200
    assert sum(float(f["quantity"]) for f in resp.json()["fills"]) >= 1.0


@pytest.mark.parametrize(
    "payload",
    [
        {"side": "sideways", "price": "1", "quantity": "1"},  # bad side
        {"side": "buy", "price": "1"},  # missing quantity
        {"side": "buy", "price": "1", "quantity": "abc"},  # non-numeric quantity
        {"side": "buy", "price": "1", "quantity": "-5"},  # non-positive quantity
    ],
)
def test_order_validation_errors(payload: dict) -> None:
    resp = client.post("/order", json=payload)
    assert resp.status_code == 400


def test_book_trades_metrics_serialization() -> None:
    book = client.get("/book", params={"product_id": "BTC-USD"}).json()
    assert set(book) == {"product_id", "bids", "asks"}
    trades = client.get("/trades", params={"product_id": "BTC-USD"}).json()
    assert set(trades) == {"product_id", "trades"}
    metrics = client.get("/metrics", params={"product_id": "BTC-USD"}).json()
    assert set(metrics) == {"product_id", "spread", "mid_price", "vwap", "imbalance"}
