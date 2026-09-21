"""Live Rich terminal dashboard for the Coinbase feed.

Renders a refreshing view of the order book, headline metrics and the recent
trade tape for a selected product. Run with::

    python -m exchange_engine.dashboard [--product BTC-USD]
"""

from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal
from typing import List, Optional

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .feed import CoinbaseFeed
from .models import BookLevel, Trade

REFRESH_HZ = 2.0  # ~500ms refresh


def _fmt(value: Optional[Decimal], places: str = "0.01") -> str:
    """Format a Decimal to fixed places, or ``--`` when missing."""
    if value is None:
        return "--"
    return str(Decimal(value).quantize(Decimal(places)))


def build_book_table(depth: dict) -> Table:
    """Build the order-book table: asks (red, high→low) above bids (green)."""
    table = Table(expand=True, show_edge=False, pad_edge=False)
    table.add_column("Price", justify="right")
    table.add_column("Quantity", justify="right")

    asks: List[BookLevel] = depth["asks"][:10]
    bids: List[BookLevel] = depth["bids"][:10]

    for lvl in reversed(asks):  # display ascending downward toward the spread
        table.add_row(
            Text(_fmt(lvl.price), style="red"),
            Text(_fmt(lvl.quantity, "0.00000001"), style="red"),
        )
    table.add_row(Text("──────", style="white"), Text("──────", style="white"))
    for lvl in bids:
        table.add_row(
            Text(_fmt(lvl.price), style="green"),
            Text(_fmt(lvl.quantity, "0.00000001"), style="green"),
        )
    return table


def build_metrics_panel(metrics: dict) -> Table:
    """Build the metrics table: spread, mid, imbalance, VWAP."""
    table = Table(expand=True, show_header=False, show_edge=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    imbalance = metrics["imbalance"]
    imb_pct = f"{Decimal(imbalance) * 100:.1f}%" if imbalance is not None else "--"
    table.add_row("Spread", Text(_fmt(metrics["spread"]), style="white"))
    table.add_row("Mid price", Text(_fmt(metrics["mid_price"]), style="white"))
    table.add_row("Imbalance", Text(imb_pct, style="magenta"))
    table.add_row("VWAP (100)", Text(_fmt(metrics["vwap"]), style="cyan"))
    return table


def build_trades_table(trades: List[Trade]) -> Table:
    """Build the recent-trades tape (last 10, newest first)."""
    table = Table(expand=True, show_edge=False)
    table.add_column("Time", justify="left")
    table.add_column("Price", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Side", justify="center")
    for trade in reversed(trades[-10:]):
        style = "green" if trade.side == "SELL" else "red"
        ts = trade.time.split("T")[-1][:12] if "T" in trade.time else trade.time
        table.add_row(
            ts,
            _fmt(trade.price),
            _fmt(trade.size, "0.00000001"),
            Text(trade.side, style=style),
        )
    return table


def build_layout(feed: CoinbaseFeed, product_id: str) -> Layout:
    """Compose the full dashboard layout for ``product_id``."""
    depth = feed.get_depth(product_id, 10)
    metrics = feed.metrics(product_id)
    trades = feed.recent_trades(product_id, 10)

    selector = "  ".join(
        f"[reverse] {pid} [/reverse]" if pid == product_id else f" {pid} "
        for pid in feed.product_ids
    )
    header = Panel(
        Text.from_markup(f"exchange-engine   {selector}", justify="center"),
        style="bold white on #00142d",
    )

    layout = Layout()
    layout.split_column(
        Layout(header, size=3, name="header"),
        Layout(name="body"),
        Layout(Panel(build_trades_table(trades), title="Recent Trades"), size=14, name="tape"),
    )
    layout["body"].split_row(
        Layout(Panel(build_book_table(depth), title=f"Order Book · {product_id}"), name="book"),
        Layout(Panel(build_metrics_panel(metrics), title="Metrics"), name="metrics"),
    )
    return layout
