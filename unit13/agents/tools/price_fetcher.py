from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import yfinance as yf


@dataclass
class PriceFetcher:
    """Fetch current or historical Bitcoin price using yfinance."""

    name: str = "price_fetcher"
    description: str = "Fetch current or historical Bitcoin price."

    def __call__(self, args: Dict[str, Any] | None = None) -> str:
        symbol = "BTC-USD"
        period = "1d"
        if args:
            symbol = args.get("symbol", symbol)
            period = args.get("period", period)
        try:
            data = yf.Ticker(symbol).history(period=period)
            if data.empty:
                return "No price data found."
            price = data["Close"].iloc[-1]
            return f"{symbol} price: {price:.2f}"
        except Exception as exc:  # pragma: no cover - network access
            return f"Error fetching price: {exc}"


if __name__ == "__main__":
    tool = PriceFetcher()
    print(tool())
