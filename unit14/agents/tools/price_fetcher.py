from __future__ import annotations

import yfinance as yf
from langchain_core.tools import BaseTool


class PriceFetcher(BaseTool):
    """Fetch current Bitcoin and gold prices."""

    name: str = "price_fetcher"
    description: str = "Fetch the latest Bitcoin and gold prices."

    def _run(self, *args, **kwargs) -> str:
        """Synchronously fetch prices."""
        try:
            btc = yf.Ticker("BTC-USD").history(period="1d")
            btc_price = float(btc["Close"].iloc[-1])
            gold = yf.Ticker("GC=F").history(period="1d")
            gold_price = float(gold["Close"].iloc[-1])
        except Exception as exc:  # pragma: no cover - network access
            return f"Error fetching prices: {exc}"

        return f"BTC: {btc_price:.2f} USD, Gold: {gold_price:.2f} USD"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)
