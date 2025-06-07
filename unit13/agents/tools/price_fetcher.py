from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import yfinance as yf
from langchain_core.tools import BaseTool


class PriceFetcher(BaseTool):
    """Fetch current or historical Bitcoin price using yfinance."""

    name: str = "price_fetcher"
    description: str = "Fetch current or historical Bitcoin price."

    def prompt_description(self) -> str:
        return f"""
{self.name}:
    Example usage: {self.name} 5d
    Return the price of bitcoin for the last 5 days in json format
""".strip()

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        period = args[0] if args else kwargs.get("period", "1d")
        try:
            data = yf.Ticker("BTC-USD").history(period=period)
            if data.empty:
                return "No price data found."
            # Convert to JSON and transform timestamps
            json_data = json.loads(data.to_json())

            # Convert Unix timestamps to YYYY-MM-DD format
            for key in json_data:
                if isinstance(json_data[key], dict):
                    new_dict = {}
                    for timestamp_str, value in json_data[key].items():
                        # Convert timestamp from milliseconds to seconds and format as YYYY-MM-DD
                        timestamp = int(timestamp_str) / 1000
                        date_str = datetime.fromtimestamp(timestamp).strftime(
                            "%Y-%m-%d"
                        )
                        new_dict[date_str] = value
                    json_data[key] = new_dict

            return json.dumps(json_data)
        except Exception as exc:  # pragma: no cover - network access
            return f"Error fetching price: {exc}"

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Use the tool asynchronously."""
        # For yfinance API calls without specific async requirements,
        # we can delegate to the sync implementation
        return self._run(*args, **kwargs)


if __name__ == "__main__":
    tool = PriceFetcher()
    print(tool._run())
