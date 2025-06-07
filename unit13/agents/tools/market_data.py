from __future__ import annotations

import json
from typing import Any, Dict

import requests
from langchain_core.tools import BaseTool


class MarketData(BaseTool):
    """Retrieve basic Bitcoin market data from CoinGecko."""

    name: str = "market_data"
    description: str = "Get Bitcoin market data."

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Use the tool."""
        url = (
            "https://api.coingecko.com/api/v3/coins/markets"
            "?vs_currency=usd&ids=bitcoin"
        )
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()[0]
            return json.dumps(data)
        except Exception as exc:  # pragma: no cover - network access
            return f"Error fetching market data: {exc}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Use the tool asynchronously."""
        # For simple HTTP requests without specific async requirements,
        # we can delegate to the sync implementation
        return self._run(*args, **kwargs)

    def prompt_description(self) -> str:
        return f'''
{self.name}:
    Example usage: {self.name}
    Return the market data of bitcoin in json format
'''.strip()


if __name__ == "__main__":
    tool = MarketData()
    print(tool.name, tool.description)
    print(tool._run())
