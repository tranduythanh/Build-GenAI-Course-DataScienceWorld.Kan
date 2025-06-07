from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import requests


@dataclass
class MarketData:
    """Retrieve basic Bitcoin market data from CoinGecko."""

    name: str = "market_data"
    description: str = "Get Bitcoin market data."

    def prompt_description(self) -> str:
        return f'''
{self.name}:
    Example usage: {self.name}
    Return the market data of bitcoin in json format
'''.strip()

    def __call__(self, args: Dict[str, Any] | None = None) -> str:
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


if __name__ == "__main__":
    tool = MarketData()
    print(tool.name, tool.description)
    print(tool())
