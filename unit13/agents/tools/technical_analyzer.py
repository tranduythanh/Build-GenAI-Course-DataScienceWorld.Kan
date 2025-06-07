from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import pandas_ta as ta
import yfinance as yf


@dataclass
class TechnicalAnalyzer:
    """Calculate basic technical indicators for Bitcoin."""

    name: str = "technical_analyzer"
    description: str = (
        "Compute simple technical indicators like SMA and RSI for Bitcoin."
    )

    def prompt_description(self) -> str:
        return f'''
{self.name}:
    Example usage: {self.name} 30d
    Return the technical analysis of bitcoin for the last 30 days in json format
'''.strip()

    def __call__(self, period: str = "30d") -> str:
        try:
            data = yf.Ticker("BTC-USD").history(period=period)
            if data.empty:
                return "No data to analyze."
            df = pd.DataFrame(data)
            df["sma20"] = ta.sma(df["Close"], length=20)
            df["rsi"] = ta.rsi(df["Close"], length=14)
            latest = df.iloc[-1]
            return json.dumps(latest.to_dict())
        except Exception as exc:  # pragma: no cover - network access
            return f"Error calculating indicators: {exc}"


if __name__ == "__main__":
    tool = TechnicalAnalyzer()
    print(tool())
