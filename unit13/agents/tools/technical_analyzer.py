from __future__ import annotations

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

    def __call__(self, args: Dict[str, Any] | None = None) -> str:
        symbol = "BTC-USD"
        period = "30d"
        if args:
            symbol = args.get("symbol", symbol)
            period = args.get("period", period)
        try:
            data = yf.Ticker(symbol).history(period=period)
            if data.empty:
                return "No data to analyze."
            df = pd.DataFrame(data)
            df["sma20"] = ta.sma(df["Close"], length=20)
            df["rsi"] = ta.rsi(df["Close"], length=14)
            latest = df.iloc[-1]
            return f"SMA20: {latest['sma20']:.2f}, RSI: {latest['rsi']:.2f}"
        except Exception as exc:  # pragma: no cover - network access
            return f"Error calculating indicators: {exc}"


if __name__ == "__main__":
    tool = TechnicalAnalyzer()
    print(tool())
