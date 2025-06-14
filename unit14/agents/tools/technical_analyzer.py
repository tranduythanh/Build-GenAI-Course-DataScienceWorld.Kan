from __future__ import annotations

import json
from typing import Any

import yfinance as yf
from langchain_core.tools import BaseTool


class TechnicalAnalyzer(BaseTool):
    """Perform basic technical analysis on Bitcoin."""

    name: str = "technical_analyzer"
    description: str = (
        "Analyze Bitcoin price using SMA20, RSI14, Bollinger Bands, and MACD indicators."  # noqa: E501
    )

    def _run(self, *args: Any, **kwargs: Any) -> str:
        period = kwargs.get("period", "3mo")
        try:
            data = yf.Ticker("BTC-USD").history(period=period)
            close = data["Close"]
            import pandas_ta as ta

            sma = ta.sma(close, length=20).iloc[-1]
            rsi = ta.rsi(close, length=14).iloc[-1]
            bb = ta.bbands(close, length=20)
            bb_upper = bb["BBU_20_2.0"].iloc[-1]
            bb_lower = bb["BBL_20_2.0"].iloc[-1]
            macd = ta.macd(close)
            macd_diff = macd["MACDh_12_26_9"].iloc[-1]

            result = {
                "SMA20": float(sma),
                "RSI": float(rsi),
                "BB_upper": float(bb_upper),
                "BB_lower": float(bb_lower),
                "MACD_diff": float(macd_diff),
            }
            return json.dumps(result)
        except Exception as exc:  # pragma: no cover - network access
            return f"Error analyzing data: {exc}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)
