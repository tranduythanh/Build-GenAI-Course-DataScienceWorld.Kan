from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pandas_ta as ta
import yfinance as yf
from langchain_core.tools import BaseTool


class TechnicalAnalyzer(BaseTool):
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

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        period = args[0] if args else kwargs.get('period', '30d')
        try:
            data = yf.Ticker("BTC-USD").history(period=period)
            if data.empty:
                return "No data to analyze."
            df = pd.DataFrame(data)
            df["sma20"] = ta.sma(df["Close"], length=20)  # type: ignore
            df["rsi"] = ta.rsi(df["Close"], length=14)  # type: ignore
            latest = df.iloc[-1]
            return json.dumps(latest.to_dict())  # type: ignore
        except Exception as exc:  # pragma: no cover - network access
            return f"Error calculating indicators: {exc}"

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Use the tool asynchronously."""
        # For data analysis without specific async requirements,
        # we can delegate to the sync implementation
        return self._run(*args, **kwargs)


if __name__ == "__main__":
    tool = TechnicalAnalyzer()
    print(tool._run())
