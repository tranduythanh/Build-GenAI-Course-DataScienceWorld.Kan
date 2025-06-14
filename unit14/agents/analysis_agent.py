from __future__ import annotations

from crewai import Agent

from .tools.price_fetcher import PriceFetcher
from .tools.technical_analyzer import TechnicalAnalyzer


def create_analysis_agent() -> Agent:
    """Create the technical analysis agent."""
    return Agent(
        role="Technical Analysis Agent",
        goal=(
            "Analyze Bitcoin price using SMA, RSI, Bollinger Bands, and MACD "
            "to forecast trend"
        ),
        tools=[PriceFetcher(), TechnicalAnalyzer()],
        verbose=True,
        backstory=(
            "You are an expert in technical analysis, capable of generating "
            "insights from price data using classic indicators."
        ),
    )
