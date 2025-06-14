from __future__ import annotations

from crewai import Agent


def create_trading_agent() -> Agent:
    """Create the trading agent."""
    return Agent(
        role="Trading Agent",
        goal=(
            "Manage portfolio and decide to buy or sell Bitcoin based on "
            "analysis and market information"
        ),
        tools=[],
        verbose=True,
        backstory=(
            "You combine technical analysis and market sentiment to make "
            "responsible trading decisions for the portfolio."
        ),
    )
