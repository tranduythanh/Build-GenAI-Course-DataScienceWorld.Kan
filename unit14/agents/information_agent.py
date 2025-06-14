from __future__ import annotations

from crewai import Agent
from crewai_tools import SerperDevTool

from .tools.price_fetcher import PriceFetcher


def create_information_agent() -> Agent:
    """Create the information agent."""
    return Agent(
        role="Information Agent",
        goal=(
            "Collect Bitcoin and gold prices and update market news "
            "to assist trading decisions"
        ),
        tools=[SerperDevTool(), PriceFetcher()],
        verbose=True,
        backstory=(
            "You specialize in gathering up-to-date market information, "
            "including prices and breaking news about Bitcoin and gold."
        ),
    )
