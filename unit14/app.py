from __future__ import annotations

import sys

from crewai import Crew, Task

from agents.analysis_agent import create_analysis_agent
from agents.information_agent import create_information_agent
from agents.trading_agent import create_trading_agent


def build_crew() -> Crew:
    info_agent = create_information_agent()
    analysis_agent = create_analysis_agent()
    trading_agent = create_trading_agent()

    info_task = Task(
        description=(
            "Gather Bitcoin and gold prices and latest market news related to "
            "'{user_question}'."
        ),
        expected_output="Summary of market data",
        agent=info_agent,
    )

    analysis_task = Task(
        description=(
            "Analyze Bitcoin price using SMA20, RSI14, Bollinger Bands, "
            "and MACD to provide a trend forecast."
        ),
        expected_output="JSON with SMA20, RSI, BB, and MACD values",
        agent=analysis_agent,
    )

    trading_task = Task(
        description=(
            "Use the collected information and analysis to answer the user's "
            "question '{user_question}' and provide a trading recommendation "
            "for a portfolio of {portfolio} USD."
        ),
        expected_output="Trading decision in markdown",
        agent=trading_agent,
    )

    crew = Crew(
        agents=[info_agent, analysis_agent, trading_agent],
        tasks=[info_task, analysis_task, trading_task],
        verbose=True,
    )
    return crew


def main() -> None:
    crew = build_crew()
    print("Bitcoin Trading System. Type a question or 'exit'.")
    for line in sys.stdin:
        question = line.strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        inputs = {"user_question": question, "portfolio": 10000}
        result = crew.kickoff(inputs=inputs)
        print(result)


if __name__ == "__main__":
    main()
