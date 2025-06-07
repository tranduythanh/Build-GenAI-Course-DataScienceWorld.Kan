"""Simple workflow using BitcoinQAAgent."""

from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage

from ..bitcoin_qa_agent import BitcoinQAAgent, default_agent


def run_query(agent: BitcoinQAAgent, question: str) -> str:
    """Run a question through the agent."""
    messages = [HumanMessage(content=question)]
    result = agent.graph.invoke({"messages": messages})
    return result["messages"][-1].content


def example_questions() -> List[str]:
    return [
        "Giá Bitcoin hện tại là bao nhiêu?",
        "RSI của Bitcoin tuần này bao nhiêu?",
    ]


if __name__ == "__main__":
    agent = default_agent()
    for q in example_questions():
        print(q)
        print(run_query(agent, q))
