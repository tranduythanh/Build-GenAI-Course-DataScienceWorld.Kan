from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Dict, List, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .tools import MarketData, NewsFetcher, PriceFetcher, TechnicalAnalyzer

# mypy: ignore-errors


class AgentState(TypedDict):
    """State for the LangGraph agent."""

    messages: Annotated[List[AnyMessage], operator.add]


@dataclass
class BitcoinQAAgent:
    """LangGraph-based agent for Bitcoin Q&A."""

    model: ChatOpenAI
    tools: List[AnyMessage]
    system: str = ""

    def __post_init__(self) -> None:
        checkpointer = SqliteSaver.from_conn_string(":memory:")
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tool_map = {t.name: t for t in self.tools}
        self.model = self.model.bind_tools(self.tools)

    def exists_action(self, state: AgentState) -> bool:
        result = state["messages"][-1]
        return len(getattr(result, "tool_calls", [])) > 0

    def call_openai(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        tool_calls = state["messages"][-1].tool_calls
        results: List[ToolMessage] = []
        for call in tool_calls:
            tool = self.tool_map.get(call["name"])
            if not tool:
                result = "bad tool name, retry"
            else:
                result = tool(call.get("args", {}))
            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    content=str(result),
                )
            )
        return {"messages": results}


def default_agent() -> BitcoinQAAgent:
    """Create a default BitcoinQAAgent with predefined tools."""
    tools = [
        TavilySearchResults(max_results=4),
        PriceFetcher(),
        TechnicalAnalyzer(),
        NewsFetcher(),
        MarketData(),
    ]
    prompt = (
        "You are a Bitcoin research assistant. Use the provided tools to "
        "answer questions about the Bitcoin market."
    )
    model = ChatOpenAI(model="gpt-3.5-turbo")
    return BitcoinQAAgent(model=model, tools=tools, system=prompt)
