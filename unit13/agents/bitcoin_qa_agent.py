from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Dict, List, TypedDict, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (AIMessage, AnyMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .tools import MarketData, NewsFetcher, PriceFetcher, TechnicalAnalyzer


class AgentState(TypedDict):
    """State for the LangGraph agent."""

    messages: Annotated[List[AnyMessage], operator.add]


class BitcoinQAAgent:
    """LangGraph-based agent for Bitcoin Q&A."""

    def __init__(
        self, model, tools: List[BaseTool], system: str = "", checkpointer=None
    ) -> None:
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tool_map: Dict[str, BaseTool] = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def exists_action(self, state: AgentState) -> bool:
        result = state["messages"][-1]
        return len(getattr(result, "tool_calls", [])) > 0

    def take_action(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Nếu model quyết định cần sử dụng tool để trả lời câu hỏi

        Nó sẽ trả về một AIMessage chứa thuộc tính tool_calls

        tool_calls là một list các tool call objects,
        mỗi object chứa thông tin về tool cần gọi (name, args, id)
        """
        last_messages = state["messages"][-1]
        if not isinstance(last_messages, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        tool_calls = getattr(last_messages, "tool_calls", [])
        results: List[ToolMessage] = []
        for t in tool_calls:
            print(
                f"\t\t\033[90mCalling tool {t['name']}, args: {t.get('args', {})}\033[0m"
            )
            tool = self.tool_map.get(t["name"])
            if not tool:
                result = "bad tool name, retry"
            else:
                result = tool.invoke(t.get("args", {}))
            print(f"\t\t\033[90mResult: {result}\033[0m")
            results.append(
                ToolMessage(
                    tool_call_id=t["id"],
                    name=t["name"],
                    content=str(result),
                )
            )
        print("\t\t\033[90mBack to model\033[0m")
        return {"messages": results}


def default_agent() -> BitcoinQAAgent:
    """Create a default BitcoinQAAgent with predefined tools."""
    tools = [
        TavilySearchResults(max_results=4),
        MarketData(),
        NewsFetcher(),
        PriceFetcher(),
        TechnicalAnalyzer(),
    ]

    # Create tool descriptions, handling both custom tools and LangChain tools
    tool_descriptions = []
    for tool in tools:
        if hasattr(tool, "prompt_description"):
            tool_descriptions.append(tool.prompt_description())  # type: ignore
        else:
            # For LangChain tools like TavilySearchResults
            tool_descriptions.append(f"{tool.name}: {tool.description}")

    prompt = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been Use Action to run one of the actions available to you - then return PAUS
Observation will be the result of running those actions.
... (this Thought/Action/Action Input/Observation can repeat N times)
""".format(
        tools="\n\n".join(tool_descriptions)
    ).strip()

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    return BitcoinQAAgent(model=model, tools=tools, system=prompt)  # type: ignore
