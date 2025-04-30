from typing import Dict, TypedDict, Annotated, Literal
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage, AIMessage

# Define the state type
class AgentState(MessagesState):
    """State for the agent workflow"""
    next: Annotated[str, "The next agent to route to"]

def get_next_node(last_message: BaseMessage, goto: str):
    """Determine the next node based on the message content"""
    if isinstance(last_message, AIMessage) and "FINAL ANSWER" in last_message.content:
        return "END"
    return goto 