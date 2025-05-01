from typing import Dict, TypedDict, Annotated
from langgraph.graph import MessagesState

# Define the state type
class AgentState(MessagesState):
    """State for the agent workflow"""
    pass 