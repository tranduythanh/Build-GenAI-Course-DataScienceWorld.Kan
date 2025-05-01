from typing import List, Optional, Any, Dict
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage

# Define the extended state
class AgentState(MessagesState):
    """State for the agent workflow"""

    # Chat history for memory or context tracking
    chat_history: Optional[List[BaseMessage]] = None

    # Steps taken by tools (raw Action/Observation pairs)
    intermediate_steps: Optional[Any] = None
