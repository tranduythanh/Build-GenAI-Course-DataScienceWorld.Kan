from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class Message(TypedDict):
    type: MessageType
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]

class ChatHistory(TypedDict):
    messages: List[Message]
    session_id: str
    created_at: datetime
    updated_at: datetime

class ToolResult(TypedDict):
    tool_name: str
    result: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]

class AnalysisResult(TypedDict):
    symbol: str
    indicators: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]

class ReportData(TypedDict):
    type: str
    content: Dict[str, Any]
    generated_at: datetime
    metadata: Optional[Dict[str, Any]] 