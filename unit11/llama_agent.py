from typing import List, Dict, Any
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings

class StockQuantAgent:
    def __init__(self, tools: List[BaseTool], api_key: str):
        self.llm = OpenAI(api_key=api_key)
        
        # Set global settings
        Settings.llm = self.llm
        
        # Initialize agent with tools
        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            verbose=True
        )

    def process_query(self, query: str) -> str:
        """
        Process a user query using the agent and return a response
        """
        try:
            # Use the agent to process the query
            response = self.agent.chat(query)
            return response.response
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Get the chat history from memory
        """
        try:
            return self.agent.chat_history
        except AttributeError:
            return []

    def clear_memory(self):
        """
        Clear the chat memory
        """
        try:
            self.agent.reset()
        except AttributeError:
            pass 