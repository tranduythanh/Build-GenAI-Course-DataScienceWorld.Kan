from typing import Dict, Any
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
from llama_index.readers.web import SimpleWebPageReader


class WebDataTool(BaseTool):
    """Tool for collecting data from web sources"""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="collect_web_data",
            description="Collect data from web sources. Provide a URL to scrape content from (e.g., 'https://finance.yahoo.com/quote/HAG')."
        )
    
    def __call__(self, input: Dict[str, Any]) -> ToolOutput:
        """Collect data from web sources"""
        try:
            # Handle wrapped input format from LlamaIndex agent
            if "input" in input and isinstance(input["input"], str):
                url = input["input"]
            else:
                url = input.get("url")
                
            if not url:
                return ToolOutput(
                    content="Missing required parameter: url",
                    tool_name=self.metadata.name,
                    raw_input=input,
                    raw_output={"error": "Missing required parameter: url"},
                    is_error=True
                )
            
            reader = SimpleWebPageReader()
            documents = reader.load_data(urls=[url])
            result = {"data": [doc.text for doc in documents]}
            
            return ToolOutput(
                content=f"Successfully collected data from {url}",
                tool_name=self.metadata.name,
                raw_input=input,
                raw_output=result
            )
        except Exception as e:
            return ToolOutput(
                content=f"Error collecting web data: {str(e)}",
                tool_name=self.metadata.name,
                raw_input=input,
                raw_output={"error": str(e)},
                is_error=True
            ) 