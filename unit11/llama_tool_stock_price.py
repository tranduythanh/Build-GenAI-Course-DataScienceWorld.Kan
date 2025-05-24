from typing import Dict, Any
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
from datetime import datetime
from llama_data_collector import DataCollector
from logger import tool_logger


class StockPriceTool(BaseTool):
    """Tool for getting stock price data"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.data_collector = DataCollector(api_key=api_key or "") if api_key else None
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="get_stock_price",
            description="Get stock price data for a given symbol. Provide the stock ticker symbol (e.g., 'AAPL', 'HAG'). Optional: start_date and end_date in YYYY-MM-DD format (defaults to 2023-01-01 to today)."
        )
    
    def __call__(self, input: Dict[str, Any]) -> ToolOutput:
        """Get stock price data for a given symbol and date range"""
        try:
            if not self.data_collector:
                return ToolOutput(
                    content="API key not configured",
                    tool_name=self.metadata.name,
                    raw_input=input,
                    raw_output={"error": "API key not configured"},
                    is_error=True
                )
            
            # Handle wrapped input format from LlamaIndex agent
            if "input" in input and isinstance(input["input"], str):
                # If input is just a string (symbol), use it with default date range
                symbol = input["input"]
                start_date = input.get("start_date", "2023-01-01")  # Default to 1 year ago
                end_date = input.get("end_date", datetime.now().strftime("%Y-%m-%d"))  # Default to today
            else:
                # Handle direct parameter format
                symbol = input.get("symbol")
                start_date = input.get("start_date")
                end_date = input.get("end_date")
            
            if not symbol:
                return ToolOutput(
                    content="Missing required parameter: symbol",
                    tool_name=self.metadata.name,
                    raw_input=input,
                    raw_output={"error": "Missing required parameter: symbol"},
                    is_error=True
                )
            
            # Provide defaults if dates are missing
            if not start_date:
                start_date = "2023-01-01"
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            tool_logger.info(f"Fetching stock data for {symbol} from {start_date} to {end_date}")
            df = self.data_collector.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            result = df.to_dict(orient="records")
            return ToolOutput(
                content=f"Successfully retrieved stock data for {symbol} from {start_date} to {end_date}",
                tool_name=self.metadata.name,
                raw_input=input,
                raw_output=result
            )
        except Exception as e:
            tool_logger.error(f"Error in get_stock_price: {str(e)}")
            return ToolOutput(
                content=f"Error getting stock price: {str(e)}",
                tool_name=self.metadata.name,
                raw_input=input,
                raw_output={"error": str(e)},
                is_error=True
            ) 