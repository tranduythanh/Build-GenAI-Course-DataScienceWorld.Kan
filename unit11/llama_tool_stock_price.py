from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
from datetime import datetime
from llama_data_collector import DataCollector
from logger import tool_logger


class StockPriceTool(BaseTool):
    """Tool for getting stock price data"""
    
    def __init__(self):
        self.data_collector = DataCollector()
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="get_stock_price",
            description="Get stock price data for a given symbol. Provide the stock ticker symbol (e.g., 'HAG'). Optional: start_date and end_date in YYYY-MM-DD format (defaults to 2023-01-01 to today)."
        )
    
    def __call__(self, input) -> ToolOutput:
        """Get stock price data for a given symbol and date range"""
        try:
            # Normalize input to always be a dictionary for raw_input
            if isinstance(input, str):
                # If input is just a string, treat it as the symbol
                symbol = input
                start_date = "2023-01-01"
                end_date = datetime.now().strftime("%Y-%m-%d")
                raw_input_dict = {"symbol": symbol, "start_date": start_date, "end_date": end_date}
            elif isinstance(input, dict):
                raw_input_dict = input
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
            else:
                return ToolOutput(
                    content="Invalid input format",
                    tool_name=self.metadata.name,
                    raw_input={"error": "Invalid input format"},
                    raw_output={"error": "Invalid input format"},
                    is_error=True
                )
            
            if not self.data_collector:
                return ToolOutput(
                    content="API key not configured",
                    tool_name=self.metadata.name,
                    raw_input=raw_input_dict,
                    raw_output={"error": "API key not configured"},
                    is_error=True
                )
            
            if not symbol:
                return ToolOutput(
                    content="Missing required parameter: symbol",
                    tool_name=self.metadata.name,
                    raw_input=raw_input_dict,
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
            
            tool_logger.info(f"Successfully retrieved stock data for {symbol} from {start_date} to {end_date}")
            tool_logger.info(f"Number of rows: {len(result)}")
            if len(result) > 5:
                tool_logger.info(f"last 5 rows: {result[-5:]}")
            else:
                tool_logger.info(f"all rows: {result}")
            
            return ToolOutput(
                content=f"Successfully retrieved stock data for {symbol} from {start_date} to {end_date}",
                tool_name=self.metadata.name,
                raw_input=raw_input_dict,
                raw_output=result
            )
        except Exception as e:
            tool_logger.error(f"Error in get_stock_price: {str(e)}")
            return ToolOutput(
                content=f"Error getting stock price: {str(e)}",
                tool_name=self.metadata.name,
                raw_input=raw_input_dict if 'raw_input_dict' in locals() else {"error": "Failed to process input"},
                raw_output={"error": str(e)},
                is_error=True
            ) 
        
if __name__ == "__main__":
    tool = StockPriceTool()
    print("\n\n\nmetadata")
    print(tool.metadata)
    print("\n\n\nHAG")
    print(tool("HAG"))
    print("\n\n\nVIC")
    print(tool("VIC"))
    print("\n\n\nVNINDEX")
    print(tool("VNINDEX"))
    