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
            description="""Get stock price data for a given symbol. 
            
            Parameters:
            - symbol (required): Stock ticker symbol (e.g., 'HAG', 'VIC', 'VNINDEX')
            - start_date (optional): Start date in YYYY-MM-DD format (default: 2023-01-01)
            - end_date (optional): End date in YYYY-MM-DD format (default: today)
            
            Examples:
            - "HAG" - Get HAG stock data from 2023-01-01 to today
            - {"symbol": "VIC", "start_date": "2023-06-01", "end_date": "2023-12-31"}
            - {"symbol": "VNINDEX", "start_date": "2024-01-01"} - end_date defaults to today"""
        )
    
    def __call__(self, input: str) -> ToolOutput:
        """Get stock price data for a given symbol and date range
        
        Args:
            input: Can be one of the following formats:
                - String: Just the symbol (e.g., "HAG", "VIC", "VNINDEX")
                - Dict with direct parameters: {"symbol": "HAG", "start_date": "2023-01-01", "end_date": "2024-01-01"}
                - Dict with wrapped input: {"input": "HAG", "start_date": "2023-01-01", "end_date": "2024-01-01"}
        
        Examples:
            tool("HAG")  # Get HAG data from 2023-01-01 to today
            tool({"symbol": "VIC", "start_date": "2023-06-01", "end_date": "2023-12-31"})
            tool({"input": "VNINDEX", "start_date": "2024-01-01"})  # end_date defaults to today
            
        Returns:
            ToolOutput with stock price data in raw_output as list of records
        """
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
    
    # Ví dụ 1: Chỉ symbol (sử dụng default date range)
    print("\n\n\n=== Ví dụ 1: Chỉ symbol (HAG) ===")
    print(tool("HAG"))
    
    print("\n\n\n=== Ví dụ 2: Chỉ symbol (VIC) ===")
    print(tool("VIC"))
    
    print("\n\n\n=== Ví dụ 3: Chỉ symbol (VNINDEX) ===")
    print(tool("VNINDEX"))
    
    # Ví dụ 4: Dictionary format với đầy đủ tham số
    print("\n\n\n=== Ví dụ 4: Dictionary format với đầy đủ tham số ===")
    print(tool({
        "symbol": "FPT", 
        "start_date": "2024-01-01", 
        "end_date": "2024-06-30"
    }))
    
    # Ví dụ 5: Dictionary format với chỉ start_date
    print("\n\n\n=== Ví dụ 5: Dictionary format với chỉ start_date ===")
    print(tool({
        "symbol": "MSN", 
        "start_date": "2024-06-01"
    }))
    
    # Ví dụ 6: Wrapped input format (như từ LlamaIndex agent)
    print("\n\n\n=== Ví dụ 6: Wrapped input format ===")
    print(tool({
        "input": "TCB",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31"
    }))
    
    # Ví dụ 7: Wrapped input format chỉ với symbol
    print("\n\n\n=== Ví dụ 7: Wrapped input format chỉ với symbol ===")
    print(tool({
        "input": "BID"
    }))
    
    # Ví dụ 8: Test với symbol khác
    print("\n\n\n=== Ví dụ 8: Symbol khác (CTG) ===")
    print(tool({
        "symbol": "CTG",
        "start_date": "2023-12-01",
        "end_date": "2023-12-31"
    }))
    