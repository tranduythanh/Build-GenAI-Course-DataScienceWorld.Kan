from typing import Dict, Any
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
from llama_index.readers.web import SimpleWebPageReader
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from llama_data_collector import DataCollector
from logger import tool_logger
from config import TECH_ANALYSIS_CONFIG

class StockPriceTool(BaseTool):
    """Tool for getting stock price data"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.data_collector = DataCollector(api_key=api_key or "") if api_key else None
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="get_stock_price",
            description="Get stock price data for a given symbol and date range. Requires symbol (stock ticker), start_date, and end_date in YYYY-MM-DD format."
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
            
            symbol = input.get("symbol")
            start_date = input.get("start_date")
            end_date = input.get("end_date")
            
            if not all([symbol, start_date, end_date]):
                return ToolOutput(
                    content="Missing required parameters: symbol, start_date, end_date",
                    tool_name=self.metadata.name,
                    raw_input=input,
                    raw_output={"error": "Missing required parameters"},
                    is_error=True
                )
                
            tool_logger.info(f"Fetching stock data for {symbol}")
            df = self.data_collector.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            result = df.to_dict(orient="records")
            return ToolOutput(
                content=f"Successfully retrieved stock data for {symbol}",
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

class TechnicalAnalysisTool(BaseTool):
    """Tool for calculating technical indicators"""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calculate_technical_indicators",
            description="Calculate technical indicators (SMA, RSI, MACD, Bollinger Bands) for stock data. Requires stock price data as input."
        )
    
    def __call__(self, input: Dict[str, Any]) -> ToolOutput:
        """Calculate technical indicators for stock data"""
        try:
            data = input.get("data")
            if not data:
                return ToolOutput(
                    content="Missing required parameter: data",
                    tool_name=self.metadata.name,
                    raw_input=input,
                    raw_output={"error": "Missing required parameter: data"},
                    is_error=True
                )
            
            # Convert dict back to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
                
            # Calculate SMA
            df.ta.sma(length=20, append=True)
            
            # Calculate RSI
            df.ta.rsi(length=14, append=True)
            
            # Calculate MACD
            df.ta.macd(append=True)
            
            # Calculate Bollinger Bands
            df.ta.bbands(length=20, append=True)
            
            result = df.to_dict(orient="records")
            return ToolOutput(
                content="Successfully calculated technical indicators",
                tool_name=self.metadata.name,
                raw_input=input,
                raw_output=result
            )
        except Exception as e:
            return ToolOutput(
                content=f"Error calculating technical indicators: {str(e)}",
                tool_name=self.metadata.name,
                raw_input=input,
                raw_output={"error": str(e)},
                is_error=True
            )

class WebDataTool(BaseTool):
    """Tool for collecting data from web sources"""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="collect_web_data",
            description="Collect data from web sources. Requires a URL as input."
        )
    
    def __call__(self, input: Dict[str, Any]) -> ToolOutput:
        """Collect data from web sources"""
        try:
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

class FinanceReportTool(BaseTool):
    """Tool for generating financial analysis reports"""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="generate_finance_report",
            description="Generate financial analysis reports. Requires data and report_type ('technical' or 'fundamental')."
        )
    
    def __call__(self, input: Dict[str, Any]) -> ToolOutput:
        """Generate financial analysis reports"""
        try:
            data = input.get("data")
            report_type = input.get("report_type", "technical")
            
            if not data:
                return ToolOutput(
                    content="Missing required parameter: data",
                    tool_name=self.metadata.name,
                    raw_input=input,
                    raw_output={"error": "Missing required parameter: data"},
                    is_error=True
                )
            
            if report_type == "technical":
                result = {
                    "type": "technical",
                    "summary": "Technical Analysis Report",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
            elif report_type == "fundamental":
                result = {
                    "type": "fundamental",
                    "summary": "Fundamental Analysis Report",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return ToolOutput(
                    content="Unsupported report type. Use 'technical' or 'fundamental'",
                    tool_name=self.metadata.name,
                    raw_input=input,
                    raw_output={"error": "Unsupported report type"},
                    is_error=True
                )
            
            return ToolOutput(
                content=f"Successfully generated {report_type} analysis report",
                tool_name=self.metadata.name,
                raw_input=input,
                raw_output=result
            )
        except Exception as e:
            return ToolOutput(
                content=f"Error generating report: {str(e)}",
                tool_name=self.metadata.name,
                raw_input=input,
                raw_output={"error": str(e)},
                is_error=True
            )

# Factory function for creating tools
def create_stock_tools(api_key: str = None):
    """Create and return a list of stock analysis tools"""
    return [
        StockPriceTool(api_key=api_key),
        TechnicalAnalysisTool(),
        WebDataTool(),
        FinanceReportTool()
    ] 