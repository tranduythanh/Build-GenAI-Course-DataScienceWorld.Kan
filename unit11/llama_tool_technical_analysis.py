from typing import Dict, Any, Union, List
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
import pandas as pd


class TechnicalAnalysisTool(BaseTool):
    """Tool for calculating technical indicators"""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calculate_technical_indicators",
            description="Calculate technical indicators (SMA, RSI, MACD, Bollinger Bands) for stock data. Requires stock price data as input."
        )
    
    def __call__(self, input: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> ToolOutput:
        """Calculate technical indicators for stock data"""
        try:
            # Normalize input to always be a dictionary for raw_input
            data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame, None]
            raw_input_dict: Dict[str, Any]
            
            if isinstance(input, (dict, list)):
                data = input
                raw_input_dict = {"data": data} if not isinstance(input, dict) else input
            elif isinstance(input, dict):
                # Handle wrapped input format from LlamaIndex agent
                if "input" in input:
                    if isinstance(input["input"], (dict, list)):
                        data = input["input"]
                    else:
                        # If input is just a string, try to get data from the main input
                        data = input.get("data")
                else:
                    data = input.get("data")
                raw_input_dict = input
            else:
                raw_input_dict = {"error": "Invalid input format"}
                data = None
                
            if not data:
                return ToolOutput(
                    content="Missing required parameter: data",
                    tool_name=self.metadata.name,
                    raw_input=raw_input_dict,
                    raw_output={"error": "Missing required parameter: data"},
                    is_error=True
                )
            
            # Convert dict back to DataFrame if needed
            df: pd.DataFrame
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
            
            result: List[Dict[str, Any]] = df.to_dict(orient="records")
            return ToolOutput(
                content="Successfully calculated technical indicators",
                tool_name=self.metadata.name,
                raw_input=raw_input_dict,
                raw_output=result
            )
        except Exception as e:
            return ToolOutput(
                content=f"Error calculating technical indicators: {str(e)}",
                tool_name=self.metadata.name,
                raw_input=raw_input_dict if 'raw_input_dict' in locals() else {"error": "Failed to process input"},
                raw_output={"error": str(e)},
                is_error=True
            ) 