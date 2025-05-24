from typing import Dict, Any
from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
import pandas as pd
import pandas_ta as ta


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
            # Handle wrapped input format from LlamaIndex agent
            if "input" in input:
                if isinstance(input["input"], (dict, list)):
                    data = input["input"]
                else:
                    # If input is just a string, try to get data from the main input
                    data = input.get("data")
            else:
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