from typing import Dict, Any, Union, List
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
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
                
            # Ensure required columns exist and convert to numeric
            if 'close' not in df.columns:
                if 'Close' in df.columns:
                    df['close'] = pd.to_numeric(df['Close'], errors='coerce')
                elif 'adjust' in df.columns:
                    df['close'] = pd.to_numeric(df['adjust'], errors='coerce')
                else:
                    # Use the last numeric column as close price
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        df['close'] = df[numeric_cols[-1]]
                    else:
                        raise ValueError("No numeric columns found for close price")
            else:
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                
            # Remove rows with NaN close prices
            df = df.dropna(subset=['close'])
                
            # Calculate SMA
            df['SMA_20'] = ta.sma(df['close'], length=20)
            
            # Calculate RSI
            df['RSI_14'] = ta.rsi(df['close'], length=14)
            
            # Calculate MACD
            macd_data = ta.macd(df['close'])
            if macd_data is not None:
                df = pd.concat([df, macd_data], axis=1)
            
            # Calculate Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20)
            if bb_data is not None:
                df = pd.concat([df, bb_data], axis=1)
            
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