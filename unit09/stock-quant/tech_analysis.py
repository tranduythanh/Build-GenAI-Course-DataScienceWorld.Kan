import pandas as pd
import pandas_ta as ta
from typing import Dict, Optional, Union

class TechAna:
    """Technical Analysis Tool for Stock Market Data"""
    
    def __init__(self):
        """
        Initialize the Technical Analysis tool
        """
        pass
        
    def analyze(self, df: pd.DataFrame, indicator: str, params: Optional[Dict] = None) -> Union[str, pd.DataFrame]:
        """
        Perform technical analysis on provided stock data
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data with 'close' column
            indicator (str): Technical indicator to calculate (RSI, MACD, SMA, Bollinger)
            params (Dict, optional): Parameters for the indicator calculation
            
        Returns:
            Union[str, pd.DataFrame]: Analysis results or error message
        """
        try:
            if df.empty:
                return "Empty DataFrame provided"
            
            if 'close' not in df.columns:
                return "DataFrame must contain 'close' column"
            
            # Convert indicator name to pandas_ta function
            indicator = indicator.lower()
            if indicator == 'rsi':
                result = ta.rsi(df['close'], **params) if params else ta.rsi(df['close'])
            elif indicator == 'macd':
                result = ta.macd(df['close'], **params) if params else ta.macd(df['close'])
            elif indicator == 'sma':
                result = ta.sma(df['close'], **params) if params else ta.sma(df['close'])
            elif indicator == 'bollinger':
                result = ta.bbands(df['close'], **params) if params else ta.bbands(df['close'])
            else:
                return f"Unsupported indicator: {indicator}. Supported indicators: RSI, MACD, SMA, Bollinger Bands"
            
            return result
        except Exception as e:
            return f"Error in technical analysis: {str(e)}"
    
    def get_analysis_string(self, df: pd.DataFrame, symbol: str, indicator: str, params: Optional[Dict] = None) -> str:
        """
        Get formatted string representation of technical analysis
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data with 'close' column
            symbol (str): Stock symbol (for display purposes)
            indicator (str): Technical indicator to calculate
            params (Dict, optional): Parameters for the indicator calculation
            
        Returns:
            str: Formatted analysis results
        """
        result = self.analyze(df, indicator, params)
        if isinstance(result, str):
            return result
        return f"Technical Analysis for {symbol} - {indicator.upper()}:\n{result.tail().to_string()}" 