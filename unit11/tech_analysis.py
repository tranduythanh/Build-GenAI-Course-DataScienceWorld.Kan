import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Optional, Union, List
import logging

logger: logging.Logger = logging.getLogger(__name__)

class TechAna:
    """Technical Analysis Tool for Stock Market Data"""
    
    def analyze(self, df: pd.DataFrame, indicator: str, params: Optional[Dict[str, Union[int, float]]] = None) -> Union[str, pd.DataFrame]:
        """
        Perform technical analysis on provided stock data
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data with 'close' column
            indicator (str): Technical indicator to calculate (RSI, MACD, SMA, Bollinger)
            params (Dict, optional): Parameters for the indicator calculation
            
        Returns:
            Union[str, pd.DataFrame]: Analysis results or error message
        """
        logger.debug(f"Analyzing {indicator} for DataFrame with shape {df.shape}")
        try:
            if df is None or df.empty:
                return "Empty DataFrame provided"
            
            # Drop 'Symbols' level from columns if it exists
            if isinstance(df.columns, pd.MultiIndex) and 'Symbols' in df.columns.names:
                df = df.droplevel('Symbols', axis=1)
            
            if 'close' not in df.columns:
                return "DataFrame must contain 'close' column"
            
            # Convert indicator name to pandas_ta function
            indicator = indicator.lower().replace(' bands', '')
            
            # Get the close price series
            close_prices_list: List[float] = df['close'].astype(np.float64).tolist()
            close_prices: pd.Series = pd.Series(close_prices_list, index=df.index)

            logger.debug(f"Close prices: {close_prices}")
            
            # Handle each indicator type
            result: Optional[Union[pd.Series, pd.DataFrame]]
            if indicator == 'rsi':
                # Default RSI parameters if none provided
                if params is None:
                    params = {'length': 14}
                result = ta.rsi(close_prices, **params)
                logger.debug(f"RSI result: {result}")
            elif indicator == 'macd':
                # Default MACD parameters if none provided
                if params is None:
                    params = {'fast': 12, 'slow': 26, 'signal': 9}
                result = ta.macd(close_prices, **params)
            elif indicator == 'sma':
                # Default SMA parameters if none provided
                if params is None:
                    params = {'length': 20}
                result = ta.sma(close_prices, **params)
            elif indicator == 'bollinger':
                # Default Bollinger Bands parameters if none provided
                if params is None:
                    params = {'length': 20, 'std': 2}
                result = ta.bbands(close_prices, **params)
            else:
                return f"Unsupported indicator: {indicator}. Supported indicators: RSI, MACD, SMA, Bollinger Bands"
            
            if result is None:
                return f"Error: No result returned for indicator {indicator}. Please check the input data."
                
            if isinstance(result, pd.Series):
                result = result.to_frame()
                
            if not isinstance(result, pd.DataFrame):
                return f"Error: Unexpected result type for indicator {indicator}. Got {type(result)}"
                
            if result.empty:
                return f"No data available for indicator {indicator}. Please check the input data."
            
            return result
        except Exception as e:
            logger.error(f"Error in technical analysis for {indicator}: {str(e)}")
            return f"Error in technical analysis for {indicator}: {str(e)}"