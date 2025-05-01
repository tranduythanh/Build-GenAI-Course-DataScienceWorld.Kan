import logging
from typing import Dict, Any, Optional
import pandas as pd

from langchain_core.tools import BaseTool
from data_collector import StockDataCollector
from finance_report import FinanceReport
from tech_analysis import TechAna

logger = logging.getLogger(__name__)

class StockPriceTool(BaseTool):
    name: str = "get_stock_price"
    description: str = "Get stock price data for a given symbol and date range"
    
    def _run(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        try:
            collector = StockDataCollector()
            return collector.fetch_stock_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Error getting stock price: {str(e)}")
            raise

class TechnicalIndicatorTool(BaseTool):
    name: str = "calculate_technical_indicator"
    description: str = "Calculate technical indicators for a given stock. Supported indicators: RSI, MACD, SMA, Bollinger Bands"
    
    def _run(self, symbol: str, indicator: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        try:
            # First get the stock data
            collector = StockDataCollector()
            stock_data = collector.fetch_stock_data(symbol, start_date, end_date)
            
            # Then use TechAna to analyze
            tech_analyzer = TechAna()
            result = tech_analyzer.analyze(stock_data, indicator)

            logger.info(f"Technical analysis completed for {symbol} - {indicator}")
            
            # Format the result for display
            if isinstance(result, str):
                # If it's an error message, return it directly
                return result
            elif isinstance(result, pd.DataFrame):
                # Format DataFrame for display
                if indicator.lower() == 'bollinger':
                    return f"Bollinger Bands Analysis for {symbol}:\n\n" + \
                           f"Upper Band (BBU):\n{result['BBU_20_2.0'].tail()}\n\n" + \
                           f"Middle Band (BBM):\n{result['BBM_20_2.0'].tail()}\n\n" + \
                           f"Lower Band (BBL):\n{result['BBL_20_2.0'].tail()}"
                elif indicator.lower() == 'rsi':
                    return f"RSI Analysis for {symbol}:\n\n" + \
                           f"RSI Values:\n{result['RSI_14'].tail()}"
                elif indicator.lower() == 'macd':
                    return f"MACD Analysis for {symbol}:\n\n" + \
                           f"MACD Line:\n{result['MACD_12_26_9'].tail()}\n\n" + \
                           f"Signal Line:\n{result['MACDs_12_26_9'].tail()}\n\n" + \
                           f"Histogram:\n{result['MACDh_12_26_9'].tail()}"
                elif indicator.lower() == 'sma':
                    return f"SMA Analysis for {symbol}:\n\n" + \
                           f"SMA Values:\n{result['SMA_20'].tail()}"
                else:
                    return f"Analysis for {symbol} - {indicator}:\n\n{result.tail()}"
            else:
                return f"Unexpected result type: {type(result)}"
        except Exception as e:
            logger.error(f"Error calculating technical indicator: {str(e)}")
            raise
