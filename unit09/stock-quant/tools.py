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
            return collector.get_stock_price(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Error getting stock price: {str(e)}")
            raise

class TechnicalIndicatorTool(BaseTool):
    name: str = "calculate_technical_indicator"
    description: str = "Calculate technical indicators for a given stock"
    
    def _run(self, symbol: str, indicator: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        try:
            collector = StockDataCollector()
            return collector.calculate_technical_indicator(symbol, indicator, start_date, end_date)
        except Exception as e:
            logger.error(f"Error calculating technical indicator: {str(e)}")
            raise

class FinancialReportTool(BaseTool):
    name: str = "get_financial_report"
    description: str = "Get financial reports for a given company"
    
    def _run(self, symbol: str, report_type: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        try:
            report = FinanceReport()
            return report.get_report(symbol, report_type, start_date, end_date)
        except Exception as e:
            logger.error(f"Error getting financial report: {str(e)}")
            raise

class CompanyInfoTool(BaseTool):
    name: str = "get_company_info"
    description: str = "Get company information for a given symbol"
    
    def _run(self, symbol: str) -> Dict[str, Any]:
        try:
            collector = StockDataCollector()
            return collector.get_company_info(symbol)
        except Exception as e:
            logger.error(f"Error getting company info: {str(e)}")
            raise

class MarketIndexTool(BaseTool):
    name: str = "get_market_index"
    description: str = "Get market index data"
    
    def _run(self, index: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        try:
            collector = StockDataCollector()
            return collector.get_market_index(index, start_date, end_date)
        except Exception as e:
            logger.error(f"Error getting market index: {str(e)}")
            raise

class StockListTool(BaseTool):
    name: str = "get_stock_list"
    description: str = "Get list of stocks in a given market"
    
    def _run(self, market: str) -> pd.DataFrame:
        try:
            collector = StockDataCollector()
            return collector.get_stock_list(market)
        except Exception as e:
            logger.error(f"Error getting stock list: {str(e)}")
            raise 