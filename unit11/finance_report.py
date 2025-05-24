import vnquant.data as dt
import pandas as pd
from typing import Dict, Optional, Union
from datetime import datetime, timedelta

class FinanceReport:
    """Financial Report Handler for Stock Market Data"""
    
    def __init__(self):
        """
        Initialize the Financial Report handler
        """
        pass
        
    def get_report(self, symbol: str, report_type: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Union[str, pd.DataFrame]:
        """
        Get financial report data
        
        Args:
            symbol (str): Stock symbol to analyze
            report_type (str): Type of report ('finance', 'business', 'cashflow', 'basic')
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            
        Returns:
            Union[str, pd.DataFrame]: Report data or error message
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
            
            # Initialize loader
            loader = dt.FinanceLoader(symbol, start_date, end_date, data_source='VND', minimal=True)
            
            # Get appropriate report
            if report_type.lower() == 'finance':
                data = loader.get_finan_report()
            elif report_type.lower() == 'business':
                data = loader.get_business_report()
            elif report_type.lower() == 'cashflow':
                data = loader.get_cashflow_report()
            elif report_type.lower() == 'basic':
                data = loader.get_basic_index()
            else:
                return f"Unsupported report type: {report_type}. Supported types: finance, business, cashflow, basic"
            
            if data.empty:
                return f"No data found for symbol {symbol}"
            
            return data
        except Exception as e:
            return f"Error in financial report: {str(e)}"
    
    def get_report_string(self, symbol: str, report_type: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        """
        Get formatted string representation of financial report
        
        Args:
            symbol (str): Stock symbol to analyze
            report_type (str): Type of report
            start_date (str, optional): Start date
            end_date (str, optional): End date
            
        Returns:
            str: Formatted report results
        """
        result = self.get_report(symbol, report_type, start_date, end_date)
        if isinstance(result, str):
            return result
        
        # Format the report for display
        report_title = {
            'finance': 'Financial Report',
            'business': 'Business Report',
            'cashflow': 'Cash Flow Report',
            'basic': 'Basic Financial Indicators'
        }.get(report_type.lower(), 'Report')
        
        return f"{report_title} for {symbol}:\n{result.head().to_string()}" 