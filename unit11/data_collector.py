from typing import Union, List, Optional
from vnquant.data import DataLoader
import pandas as pd
from datetime import datetime, timedelta
import logging

class StockDataCollector:
    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)

    def fetch_stock_data(self, symbol: Union[str, List[str]], start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch stock data for a given symbol or list of symbols from CAFE"""
        if start_date is None:
            start_date_dt: datetime = datetime.now() - timedelta(days=365)
        else:
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
            
        if end_date is None:
            end_date_dt: datetime = datetime.now()
        else:
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date
        
        if end_date_dt - start_date_dt < timedelta(days=30):
            # min time delta between start_date and end_date must be at least 30 days
            start_date_dt = end_date_dt - timedelta(days=30)
        
        # Convert to string format for the API call
        start_date_str: str = start_date_dt.strftime('%Y-%m-%d')
        end_date_str: str = end_date_dt.strftime('%Y-%m-%d')

        try:
            # Handle both single symbol (string) and multiple symbols (list)
            symbols: Union[str, List[str]]
            if isinstance(symbol, list):
                symbols = symbol
            else:
                symbols = symbol
                
            # Use the DataLoader from vnquant for CAFE data
            loader: DataLoader = DataLoader(
                symbols=symbols,
                start=start_date_str,
                end=end_date_str,
                data_source='CAFE',
                minimal=True,  # Only get essential columns
                table_style='levels'  # Return data in multi-level format
            )
            data: pd.DataFrame = loader.download()

            if data is None or len(data) == 0:
                raise ValueError(f"No data found for symbol {symbol}")

            self.logger.info(f"Successfully fetched data for symbol {symbol} from CAFE")
            return data

        except Exception as e:
            self.logger.error(f"Error fetching data for symbol {symbol}: {str(e)}")
            raise

    def fetch_all_symbols(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch data for all available symbols"""
        try:
            # For demo purposes, use a few common symbols
            symbols: List[str] = ['VIC', 'VNM', 'FPT', 'HAG', 'MSN']
            return self.fetch_stock_data(symbols, start_date, end_date)
        except Exception as e:
            self.logger.error(f"Error fetching all symbols: {str(e)}")
            raise

    def update_stock_data(self, symbol: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """Update stock data for given symbol or all symbols"""
        try:
            if symbol:
                # Update single symbol
                data: pd.DataFrame = self.fetch_stock_data(symbol, start_date, end_date)
                return data is not None and len(data) > 0
            else:
                # Update all symbols
                data = self.fetch_all_symbols(start_date, end_date)
                return data is not None and len(data) > 0
        except Exception as e:
            self.logger.error(f"Error updating stock data: {str(e)}")
            return False

    def close(self) -> None:
        """Close any resources (placeholder method for compatibility)"""
        self.logger.info("Closing StockDataCollector resources")
        # No actual resources to close in this implementation
        pass

    