from vnquant.data import DataLoader
import pandas as pd
from datetime import datetime, timedelta
import logging

class StockDataCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fetch_stock_data(self, symbol, start_date=None, end_date=None):
        """Fetch stock data for a given symbol from CAFE"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        # Convert string dates to datetime objects if they are strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if end_date - start_date < timedelta(days=30):
            # min time delta between start_date and end_date must be at least 30 days
            start_date = end_date - timedelta(days=30)
        
        # Convert to string format for the API call
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        try:
            # Use the DataLoader from vnquant for CAFE data
            loader = DataLoader(
                symbols=symbol,
                start=start_date_str,
                end=end_date_str,
                data_source='CAFE',
                minimal=True,  # Only get essential columns
                table_style='levels'  # Return data in multi-level format
            )
            data = loader.download()

            if data is None or len(data) == 0:
                raise ValueError(f"No data found for symbol {symbol}")

            self.logger.info(f"Successfully fetched data for symbol {symbol} from CAFE")
            return data

        except Exception as e:
            self.logger.error(f"Error fetching data for symbol {symbol}: {str(e)}")
            raise

    