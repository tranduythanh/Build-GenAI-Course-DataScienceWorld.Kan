from vnquant.data import DataLoader
import pandas as pd
from datetime import datetime, timedelta
import logging
import requests
from vnquant.configs import (
    CAFE_SYMBOLS_API,
    HEADERS
)

class StockDataCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fetch_stock_data(self, symbol, start_date=None, end_date=None):
        """Fetch stock data for a given symbol from CAFE"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            # Use the DataLoader from vnquant for CAFE data
            loader = DataLoader(
                symbols=symbol,
                start=start_date,
                end=end_date,
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

    def fetch_all_symbols(self, start_date=None, end_date=None):
        """Fetch data for all available symbols from CAFE"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            # Get all symbols from CAFE
            response = requests.get(CAFE_SYMBOLS_API, headers=HEADERS)
            if response.status_code != 200:
                raise ValueError(f"Failed to fetch symbols from CAFE API. Status code: {response.status_code}")
            
            response_data = response.json()
            if not response_data or 'Data' not in response_data or 'Data' not in response_data['Data']:
                raise ValueError("Invalid response format from CAFE API")
            
            symbols = [item['Symbol'] for item in response_data['Data']['Data']]
            if not symbols:
                raise ValueError("No symbols found in CAFE API response")

            self.logger.info(f"Found {len(symbols)} symbols from CAFE")

            # Use DataLoader to fetch data for all symbols at once
            loader = DataLoader(
                symbols=symbols,
                start=start_date,
                end=end_date,
                data_source='CAFE',
                minimal=True,
                table_style='levels'
            )

            self.logger.info(f"Fetching data for {len(symbols)} symbols from CAFE")
            data = loader.download()

            if data is None or len(data) == 0:
                raise ValueError("No data was successfully fetched for any symbols")

            self.logger.info(f"Successfully fetched data for {len(symbols)} symbols")
            return data

        except Exception as e:
            self.logger.error(f"Error fetching all symbols: {str(e)}")
            raise

    def update_stock_data(self, symbol=None):
        """Update stock data for a specific symbol or all symbols from CAFE"""
        try:
            if symbol:
                self.fetch_stock_data(symbol)
            else:
                self.fetch_all_symbols()
            return True
        except Exception as e:
            self.logger.error(f"Error updating stock data: {str(e)}")
            return False