from data_collector import StockDataCollector
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_stock_data(data, symbol):
    """Helper function to print stock data in a formatted way"""
    print(f"\n=== Stock Data for {symbol} from CAFE ===")
    print(f"Data Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Number of trading days: {len(data)}")
    print("\nFirst 5 days of data:")
    print(data.head())
    print("\nLast 5 days of data:")
    print(data.tail())
    print("\nSummary Statistics:")
    print(data.describe())

def main():
    # Initialize the collector
    logger.info("Initializing StockDataCollector...")
    collector = StockDataCollector()
    
    # Set default date range to 1 year
    default_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    default_end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Example 1: Fetch data for a single stock (VIC) with default parameters
        logger.info("Example 1: Fetching VIC data with default parameters")
        vic_data = collector.fetch_stock_data('VIC', default_start_date, default_end_date)
        print_stock_data(vic_data, 'VIC')
        
        # Example 2: Fetch data for a specific date range (last 30 days)
        logger.info("Example 2: Fetching VIC data for specific date range")
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        vic_data_recent = collector.fetch_stock_data('VIC', start_date, end_date)
        print_stock_data(vic_data_recent, 'VIC')
        
        # Example 3: Fetch data for multiple stocks with default date range
        logger.info("Example 3: Fetching data for multiple stocks")
        stocks = ['VIC', 'VNM', 'FPT']
        for stock in stocks:
            try:
                logger.info(f"Fetching data for {stock}...")
                data = collector.fetch_stock_data(stock, default_start_date, default_end_date)
                print_stock_data(data, stock)
            except Exception as e:
                logger.error(f"Error fetching data for {stock}: {str(e)}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        # Clean up resources
        logger.info("Cleaning up resources...")
        collector.close()

if __name__ == "__main__":
    main() 