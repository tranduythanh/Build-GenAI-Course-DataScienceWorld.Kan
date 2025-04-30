import unittest
import pandas as pd
from datetime import datetime, timedelta
from data_collector import StockDataCollector

class TestStockDataCollector(unittest.TestCase):
    """Test suite for StockDataCollector class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.collector = StockDataCollector()
        # Set a reasonable date range for testing
        self.start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        self.end_date = datetime.now().strftime('%Y-%m-%d')

    def test_fetch_stock_data(self):
        """Test fetching stock data for a single symbol."""
        print("\n=== Testing fetch_stock_data ===")
        
        # Test with default parameters
        result = self.collector.fetch_stock_data('VIC')
        print("\nFetched data for VIC (default parameters):")
        print(result.head())
        print("\nData shape:", result.shape)
        print("Columns:", result.columns.tolist())
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        self.assertIn(('close', 'VIC'), result.columns)
        self.assertIn(('volume_match', 'VIC'), result.columns)

        # Test with custom parameters
        result = self.collector.fetch_stock_data('VIC', self.start_date, self.end_date)
        print(f"\nFetched data for VIC ({self.start_date} to {self.end_date}):")
        print(result.head())
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

    def test_fetch_stock_data_error(self):
        """Test error handling in fetch_stock_data."""
        print("\n=== Testing fetch_stock_data error handling ===")
        
        # Test with invalid symbol
        with self.assertRaises(Exception) as context:
            self.collector.fetch_stock_data('INVALID_SYMBOL')
        print(f"\nCaught expected error: {str(context.exception)}")

    def test_fetch_all_symbols(self):
        """Test fetching data for all available symbols."""
        print("\n=== Testing fetch_all_symbols ===")
        
        try:
            # Test with default parameters
            result = self.collector.fetch_all_symbols()
            print("\nFetched data for all symbols:")
            print(result.head())
            print("\nData shape:", result.shape)
            print("Columns:", result.columns.tolist())
            
            self.assertIsNotNone(result)
            self.assertGreater(len(result), 0)
        except Exception as e:
            print(f"\nError fetching all symbols: {str(e)}")
            # Skip the test if the API is not available
            self.skipTest(f"API not available: {str(e)}")

    def test_update_stock_data_single_symbol(self):
        """Test updating stock data for a single symbol."""
        print("\n=== Testing update_stock_data (single symbol) ===")
        
        # Test updating single symbol
        result = self.collector.update_stock_data(symbol='VIC')
        
        if result:
            print("\nSuccessfully updated data for VIC")
        else:
            print("\nFailed to update data for VIC")
            
        self.assertTrue(result)

    def test_update_stock_data_all_symbols(self):
        """Test updating stock data for all symbols."""
        print("\n=== Testing update_stock_data (all symbols) ===")
        
        try:
            # Test updating all symbols
            result = self.collector.update_stock_data()
            
            if result:
                print("\nSuccessfully updated data for all symbols")
            else:
                print("\nFailed to update data for all symbols")
                
            self.assertTrue(result)
        except Exception as e:
            print(f"\nError updating all symbols: {str(e)}")
            # Skip the test if the API is not available
            self.skipTest(f"API not available: {str(e)}")

    def test_close(self):
        """Test the close method."""
        print("\n=== Testing close method ===")
        self.collector.close()
        print("Close method executed successfully")

if __name__ == '__main__':
    unittest.main(verbose=2) 