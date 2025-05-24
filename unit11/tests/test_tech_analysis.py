import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tech_analysis import TechAna
from data_collector import StockDataCollector

class TestTechAna(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.tech_ana = TechAna()
        self.collector = StockDataCollector()
        
        # Set date range to last 100 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')


        try:
            self.df = self.collector.fetch_stock_data(['VIC'], start_date, end_date)
            print("==================================")
            print(self.df)
            print("==================================")
        except Exception as e:
            print(f"Error fetching real data: {str(e)}")
            raise

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        result = self.tech_ana.analyze(empty_df, 'RSI')
        self.assertEqual(result, "Empty DataFrame provided")

    def test_missing_close_column(self):
        """Test handling of DataFrame without close column"""
        df_no_close = pd.DataFrame({'open': [100, 101, 102]})
        result = self.tech_ana.analyze(df_no_close, 'RSI')
        self.assertEqual(result, "DataFrame must contain 'close' column")

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        result = self.tech_ana.analyze(self.df, 'RSI')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue('RSI_14' in result.columns)

    def test_macd_calculation(self):
        """Test MACD calculation"""
        result = self.tech_ana.analyze(self.df, 'MACD')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']))

    def test_sma_calculation(self):
        """Test SMA calculation"""
        result = self.tech_ana.analyze(self.df, 'SMA')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue('SMA_20' in result.columns)

    def test_bollinger_calculation(self):
        """Test Bollinger Bands calculation"""
        result = self.tech_ana.analyze(self.df, 'Bollinger')
        print("\nBollinger Bands columns:", result.columns.tolist())
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in ['BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0']))

    def test_unsupported_indicator(self):
        """Test handling of unsupported indicator"""
        result = self.tech_ana.analyze(self.df, 'UNKNOWN')
        self.assertTrue("Unsupported indicator" in result)

    def test_custom_parameters(self):
        """Test indicators with custom parameters"""
        params = {'length': 14, 'signal': 9}
        result = self.tech_ana.analyze(self.df, 'RSI', params)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('RSI_14' in result.columns)

if __name__ == '__main__':
    unittest.main()