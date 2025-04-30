import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tech_analysis import TechAna

class TestTechAna(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.tech_ana = TechAna()
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'close': np.random.normal(100, 10, 100)
        }, index=dates)

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
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.df))

    def test_macd_calculation(self):
        """Test MACD calculation"""
        result = self.tech_ana.analyze(self.df, 'MACD')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']))

    def test_sma_calculation(self):
        """Test SMA calculation"""
        result = self.tech_ana.analyze(self.df, 'SMA')
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.df))

    def test_bollinger_calculation(self):
        """Test Bollinger Bands calculation"""
        result = self.tech_ana.analyze(self.df, 'Bollinger')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in ['BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0']))

    def test_unsupported_indicator(self):
        """Test handling of unsupported indicator"""
        result = self.tech_ana.analyze(self.df, 'UNKNOWN')
        self.assertTrue("Unsupported indicator" in result)

    def test_custom_parameters(self):
        """Test indicators with custom parameters"""
        params = {'length': 14, 'signal': 9}
        result = self.tech_ana.analyze(self.df, 'RSI', params)
        self.assertIsInstance(result, pd.Series)

    def test_analysis_string(self):
        """Test get_analysis_string method"""
        result = self.tech_ana.get_analysis_string(self.df, 'AAPL', 'RSI')
        self.assertIsInstance(result, str)
        self.assertTrue('Technical Analysis for AAPL - RSI:' in result)

    def test_analysis_string_error(self):
        """Test get_analysis_string with invalid data"""
        empty_df = pd.DataFrame()
        result = self.tech_ana.get_analysis_string(empty_df, 'AAPL', 'RSI')
        self.assertEqual(result, "Empty DataFrame provided")

if __name__ == '__main__':
    unittest.main()