import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from typing import Union

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tech_analysis import TechAna
from data_collector import StockDataCollector

class TestTechAna(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data"""
        self.tech_ana: TechAna = TechAna()
        self.collector: StockDataCollector = StockDataCollector()
        
        # Set date range to last 100 days
        end_date: str = datetime.now().strftime('%Y-%m-%d')
        start_date: str = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')

        try:
            self.df: pd.DataFrame = self.collector.fetch_stock_data(['VIC'], start_date, end_date)
            print("==================================")
            print(self.df)
            print("==================================")
        except Exception as e:
            print(f"Error fetching real data: {str(e)}")
            raise

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame"""
        empty_df: pd.DataFrame = pd.DataFrame()
        result: Union[str, pd.DataFrame] = self.tech_ana.analyze(empty_df, 'RSI')
        self.assertEqual(result, "Empty DataFrame provided")

    def test_missing_close_column(self) -> None:
        """Test handling of DataFrame without close column"""
        df_no_close: pd.DataFrame = pd.DataFrame({'open': [100, 101, 102]})
        result: Union[str, pd.DataFrame] = self.tech_ana.analyze(df_no_close, 'RSI')
        self.assertEqual(result, "DataFrame must contain 'close' column")

    def test_rsi_calculation(self) -> None:
        """Test RSI calculation"""
        result: Union[str, pd.DataFrame] = self.tech_ana.analyze(self.df, 'RSI')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue('RSI_14' in result.columns)

    def test_macd_calculation(self) -> None:
        """Test MACD calculation"""
        result: Union[str, pd.DataFrame] = self.tech_ana.analyze(self.df, 'MACD')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']))

    def test_sma_calculation(self) -> None:
        """Test SMA calculation"""
        result: Union[str, pd.DataFrame] = self.tech_ana.analyze(self.df, 'SMA')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue('SMA_20' in result.columns)

    def test_bollinger_calculation(self) -> None:
        """Test Bollinger Bands calculation"""
        result: Union[str, pd.DataFrame] = self.tech_ana.analyze(self.df, 'Bollinger')
        print("\nBollinger Bands columns:", result.columns.tolist())
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in ['BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0']))

    def test_unsupported_indicator(self) -> None:
        """Test handling of unsupported indicator"""
        result: Union[str, pd.DataFrame] = self.tech_ana.analyze(self.df, 'UNKNOWN')
        self.assertTrue("Unsupported indicator" in result)

    def test_custom_parameters(self) -> None:
        """Test indicators with custom parameters"""
        params: dict = {'length': 14, 'signal': 9}
        result: Union[str, pd.DataFrame] = self.tech_ana.analyze(self.df, 'RSI', params)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('RSI_14' in result.columns)

if __name__ == '__main__':
    unittest.main()