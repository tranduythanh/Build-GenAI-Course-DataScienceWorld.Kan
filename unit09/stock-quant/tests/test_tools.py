import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
from tools import (
    StockPriceTool,
    TechnicalIndicatorTool,
    FinancialReportTool,
    CompanyInfoTool,
    MarketIndexTool,
    StockListTool
)

class TestStockPriceTool(unittest.TestCase):
    def setUp(self):
        self.tool = StockPriceTool()
        self.symbol = "VIC"
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-31"

    @patch('tools.StockDataCollector')
    def test_get_stock_price(self, mock_collector):
        # Setup mock
        mock_df = pd.DataFrame({
            'date': pd.date_range(start=self.start_date, end=self.end_date),
            'close': [100.0] * 31
        })
        mock_collector.return_value.get_stock_price.return_value = mock_df

        # Test
        result = self.tool._run(self.symbol, self.start_date, self.end_date)
        
        # Verify
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 31)
        mock_collector.return_value.get_stock_price.assert_called_once_with(
            self.symbol, self.start_date, self.end_date
        )

class TestTechnicalIndicatorTool(unittest.TestCase):
    def setUp(self):
        self.tool = TechnicalIndicatorTool()
        self.symbol = "VIC"
        self.indicator = "RSI"
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-31"

    @patch('tools.StockDataCollector')
    def test_calculate_technical_indicator(self, mock_collector):
        # Setup mock
        mock_df = pd.DataFrame({
            'date': pd.date_range(start=self.start_date, end=self.end_date),
            'RSI': [50.0] * 31
        })
        mock_collector.return_value.calculate_technical_indicator.return_value = mock_df

        # Test
        result = self.tool._run(self.symbol, self.indicator, self.start_date, self.end_date)
        
        # Verify
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 31)
        mock_collector.return_value.calculate_technical_indicator.assert_called_once_with(
            self.symbol, self.indicator, self.start_date, self.end_date
        )

class TestFinancialReportTool(unittest.TestCase):
    def setUp(self):
        self.tool = FinancialReportTool()
        self.symbol = "VIC"
        self.report_type = "income_statement"
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-31"

    @patch('tools.FinanceReport')
    def test_get_financial_report(self, mock_report):
        # Setup mock
        mock_df = pd.DataFrame({
            'date': pd.date_range(start=self.start_date, end=self.end_date),
            'revenue': [1000000] * 31
        })
        mock_report.return_value.get_report.return_value = mock_df

        # Test
        result = self.tool._run(self.symbol, self.report_type, self.start_date, self.end_date)
        
        # Verify
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 31)
        mock_report.return_value.get_report.assert_called_once_with(
            self.symbol, self.report_type, self.start_date, self.end_date
        )

class TestCompanyInfoTool(unittest.TestCase):
    def setUp(self):
        self.tool = CompanyInfoTool()
        self.symbol = "VIC"

    @patch('tools.StockDataCollector')
    def test_get_company_info(self, mock_collector):
        # Setup mock
        mock_info = {
            "name": "Vingroup",
            "sector": "Real Estate",
            "market_cap": 1000000000
        }
        mock_collector.return_value.get_company_info.return_value = mock_info

        # Test
        result = self.tool._run(self.symbol)
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], "Vingroup")
        mock_collector.return_value.get_company_info.assert_called_once_with(self.symbol)

class TestMarketIndexTool(unittest.TestCase):
    def setUp(self):
        self.tool = MarketIndexTool()
        self.index = "VNINDEX"
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-31"

    @patch('tools.StockDataCollector')
    def test_get_market_index(self, mock_collector):
        # Setup mock
        mock_df = pd.DataFrame({
            'date': pd.date_range(start=self.start_date, end=self.end_date),
            'close': [1000.0] * 31
        })
        mock_collector.return_value.get_market_index.return_value = mock_df

        # Test
        result = self.tool._run(self.index, self.start_date, self.end_date)
        
        # Verify
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 31)
        mock_collector.return_value.get_market_index.assert_called_once_with(
            self.index, self.start_date, self.end_date
        )

class TestStockListTool(unittest.TestCase):
    def setUp(self):
        self.tool = StockListTool()
        self.market = "HOSE"

    @patch('tools.StockDataCollector')
    def test_get_stock_list(self, mock_collector):
        # Setup mock
        mock_df = pd.DataFrame({
            'symbol': ['VIC', 'VHM', 'VRE'],
            'name': ['Vingroup', 'Vinhomes', 'Vincom Retail']
        })
        mock_collector.return_value.get_stock_list.return_value = mock_df

        # Test
        result = self.tool._run(self.market)
        
        # Verify
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        mock_collector.return_value.get_stock_list.assert_called_once_with(self.market)

if __name__ == '__main__':
    unittest.main() 