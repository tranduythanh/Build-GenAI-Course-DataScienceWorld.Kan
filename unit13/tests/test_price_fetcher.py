import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

TOOL_DIR = os.path.join(os.path.dirname(__file__), "..", "agents", "tools")
sys.path.insert(0, TOOL_DIR)

from price_fetcher import PriceFetcher  # noqa: E402


class DummyTicker:
    def __init__(self):
        self.data = pd.DataFrame({"Close": [30000.0]})

    def history(self, period: str = "1d"):
        return self.data


class TestPriceFetcher(unittest.TestCase):
    @patch("price_fetcher.yf.Ticker")
    def test_fetch_price(self, mock_ticker):
        mock_ticker.return_value = DummyTicker()
        tool = PriceFetcher()
        result = tool()
        self.assertIn("price", result)


if __name__ == "__main__":
    unittest.main()
