import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

TOOL_DIR = os.path.join(os.path.dirname(__file__), "..", "agents", "tools")
sys.path.insert(0, TOOL_DIR)

from price_fetcher import PriceFetcher  # noqa: E402


class DummyTicker:
    def __init__(self, price: float) -> None:
        self.data = pd.DataFrame({"Close": [price]})

    def history(self, period: str = "1d"):
        return self.data


class TestPriceFetcher(unittest.TestCase):
    @patch("price_fetcher.yf.Ticker")
    def test_fetch_price(self, mock_ticker):
        mock_ticker.side_effect = [DummyTicker(30000.0), DummyTicker(2000.0)]
        tool = PriceFetcher()
        result = tool.invoke("")
        self.assertIn("BTC", result)
        self.assertIn("Gold", result)


if __name__ == "__main__":
    unittest.main()
