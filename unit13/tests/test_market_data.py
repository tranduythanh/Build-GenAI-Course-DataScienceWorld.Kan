import os
import sys
import unittest
from unittest.mock import patch

TOOL_DIR = os.path.join(os.path.dirname(__file__), "..", "agents", "tools")
sys.path.insert(0, TOOL_DIR)

from market_data import MarketData  # noqa: E402


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return [self._data]

    def raise_for_status(self):
        pass


class TestMarketData(unittest.TestCase):
    @patch("market_data.requests.get")
    def test_market_data(self, mock_get):
        mock_get.return_value = DummyResponse(
            {"market_cap": 1, "total_volume": 2}
        )  # noqa: E501
        tool = MarketData()
        result = tool()
        self.assertIn("Market Cap", result)


if __name__ == "__main__":
    unittest.main()
