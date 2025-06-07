import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

TOOL_DIR = os.path.join(os.path.dirname(__file__), "..", "agents", "tools")
sys.path.insert(0, TOOL_DIR)

from news_fetcher import NewsFetcher  # noqa: E402


class TestNewsFetcher(unittest.TestCase):
    @patch("news_fetcher.feedparser.parse")
    def test_fetch_news(self, mock_parse):
        mock_parse.return_value = SimpleNamespace(
            entries=[SimpleNamespace(title="a")]
        )  # noqa: E501
        tool = NewsFetcher()
        result = tool()
        self.assertIn("a", result)


if __name__ == "__main__":
    unittest.main()
