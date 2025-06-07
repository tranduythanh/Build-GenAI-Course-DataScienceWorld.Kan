import os
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

TOOL_DIR = os.path.join(os.path.dirname(__file__), "..", "agents", "tools")
sys.path.insert(0, TOOL_DIR)

mock_ta = types.SimpleNamespace(
    sma=lambda x, length=20: [10.0] * len(x),
    rsi=lambda x, length=14: [50.0] * len(x),
)


class DummyTicker:
    def __init__(self):
        self.data = pd.DataFrame({"Close": [1.0] * 30})

    def history(self, period: str = "1d"):
        return self.data


class TestTechnicalAnalyzer(unittest.TestCase):
    def test_analyze(self):
        with patch.dict(sys.modules, {"pandas_ta": mock_ta}):
            import technical_analyzer as ta_mod
        with patch.object(ta_mod.yf, "Ticker", return_value=DummyTicker()):
            tool = ta_mod.TechnicalAnalyzer()
            result = tool()
            self.assertIn("SMA20", result)
            self.assertIn("RSI", result)


if __name__ == "__main__":
    unittest.main()
