import os
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

TOOL_DIR = os.path.join(os.path.dirname(__file__), "..", "agents", "tools")
sys.path.insert(0, TOOL_DIR)

mock_ta = types.SimpleNamespace(
    sma=lambda x, length=20: pd.Series([10.0] * len(x)),
    rsi=lambda x, length=14: pd.Series([50.0] * len(x)),
    bbands=lambda x, length=20: pd.DataFrame(
        {
            "BBU_20_2.0": [12.0] * len(x),
            "BBL_20_2.0": [8.0] * len(x),
        }
    ),
    macd=lambda x: pd.DataFrame({"MACDh_12_26_9": [0.5] * len(x)}),
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
                result = tool.invoke("")
                self.assertIn("SMA20", result)
                self.assertIn("RSI", result)
                self.assertIn("BB_upper", result)
                self.assertIn("BB_lower", result)
                self.assertIn("MACD_diff", result)


if __name__ == "__main__":
    unittest.main()
