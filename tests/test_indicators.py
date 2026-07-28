import pandas as pd
import unittest

from src.indicators import add_ema_indicators, validate_ohlcv_columns


class IndicatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame(
            {
                "open_time": pd.date_range("2026-01-01", periods=20, freq="min", tz="UTC"),
                "open": [100 + i for i in range(20)],
                "high": [101 + i for i in range(20)],
                "low": [99 + i for i in range(20)],
                "close": [100 + i for i in range(20)],
                "volume": [1000] * 20,
                "close_time": pd.date_range("2026-01-01", periods=20, freq="min", tz="UTC"),
                "number_of_trades": [10] * 20,
            }
        )

    def test_validate_ohlcv_columns(self) -> None:
        validate_ohlcv_columns(self.dataframe)

    def test_add_ema_indicators_returns_copy(self) -> None:
        result = add_ema_indicators(self.dataframe, fast_length=9, slow_length=15, trend_length=20, atr_length=14)
        self.assertIsNot(result, self.dataframe)
        self.assertIn("ema_9", result.columns)
        self.assertIn("ema_15", result.columns)
        self.assertIn("ema_200", result.columns)
        self.assertIn("atr_14", result.columns)

    def test_add_ema_indicators_raises_for_missing_columns(self) -> None:
        bad_frame = self.dataframe.drop(columns=["close"])
        with self.assertRaises(ValueError):
            add_ema_indicators(bad_frame)


if __name__ == "__main__":
    unittest.main()
