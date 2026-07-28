import unittest

import pandas as pd

from src.backtester import run_backtest
from src.data_loader import get_pandas_frequency
from src.indicators import add_indicators
from src.strategy import generate_signals


class TestMVP(unittest.TestCase):
    def test_pandas_interval_conversion(self) -> None:
        self.assertEqual(get_pandas_frequency("1m"), "1min")
        self.assertEqual(get_pandas_frequency("1h"), "1h")

    def test_ema_crossover_detection(self) -> None:
        dataframe = pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2024-01-01",
                    periods=30,
                    freq="min",
                ),
                "open": [100 + i for i in range(30)],
                "high": [100 + i for i in range(30)],
                "low": [100 + i for i in range(30)],
                "close": [100 + i for i in range(30)],
                "volume": [1] * 30,
            }
        )

        with_indicators = add_indicators(dataframe.copy())

        result = generate_signals(
            with_indicators,
            use_ema_200_filter=False,
        )

        self.assertIn("signal", result.columns)
        self.assertGreaterEqual(result["signal"].abs().sum(), 1)

    def test_no_look_ahead_entry_behavior(self) -> None:
        dataframe = pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2024-01-01",
                    periods=4,
                    freq="min",
                ),
                "open": [100, 100, 100, 100],
                "high": [100, 100, 100, 100],
                "low": [100, 100, 100, 100],
                "close": [100, 100, 100, 100],
                "volume": [1, 1, 1, 1],
                "signal": [0, 1, 0, 0],
            }
        )

        trades, _ = run_backtest(
            dataframe,
            initial_balance=1000.0,
            fee_rate=0.0,
        )

        self.assertEqual(len(trades), 1)

        self.assertEqual(
            trades[0]["entry_time"],
            dataframe["open_time"].iloc[1],
        )

        self.assertEqual(
            trades[0]["entry_price"],
            100.0,
        )

    def test_profit_calculation(self) -> None:
        dataframe = pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2024-01-01",
                    periods=3,
                    freq="min",
                ),
                "open": [100, 100, 100],
                "high": [100, 110, 110],
                "low": [100, 90, 100],
                "close": [100, 110, 110],
                "volume": [1, 1, 1],
                "signal": [0, 1, -1],
            }
        )

        trades, _ = run_backtest(
            dataframe,
            initial_balance=1000.0,
            fee_rate=0.0,
        )

        self.assertEqual(len(trades), 1)

        self.assertAlmostEqual(
            trades[0]["entry_price"],
            100.0,
        )

        self.assertAlmostEqual(
            trades[0]["exit_price"],
            110.0,
        )

        self.assertAlmostEqual(
            trades[0]["net_profit"],
            100.0,
        )


if __name__ == "__main__":
    unittest.main()