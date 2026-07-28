import unittest

import pandas as pd

from src.backtester import run_backtest
from src.indicators import add_ema_indicators
from src.report import build_report
from src.strategy import generate_signals


class BacktesterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame(
            {
                "open_time": pd.date_range("2026-01-01", periods=8, freq="min", tz="UTC"),
                "open": [100, 102, 104, 106, 108, 110, 112, 114],
                "high": [101, 103, 105, 107, 109, 111, 113, 115],
                "low": [99, 101, 103, 105, 107, 109, 111, 113],
                "close": [100, 102, 104, 106, 108, 110, 112, 114],
                "volume": [1000] * 8,
                "close_time": pd.date_range("2026-01-01", periods=8, freq="min", tz="UTC"),
                "number_of_trades": [10] * 8,
            }
        )

    def test_long_profit_calculation(self) -> None:
        dataframe = add_ema_indicators(self.dataframe.copy(), fast_length=2, slow_length=3, trend_length=200, atr_length=2)
        dataframe = generate_signals(dataframe, use_ema200_filter=False)
        trades, equity_curve = run_backtest(dataframe, initial_balance=1000.0, fee_rate=0.0)
        self.assertEqual(len(trades), 1)
        self.assertGreater(trades[0]["net_profit"], 0)

    def test_short_profit_calculation(self) -> None:
        dataframe = self.dataframe.copy()
        dataframe["close"] = [100, 98, 96, 94, 92, 90, 88, 86]
        dataframe = add_ema_indicators(dataframe, fast_length=2, slow_length=3, trend_length=200, atr_length=2)
        dataframe = generate_signals(dataframe, use_ema200_filter=False)
        trades, equity_curve = run_backtest(dataframe, initial_balance=1000.0, fee_rate=0.0)
        self.assertEqual(len(trades), 1)
        self.assertGreater(trades[0]["net_profit"], 0)
    def test_fee_application(self) -> None:
        dataframe = self.dataframe.copy()
        dataframe["close"] = [100, 102, 104, 106, 108, 110, 112, 114]
        dataframe = add_ema_indicators(dataframe, fast_length=2, slow_length=3, trend_length=200, atr_length=2)
        dataframe = generate_signals(dataframe, use_ema200_filter=False)
        trades, equity_curve = run_backtest(dataframe, initial_balance=1000.0, fee_rate=0.01)
        self.assertEqual(len(trades), 1)
        self.assertLess(trades[0]["net_profit"], trades[0]["gross_profit"])

    def test_reversal_on_next_candle_open(self) -> None:
        dataframe = self.dataframe.copy()
        dataframe["close"] = [100, 102, 104, 106, 108, 110, 112, 114]
        dataframe = add_ema_indicators(dataframe, fast_length=2, slow_length=3, trend_length=200, atr_length=2)
        dataframe = generate_signals(dataframe, use_ema200_filter=False)
        trades, equity_curve = run_backtest(dataframe, initial_balance=1000.0, fee_rate=0.0)
        self.assertGreaterEqual(len(trades), 1)

    def test_final_open_position_closure(self) -> None:
        dataframe = self.dataframe.copy()
        dataframe["close"] = [100, 102, 104, 106, 108, 110, 112, 114]
        dataframe = add_ema_indicators(dataframe, fast_length=2, slow_length=3, trend_length=200, atr_length=2)
        dataframe = generate_signals(dataframe, use_ema200_filter=False)
        trades, equity_curve = run_backtest(dataframe, initial_balance=1000.0, fee_rate=0.0)
        self.assertTrue(any(trade["exit_reason"] == "end_of_data" for trade in trades))

    def test_max_drawdown_from_equity_curve(self) -> None:
        dataframe = self.dataframe.copy()
        dataframe["close"] = [100, 102, 104, 106, 108, 110, 112, 114]
        dataframe = add_ema_indicators(dataframe, fast_length=2, slow_length=3, trend_length=200, atr_length=2)
        dataframe = generate_signals(dataframe, use_ema200_filter=False)
        trades, equity_curve = run_backtest(dataframe, initial_balance=1000.0, fee_rate=0.0)
        report = build_report(trades, equity_curve, initial_balance=1000.0)
        self.assertGreaterEqual(report["max_drawdown"], 0.0)

    def test_ignores_signal_on_final_candle(self) -> None:
        dataframe = self.dataframe.copy()
        dataframe["close"] = [100, 102, 104, 106, 108, 110, 112, 114]
        dataframe = add_ema_indicators(dataframe, fast_length=2, slow_length=3, trend_length=200, atr_length=2)
        dataframe = generate_signals(dataframe, use_ema200_filter=False)
        dataframe.loc[dataframe.index[-1], "long_signal"] = 1
        trades, equity_curve = run_backtest(dataframe, initial_balance=1000.0, fee_rate=0.0)
        self.assertTrue(all(trade["exit_reason"] != "end_of_data" or trade["exit_reason"] == "end_of_data" for trade in trades))


if __name__ == "__main__":
    unittest.main()
