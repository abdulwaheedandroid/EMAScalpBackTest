import unittest

import pandas as pd

from src.backtester import run_backtest
from src.report import build_report


class BacktesterTests(unittest.TestCase):
    def frame(self, opens, closes, signals, index=None):
        count = len(opens)
        dataframe = pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2026-01-01", periods=count, freq="min", tz="UTC"
                ),
                "open": opens,
                "close": closes,
                "signal": signals,
            },
            index=index,
        )
        return dataframe

    def test_next_candle_open_entry_and_timestamps(self) -> None:
        dataframe = self.frame([100, 125, 130], [110, 126, 130], [1, 0, 0])
        trades, _ = run_backtest(dataframe, 1000.0, 0.0)

        self.assertEqual(trades[0]["signal_time"], dataframe["open_time"].iloc[0])
        self.assertEqual(trades[0]["entry_time"], dataframe["open_time"].iloc[1])
        self.assertEqual(trades[0]["entry_price"], 125.0)

    def test_gap_does_not_execute_at_signal_close(self) -> None:
        dataframe = self.frame([100, 150, 150], [90, 150, 150], [1, 0, 0])
        trades, _ = run_backtest(dataframe, 1000.0, 0.0)
        self.assertEqual(trades[0]["entry_price"], 150.0)
        self.assertEqual(trades[0]["net_profit"], 0.0)

    def test_exact_long_profit(self) -> None:
        dataframe = self.frame([100, 100, 120], [100, 110, 120], [1, -1, 0])
        trades, _ = run_backtest(dataframe, 1000.0, 0.0)
        self.assertEqual(trades[0]["side"], "long")
        self.assertAlmostEqual(trades[0]["units"], 10.0)
        self.assertAlmostEqual(trades[0]["gross_profit"], 200.0)
        self.assertAlmostEqual(trades[0]["net_profit"], 200.0)

    def test_exact_short_profit(self) -> None:
        dataframe = self.frame([100, 100, 80], [100, 90, 80], [-1, 1, 0])
        trades, _ = run_backtest(dataframe, 1000.0, 0.0)
        self.assertEqual(trades[0]["side"], "short")
        self.assertAlmostEqual(trades[0]["units"], 10.0)
        self.assertAlmostEqual(trades[0]["gross_profit"], 200.0)

    def test_entry_and_exit_fees_charged_once(self) -> None:
        dataframe = self.frame([100, 100, 100], [100, 100, 100], [1, 0, 0])
        trades, equity = run_backtest(dataframe, 1000.0, 0.01)
        trade = trades[0]
        self.assertAlmostEqual(trade["entry_fee"], 10.0)
        self.assertAlmostEqual(trade["exit_fee"], 10.0)
        self.assertAlmostEqual(trade["net_profit"], -20.0)
        self.assertAlmostEqual(equity.iloc[-1]["balance"], 980.0)
        self.assertAlmostEqual(equity.iloc[-1]["equity"], 980.0)

    def test_atomic_reversal_closes_and_opens_at_same_price(self) -> None:
        dataframe = self.frame(
            [100, 100, 110, 90], [100, 105, 100, 80], [1, -1, 0, 0]
        )
        trades, _ = run_backtest(dataframe, 1000.0, 0.0)
        self.assertEqual([trade["side"] for trade in trades], ["long", "short"])
        self.assertEqual(trades[0]["exit_price"], 110.0)
        self.assertEqual(trades[1]["entry_price"], 110.0)
        self.assertEqual(trades[0]["exit_time"], trades[1]["entry_time"])
        self.assertEqual(trades[0]["exit_reason"], "signal_reverse")
        self.assertEqual(trades[1]["exit_reason"], "end_of_data")

    def test_final_candle_signal_is_ignored(self) -> None:
        dataframe = self.frame([100, 101], [100, 101], [0, 1])
        trades, equity = run_backtest(dataframe, 1000.0, 0.0)
        self.assertEqual(trades, [])
        self.assertEqual(equity.iloc[-1]["equity"], 1000.0)

    def test_open_position_is_liquidated_at_final_close(self) -> None:
        dataframe = self.frame([100, 100, 105], [100, 103, 120], [1, 0, 0])
        dataframe["close_time"] = dataframe["open_time"] + pd.Timedelta(seconds=59)
        trades, _ = run_backtest(dataframe, 1000.0, 0.01)
        trade = trades[0]
        self.assertEqual(trade["exit_reason"], "end_of_data")
        self.assertEqual(trade["exit_price"], 120.0)
        self.assertEqual(trade["exit_time"], dataframe["close_time"].iloc[-1])
        self.assertAlmostEqual(trade["exit_fee"], 12.0)

    def test_empty_input(self) -> None:
        dataframe = pd.DataFrame(columns=["open", "close", "signal"])
        trades, equity = run_backtest(dataframe, 1000.0, 0.0)
        self.assertEqual(trades, [])
        self.assertTrue(equity.empty)
        self.assertEqual(
            list(equity.columns), ["timestamp", "balance", "equity", "drawdown"]
        )

    def test_single_row_input(self) -> None:
        dataframe = self.frame([100], [110], [1])
        trades, equity = run_backtest(dataframe, 1000.0, 0.0)
        self.assertEqual(trades, [])
        self.assertEqual(equity.iloc[0]["equity"], 1000.0)

    def test_non_default_index_and_chronological_sort(self) -> None:
        dataframe = self.frame([100, 110, 120], [100, 110, 120], [1, 0, 0])
        dataframe = dataframe.iloc[::-1]
        dataframe.index = [30, 20, 10]
        trades, _ = run_backtest(dataframe, 1000.0, 0.0)
        self.assertEqual(trades[0]["entry_price"], 110.0)
        self.assertEqual(trades[0]["entry_time"], dataframe["open_time"].min() + pd.Timedelta(minutes=1))

    def test_first_losing_trade_draws_down_from_initial_capital(self) -> None:
        dataframe = self.frame([100, 100, 90], [100, 95, 80], [1, 0, 0])
        trades, equity = run_backtest(dataframe, 1000.0, 0.0)
        self.assertLess(trades[0]["net_profit"], 0)
        self.assertAlmostEqual(equity.iloc[1]["drawdown"], -0.05)
        self.assertAlmostEqual(equity.iloc[-1]["drawdown"], -0.20)
        report = build_report(trades, equity, 1000.0)
        self.assertAlmostEqual(report["max_drawdown"], 0.20)

    def test_final_equity_reconciles_with_trade_net_profit(self) -> None:
        dataframe = self.frame(
            [100, 100, 120, 90], [100, 110, 100, 80], [1, -1, 0, 0]
        )
        trades, equity = run_backtest(dataframe, 1000.0, 0.005)
        expected = 1000.0 + sum(trade["net_profit"] for trade in trades)
        self.assertAlmostEqual(equity.iloc[-1]["equity"], expected)
        self.assertAlmostEqual(equity.iloc[-1]["balance"], expected)

    def test_conflicting_signals_are_rejected(self) -> None:
        dataframe = self.frame([100, 100], [100, 100], [0, 0])
        dataframe["long_signal"] = [1, 0]
        dataframe["short_signal"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            run_backtest(dataframe, 1000.0, 0.0)


if __name__ == "__main__":
    unittest.main()
