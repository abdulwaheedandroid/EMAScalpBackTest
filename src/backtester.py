from __future__ import annotations

from typing import Any

import pandas as pd


def run_backtest(
    dataframe: pd.DataFrame,
    initial_balance: float,
    fee_rate: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run a simple EMA crossover backtest with no look-ahead bias."""
    if "long_signal" not in dataframe.columns or "short_signal" not in dataframe.columns:
        raise ValueError("Signal columns are required before backtesting")

    trades: list[dict[str, Any]] = []
    balance = float(initial_balance)
    equity_curve_rows: list[dict[str, Any]] = []
    open_position: dict[str, Any] | None = None

    for index, row in dataframe.iterrows():
        timestamp = row.get("open_time")
        current_close = float(row["close"])
        current_open = float(row["open"])

        if index < len(dataframe) - 1:
            next_open = float(dataframe.iloc[index + 1]["open"])
        else:
            next_open = None

        if open_position is not None:
            unrealized_pnl = open_position["gross_unrealized"]
            if open_position["side"] == "long":
                unrealized_pnl = (current_close - open_position["entry_price"]) * open_position["units"]
            else:
                unrealized_pnl = (open_position["entry_price"] - current_close) * open_position["units"]

            open_position["unrealized_pnl"] = unrealized_pnl
            open_position["current_price"] = current_close

        if index == len(dataframe) - 1:
            if open_position is not None:
                exit_price = current_close
                exit_fee = exit_price * open_position["units"] * fee_rate
                gross_profit = (exit_price - open_position["entry_price"]) * open_position["units"] if open_position["side"] == "long" else (open_position["entry_price"] - exit_price) * open_position["units"]
                net_profit = gross_profit - exit_fee
                balance += net_profit
                trade = {
                    "entry_time": open_position["entry_time"],
                    "exit_time": timestamp,
                    "side": open_position["side"],
                    "entry_price": open_position["entry_price"],
                    "exit_price": exit_price,
                    "gross_profit": gross_profit,
                    "net_profit": net_profit,
                    "entry_fee": open_position["entry_fee"],
                    "exit_fee": exit_fee,
                    "return_pct": net_profit / max(open_position["entry_cost"], 1e-9),
                    "exit_reason": "end_of_data",
                }
                trades.append(trade)
                open_position = None

        if next_open is None:
            equity_curve_rows.append(
                {
                    "timestamp": timestamp,
                    "balance": balance,
                    "equity": balance,
                    "drawdown": 0.0,
                }
            )
            continue

        if open_position is None:
            if int(row["long_signal"]) == 1 and int(row["short_signal"]) != 1:
                entry_price = next_open
                units = balance / entry_price
                entry_fee = entry_price * units * fee_rate
                balance -= entry_fee
                open_position = {
                    "side": "long",
                    "entry_time": timestamp,
                    "entry_price": entry_price,
                    "units": units,
                    "entry_fee": entry_fee,
                    "entry_cost": entry_price * units + entry_fee,
                    "gross_unrealized": 0.0,
                    "unrealized_pnl": 0.0,
                }
            elif int(row["short_signal"]) == 1 and int(row["long_signal"]) != 1:
                entry_price = next_open
                units = balance / entry_price
                entry_fee = entry_price * units * fee_rate
                balance -= entry_fee
                open_position = {
                    "side": "short",
                    "entry_time": timestamp,
                    "entry_price": entry_price,
                    "units": units,
                    "entry_fee": entry_fee,
                    "entry_cost": entry_price * units + entry_fee,
                    "gross_unrealized": 0.0,
                    "unrealized_pnl": 0.0,
                }
        else:
            if open_position["side"] == "long" and int(row["short_signal"]) == 1:
                exit_price = next_open
                exit_fee = exit_price * open_position["units"] * fee_rate
                gross_profit = (exit_price - open_position["entry_price"]) * open_position["units"]
                net_profit = gross_profit - exit_fee - open_position["entry_fee"]
                balance += net_profit
                trades.append(
                    {
                        "entry_time": open_position["entry_time"],
                        "exit_time": timestamp,
                        "side": open_position["side"],
                        "entry_price": open_position["entry_price"],
                        "exit_price": exit_price,
                        "gross_profit": gross_profit,
                        "net_profit": net_profit,
                        "entry_fee": open_position["entry_fee"],
                        "exit_fee": exit_fee,
                        "return_pct": net_profit / max(open_position["entry_cost"], 1e-9),
                        "exit_reason": "signal_reverse",
                    }
                )
                open_position = None
                if int(row["long_signal"]) == 1 and int(row["short_signal"]) != 1:
                    entry_price = next_open
                    units = balance / entry_price
                    entry_fee = entry_price * units * fee_rate
                    balance -= entry_fee
                    open_position = {
                        "side": "long",
                        "entry_time": timestamp,
                        "entry_price": entry_price,
                        "units": units,
                        "entry_fee": entry_fee,
                        "entry_cost": entry_price * units + entry_fee,
                        "gross_unrealized": 0.0,
                        "unrealized_pnl": 0.0,
                    }
            elif open_position["side"] == "short" and int(row["long_signal"]) == 1:
                exit_price = next_open
                exit_fee = exit_price * open_position["units"] * fee_rate
                gross_profit = (open_position["entry_price"] - exit_price) * open_position["units"]
                net_profit = gross_profit - exit_fee - open_position["entry_fee"]
                balance += net_profit
                trades.append(
                    {
                        "entry_time": open_position["entry_time"],
                        "exit_time": timestamp,
                        "side": open_position["side"],
                        "entry_price": open_position["entry_price"],
                        "exit_price": exit_price,
                        "gross_profit": gross_profit,
                        "net_profit": net_profit,
                        "entry_fee": open_position["entry_fee"],
                        "exit_fee": exit_fee,
                        "return_pct": net_profit / max(open_position["entry_cost"], 1e-9),
                        "exit_reason": "signal_reverse",
                    }
                )
                open_position = None
                if int(row["short_signal"]) == 1 and int(row["long_signal"]) != 1:
                    entry_price = next_open
                    units = balance / entry_price
                    entry_fee = entry_price * units * fee_rate
                    balance -= entry_fee
                    open_position = {
                        "side": "short",
                        "entry_time": timestamp,
                        "entry_price": entry_price,
                        "units": units,
                        "entry_fee": entry_fee,
                        "entry_cost": entry_price * units + entry_fee,
                        "gross_unrealized": 0.0,
                        "unrealized_pnl": 0.0,
                    }

        if open_position is None:
            equity = balance
        else:
            if open_position["side"] == "long":
                unrealized_pnl = (current_close - open_position["entry_price"]) * open_position["units"]
            else:
                unrealized_pnl = (open_position["entry_price"] - current_close) * open_position["units"]
            equity = balance + unrealized_pnl

        equity_curve_rows.append(
            {
                "timestamp": timestamp,
                "balance": balance,
                "equity": equity,
                "drawdown": 0.0,
            }
        )

    equity_curve = pd.DataFrame(equity_curve_rows)
    equity_curve["running_peak"] = equity_curve["equity"].cummax()
    equity_curve["drawdown"] = (equity_curve["equity"] - equity_curve["running_peak"]) / equity_curve["running_peak"]
    equity_curve = equity_curve[["timestamp", "balance", "equity", "drawdown"]]
    return trades, equity_curve
