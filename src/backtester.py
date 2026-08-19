from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _open_position(side: str, signal_time: Any, entry_time: Any,
                   entry_price: float, balance: float,
                   fee_rate: float) -> tuple[dict[str, Any], float]:
    """Open a full-balance notional position and charge its entry fee."""
    units = balance / entry_price
    entry_fee = entry_price * units * fee_rate
    position = {
        "side": side, "signal_time": signal_time, "entry_time": entry_time,
        "entry_price": entry_price, "units": units, "entry_fee": entry_fee,
        "entry_notional": entry_price * units,
    }
    return position, balance - entry_fee


def _close_position(position: dict[str, Any], exit_signal_time: Any,
                    exit_time: Any, exit_price: float, balance: float,
                    fee_rate: float,
                    exit_reason: str) -> tuple[dict[str, Any], float]:
    """Close a position, charging only the exit fee to account balance."""
    units = position["units"]
    exit_fee = exit_price * units * fee_rate
    if position["side"] == "long":
        gross_profit = (exit_price - position["entry_price"]) * units
    else:
        gross_profit = (position["entry_price"] - exit_price) * units

    net_profit = gross_profit - position["entry_fee"] - exit_fee
    # Entry fee was deducted when opening. Add only gross P/L and exit fee.
    new_balance = balance + gross_profit - exit_fee
    trade = {
        "signal_time": position["signal_time"],
        "entry_time": position["entry_time"],
        "exit_signal_time": exit_signal_time,
        "exit_time": exit_time,
        "side": position["side"],
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "units": units,
        "gross_profit": gross_profit,
        "net_profit": net_profit,
        "entry_fee": position["entry_fee"],
        "exit_fee": exit_fee,
        "return_pct": net_profit / max(position["entry_notional"], 1e-9),
        "exit_reason": exit_reason,
    }
    return trade, new_balance


def _unrealized_pnl(position: dict[str, Any], price: float) -> float:
    if position["side"] == "long":
        return (price - position["entry_price"]) * position["units"]
    return (position["entry_price"] - price) * position["units"]


def _prepare_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize input without mutating the caller's frame."""
    working = dataframe.copy()
    missing_prices = {"open", "close"}.difference(working.columns)
    if missing_prices:
        raise ValueError(f"Missing required price columns: {sorted(missing_prices)}")

    if "long_signal" not in working.columns or "short_signal" not in working.columns:
        if "signal" not in working.columns:
            raise ValueError(
                "Either 'signal' or both 'long_signal' and "
                "'short_signal' columns are required before backtesting"
            )
        working["long_signal"] = (working["signal"] == 1).astype(int)
        working["short_signal"] = (working["signal"] == -1).astype(int)

    if "open_time" in working.columns:
        working = working.sort_values("open_time", kind="stable")
    working = working.reset_index(drop=True)

    valid_long = working["long_signal"].isin([0, 1])
    valid_short = working["short_signal"].isin([0, 1])
    if not (valid_long.all() and valid_short.all()):
        raise ValueError("Long and short signal columns must contain only 0 or 1")

    conflicting = (working["long_signal"] == 1) & (working["short_signal"] == 1)
    if conflicting.any():
        positions = working.index[conflicting].tolist()
        raise ValueError(
            "Conflicting long and short signals at row positions: "
            f"{positions}"
        )

    if not working.empty:
        prices = working[["open", "close"]].apply(pd.to_numeric, errors="coerce")
        finite = prices.apply(lambda column: column.map(math.isfinite))
        if not finite.all().all() or (prices <= 0).any().any():
            raise ValueError("Open and close prices must be finite positive numbers")
        working[["open", "close"]] = prices
    return working


def run_backtest(dataframe: pd.DataFrame, initial_balance: float,
                 fee_rate: float) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run a next-candle-open EMA signal backtest."""
    if not math.isfinite(initial_balance) or initial_balance <= 0:
        raise ValueError("initial_balance must be a finite positive number")
    if not math.isfinite(fee_rate) or fee_rate < 0:
        raise ValueError("fee_rate must be finite and non-negative")

    working = _prepare_dataframe(dataframe)
    equity_columns = ["timestamp", "balance", "equity", "drawdown"]
    if working.empty:
        return [], pd.DataFrame(columns=equity_columns)

    trades: list[dict[str, Any]] = []
    balance = float(initial_balance)
    open_position: dict[str, Any] | None = None
    equity_rows: list[dict[str, Any]] = []

    for row_position in range(len(working)):
        row = working.iloc[row_position]
        timestamp = row.get("open_time", row_position)

        # Only the preceding completed candle's signal can execute now.
        if row_position > 0:
            signal_row = working.iloc[row_position - 1]
            signal_time = signal_row.get("open_time", row_position - 1)
            long_signal = int(signal_row["long_signal"]) == 1
            short_signal = int(signal_row["short_signal"]) == 1
            side = "long" if long_signal else "short" if short_signal else None
            execution_price = float(row["open"])

            if open_position is not None and side != open_position["side"] and side:
                trade, balance = _close_position(
                    open_position, signal_time, timestamp, execution_price,
                    balance, fee_rate, "signal_reverse",
                )
                trades.append(trade)
                open_position = None
                open_position, balance = _open_position(
                    side, signal_time, timestamp, execution_price,
                    balance, fee_rate,
                )
            elif open_position is None and side:
                open_position, balance = _open_position(
                    side, signal_time, timestamp, execution_price,
                    balance, fee_rate,
                )

        current_close = float(row["close"])
        if row_position == len(working) - 1 and open_position is not None:
            final_exit_time = row.get("close_time", timestamp)
            trade, balance = _close_position(
                open_position, None, final_exit_time, current_close,
                balance, fee_rate, "end_of_data",
            )
            trades.append(trade)
            open_position = None

        equity = balance
        if open_position is not None:
            equity += _unrealized_pnl(open_position, current_close)
        equity_rows.append(
            {"timestamp": timestamp, "balance": balance, "equity": equity}
        )

    equity_curve = pd.DataFrame(equity_rows)
    running_peak = equity_curve["equity"].cummax().clip(lower=initial_balance)
    equity_curve["drawdown"] = (
        equity_curve["equity"] - running_peak
    ) / running_peak
    return trades, equity_curve[equity_columns]
