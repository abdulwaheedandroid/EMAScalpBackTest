from __future__ import annotations

from typing import Any

import pandas as pd


def build_report(
    trades: list[dict[str, Any]],
    equity_curve: pd.DataFrame,
    initial_balance: float,
) -> dict[str, Any]:
    """Build a simple report from the backtest trades and equity curve."""
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "final_balance": float(initial_balance),
        }

    net_profits = [float(trade["net_profit"]) for trade in trades]
    gross_winning = sum(value for value in net_profits if value > 0)
    gross_losing = abs(sum(value for value in net_profits if value < 0))
    winning_trades = sum(1 for value in net_profits if value > 0)
    losing_trades = sum(1 for value in net_profits if value < 0)

    profit_factor = gross_winning / gross_losing if gross_losing > 0 else float("inf") if gross_winning > 0 else 0.0
    final_balance = float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else float(initial_balance)
    total_return_pct = (final_balance / initial_balance - 1.0) * 100.0 if initial_balance else 0.0
    max_drawdown = float(equity_curve["drawdown"].min()) if not equity_curve.empty else 0.0
    max_drawdown = abs(max_drawdown) if max_drawdown < 0 else 0.0

    return {
        "total_trades": len(trades),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": winning_trades / len(trades) if trades else 0.0,
        "net_profit": sum(net_profits),
        "total_return_pct": total_return_pct,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "final_balance": final_balance,
    }
