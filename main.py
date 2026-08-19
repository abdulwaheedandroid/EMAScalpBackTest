import logging
from pathlib import Path

import pandas as pd

from config import (
    END_DATE,
    FAST_EMA_LENGTH,
    INITIAL_BALANCE,
    INTERVAL,
    SLOW_EMA_LENGTH,
    START_DATE,
    SYMBOL,
    TRADING_FEE_RATE,
    TREND_EMA_LENGTH,
    USE_EMA200_FILTER,
)
from src.backtester import run_backtest
from src.data_loader import load_or_download_data
from src.indicators import add_ema_indicators
from src.report import build_report
from src.strategy import generate_signals


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataframe = load_or_download_data(
        symbol=SYMBOL,
        interval=INTERVAL,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    dataframe = add_ema_indicators(
        dataframe,
        fast_length=FAST_EMA_LENGTH,
        slow_length=SLOW_EMA_LENGTH,
        trend_length=TREND_EMA_LENGTH,
        atr_length=14,
    )

    dataframe = generate_signals(
        dataframe,
        use_ema200_filter=USE_EMA200_FILTER,
    )

    trades, equity_curve = run_backtest(
        dataframe,
        initial_balance=INITIAL_BALANCE,
        fee_rate=TRADING_FEE_RATE,
    )

    report = build_report(trades, equity_curve, initial_balance=INITIAL_BALANCE)

    reports_directory = Path("reports")
    reports_directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(trades).to_csv(reports_directory / "trades.csv", index=False)
    equity_curve.to_csv(reports_directory / "equity_curve.csv", index=False)

    print("\nDownload completed successfully.")
    print(f"Rows: {len(dataframe):,}")
    print(f"First candle: {dataframe['open_time'].min()}")
    print(f"Last candle: {dataframe['open_time'].max()}")
    print("\nReport:")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
