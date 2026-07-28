from __future__ import annotations

import pandas as pd

REQUIRED_OHLCV_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def validate_ohlcv_columns(dataframe: pd.DataFrame) -> None:
    """Ensure the dataframe contains the required OHLCV columns."""
    missing_columns = sorted(REQUIRED_OHLCV_COLUMNS.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")


def add_ema_indicators(
    dataframe: pd.DataFrame,
    fast_length: int = 9,
    slow_length: int = 15,
    trend_length: int = 200,
    atr_length: int = 14,
) -> pd.DataFrame:
    """Return a copy of the dataframe with EMA and ATR columns added."""
    validate_ohlcv_columns(dataframe)

    enriched = dataframe.copy()

    # Keep the original column names for compatibility
    enriched["ema_9"] = (
        enriched["close"]
        .ewm(span=fast_length, adjust=False)
        .mean()
    )

    enriched["ema_15"] = (
        enriched["close"]
        .ewm(span=slow_length, adjust=False)
        .mean()
    )

    enriched["ema_200"] = (
        enriched["close"]
        .ewm(span=trend_length, adjust=False)
        .mean()
    )

    high_low_range = enriched["high"] - enriched["low"]
    high_close_range = (enriched["high"] - enriched["close"].shift()).abs()
    low_close_range = (enriched["low"] - enriched["close"].shift()).abs()

    true_range = pd.concat(
        [
            high_low_range,
            high_close_range,
            low_close_range,
        ],
        axis=1,
    ).max(axis=1)

    enriched["atr_14"] = true_range.rolling(
        window=atr_length,
        min_periods=atr_length,
    ).mean()

    return enriched


def add_indicators(
    dataframe: pd.DataFrame,
    fast_length: int = 9,
    slow_length: int = 15,
    trend_length: int = 200,
    atr_length: int = 14,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper.

    Existing code and tests may still call add_indicators().
    """
    return add_ema_indicators(
        dataframe=dataframe,
        fast_length=fast_length,
        slow_length=slow_length,
        trend_length=trend_length,
        atr_length=atr_length,
    )