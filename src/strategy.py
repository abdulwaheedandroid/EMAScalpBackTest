from __future__ import annotations

import pandas as pd


def generate_signals(
    dataframe: pd.DataFrame,
    use_ema200_filter: bool = False,
) -> pd.DataFrame:
    """Generate simple EMA crossover signals for the MVP strategy."""
    if "ema_9" not in dataframe.columns or "ema_15" not in dataframe.columns:
        raise ValueError("EMA columns are required before generating signals")

    signals = dataframe.copy()
    signals["long_signal"] = 0
    signals["short_signal"] = 0

    long_condition = (signals["ema_9"] > signals["ema_15"]) & (signals["ema_9"].shift(1) <= signals["ema_15"].shift(1))
    short_condition = (signals["ema_9"] < signals["ema_15"]) & (signals["ema_9"].shift(1) >= signals["ema_15"].shift(1))

    if use_ema200_filter and "ema_200" in signals.columns:
        long_condition &= signals["ema_9"] > signals["ema_200"]
        short_condition &= signals["ema_9"] < signals["ema_200"]

    signals.loc[long_condition, "long_signal"] = 1
    signals.loc[short_condition, "short_signal"] = 1

    return signals
