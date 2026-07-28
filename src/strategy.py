from __future__ import annotations

import pandas as pd


def generate_signals(
    dataframe: pd.DataFrame,
    use_ema200_filter: bool = False,
    use_ema_200_filter: bool | None = None,
) -> pd.DataFrame:
    """
    Generate EMA crossover signals.

    Supports both parameter names for backward compatibility:
    - use_ema200_filter
    - use_ema_200_filter
    """

    if use_ema_200_filter is not None:
        use_ema200_filter = use_ema_200_filter

    required_columns = {"ema_9", "ema_15"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"EMA columns are required before generating signals: "
            f"{sorted(missing_columns)}"
        )

    result = dataframe.copy()

    previous_fast = result["ema_9"].shift(1)
    previous_slow = result["ema_15"].shift(1)

    long_condition = (
        (previous_fast <= previous_slow)
        & (result["ema_9"] > result["ema_15"])
    )

    short_condition = (
        (previous_fast >= previous_slow)
        & (result["ema_9"] < result["ema_15"])
    )

    if use_ema200_filter:
        if "ema_200" not in result.columns:
            raise ValueError(
                "ema_200 column is required when EMA 200 filter is enabled"
            )

        long_condition &= result["close"] > result["ema_200"]
        short_condition &= result["close"] < result["ema_200"]

    # Use integers to remain compatible with the existing tests.
    result["long_signal"] = long_condition.astype(int)
    result["short_signal"] = short_condition.astype(int)

    # Maintain the original combined signal format.
    result["signal"] = 0
    result.loc[result["long_signal"] == 1, "signal"] = 1
    result.loc[result["short_signal"] == 1, "signal"] = -1

    return result