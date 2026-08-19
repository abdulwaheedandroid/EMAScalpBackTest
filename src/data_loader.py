from __future__ import annotations

import math
import logging
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)
_HTTP_SESSION = requests.Session()
MAX_REQUEST_ATTEMPTS = 3
INITIAL_RETRY_DELAY_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 30.0

BINANCE_KLINES_URLS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
    "https://api4.binance.com/api/v3/klines",
]

INTERVALS = {
    "1m": (60_000, "1min"),
    "3m": (3 * 60_000, "3min"),
    "5m": (5 * 60_000, "5min"),
    "15m": (15 * 60_000, "15min"),
    "30m": (30 * 60_000, "30min"),
    "1h": (60 * 60_000, "1h"),
    "4h": (4 * 60 * 60_000, "4h"),
    "1d": (24 * 60 * 60_000, "1D"),
}

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def _get_current_utc_time() -> pd.Timestamp:
    """Return the current UTC time through a patchable boundary."""
    return pd.Timestamp.now(tz="UTC")


def _as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _filter_completed_range(
    dataframe: pd.DataFrame,
    start: pd.Timestamp,
    exclusive_end: pd.Timestamp,
) -> pd.DataFrame:
    """Keep only candles fully contained in an exclusive UTC range."""
    if dataframe.empty:
        return dataframe.copy()

    result = dataframe.copy()
    result["open_time"] = pd.to_datetime(result["open_time"], utc=True)
    result["close_time"] = pd.to_datetime(result["close_time"], utc=True)
    completed = (
        (result["open_time"] >= start)
        & (result["open_time"] < exclusive_end)
        & (result["close_time"] < exclusive_end)
    )
    return result.loc[completed].reset_index(drop=True)


def validate_candle_data(dataframe: pd.DataFrame, interval: str) -> None:
    """Reject malformed, non-chronological, or discontinuous candle data."""
    interval_ms = get_interval_milliseconds(interval)
    required = {
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "number_of_trades",
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required candle columns: {missing}")
    if dataframe.empty:
        return

    open_times = pd.to_datetime(dataframe["open_time"], utc=True, errors="coerce")
    close_times = pd.to_datetime(dataframe["close_time"], utc=True, errors="coerce")
    if open_times.isna().any() or close_times.isna().any():
        raise ValueError("Candle timestamps must be valid UTC timestamps")
    if open_times.duplicated().any():
        raise ValueError("Candle open_time values must be unique")
    if not open_times.is_monotonic_increasing:
        raise ValueError("Candles must be sorted chronologically")

    numeric_columns = [
        "open", "high", "low", "close", "volume", "number_of_trades"
    ]
    numeric = dataframe[numeric_columns].apply(pd.to_numeric, errors="coerce")
    finite = numeric.apply(lambda column: column.map(math.isfinite))
    if not finite.all().all():
        raise ValueError("Candle numeric values must be finite")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (numeric[["volume", "number_of_trades"]] < 0).any().any():
        raise ValueError("Volume and number_of_trades must be non-negative")
    if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Candle high must be at least open, close, and low")
    if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Candle low must be at most open, close, and high")

    interval_delta = pd.Timedelta(milliseconds=interval_ms)
    if (close_times < open_times).any() or (
        close_times >= open_times + interval_delta
    ).any():
        raise ValueError("Candle close_time must fall within its interval")
    gaps = open_times.diff().dropna() != interval_delta
    if gaps.any():
        raise ValueError("Candle data contains missing or irregular intervals")


def _write_cache_atomically(dataframe: pd.DataFrame, filepath: Path) -> None:
    """Replace a CSV cache only after its complete temporary file is written."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=filepath.parent,
            prefix=f".{filepath.name}.",
            suffix=".tmp",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            dataframe.to_csv(temporary_file, index=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, filepath)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def date_to_milliseconds(date_string: str) -> int:
    """Convert a UTC date string into Unix milliseconds."""
    timestamp = _as_utc_timestamp(date_string)
    return int(timestamp.timestamp() * 1000)


def get_interval_milliseconds(interval: str) -> int:
    """Return a Binance candle interval in milliseconds."""
    if interval not in INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")
    return INTERVALS[interval][0]


def get_pandas_frequency(interval: str) -> str:
    """Map Binance-style intervals to pandas frequency strings."""
    if interval not in INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")
    return INTERVALS[interval][1]


def request_klines(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    limit: int = 1000,
) -> list:
    """Download one batch using available Binance endpoints."""
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit,
    }

    last_error: Exception | None = None

    for url in BINANCE_KLINES_URLS:
        for attempt in range(MAX_REQUEST_ATTEMPTS):
            try:
                response = _HTTP_SESSION.get(
                    url,
                    params=params,
                    timeout=30,
                )
            except (requests.ConnectionError, requests.Timeout) as error:
                last_error = error
                logger.warning(
                    "Binance endpoint request failed (%s/%s): %s: %s",
                    attempt + 1,
                    MAX_REQUEST_ATTEMPTS,
                    url,
                    error,
                )
                if attempt < MAX_REQUEST_ATTEMPTS - 1:
                    delay = min(
                        INITIAL_RETRY_DELAY_SECONDS * (2 ** attempt),
                        MAX_RETRY_DELAY_SECONDS,
                    )
                    time.sleep(delay)
                continue

            if response.status_code in {418, 429} or response.status_code >= 500:
                last_error = requests.HTTPError(
                    f"Retryable Binance HTTP status {response.status_code}",
                    response=response,
                )
                logger.warning(
                    "Retryable Binance response (%s/%s): %s returned %s",
                    attempt + 1,
                    MAX_REQUEST_ATTEMPTS,
                    url,
                    response.status_code,
                )
                if attempt < MAX_REQUEST_ATTEMPTS - 1:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else (
                            INITIAL_RETRY_DELAY_SECONDS * (2 ** attempt)
                        )
                    except ValueError:
                        delay = INITIAL_RETRY_DELAY_SECONDS * (2 ** attempt)
                    time.sleep(min(max(delay, 0.0), MAX_RETRY_DELAY_SECONDS))
                continue

            # Other 4xx responses represent a bad request or another
            # permanent failure that changing endpoints will not fix.
            response.raise_for_status()

            try:
                data = response.json()
            except (requests.JSONDecodeError, ValueError) as error:
                last_error = error
                logger.warning("Malformed JSON from Binance endpoint: %s", url)
                break

            if isinstance(data, dict) and "code" in data:
                raise RuntimeError(f"Binance API error: {data}")
            if not isinstance(data, list):
                last_error = RuntimeError(
                    f"Unexpected Binance response from {url}: {data}"
                )
                logger.warning("Unexpected response shape from: %s", url)
                break

            return data

        logger.warning("Moving to the next Binance endpoint after: %s", url)

    raise ConnectionError(
        "Could not connect to any Binance endpoint. "
        "Check your internet, DNS, VPN, firewall, or ISP."
    ) from last_error


def clean_klines(raw_klines: list) -> pd.DataFrame:
    """Convert the Binance response into a clean DataFrame."""
    dataframe = pd.DataFrame(
        raw_klines,
        columns=KLINE_COLUMNS,
    )

    if dataframe.empty:
        return dataframe

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    dataframe[numeric_columns] = dataframe[
        numeric_columns
    ].astype(float)

    dataframe["number_of_trades"] = dataframe[
        "number_of_trades"
    ].astype(int)

    dataframe["open_time"] = pd.to_datetime(
        dataframe["open_time"],
        unit="ms",
        utc=True,
    )

    dataframe["close_time"] = pd.to_datetime(
        dataframe["close_time"],
        unit="ms",
        utc=True,
    )

    dataframe = dataframe[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "number_of_trades",
        ]
    ]

    dataframe = dataframe.drop_duplicates(
        subset=["open_time"]
    )

    dataframe = dataframe.sort_values(
        "open_time"
    ).reset_index(drop=True)

    return dataframe


def download_historical_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download all candles between the supplied UTC times."""
    start_time = date_to_milliseconds(start_date)
    exclusive_end_time = date_to_milliseconds(end_date)

    if start_time >= exclusive_end_time:
        return pd.DataFrame()

    interval_ms = get_interval_milliseconds(interval)

    all_klines: list = []
    current_start = start_time

    while current_start < exclusive_end_time:
        logger.info(
            "Downloading from: %s",
            pd.to_datetime(
                current_start,
                unit="ms",
                utc=True,
            ),
        )

        batch = request_klines(
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            # Binance's endTime is inclusive. Subtract one millisecond so
            # end_date remains an exclusive completed-candle boundary.
            end_time=exclusive_end_time - 1,
        )

        if not batch:
            break

        all_klines.extend(batch)

        last_open_time = int(batch[-1][0])
        next_start = last_open_time + interval_ms

        if next_start <= current_start:
            raise RuntimeError(
                "Downloader did not advance to the next candle."
            )

        current_start = next_start

        time.sleep(0.15)

    cleaned = clean_klines(all_klines)
    completed = _filter_completed_range(
        cleaned,
        _as_utc_timestamp(start_date),
        _as_utc_timestamp(end_date),
    )
    validate_candle_data(completed, interval)
    return completed


def load_or_download_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str | None = None,
    data_directory: str = "data",
) -> pd.DataFrame:
    """
    Load cached data and automatically append missing candles.

    When end_date is None, data is updated to the latest fully
    completed candle in UTC.
    """
    directory = Path(data_directory)
    directory.mkdir(parents=True, exist_ok=True)

    filepath = directory / f"{symbol.upper()}_{interval}.csv"

    get_interval_milliseconds(interval)
    requested_start = _as_utc_timestamp(start_date)

    if end_date is None:
        current_time = _get_current_utc_time()

        # Exclude the candle that may still be forming.
        frequency = get_pandas_frequency(interval)
        requested_end = current_time.floor(frequency)
    else:
        requested_end = _as_utc_timestamp(end_date)

    if filepath.exists():
        logger.info("Loading cached data: %s", filepath)

        existing_data = pd.read_csv(
            filepath,
            parse_dates=["open_time", "close_time"],
        )

        existing_data = (
            existing_data
            .sort_values("open_time")
            .drop_duplicates(subset=["open_time"])
            .reset_index(drop=True)
        )
        validate_candle_data(existing_data, interval)

        if existing_data.empty:
            next_start = requested_start
        else:
            # Re-download the cached tail because it may have been saved
            # while still forming by an older version of the loader.
            next_start = existing_data["open_time"].max()

        if next_start >= requested_end:
            logger.info("Cached data is already up to date.")
            completed_data = _filter_completed_range(
                existing_data, requested_start, requested_end
            )
            # Persist removal of a boundary candle cached by older versions.
            if len(completed_data) != len(existing_data):
                validate_candle_data(completed_data, interval)
                _write_cache_atomically(completed_data, filepath)
            return completed_data

        logger.info(
            "Updating data from %s to %s",
            next_start,
            requested_end,
        )

        new_data = download_historical_data(
            symbol=symbol,
            interval=interval,
            start_date=next_start.isoformat(),
            end_date=requested_end.isoformat(),
        )

        if new_data.empty:
            logger.info("No new completed candles were available.")
            return _filter_completed_range(
                existing_data, requested_start, requested_end
            )

        new_data = _filter_completed_range(
            new_data, next_start, requested_end
        )
        existing_open_times = set(existing_data["open_time"])
        added_count = int(
            (~new_data["open_time"].isin(existing_open_times)).sum()
        )

        combined_data = pd.concat(
            [existing_data, new_data],
            ignore_index=True,
        )

        combined_data = (
            combined_data
            .drop_duplicates(subset=["open_time"], keep="last")
            .sort_values("open_time")
            .reset_index(drop=True)
        )
        validate_candle_data(combined_data, interval)
        _write_cache_atomically(combined_data, filepath)

        logger.info(
            "Added %s candles and refreshed the cached tail. Total candles: %s",
            f"{added_count:,}",
            f"{len(combined_data):,}",
        )

        return _filter_completed_range(
            combined_data, requested_start, requested_end
        )

    logger.info("No cached file found. Starting full download.")

    dataframe = download_historical_data(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=requested_end.isoformat(),
    )

    dataframe = _filter_completed_range(
        dataframe, requested_start, requested_end
    )

    if dataframe.empty:
        raise RuntimeError(
            "No candles were downloaded. "
            "Check the symbol, date range, and connection."
        )

    validate_candle_data(dataframe, interval)
    _write_cache_atomically(dataframe, filepath)

    logger.info(
        "Saved %s candles to: %s",
        f"{len(dataframe):,}",
        filepath,
    )

    return dataframe
