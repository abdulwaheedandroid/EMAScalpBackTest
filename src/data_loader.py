from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BINANCE_KLINES_URLS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
    "https://api4.binance.com/api/v3/klines",
]

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


def date_to_milliseconds(date_string: str) -> int:
    """Convert a UTC date string into Unix milliseconds."""
    timestamp = pd.Timestamp(date_string)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return int(timestamp.timestamp() * 1000)


def get_interval_milliseconds(interval: str) -> int:
    """Return a Binance candle interval in milliseconds."""
    intervals = {
        "1m": 60_000,
        "3m": 3 * 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }

    if interval not in intervals:
        raise ValueError(f"Unsupported interval: {interval}")

    return intervals[interval]


def get_pandas_frequency(interval: str) -> str:
    """Map Binance-style intervals to pandas frequency strings."""
    frequencies = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }

    if interval not in frequencies:
        raise ValueError(f"Unsupported interval: {interval}")

    return frequencies[interval]


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
        try:
            response = requests.get(
                url,
                params=params,
                timeout=30,
            )

            response.raise_for_status()

            data = response.json()

            if not isinstance(data, list):
                raise RuntimeError(
                    f"Unexpected Binance response: {data}"
                )

            return data

        except requests.RequestException as error:
            last_error = error
            print(f"Endpoint failed: {url}")
            print(f"Reason: {error}")

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
    end_time = date_to_milliseconds(end_date)

    if start_time >= end_time:
        return pd.DataFrame()

    interval_ms = get_interval_milliseconds(interval)

    all_klines: list = []
    current_start = start_time

    while current_start < end_time:
        print(
            "Downloading from:",
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
            end_time=end_time,
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

    return clean_klines(all_klines)


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

    interval_ms = get_interval_milliseconds(interval)

    if end_date is None:
        current_time = pd.Timestamp.now(tz="UTC")

        # Exclude the candle that may still be forming.
        frequency = get_pandas_frequency(interval)
        requested_end = current_time.floor(frequency)
    else:
        requested_end = pd.Timestamp(end_date)

        if requested_end.tzinfo is None:
            requested_end = requested_end.tz_localize("UTC")
        else:
            requested_end = requested_end.tz_convert("UTC")

    if filepath.exists():
        print(f"Loading cached data: {filepath}")

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

        if existing_data.empty:
            next_start = pd.Timestamp(start_date)

            if next_start.tzinfo is None:
                next_start = next_start.tz_localize("UTC")
            else:
                next_start = next_start.tz_convert("UTC")
        else:
            last_open_time = existing_data[
                "open_time"
            ].max()

            next_start = last_open_time + pd.Timedelta(
                milliseconds=interval_ms
            )

        if next_start >= requested_end:
            print("Cached data is already up to date.")
            return existing_data

        print(
            f"Updating data from {next_start} "
            f"to {requested_end}"
        )

        new_data = download_historical_data(
            symbol=symbol,
            interval=interval,
            start_date=next_start.isoformat(),
            end_date=requested_end.isoformat(),
        )

        if new_data.empty:
            print("No new completed candles were available.")
            return existing_data

        combined_data = pd.concat(
            [existing_data, new_data],
            ignore_index=True,
        )

        combined_data = (
            combined_data
            .drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        combined_data.to_csv(filepath, index=False)

        print(
            f"Added {len(new_data):,} candles. "
            f"Total candles: {len(combined_data):,}"
        )

        return combined_data

    print("No cached file found. Starting full download.")

    dataframe = download_historical_data(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=requested_end.isoformat(),
    )

    if dataframe.empty:
        raise RuntimeError(
            "No candles were downloaded. "
            "Check the symbol, date range, and connection."
        )

    dataframe.to_csv(filepath, index=False)

    print(
        f"Saved {len(dataframe):,} candles to: {filepath}"
    )

    return dataframe