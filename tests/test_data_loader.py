import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd
import requests

from src.data_loader import (
    _write_cache_atomically,
    download_historical_data,
    get_pandas_frequency,
    load_or_download_data,
    request_klines,
    validate_candle_data,
)


class DataLoaderTests(unittest.TestCase):
    @staticmethod
    def raw_kline(open_time: str, close: float = 100.0) -> list:
        opened = pd.Timestamp(open_time)
        if opened.tzinfo is None:
            opened = opened.tz_localize("UTC")
        open_ms = int(opened.timestamp() * 1000)
        return [
            open_ms, "100", "101", "99", str(close), "1",
            open_ms + 59_999, "100", 10, "0.5", "50", "0",
        ]

    @staticmethod
    def candle_frame(times: list[str], closes: list[float]) -> pd.DataFrame:
        opened = pd.to_datetime(times, utc=True)
        return pd.DataFrame(
            {
                "open_time": opened,
                "open": [100.0] * len(times),
                "high": [max(101.0, close) for close in closes],
                "low": [min(99.0, close) for close in closes],
                "close": closes,
                "volume": [1.0] * len(times),
                "close_time": opened + pd.Timedelta(seconds=59, milliseconds=999),
                "number_of_trades": [10] * len(times),
            }
        )

    def test_pandas_interval_conversion_for_one_minute(self) -> None:
        self.assertEqual(get_pandas_frequency("1m"), "1min")

    def test_pandas_interval_conversion_rejects_unknown_interval(self) -> None:
        with self.assertRaises(ValueError):
            get_pandas_frequency("2m")

    @patch("src.data_loader.time.sleep")
    @patch("src.data_loader.request_klines")
    def test_download_treats_end_as_exclusive(self, request, _sleep) -> None:
        request.return_value = [
            self.raw_kline("2026-01-01 00:00:00"),
            self.raw_kline("2026-01-01 00:01:00"),
        ]

        result = download_historical_data(
            "BTCUSDT", "1m", "2026-01-01 00:00:00", "2026-01-01 00:01:00"
        )

        self.assertEqual(result["open_time"].tolist(), [pd.Timestamp("2026-01-01 00:00:00", tz="UTC")])
        self.assertEqual(request.call_args.kwargs["end_time"], 1767225659999)

    @patch("src.data_loader.download_historical_data")
    def test_cached_tail_is_refetched_and_replaced(self, download) -> None:
        cached = self.candle_frame(
            ["2026-01-01 00:00:00", "2026-01-01 00:01:00"],
            [100.0, 101.0],
        )
        download.return_value = self.candle_frame(
            ["2026-01-01 00:01:00", "2026-01-01 00:02:00"],
            [111.0, 102.0],
        )

        with TemporaryDirectory() as directory:
            cache = Path(directory) / "BTCUSDT_1m.csv"
            cached.to_csv(cache, index=False)
            result = load_or_download_data(
                "BTCUSDT", "1m", "2026-01-01", "2026-01-01 00:03:00",
                data_directory=directory,
            )

        refreshed = result.loc[result["open_time"] == pd.Timestamp("2026-01-01 00:01:00", tz="UTC")]
        self.assertEqual(len(result), 3)
        self.assertEqual(refreshed.iloc[0]["close"], 111.0)
        self.assertEqual(download.call_args.kwargs["start_date"], "2026-01-01T00:01:00+00:00")

    @patch("src.data_loader._get_current_utc_time")
    @patch("src.data_loader.download_historical_data")
    def test_dynamic_download_excludes_current_boundary_candle(self, download, now) -> None:
        now.return_value = pd.Timestamp("2026-01-01 00:03:40", tz="UTC")
        download.return_value = self.candle_frame(
            ["2026-01-01 00:02:00", "2026-01-01 00:03:00"],
            [102.0, 103.0],
        )

        with TemporaryDirectory() as directory:
            result = load_or_download_data(
                "BTCUSDT", "1m", "2026-01-01 00:02:00",
                data_directory=directory,
            )

        self.assertEqual(result["open_time"].tolist(), [pd.Timestamp("2026-01-01 00:02:00", tz="UTC")])
        self.assertEqual(download.call_args.kwargs["end_date"], "2026-01-01T00:03:00+00:00")

    def test_validation_rejects_invalid_ohlc_relationships(self) -> None:
        dataframe = self.candle_frame(["2026-01-01 00:00:00"], [105.0])
        dataframe.loc[0, "high"] = 101.0
        with self.assertRaisesRegex(ValueError, "Candle high"):
            validate_candle_data(dataframe, "1m")

    def test_validation_rejects_missing_intervals(self) -> None:
        dataframe = self.candle_frame(
            ["2026-01-01 00:00:00", "2026-01-01 00:02:00"],
            [100.0, 100.0],
        )
        with self.assertRaisesRegex(ValueError, "missing or irregular"):
            validate_candle_data(dataframe, "1m")

    def test_atomic_cache_write_leaves_only_complete_destination(self) -> None:
        dataframe = self.candle_frame(["2026-01-01 00:00:00"], [100.0])
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "BTCUSDT_1m.csv"
            _write_cache_atomically(dataframe, destination)
            restored = pd.read_csv(destination)
            remaining_files = list(Path(directory).iterdir())

        self.assertEqual(restored["close"].tolist(), [100.0])
        self.assertEqual([path.name for path in remaining_files], [destination.name])

    @patch("src.data_loader.time.sleep")
    @patch("src.data_loader._HTTP_SESSION.get")
    def test_request_retries_transient_connection_failure(self, get, sleep) -> None:
        success = Mock(status_code=200, headers={})
        success.json.return_value = []
        success.raise_for_status.return_value = None
        get.side_effect = [requests.ConnectionError("offline"), success]

        result = request_klines("BTCUSDT", "1m", 1, 2)

        self.assertEqual(result, [])
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(0.5)

    @patch("src.data_loader.time.sleep")
    @patch("src.data_loader._HTTP_SESSION.get")
    def test_request_respects_retry_after_for_rate_limit(self, get, sleep) -> None:
        limited = Mock(status_code=429, headers={"Retry-After": "2"})
        success = Mock(status_code=200, headers={})
        success.json.return_value = []
        success.raise_for_status.return_value = None
        get.side_effect = [limited, success]

        request_klines("BTCUSDT", "1m", 1, 2)

        sleep.assert_called_once_with(2.0)

    @patch("src.data_loader._HTTP_SESSION.get")
    def test_request_does_not_retry_permanent_client_error(self, get) -> None:
        response = Mock(status_code=400, headers={})
        response.raise_for_status.side_effect = requests.HTTPError("bad request")
        get.return_value = response

        with self.assertRaises(requests.HTTPError):
            request_klines("INVALID", "1m", 1, 2)

        self.assertEqual(get.call_count, 1)

    @patch("src.data_loader._HTTP_SESSION.get")
    def test_request_falls_back_after_malformed_response(self, get) -> None:
        malformed = Mock(status_code=200, headers={})
        malformed.raise_for_status.return_value = None
        malformed.json.return_value = "not a kline list"
        success = Mock(status_code=200, headers={})
        success.raise_for_status.return_value = None
        success.json.return_value = []
        get.side_effect = [malformed, success]

        result = request_klines("BTCUSDT", "1m", 1, 2)

        self.assertEqual(result, [])
        self.assertEqual(get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
