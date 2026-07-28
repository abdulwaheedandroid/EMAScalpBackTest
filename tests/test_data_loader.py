import unittest

from src.data_loader import get_pandas_frequency


class DataLoaderTests(unittest.TestCase):
    def test_pandas_interval_conversion_for_one_minute(self) -> None:
        self.assertEqual(get_pandas_frequency("1m"), "1min")

    def test_pandas_interval_conversion_rejects_unknown_interval(self) -> None:
        with self.assertRaises(ValueError):
            get_pandas_frequency("2m")


if __name__ == "__main__":
    unittest.main()
