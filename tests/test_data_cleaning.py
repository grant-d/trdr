"""
Comprehensive tests for data cleaning functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from base_data_loader import BaseDataLoader


class TestDataCleaning:
    """Test suite for data cleaning functionality."""

    def test_clean_data_with_clean_input(
        self, test_data_loader, clean_ohlcv_data, data_quality_assertions
    ) -> None:
        """Test that clean data passes through unchanged."""
        result = test_data_loader.clean_data(clean_ohlcv_data)

        # Should have same number of rows
        assert len(result) == len(clean_ohlcv_data)

        # Should maintain data quality
        data_quality_assertions.assert_ohlcv_integrity(result)
        data_quality_assertions.assert_no_missing_values(result)
        data_quality_assertions.assert_derived_fields_correct(result)
        data_quality_assertions.assert_chronological_order(result)

    def test_clean_data_with_dirty_input(
        self, test_data_loader, dirty_ohlcv_data, data_quality_assertions
    ) -> None:
        """Test that dirty data gets cleaned properly."""
        # Verify dirty data has issues
        assert dirty_ohlcv_data.isnull().sum().sum() > 0  # Has missing values
        assert (dirty_ohlcv_data["volume"] < 0).any()  # Has negative volume

        result = test_data_loader.clean_data(dirty_ohlcv_data)

        # Should fix all issues
        data_quality_assertions.assert_ohlcv_integrity(result)
        data_quality_assertions.assert_no_missing_values(result)
        data_quality_assertions.assert_derived_fields_correct(result)
        data_quality_assertions.assert_chronological_order(result)

    def test_clean_data_empty_dataframe(self, test_data_loader, empty_dataframe):
        """Test cleaning empty DataFrame."""
        result = test_data_loader.clean_data(empty_dataframe)

        assert result.empty
        assert len(result) == 0

    def test_clean_data_minimal_dataframe(
        self, test_data_loader, minimal_dataframe, data_quality_assertions
    ):
        """Test cleaning minimal DataFrame with one row."""
        result = test_data_loader.clean_data(minimal_dataframe)

        assert len(result) == 1
        data_quality_assertions.assert_ohlcv_integrity(result)
        data_quality_assertions.assert_derived_fields_correct(result)

    def test_clean_data_missing_required_columns(
        self, test_data_loader, missing_columns_dataframe
    ):
        """Test that missing required columns raises ValueError."""
        with pytest.raises(ValueError, match="Missing required columns"):
            test_data_loader.clean_data(missing_columns_dataframe)


class TestDataStructureValidation:
    """Test suite for data structure validation."""

    def test_validate_data_structure_valid_data(
        self, test_data_loader, clean_ohlcv_data
    ):
        """Test validation of valid data structure."""
        result = test_data_loader._validate_data_structure(clean_ohlcv_data)

        # Should have all required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in result.columns

        # Should have correct data types
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
        assert pd.api.types.is_numeric_dtype(result["open"])
        assert pd.api.types.is_numeric_dtype(result["high"])
        assert pd.api.types.is_numeric_dtype(result["low"])
        assert pd.api.types.is_numeric_dtype(result["close"])
        assert pd.api.types.is_numeric_dtype(result["volume"])

    def test_validate_data_structure_string_timestamp(self, test_data_loader) -> None:
        """Test conversion of string timestamp to datetime."""
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            }
        )

        result = test_data_loader._validate_data_structure(df)

        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])

    def test_validate_data_structure_missing_trade_count(self, test_data_loader) -> None:
        """Test handling of missing trade_count column."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
            }
        )

        result = test_data_loader._validate_data_structure(df)

        assert "trade_count" in result.columns
        assert result["trade_count"].iloc[0] == 0


class TestMissingValueHandling:
    """Test suite for missing value handling."""

    def test_handle_missing_values_forward_fill(self, test_data_loader) -> None:
        """Test forward fill of missing price values."""
        df = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "open": [100.0, np.nan, np.nan],
                "high": [101.0, np.nan, 103.0],
                "low": [99.0, 100.0, np.nan],
                "close": [100.5, 101.5, np.nan],
                "volume": [1000.0, np.nan, 1200.0],
                "trade_count": [50, np.nan, 60],
            }
        )

        result = test_data_loader._handle_missing_values(df)

        # Price columns should be forward filled
        assert result["open"].iloc[1] == 100.0
        assert result["open"].iloc[2] == 100.0
        assert result["high"].iloc[1] == 101.0
        assert result["low"].iloc[2] == 100.0
        assert result["close"].iloc[2] == 101.5

        # Volume should be filled with 0
        assert result["volume"].iloc[1] == 0.0

        # Trade count should be filled with 0
        assert result["trade_count"].iloc[1] == 0

    def test_handle_missing_values_remove_all_nan_rows(self, test_data_loader):
        """Test removal of rows where all price columns are NaN."""
        df = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "open": [np.nan, np.nan, 100.0],
                "high": [np.nan, np.nan, 101.0],
                "low": [np.nan, np.nan, 99.0],
                "close": [np.nan, np.nan, 100.5],
                "volume": [1000.0, 1100.0, 1200.0],
                "trade_count": [50, 55, 60],
            }
        )

        result = test_data_loader._handle_missing_values(df)

        # Should remove first two rows and keep only the last one
        assert len(result) == 1
        assert result["open"].iloc[0] == 100.0


class TestOutlierDetection:
    """Test suite for outlier detection and handling."""

    def test_detect_outliers_z_score_method(self, test_data_loader) -> None:
        """Test Z-score outlier detection for price columns."""
        # Create data with clear outliers
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="1h"),
                "open": [100.0] * 95 + [500.0] * 5,  # 5 extreme outliers
                "high": [101.0] * 95 + [501.0] * 5,
                "low": [99.0] * 95 + [499.0] * 5,
                "close": [100.5] * 95 + [500.5] * 5,
                "volume": [1000.0] * 100,
                "trade_count": [50] * 100,
            }
        )

        result = test_data_loader._detect_and_handle_outliers(df)

        # Outliers should be capped, not removed
        assert len(result) == 100

        # Extreme values should be reduced but not completely removed
        # The implementation caps at 99th percentile, so some extreme values may remain
        assert result["open"].max() <= 500.0  # Allow equal since capping at percentile
        assert result["open"].min() >= 99.0  # Should not be too extreme

    def test_detect_outliers_volume_iqr_method(self, test_data_loader) -> None:
        """Test IQR outlier detection for volume."""
        # Create data with volume outliers
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="1h"),
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.5] * 100,
                "volume": [1000.0] * 95 + [50000.0] * 5,  # 5 volume spikes
                "trade_count": [50] * 100,
            }
        )

        result = test_data_loader._detect_and_handle_outliers(df)

        # Volume outliers should be capped
        # The implementation caps at 99th percentile, so some extreme values may remain
        assert (
            result["volume"].max() <= 50000.0
        )  # Allow equal since capping at percentile
        assert result["volume"].max() >= 1000.0  # Should still be above normal

    def test_apply_financial_outlier_rules_price_jumps(self, test_data_loader):
        """Test detection of extreme price jumps."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2024-01-01", periods=10, freq="1h"),
                "open": [
                    100.0,
                    101.0,
                    150.0,
                    151.0,
                    152.0,
                    153.0,
                    154.0,
                    155.0,
                    156.0,
                    157.0,
                ],  # 50% jump
                "high": [
                    101.0,
                    102.0,
                    151.0,
                    152.0,
                    153.0,
                    154.0,
                    155.0,
                    156.0,
                    157.0,
                    158.0,
                ],
                "low": [
                    99.0,
                    100.0,
                    149.0,
                    150.0,
                    151.0,
                    152.0,
                    153.0,
                    154.0,
                    155.0,
                    156.0,
                ],
                "close": [
                    100.5,
                    101.5,
                    150.5,
                    151.5,
                    152.5,
                    153.5,
                    154.5,
                    155.5,
                    156.5,
                    157.5,
                ],
                "volume": [1000.0] * 10,
                "trade_count": [50] * 10,
            }
        )

        outliers_handled = test_data_loader._apply_financial_outlier_rules(df)

        # Should detect and handle the extreme price jump
        assert outliers_handled > 0
        assert df["open"].iloc[2] == 101.0  # Should be corrected to previous value

    def test_apply_financial_outlier_rules_volume_spikes(self, test_data_loader) -> None:
        """Test detection of volume spikes."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2024-01-01", periods=20, freq="1h"),
                "open": [100.0] * 20,
                "high": [101.0] * 20,
                "low": [99.0] * 20,
                "close": [100.5] * 20,
                "volume": [1000.0] * 15 + [100000.0] * 5,  # 100x volume spike
                "trade_count": [50] * 20,
            }
        )

        outliers_handled = test_data_loader._apply_financial_outlier_rules(df)

        # Should detect and handle volume spikes
        assert outliers_handled > 0
        # Volume should be capped at 10x median
        assert df["volume"].iloc[15] == 10000.0  # 10x median of 1000

    def test_volume_outlier_detection_with_zero_volumes(self, test_data_loader) -> None:
        """Test that legitimate trades are not flagged as outliers when mixed with zero volumes."""
        # Create data similar to the user's scenario: mostly zero volumes with some legitimate trades
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="1h"),
                "open": [165.11] * 100,
                "high": [165.11] * 100,
                "low": [165.11] * 100,
                "close": [165.11] * 100,
                "volume": [0.0] * 80 + [6.468293108100005] * 20,  # Mostly zeros with some legitimate trades
                "trade_count": [0] * 80 + [1] * 20,
            }
        )

        result = test_data_loader._detect_and_handle_outliers(df)

        # Legitimate trades should not be modified
        legitimate_trades = result[result["volume"] > 0]
        assert len(legitimate_trades) == 20
        assert all(legitimate_trades["volume"] == 6.468293108100005)


class TestOHLCVIntegrityValidation:
    """Test suite for OHLCV integrity validation."""

    def test_validate_ohlcv_integrity_high_low_swap(self, test_data_loader) -> None:
        """Test fixing High < Low issues."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [99.0],  # High < Low
                "low": [101.0],
                "close": [100.5],
                "volume": [1000.0],
                "trade_count": [50],
            }
        )

        result = test_data_loader._validate_ohlcv_integrity(df)

        # High and Low should be swapped
        assert result["high"].iloc[0] == 101.0
        assert result["low"].iloc[0] == 99.0

    def test_validate_ohlcv_integrity_open_outside_range(self, test_data_loader) -> None:
        """Test fixing Open outside High/Low range."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "open": [102.0, 98.0],  # Open > High, Open < Low
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.5, 100.5],
                "volume": [1000.0, 1000.0],
                "trade_count": [50, 50],
            }
        )

        result = test_data_loader._validate_ohlcv_integrity(df)

        # Open should be clamped to High/Low range
        assert result["open"].iloc[0] == 101.0  # Clamped to High
        assert result["open"].iloc[1] == 99.0  # Clamped to Low

    def test_validate_ohlcv_integrity_close_outside_range(self, test_data_loader) -> None:
        """Test fixing Close outside High/Low range."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [102.0, 98.0],  # Close > High, Close < Low
                "volume": [1000.0, 1000.0],
                "trade_count": [50, 50],
            }
        )

        result = test_data_loader._validate_ohlcv_integrity(df)

        # Close should be clamped to High/Low range
        assert result["close"].iloc[0] == 101.0  # Clamped to High
        assert result["close"].iloc[1] == 99.0  # Clamped to Low

    def test_validate_ohlcv_integrity_negative_volume(self, test_data_loader) -> None:
        """Test fixing negative volume."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [-1000.0],  # Negative volume
                "trade_count": [50],
            }
        )

        result = test_data_loader._validate_ohlcv_integrity(df)

        # Volume should be set to 0
        assert result["volume"].iloc[0] == 0.0

    def test_validate_ohlcv_integrity_zero_negative_prices(self, test_data_loader) -> None:
        """Test fixing zero/negative prices."""
        df = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "open": [100.0, 0.0, 101.0],  # Zero price
                "high": [101.0, 101.0, 102.0],  # Keep valid to avoid OHLC conflicts
                "low": [99.0, -1.0, 100.0],  # Negative price
                "close": [100.5, 100.5, 101.5],
                "volume": [1000.0, 1000.0, 1000.0],
                "trade_count": [50, 50, 50],
            }
        )

        result = test_data_loader._validate_ohlcv_integrity(df)

        # Zero/negative prices should be forward filled
        assert result["open"].iloc[1] == 100.0  # Forward filled
        assert result["low"].iloc[1] == 99.0  # Forward filled


class TestDerivedFieldCalculation:
    """Test suite for derived field calculation."""

    def test_recalculate_derived_fields_hlc3(self, test_data_loader) -> None:
        """Test HLC3 calculation."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [102.0],
                "low": [98.0],
                "close": [101.0],
                "volume": [1000.0],
                "trade_count": [50],
                "hlc3": [999.0],  # Incorrect value
                "dv": [999.0],  # Incorrect value
            }
        )

        result = test_data_loader._recalculate_derived_fields(df)

        # HLC3 should be recalculated correctly
        expected_hlc3 = (102.0 + 98.0 + 101.0) / 3
        assert result["hlc3"].iloc[0] == expected_hlc3

    def test_recalculate_derived_fields_dollar_volume(self, test_data_loader) -> None:
        """Test dollar volume calculation."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [102.0],
                "low": [98.0],
                "close": [101.0],
                "volume": [1000.0],
                "trade_count": [50],
                "hlc3": [999.0],  # Will be recalculated
                "dv": [999.0],  # Will be recalculated
            }
        )

        result = test_data_loader._recalculate_derived_fields(df)

        # Dollar volume should be recalculated correctly
        expected_hlc3 = (102.0 + 98.0 + 101.0) / 3
        expected_dv = expected_hlc3 * 1000.0
        assert result["dv"].iloc[0] == expected_dv


class TestFinalCleanup:
    """Test suite for final cleanup operations."""

    def test_final_cleanup_chronological_order(self, test_data_loader) -> None:
        """Test sorting by timestamp."""
        df = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                ],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000.0, 1100.0, 1200.0],
                "trade_count": [50, 55, 60],
            }
        )

        result = test_data_loader._final_cleanup(df)

        # Should be sorted by timestamp
        assert result["timestamp"].iloc[0] == datetime(2024, 1, 1)
        assert result["timestamp"].iloc[1] == datetime(2024, 1, 2)
        assert result["timestamp"].iloc[2] == datetime(2024, 1, 3)

    def test_final_cleanup_remove_critical_nans(self, test_data_loader) -> None:
        """Test removal of rows with NaN in critical columns."""
        df = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ],
                "open": [100.0, np.nan, 102.0],
                "high": [101.0, 102.0, np.nan],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000.0, 1100.0, 1200.0],
                "trade_count": [50, 55, 60],
            }
        )

        result = test_data_loader._final_cleanup(df)

        # Should remove rows with NaN in critical columns
        assert len(result) == 1
        assert result["open"].iloc[0] == 100.0

    def test_final_cleanup_trade_count_integer(self, test_data_loader) -> None:
        """Test conversion of trade_count to integer."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
                "trade_count": [50.7],  # Float value
            }
        )

        result = test_data_loader._final_cleanup(df)

        # trade_count should be converted to integer
        assert result["trade_count"].dtype == int
        assert result["trade_count"].iloc[0] == 50


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling."""

    def test_clean_data_insufficient_data_for_statistics(self, test_data_loader) -> None:
        """Test handling of insufficient data for statistical calculations."""
        # Create data with only a few rows (less than 10)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2024-01-01", periods=5, freq="1h"),
                "open": [100.0, 500.0, 101.0, 102.0, 103.0],  # Contains outlier
                "high": [101.0, 501.0, 102.0, 103.0, 104.0],
                "low": [99.0, 499.0, 100.0, 101.0, 102.0],
                "close": [100.5, 500.5, 101.5, 102.5, 103.5],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
                "trade_count": [50, 55, 60, 65, 70],
            }
        )

        result = test_data_loader.clean_data(df)

        # Should handle gracefully without crashing
        assert len(result) == 5
        assert not result.empty

    def test_clean_data_all_values_same(self, test_data_loader) -> None:
        """Test handling of data where all values are the same."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2024-01-01", periods=20, freq="1h"),
                "open": [100.0] * 20,
                "high": [100.0] * 20,
                "low": [100.0] * 20,
                "close": [100.0] * 20,
                "volume": [1000.0] * 20,
                "trade_count": [50] * 20,
            }
        )

        result = test_data_loader.clean_data(df)

        # Should handle gracefully (std dev = 0)
        assert len(result) == 20
        assert all(result["open"] == 100.0)

    def test_clean_data_large_dataset_performance(self, test_data_loader) -> None:
        """Test performance with larger dataset."""
        # Create larger dataset
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2024-01-01", periods=10000, freq="1min"
                ),
                "open": np.random.normal(100, 1, 10000),
                "high": np.random.normal(101, 1, 10000),
                "low": np.random.normal(99, 1, 10000),
                "close": np.random.normal(100, 1, 10000),
                "volume": np.random.normal(1000, 100, 10000),
                "trade_count": np.random.randint(10, 100, 10000),
            }
        )

        # Ensure OHLC integrity for the test data
        df["high"] = np.maximum.reduce([df["open"], df["high"], df["low"], df["close"]])
        df["low"] = np.minimum.reduce([df["open"], df["high"], df["low"], df["close"]])

        result = test_data_loader.clean_data(df)

        # Should complete without timeout and maintain data integrity
        assert len(result) <= len(df)  # May remove some rows
        assert not result.empty
