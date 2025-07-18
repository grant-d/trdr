"""
Unit tests for fractional differentiation functionality.
"""

from typing import Optional
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from base_data_loader import BaseDataLoader


class MockDataLoader(BaseDataLoader):
    """Mock data loader for testing."""

    def setup_clients(self) -> None:
        """No clients needed for mock."""
        pass

    def fetch_bars(
        self, start: datetime, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Return mock data."""
        # Create 500 data points
        dates = pd.date_range(start=start, periods=500, freq="1min")

        # Create trending price data with noise
        trend = np.linspace(100, 110, 500)
        noise = np.random.normal(0, 0.5, 500)
        prices = trend + noise

        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices * 0.999,
                "high": prices * 1.001,
                "low": prices * 0.998,
                "close": prices,
                "volume": np.random.uniform(1000, 5000, 500),
                "trade_count": np.random.randint(10, 100, 500),
                "hlc3": (prices * 1.001 + prices * 0.998 + prices) / 3,
                "dv": ((prices * 1.001 + prices * 0.998 + prices) / 3)
                * np.random.uniform(1000, 5000, 500),
            }
        )


@pytest.fixture
def mock_config():
    """Create mock configuration."""

    class MockConfig:
        symbol = "TEST/USD"
        timeframe = "1m"
        min_bars = 100
        paper_mode = True

    return MockConfig()


@pytest.fixture
def mock_loader(mock_config) -> MockDataLoader:
    """Create mock data loader."""
    return MockDataLoader(mock_config)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1min")
    trend = np.linspace(100, 110, 500)
    noise = np.random.normal(0, 0.5, 500)
    prices = trend + noise

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices * 0.999,
            "high": prices * 1.001,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.random.uniform(1000, 5000, 500),
            "trade_count": np.random.randint(10, 100, 500),
            "hlc3": (prices * 1.001 + prices * 0.998 + prices) / 3,
        }
    )


def test_frac_diff_auto_fit(mock_loader, sample_data) -> None:
    """Test auto-fit fractional differentiation."""
    # Apply fractional differentiation
    result, orders = mock_loader.frac_diff(sample_data, columns=["close"])

    # Check that new columns were created
    assert "close_fd" in result.columns
    assert "close" in orders

    # Check that rows with NaN were dropped (default behavior)
    assert len(result) < len(sample_data)
    # Check that original close values are preserved in the remaining rows
    assert all(col in result.columns for col in ["close", "open", "high", "low"])

    # Check that some values are non-null
    assert result["close_fd"].notna().sum() > 0

    # Check that order is a reasonable value
    assert 0 <= orders["close"] <= 1.0


def test_frac_diff_multiple_columns(mock_loader, sample_data) -> None:
    """Test fractional differentiation on multiple columns."""
    # Apply to multiple columns
    result, orders = mock_loader.frac_diff(
        sample_data, columns=["open", "high", "low", "close"]
    )

    # Check that all columns were processed
    for col in ["open", "high", "low", "close"]:
        assert f"{col}_fd" in result.columns
        assert col in orders


def test_frac_diff_default_columns(mock_loader, sample_data) -> None:
    """Test default column selection."""
    # Apply without specifying columns
    result, orders = mock_loader.frac_diff(sample_data)

    # Should process open, high, low, close by default
    expected_cols = ["open", "high", "low", "close"]
    for col in expected_cols:
        assert f"{col}_fd" in result.columns


def test_frac_diff_insufficient_data(mock_loader) -> None:
    """Test handling of insufficient data."""
    # Create very small dataset
    small_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=50, freq="1min"),
            "close": np.random.uniform(100, 110, 50),
        }
    )

    # Should warn but not fail
    with pytest.warns(UserWarning, match="insufficient data"):
        result, orders = mock_loader.frac_diff(small_data, columns=["close"])

    # Should return original data unchanged
    assert "close_fd" not in result.columns
    assert len(result) == len(small_data)


def test_frac_diff_missing_columns(mock_loader, sample_data) -> None:
    """Test handling of missing columns."""
    # Try to process non-existent columns
    with pytest.warns(UserWarning, match="No specified columns found"):
        result, orders = mock_loader.frac_diff(
            sample_data, columns=["nonexistent1", "nonexistent2"]
        )

    # Should return original data unchanged
    assert result.equals(sample_data)
    assert orders == {}


def test_frac_diff_preserves_dtypes(mock_loader, sample_data) -> None:
    """Test that data types are preserved."""
    # Apply fractional differentiation
    result, orders = mock_loader.frac_diff(sample_data, columns=["close"])

    # Check that original column dtypes are preserved
    for col in sample_data.columns:
        assert result[col].dtype == sample_data[col].dtype


def test_frac_diff_alignment(mock_loader, sample_data) -> None:
    """Test that fractionally differentiated values are properly aligned."""
    # Apply fractional differentiation with drop_na=False to test alignment
    result, orders = mock_loader.frac_diff(
        sample_data, columns=["close"], drop_na=False
    )

    # Check alignment - NaN values should be at the beginning
    if "close_fd" in result.columns:
        fracdiff_series = result["close_fd"]

        # Find first non-NaN value
        first_valid_idx = fracdiff_series.first_valid_index()

        if first_valid_idx is not None and first_valid_idx > 0:
            # All values before first_valid_idx should be NaN
            assert fracdiff_series.iloc[:first_valid_idx].isna().all()

            # Should have some non-NaN values after
            assert fracdiff_series.iloc[first_valid_idx:].notna().any()

    # Test with drop_na=True (default)
    result_dropped, _ = mock_loader.frac_diff(
        sample_data, columns=["close"], drop_na=True
    )

    # Should have no NaN values in the result
    assert not result_dropped["close_fd"].isna().any()


def test_frac_diff_overwrite(mock_loader, sample_data) -> None:
    """Test overwrite functionality."""
    # Store original close values
    original_close = sample_data["close"].copy()

    # Apply fractional differentiation with overwrite
    result, orders = mock_loader.frac_diff(
        sample_data.copy(),  # Use copy to preserve original
        columns=["close"],
        overwrite=True,
    )

    # Check that original column was overwritten
    assert "close" in result.columns
    assert "close_fd" not in result.columns

    # Check that order was recorded
    assert "close" in orders
    assert 0 <= orders["close"] <= 1.0

    # Check that values changed
    # Some values should be different (except possibly NaN values at start)
    valid_indices = result["close"].notna()
    if valid_indices.any():
        assert not result.loc[valid_indices, "close"].equals(
            original_close.loc[valid_indices]
        )


def test_frac_diff_with_dv(mock_loader) -> None:
    """Test fractional differentiation with dollar volume."""
    # Create data with dv column
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1min")
    prices = np.linspace(100, 110, 500) + np.random.normal(0, 0.5, 500)
    volume = np.random.uniform(1000, 5000, 500)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices * 1.001,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": volume,
            "trade_count": np.random.randint(10, 100, 500),
            "hlc3": (prices * 1.002 + prices * 0.998 + prices) / 3,
            "dv": ((prices * 1.002 + prices * 0.998 + prices) / 3) * volume,
        }
    )

    # Apply fractional differentiation without specifying columns
    result, orders = mock_loader.frac_diff(data)

    # Check that all default columns were processed
    for col in ["open", "high", "low", "close"]:
        assert f"{col}_fd" in result.columns
        assert col in orders
