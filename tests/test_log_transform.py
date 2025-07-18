"""
Test log transformation functionality.
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
        dates = pd.date_range(start=start, periods=200, freq="1min")
        prices = np.linspace(100, 110, 200) + np.random.normal(0, 0.5, 200)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices * 0.999,
                "high": prices * 1.001,
                "low": prices * 0.998,
                "close": prices,
                "volume": np.random.uniform(1000, 5000, 200),
                "trade_count": np.random.randint(10, 100, 200),
            }
        )


@pytest.fixture
def mock_config():
    """Create mock configuration."""

    class MockConfig:
        symbol = "TEST"
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
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="1min"),
            "open": np.random.uniform(100, 110, 100),
            "high": np.random.uniform(101, 111, 100),
            "low": np.random.uniform(99, 109, 100),
            "close": np.random.uniform(100, 110, 100),
            "volume": np.random.uniform(1000, 5000, 100),
        }
    )


def test_log_transform_default_columns(mock_loader, sample_data):
    """Test log transformation with default columns (volume only)."""
    result = mock_loader.log_transform_volume(sample_data)

    # Check that volume_log was created
    assert "volume_log" in result.columns

    # Check that dv_log was NOT created (not in defaults)
    assert "dv_log" not in result.columns

    # Check original columns still exist
    assert "volume" in result.columns

    # Verify first row was dropped
    assert len(result) == len(sample_data) - 1

    # Verify log transformation (not returns)
    expected_log = np.log(sample_data["volume"].iloc[1:] + 1e-8)
    np.testing.assert_allclose(
        result["volume_log"].values, expected_log.values, rtol=1e-10
    )


def test_log_transform_overwrite(mock_loader, sample_data):
    """Test log transformation with overwrite."""
    original_volume = sample_data["volume"].copy()

    result = mock_loader.log_transform_volume(sample_data, overwrite=True)

    # Check that volume was overwritten
    assert "volume" in result.columns
    assert "volume_log" not in result.columns

    # Verify first row was dropped
    assert len(result) == len(sample_data) - 1

    # Verify values changed
    assert not np.array_equal(result["volume"], original_volume[1:])

    # Verify log transformation
    expected_log = np.log(original_volume.iloc[1:] + 1e-8)
    np.testing.assert_allclose(result["volume"].values, expected_log.values, rtol=1e-10)


def test_log_transform_custom_columns(mock_loader):
    """Test log transformation with custom columns."""
    # Create sample data with custom volume-like columns
    data = pd.DataFrame(
        {
            "volume": [1000, 2000, 3000, 4000, 5000],
            "buy_volume": [500, 1000, 1500, 2000, 2500],
            "sell_volume": [500, 1000, 1500, 2000, 2500],
        }
    )

    result = mock_loader.log_transform_volume(
        data, columns=["buy_volume", "sell_volume"]
    )

    # Check that custom logs were created
    assert "buy_volume_log" in result.columns
    assert "sell_volume_log" in result.columns

    # Check that volume_log was NOT created
    assert "volume_log" not in result.columns

    # Verify first row was dropped
    assert len(result) == len(data) - 1

    # Verify log transformation
    expected_buy_log = np.log(data["buy_volume"].iloc[1:] + 1e-8)
    np.testing.assert_allclose(
        result["buy_volume_log"].values, expected_buy_log.values, rtol=1e-10
    )


def test_log_transform_missing_columns(mock_loader, sample_data):
    """Test log transformation with missing columns."""
    with pytest.warns(UserWarning, match="No specified volume columns found"):
        result = mock_loader.log_transform_volume(sample_data, columns=["nonexistent"])

    # Should return original data unchanged
    assert result.equals(sample_data)


def test_log_transform_with_zeros(mock_loader):
    """Test log transformation handles zero values correctly."""
    data = pd.DataFrame({"volume": [0, 100, 1000, 0, 5000]})

    result = mock_loader.log_transform_volume(data)

    # First row is dropped, so we have 4 rows
    assert len(result) == 4

    # Check no NaN or inf values
    assert not result["volume_log"].isna().any()
    assert not np.isinf(result["volume_log"]).any()

    # Verify epsilon was applied to zero values
    # Original indices [1, 2, 3, 4] become [0, 1, 2, 3] after drop
    assert result["volume_log"].iloc[0] == np.log(100 + 1e-8)  # was index 1
    assert result["volume_log"].iloc[2] == np.log(1e-8)  # was index 3 (zero value)
