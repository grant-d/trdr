"""
Test to verify NaN handling fix in clean_data method.
"""

from typing import Optional
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from base_data_loader import BaseDataLoader


class MockDataLoader(BaseDataLoader):
    """Mock data loader for testing."""

    def setup_clients(self) -> None:
        """No clients needed for mock."""
        pass

    def fetch_bars(
        self, start: datetime, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Return mock data simulating quiet exchange with zero volume bars."""
        # Simulate real scenario: quiet exchange with many zero-volume bars
        dates = pd.date_range(start=start, periods=100, freq="1min")

        # Create prices that change occasionally (forward-filled in reality)
        base_prices = [107850.0] * 20 + [107860.0] * 30 + [107880.0] * 50

        data = []
        for i, (date, price) in enumerate(zip(dates, base_prices)):
            # Most bars have zero volume (quiet exchange)
            volume = 0.0 if i % 10 != 0 else np.random.uniform(0.01, 0.1)

            data.append(
                {
                    "timestamp": date,
                    "open": price,
                    "high": price + np.random.uniform(0, 10),
                    "low": price - np.random.uniform(0, 10),
                    "close": price + np.random.uniform(-5, 5),
                    "volume": volume,
                    "trade_count": int(volume * 100) if volume > 0 else 0,
                }
            )

        return pd.DataFrame(data)


@pytest.fixture
def mock_config():
    """Create mock configuration."""

    class MockConfig:
        symbol = "BTCUSD"
        timeframe = "1m"
        min_bars = 100
        paper_mode = True

    return MockConfig()


@pytest.fixture
def mock_loader(mock_config) -> MockDataLoader:
    """Create mock data loader."""
    return MockDataLoader(mock_config)


def test_no_nan_after_cleaning_quiet_exchange(mock_loader):
    """Test that clean_data doesn't produce NaN values for quiet exchange data."""
    # Fetch data simulating quiet exchange
    df = mock_loader.fetch_bars(datetime.now(timezone.utc))

    # Verify we have zero-volume bars
    zero_volume_count = (df["volume"] == 0).sum()
    assert (
        zero_volume_count > 50
    ), f"Expected many zero-volume bars, got {zero_volume_count}"

    # Clean the data
    cleaned_df = mock_loader.clean_data(df)

    # Check for NaN values
    nan_count = cleaned_df.isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaN values after cleaning"

    # Verify all columns exist and have no NaN
    expected_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
    ]
    for col in expected_columns:
        assert col in cleaned_df.columns, f"Missing column: {col}"
        assert not cleaned_df[col].isna().any(), f"Found NaN in column: {col}"


def test_no_nan_with_identical_prices(mock_loader):
    """Test handling of identical prices (zero std deviation scenario)."""
    # Create data with identical prices
    dates = pd.date_range(start=datetime.now(timezone.utc), periods=50, freq="1min")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [107850.0] * 50,
            "high": [107850.0] * 50,
            "low": [107850.0] * 50,
            "close": [107850.0] * 50,
            "volume": [0.0] * 50,
            "trade_count": [0] * 50,
        }
    )

    # Clean the data
    cleaned_df = mock_loader.clean_data(df)

    # Check for NaN values
    nan_count = cleaned_df.isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaN values with identical prices"


def test_pct_change_nan_handling(mock_loader):
    """Test that pct_change NaN in first row is handled correctly."""
    # Create minimal data
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=datetime.now(timezone.utc), periods=5, freq="1min"
            ),
            "open": [
                100.0,
                120.0,
                130.0,
                140.0,
                150.0,
            ],  # 20% jump would trigger outlier
            "high": [101.0, 121.0, 131.0, 141.0, 151.0],
            "low": [99.0, 119.0, 129.0, 139.0, 149.0],
            "close": [100.5, 120.5, 130.5, 140.5, 150.5],
            "volume": [100.0, 200.0, 300.0, 400.0, 500.0],
            "trade_count": [10, 20, 30, 40, 50],
        }
    )

    # Clean the data
    cleaned_df = mock_loader.clean_data(df)

    # Check for NaN values
    nan_count = cleaned_df.isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaN values after handling pct_change"

    # First row should still have valid data
    assert not cleaned_df.iloc[0].isna().any(), "First row should not have NaN values"
