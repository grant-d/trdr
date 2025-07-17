"""
Pytest configuration and fixtures for testing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Generator, Optional
import tempfile
import os

from config_manager import Config
from base_data_loader import BaseDataLoader


class TestDataLoader(BaseDataLoader):
    """Test implementation of BaseDataLoader for testing purposes."""

    def setup_clients(self) -> None:
        """Test implementation - no clients needed."""
        pass

    def fetch_bars(
        self, start: datetime, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Test implementation - return empty DataFrame."""
        return pd.DataFrame()


@pytest.fixture
def test_config() -> Generator[Config, Any, None]:
    """Create a test configuration object."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config_path = f.name
        # Write empty JSON to avoid parse error
        f.write("{}")

    try:
        config = Config(
            config_path=config_path, symbol="TEST", timeframe="1h", min_bars=100
        )
        yield config
    finally:
        # Clean up
        if os.path.exists(config_path):
            os.unlink(config_path)


@pytest.fixture
def test_data_loader(test_config) -> TestDataLoader:
    """Create a test data loader instance."""
    return TestDataLoader(test_config)


@pytest.fixture
def clean_ohlcv_data() -> pd.DataFrame:
    """Create clean OHLCV test data without any issues."""
    timestamps = pd.date_range(
        start=datetime(2024, 1, 1), end=datetime(2024, 1, 2), freq="1h"
    )

    data = []
    base_price = 100.0

    for ts in timestamps:
        # Create realistic OHLC progression
        open_price = base_price + np.random.normal(0, 0.1)
        high_price = open_price + abs(np.random.normal(0.5, 0.1))
        low_price = open_price - abs(np.random.normal(0.5, 0.1))
        close_price = open_price + np.random.normal(0, 0.2)

        # Ensure OHLC integrity
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        volume = abs(np.random.normal(1000, 100))
        trade_count = int(abs(np.random.normal(50, 10)))

        data.append(
            {
                "timestamp": ts,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "trade_count": trade_count,
                "hlc3": (high_price + low_price + close_price) / 3,
                "dv": ((high_price + low_price + close_price) / 3) * volume,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def dirty_ohlcv_data() -> pd.DataFrame:
    """Create OHLCV test data with various quality issues."""
    timestamps = pd.date_range(
        start=datetime(2024, 1, 1), end=datetime(2024, 1, 2), freq="1h"
    )

    data = []
    base_price = 100.0

    for ts in timestamps:
        open_price = base_price + np.random.normal(0, 0.1)
        high_price = open_price + abs(np.random.normal(0.5, 0.1))
        low_price = open_price - abs(np.random.normal(0.5, 0.1))
        close_price = open_price + np.random.normal(0, 0.2)

        volume = abs(np.random.normal(1000, 100))
        trade_count = int(abs(np.random.normal(50, 10)))

        data.append(
            {
                "timestamp": ts,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "trade_count": trade_count,
                "hlc3": (high_price + low_price + close_price) / 3,
                "dv": ((high_price + low_price + close_price) / 3) * volume,
            }
        )

    df = pd.DataFrame(data)

    # Introduce specific data quality issues
    if len(df) >= 20:  # Ensure we have enough data
        # 1. Missing values (NaN)
        df.loc[1:3, "open"] = np.nan
        df.loc[5:7, "volume"] = np.nan
        df.loc[9, "trade_count"] = np.nan

        # 2. Outliers - extreme price jumps
        def safe_float(val, default: float = 1.0) -> float:
            try:
                v = pd.to_numeric(val, errors="coerce")
                if np.isnan(v):
                    return float(default)
                return float(v)
            except Exception:
                return float(default)

        df.loc[10, "close"] = safe_float(df.loc[10, "close"]) * 5.0  # 5x price jump
        df.loc[11, "open"] = safe_float(df.loc[11, "open"]) * 0.2  # 80% price drop

        # 3. Volume outliers
        df.loc[12, "volume"] = (
            safe_float(df.loc[12, "volume"]) * 50.0
        )  # 50x volume spike

        # 4. OHLCV integrity issues
        df.loc[13, "high"] = safe_float(df.loc[13, "low"]) - 1.0  # High < Low
        df.loc[14, "open"] = safe_float(df.loc[14, "high"]) + 2.0  # Open > High
        df.loc[15, "close"] = safe_float(df.loc[15, "low"]) - 1.0  # Close < Low

        # 5. Negative volume
        df.loc[16, "volume"] = -100.0

        # 6. Zero/negative prices
        df.loc[17, "low"] = 0.0
        df.loc[18, "close"] = -5.0

    return df


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame for testing edge cases."""
    return pd.DataFrame()


@pytest.fixture
def minimal_dataframe() -> pd.DataFrame:
    """Create a minimal DataFrame with just one row."""
    return pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
            "trade_count": [50],
            "hlc3": [100.17],
            "dv": [100170.0],
        }
    )


@pytest.fixture
def missing_columns_dataframe() -> pd.DataFrame:
    """Create a DataFrame missing required columns."""
    return pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)],
            "open": [100.0],
            "high": [101.0],
            # Missing 'low', 'close', 'volume'
        }
    )


class DataQualityAssertions:
    """Helper class for data quality assertions."""

    @staticmethod
    def assert_ohlcv_integrity(df: pd.DataFrame) -> None:
        """Assert OHLCV data integrity."""
        assert (df["high"] >= df["low"]).all(), "High should be >= Low"
        assert (df["open"] >= df["low"]).all(), "Open should be >= Low"
        assert (df["open"] <= df["high"]).all(), "Open should be <= High"
        assert (df["close"] >= df["low"]).all(), "Close should be >= Low"
        assert (df["close"] <= df["high"]).all(), "Close should be <= High"
        assert (df["volume"] >= 0).all(), "Volume should be >= 0"
        assert (
            (df[["open", "high", "low", "close"]] > 0).all().all()
        ), "All prices should be > 0"

    @staticmethod
    def assert_no_missing_values(df: pd.DataFrame) -> None:
        """Assert no missing values in critical columns."""
        critical_cols = ["open", "high", "low", "close", "volume"]
        for col in critical_cols:
            if col in df.columns:
                assert (
                    not df[col].isnull().any()
                ), f"Column {col} should not have missing values"

    @staticmethod
    def assert_derived_fields_correct(df: pd.DataFrame) -> None:
        """Assert derived fields are calculated correctly."""
        expected_hlc3 = (df["high"] + df["low"] + df["close"]) / 3
        expected_dv = expected_hlc3 * df["volume"]

        assert np.allclose(
            df["hlc3"], expected_hlc3, rtol=1e-10
        ), "HLC3 calculation incorrect"
        assert np.allclose(
            df["dv"], expected_dv, rtol=1e-10
        ), "Dollar volume calculation incorrect"

    @staticmethod
    def assert_chronological_order(df: pd.DataFrame) -> None:
        """Assert data is in chronological order."""
        if len(df) > 1:
            assert df[
                "timestamp"
            ].is_monotonic_increasing, "Data should be in chronological order"


@pytest.fixture
def data_quality_assertions() -> DataQualityAssertions:
    """Provide data quality assertion helpers."""
    return DataQualityAssertions()
