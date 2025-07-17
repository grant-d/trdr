"""
Test load_data and transform method integration.
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
            }
        )


@pytest.fixture
def mock_config(tmp_path):
    """Create mock configuration with temp directory."""

    class MockConfig:
        symbol = "TEST/USD"
        timeframe = "1m"
        min_bars = 100
        paper_mode = True

    return MockConfig()


@pytest.fixture
def mock_loader(mock_config, tmp_path, monkeypatch):
    """Create mock data loader."""
    loader = MockDataLoader(mock_config)

    # Mock the CSV path to use temp directory
    def mock_get_csv_path(suffix=None):
        if suffix:
            return str(tmp_path / f"test_data.{suffix}.csv")
        return str(tmp_path / "test_data.csv")

    monkeypatch.setattr(loader, "get_csv_path", mock_get_csv_path)
    return loader


def test_load_data_without_transformations(mock_loader):
    """Test load_data returns raw data without transformations."""
    df = mock_loader.load_data()
    df = mock_loader.clean_data(df)

    # Check all expected columns exist
    expected_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "hlc3",
        "dv",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Check no transformation columns exist
    assert not any("_fd" in col for col in df.columns)
    assert not any("_log" in col for col in df.columns)
    assert not any("_lr" in col for col in df.columns)


def test_transform_with_frac_diff_overwrite(mock_loader):
    """Test transform with fractional differentiation overwrite mode."""
    df_base = mock_loader.load_data()
    df_base = mock_loader.clean_data(df_base)
    original_close = df_base["close"].copy()

    df = mock_loader.transform(df_base.copy(), frac_diff=True)

    # Original columns should exist but be transformed
    assert "close" in df.columns
    assert "open" in df.columns

    # No new columns should be created
    assert "close_fd" not in df.columns

    # Values should be different (fractionally differentiated)
    # Note: May have fewer rows due to lag requirements
    assert len(df) <= len(df_base)


def test_transform_with_frac_diff_add(mock_loader):
    """Test transform with fractional differentiation in add mode."""
    df_base = mock_loader.load_data()
    df_base = mock_loader.clean_data(df_base)
    df = mock_loader.transform(df_base, frac_diff="_fd")

    # Original columns should still exist
    assert "close" in df.columns
    assert "open" in df.columns

    # New columns should be created
    assert "close_fd" in df.columns
    assert "open_fd" in df.columns
    assert "high_fd" in df.columns
    assert "low_fd" in df.columns

    # DataFrame may have fewer rows due to NaN dropping
    assert len(df) <= len(df_base)


def test_transform_invalid_mode(mock_loader):
    """Test transform with invalid parameter type raises error."""
    df_base = mock_loader.load_data()
    df_base = mock_loader.clean_data(df_base)

    with pytest.raises(ValueError, match="Invalid frac_diff type"):
        mock_loader.transform(df_base, frac_diff=123)


def test_transform_with_empty_data(mock_loader):
    """Test transform handles empty DataFrames gracefully."""
    empty_df = pd.DataFrame()
    result = mock_loader.transform(empty_df, frac_diff=True)

    assert result.empty
    assert len(result) == 0


def test_transform_preserves_cleaning(mock_loader):
    """Test that transform doesn't undo data cleaning."""
    # Load and clean data
    df_raw = mock_loader.load_data()
    df_clean = mock_loader.clean_data(df_raw)

    # Apply transformation
    df_transformed = mock_loader.transform(df_clean, frac_diff="_fd")

    # Check that cleaned properties are preserved
    # (e.g., no NaN values in critical columns)
    critical_cols = ["open", "high", "low", "close"]
    for col in critical_cols:
        if col in df_transformed.columns:
            assert not df_transformed[col].isna().any()


def test_load_and_transform_workflow(mock_loader):
    """Test the complete workflow of load then transform."""
    # Step 1: Load data
    df_base = mock_loader.load_data()

    # Step 2: Clean data
    df_base = mock_loader.clean_data(df_base)

    # Step 3: Apply transformations
    df_transformed = mock_loader.transform(df_base, frac_diff="_fd", log_volume="_lr")

    # Check both transformations were applied
    assert "close_fd" in df_transformed.columns
    assert "volume_lr" in df_transformed.columns

    # Check original columns still exist
    assert "close" in df_transformed.columns
    assert "volume" in df_transformed.columns

    # Check row count (should be reduced due to log volume drop_first)
    assert len(df_transformed) < len(df_base)


def test_clean_data_saves_to_clean_csv(mock_loader, tmp_path, monkeypatch):
    """Test that clean_data saves to .clean.csv file."""

    # Mock the CSV path
    def mock_get_csv_path(suffix=None):
        if suffix:
            return str(tmp_path / f"test_data.{suffix}.csv")
        return str(tmp_path / "test_data.csv")

    monkeypatch.setattr(mock_loader, "get_csv_path", mock_get_csv_path)

    # Load and clean data
    df_raw = mock_loader.load_data()
    df_clean = mock_loader.clean_data(df_raw, stage_data=True)

    # Check that clean CSV was created
    clean_csv_path = str(tmp_path / "test_data.clean.csv")
    assert pd.read_csv(clean_csv_path).shape[0] > 0

    # Check it has same number of rows as cleaned data
    clean_df_from_file = pd.read_csv(clean_csv_path)
    assert len(clean_df_from_file) == len(df_clean)


def test_load_data_stage_data_false(mock_loader, tmp_path, monkeypatch):
    """Test that load_data with stage_data=False doesn't save to CSV."""

    # Mock the CSV path
    def mock_get_csv_path(suffix=None):
        if suffix:
            return str(tmp_path / f"test_data.{suffix}.csv")
        return str(tmp_path / "test_data.csv")

    monkeypatch.setattr(mock_loader, "get_csv_path", mock_get_csv_path)

    # Load data without staging
    df = mock_loader.load_data(stage_data=False)

    # Check that data was loaded
    assert len(df) > 0

    # Check that CSV was NOT created
    csv_path = tmp_path / "test_data.csv"
    assert not csv_path.exists()
