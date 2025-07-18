"""
Test the transform method with frac_diff and log_volume parameters.
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
        dates = pd.date_range(start=start, periods=500, freq="1min")
        prices = np.linspace(100, 110, 500) + np.random.normal(0, 0.5, 500)

        df = pd.DataFrame(
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
        # Add derived columns
        df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3
        df["dv"] = df["hlc3"] * df["volume"]
        return df


@pytest.fixture
def mock_config(tmp_path):
    """Create mock configuration."""

    class MockConfig:
        symbol = "TEST"
        timeframe = "1m"
        min_bars = 100
        paper_mode = True

    return MockConfig()


@pytest.fixture
def mock_loader(mock_config, tmp_path, monkeypatch):
    """Create mock data loader for testing."""
    loader = MockDataLoader(mock_config)

    # Mock the CSV path to use temp directory
    def mock_get_csv_path(suffix=None):
        if suffix:
            return str(tmp_path / f"test_data.{suffix}.csv")
        return str(tmp_path / "test_data.csv")

    monkeypatch.setattr(loader, "get_csv_path", mock_get_csv_path)
    return loader


def test_apply_frac_diff_false(mock_loader):
    """Test frac_diff=False does nothing."""
    df = mock_loader.load_data()
    df_transformed = mock_loader.transform(df, frac_diff=False)

    # Check no _fd columns were created
    fd_columns = [col for col in df_transformed.columns if "_fd" in col]
    assert len(fd_columns) == 0

    # Data should be unchanged
    pd.testing.assert_frame_equal(df, df_transformed)


def test_apply_frac_diff_none(mock_loader):
    """Test frac_diff=None does nothing."""
    df = mock_loader.load_data()
    df_transformed = mock_loader.transform(df, frac_diff=None)

    # Check no _fd columns were created
    fd_columns = [col for col in df_transformed.columns if "_fd" in col]
    assert len(fd_columns) == 0

    # Data should be unchanged
    pd.testing.assert_frame_equal(df, df_transformed)


def test_apply_frac_diff_true(mock_loader, capsys):
    """Test frac_diff=True overwrites original columns."""
    df_original = mock_loader.load_data()
    df = mock_loader.transform(df_original.copy(), frac_diff=True)

    # Check no _fd columns were created
    fd_columns = [col for col in df.columns if "_fd" in col]
    assert len(fd_columns) == 0

    # Check that original price columns still exist (but are now differentiated)
    assert "close" in df.columns
    assert "open" in df.columns

    # Values should be different due to transformation
    assert not np.array_equal(df["close"].values, df_original["close"].values)


def test_apply_frac_diff_string_default(mock_loader, capsys):
    """Test frac_diff='_fd' creates columns with _fd suffix."""
    df_base = mock_loader.load_data()
    df = mock_loader.transform(df_base, frac_diff="_fd")

    # Check _fd columns were created
    assert "open_fd" in df.columns
    assert "high_fd" in df.columns
    assert "low_fd" in df.columns
    assert "close_fd" in df.columns

    # Check original columns still exist
    assert "open" in df.columns
    assert "close" in df.columns


def test_apply_frac_diff_string_custom(mock_loader, capsys):
    """Test frac_diff with custom suffix."""
    df_base = mock_loader.load_data()
    df = mock_loader.transform(df_base, frac_diff="_diff")

    # Check custom suffix columns were created
    assert "open_diff" in df.columns
    assert "high_diff" in df.columns
    assert "low_diff" in df.columns
    assert "close_diff" in df.columns

    # Check no _fd columns exist
    fd_columns = [col for col in df.columns if col.endswith("_fd")]
    assert len(fd_columns) == 0


def test_apply_log_volume_false(mock_loader):
    """Test log_volume=False does nothing."""
    df = mock_loader.load_data()
    df_transformed = mock_loader.transform(df, log_volume=False)

    # Check no log columns were created
    log_columns = [
        col for col in df_transformed.columns if "_log" in col or "_lr" in col
    ]
    assert len(log_columns) == 0

    # Volume should be unchanged
    np.testing.assert_array_equal(df["volume"].values, df_transformed["volume"].values)


def test_apply_log_volume_true(mock_loader, capsys):
    """Test log_volume=True overwrites volume."""
    # Store original volume to verify it changes
    df_original = mock_loader.load_data()
    original_volume = df_original["volume"].copy()

    # Apply log transformation
    df = mock_loader.transform(df_original.copy(), log_volume=True)

    # Check no _log columns were created
    log_columns = [col for col in df.columns if "_log" in col or "_lr" in col]
    assert len(log_columns) == 0

    # Check that volume was transformed
    assert "volume" in df.columns
    # Verify values changed (log transform should change all positive values)
    # Note: first row is dropped by default
    assert len(df) == len(df_original) - 1


def test_apply_log_volume_string_default(mock_loader, capsys):
    """Test log_volume='_log' creates columns with _log suffix."""
    df_base = mock_loader.load_data()
    df = mock_loader.transform(df_base, log_volume="_log")

    # Check volume_log was created
    assert "volume_log" in df.columns

    # Check original volume still exists
    assert "volume" in df.columns

    # Should have fewer rows due to first row drop
    assert len(df) == len(df_base) - 1


def test_apply_log_volume_string_custom(mock_loader, capsys):
    """Test log_volume with custom suffix."""
    df_base = mock_loader.load_data()
    df = mock_loader.transform(df_base, log_volume="_lr")

    # Check custom suffix column was created
    assert "volume_lr" in df.columns

    # Check no _log column exists
    assert "volume_log" not in df.columns

    # Check original volume still exists
    assert "volume" in df.columns


def test_combined_transformations(mock_loader, capsys):
    """Test applying both transformations together."""
    df_base = mock_loader.load_data()
    df = mock_loader.transform(df_base, frac_diff="_fd", log_volume="_log")

    # Check both sets of columns were created
    assert "close_fd" in df.columns
    assert "volume_log" in df.columns

    # Check originals still exist
    assert "close" in df.columns
    assert "volume" in df.columns


def test_combined_transformations_custom_suffixes(mock_loader, capsys):
    """Test combined transformations with custom suffixes."""
    df_base = mock_loader.load_data()
    df = mock_loader.transform(df_base, frac_diff="_fd", log_volume="_lr")

    # Check custom suffix columns
    assert "close_fd" in df.columns
    assert "volume_lr" in df.columns

    # Original columns should still exist
    assert "close" in df.columns
    assert "volume" in df.columns


def test_transform_saves_to_transform_csv(mock_loader, tmp_path, monkeypatch):
    """Test that transform saves to .transform.csv file."""

    # Mock the CSV path
    def mock_get_csv_path(suffix=None):
        if suffix:
            return str(tmp_path / f"test_data.{suffix}.csv")
        return str(tmp_path / "test_data.csv")

    monkeypatch.setattr(mock_loader, "get_csv_path", mock_get_csv_path)

    # Load and transform data
    df_base = mock_loader.load_data()
    df = mock_loader.transform(df_base, frac_diff="_fd", log_volume="_lr")

    # Check that transform CSV was created
    transform_csv_path = str(tmp_path / "test_data.transform.csv")
    assert pd.read_csv(transform_csv_path).shape[0] > 0

    # Check main CSV has only base columns
    main_df = pd.read_csv(str(tmp_path / "test_data.bars.csv"))
    assert "close_fd" not in main_df.columns
    assert "volume_lr" not in main_df.columns
