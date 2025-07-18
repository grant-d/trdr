import pytest
import pandas as pd
import numpy as np
from indicators import (
    calculate_atr,
    calculate_macd,
    calculate_rsi,
    calculate_stddev,
    calculate_trend_factor,
    calculate_volatility_factor,
    calculate_exhaustion_factor,
    calculate_mss,
    calculate_dynamic_mss,
    MacdResult,
    DynamicWeights,
)


# Helper to create OHLC test DataFrame
def make_ohlc_df(size=100):
    np.random.seed(42)
    close = np.cumsum(np.random.randn(size)) + 100
    high = close + np.random.rand(size) * 2
    low = close - np.random.rand(size) * 2
    df = pd.DataFrame(
        {
            "open": close + np.random.randn(size),
            "high": high,
            "low": low,
            "close": close,
        }
    )
    return df


def test_calculate_atr_basic():
    df = make_ohlc_df()
    atr = calculate_atr(df, period=14)
    assert isinstance(atr, pd.Series)
    assert len(atr) == len(df)
    assert atr.isna().sum() < len(df)  # Should have non-NaN values


def test_calculate_macd_basic():
    df = make_ohlc_df()
    result = calculate_macd(df)
    assert isinstance(result, MacdResult)
    assert len(result.macd) == len(df)
    assert len(result.signal) == len(df)
    assert len(result.histogram) == len(df)
    assert len(result.histogram_normalized) == len(df)


def test_calculate_rsi_basic():
    df = make_ohlc_df()
    rsi = calculate_rsi(df, period=14)
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(df)
    assert rsi.min() >= 0 or np.isnan(rsi.min())
    assert rsi.max() <= 100 or np.isnan(rsi.max())
    rsi_norm = calculate_rsi(df, period=14, normalized=True)
    assert rsi_norm.min() >= -100 or np.isnan(rsi_norm.min())
    assert rsi_norm.max() <= 100 or np.isnan(rsi_norm.max())


def test_calculate_stddev_basic():
    df = make_ohlc_df()
    stddev = calculate_stddev(df, period=20)
    assert isinstance(stddev, pd.Series)
    assert len(stddev) == len(df)
    assert stddev.isna().sum() < len(df)


def test_calculate_trend_factor_basic():
    df = make_ohlc_df()
    trend = calculate_trend_factor(df, lookback=20)
    assert isinstance(trend, pd.Series)
    assert len(trend) == len(df)


def test_calculate_volatility_factor_basic():
    df = make_ohlc_df()
    vol = calculate_volatility_factor(df, lookback=20)
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(df)
    # Only check non-NaN values for non-negativity
    assert (vol[vol.notna()] >= 0).all()


def test_calculate_exhaustion_factor_basic():
    df = make_ohlc_df()
    exhaustion = calculate_exhaustion_factor(df, lookback=20)
    assert isinstance(exhaustion, pd.Series)
    assert len(exhaustion) == len(df)
    assert exhaustion.min() >= -100 or np.isnan(exhaustion.min())
    assert exhaustion.max() <= 100 or np.isnan(exhaustion.max())


def test_calculate_mss_basic():
    df = make_ohlc_df()
    result = calculate_mss(df, lookback=20)
    assert hasattr(result, "results")
    assert hasattr(result, "regimes")
    assert isinstance(result.results, pd.DataFrame)
    assert isinstance(result.regimes, pd.Series)
    assert "mss" in result.results.columns
    assert "regime" in result.results.columns


def test_calculate_dynamic_mss_basic():
    df = make_ohlc_df()
    result = calculate_dynamic_mss(df, lookback=20)
    assert hasattr(result, "results")
    assert hasattr(result, "regimes")
    assert isinstance(result.results, pd.DataFrame)
    assert isinstance(result.regimes, pd.Series)
    assert "mss" in result.results.columns
    assert "regime" in result.results.columns
