"""
Data processing utilities for Helios Trader
Includes dollar bars conversion and data preparation
"""

import pandas as pd
import numpy as np
# Type hints are handled by pandas


def create_dollar_bars(
    df: pd.DataFrame, dollar_threshold: float = 1000000
) -> pd.DataFrame:
    """
    Convert OHLCV data to dollar bars

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: timestamp, open, high, low, close, volume
    dollar_threshold : float
        Dollar volume threshold for each bar

    Returns:
    --------
    pd.DataFrame
        Dollar bars with OHLCV data
    """
    df = df.copy()
    df["dollar_volume"] = df["close"] * df["volume"]

    bars = []
    current_bar = {
        "open": None,
        "high": -np.inf,
        "low": np.inf,
        "close": None,
        "volume": 0,
        "dollar_volume": 0,
        "timestamp": None,
        "bar_count": 0,
    }

    for idx, row in df.iterrows():
        if current_bar["open"] is None:
            current_bar["open"] = row.loc["open"]
            current_bar["timestamp"] = idx  # Use index as timestamp

        current_bar["high"] = max(current_bar["high"], row.loc["high"])
        current_bar["low"] = min(current_bar["low"], row.loc["low"])
        current_bar["close"] = row.loc["close"]
        current_bar["volume"] += row.loc["volume"]
        current_bar["dollar_volume"] += row.loc["dollar_volume"]
        current_bar["bar_count"] += 1

        if current_bar["dollar_volume"] >= dollar_threshold:
            bars.append(
                {
                    "timestamp": current_bar["timestamp"],
                    "open": current_bar["open"],
                    "high": current_bar["high"],
                    "low": current_bar["low"],
                    "close": current_bar["close"],
                    "volume": current_bar["volume"],
                    "dollar_volume": current_bar["dollar_volume"],
                    "bar_count": current_bar["bar_count"],
                }
            )

            current_bar = {
                "open": None,
                "high": -np.inf,
                "low": np.inf,
                "close": None,
                "volume": 0,
                "dollar_volume": 0,
                "timestamp": None,
                "bar_count": 0,
            }

    # Add the last bar if it has data
    if current_bar["open"] is not None:
        bars.append(
            {
                "timestamp": current_bar["timestamp"],
                "open": current_bar["open"],
                "high": current_bar["high"],
                "low": current_bar["low"],
                "close": current_bar["close"],
                "volume": current_bar["volume"],
                "dollar_volume": current_bar["dollar_volume"],
                "bar_count": current_bar["bar_count"],
            }
        )

    dollar_bars_df = pd.DataFrame(bars)
    if not dollar_bars_df.empty:
        dollar_bars_df["timestamp"] = pd.to_datetime(dollar_bars_df["timestamp"])
        dollar_bars_df.set_index("timestamp", inplace=True)

    return dollar_bars_df


def prepare_data(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Prepare data for analysis with rolling normalizations suitable for live trading
    
    All normalizations use only historical data (no look-ahead bias)

    Parameters:
    -----------
    df : pd.DataFrame
        Raw OHLCV data
    lookback : int
        Number of periods for rolling calculations

    Returns:
    --------
    pd.DataFrame
        Prepared data with normalized columns
    """
    df = df.copy()

    # Ensure numeric types
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with NaN values in critical columns
    df.dropna(subset=["close"], inplace=True)

    # Sort by index to ensure chronological order
    df.sort_index(inplace=True)
    
    # Add HLC3 (more stable than close for some calculations)
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Log returns - useful for volatility estimation and risk modeling
    # Comment: We use simple returns for trading signals (more intuitive),
    # but log returns are available for statistical analysis if needed
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    df['hlc3_returns'] = df['hlc3'].pct_change()
    
    # Rolling normalized returns (z-score)
    # Using expanding window for early periods, then fixed rolling window
    # This ensures we always have some normalization, even for early rows
    
    # Calculate both rolling and expanding statistics
    returns_mean = df['returns'].rolling(lookback, min_periods=1).mean()
    returns_std = df['returns'].rolling(lookback, min_periods=1).std()
    
    # For the first lookback periods, use expanding window
    # Using where() is more efficient than fillna() for this use case
    mask = returns_mean.isna()
    if mask.any():
        expanding_mean = df['returns'].expanding(min_periods=1).mean()
        expanding_std = df['returns'].expanding(min_periods=1).std()
        returns_mean = returns_mean.where(~mask, expanding_mean)
        returns_std = returns_std.where(returns_std.notna(), expanding_std)
    
    # Avoid division by zero - if std is too small, use a minimum value
    returns_std_safe = returns_std.clip(lower=1e-8)
    df['returns_z'] = (df['returns'] - returns_mean) / returns_std_safe
    
    # Rolling normalized volume (percentage of average)
    # This shows volume relative to recent average (1.0 = average volume)
    volume_ma = df['volume'].rolling(lookback, min_periods=1).mean()
    
    # Use expanding window for early periods (more efficient with where())
    mask_vol = volume_ma.isna()
    if mask_vol.any():
        expanding_vol_ma = df['volume'].expanding(min_periods=1).mean()
        volume_ma = volume_ma.where(~mask_vol, expanding_vol_ma)
    
    # Avoid division by zero
    volume_ma_safe = volume_ma.clip(lower=1e-8)
    df['volume_n'] = df['volume'] / volume_ma_safe
    
    # Rolling z-score normalized volume (for spike detection)
    volume_mean = df['volume'].rolling(lookback, min_periods=1).mean()
    volume_std = df['volume'].rolling(lookback, min_periods=1).std()
    
    # Use expanding window for early periods
    mask_vol_stats = volume_mean.isna() | volume_std.isna()
    if mask_vol_stats.any():
        expanding_vol_mean = df['volume'].expanding(min_periods=1).mean()
        expanding_vol_std = df['volume'].expanding(min_periods=1).std()
        volume_mean = volume_mean.where(~mask_vol_stats, expanding_vol_mean)
        volume_std = volume_std.where(~mask_vol_stats, expanding_vol_std)
    
    # Avoid division by zero
    volume_std_safe = volume_std.clip(lower=1e-8)
    df['volume_z'] = (df['volume'] - volume_mean) / volume_std_safe
    
    # Clean up edge cases (inf values from division, NaN from insufficient data)
    df['returns_z'] = df['returns_z'].replace([np.inf, -np.inf], 0)
    df['returns_z'] = df['returns_z'].fillna(0)
    
    df['volume_n'] = df['volume_n'].replace([np.inf, -np.inf], 1)
    df['volume_n'] = df['volume_n'].fillna(1)
    
    df['volume_z'] = df['volume_z'].replace([np.inf, -np.inf], 0)
    df['volume_z'] = df['volume_z'].fillna(0)
    
    # Add cumulative returns for performance tracking
    df['cumulative_returns'] = (1 + df['returns'].fillna(0)).cumprod() - 1

    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with high, low, close columns
    period : int
        Period for ATR calculation

    Returns:
    --------
    pd.Series
        ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr
