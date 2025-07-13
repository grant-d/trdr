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
    Prepare data for analysis by ensuring proper types and removing NaNs

    Parameters:
    -----------
    df : pd.DataFrame
        Raw OHLCV data
    lookback : int
        Number of periods for rolling calculations

    Returns:
    --------
    pd.DataFrame
        Prepared data
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

    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various return metrics

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'close' column

    Returns:
    --------
    pd.DataFrame
        DataFrame with return columns added
    """
    df = df.copy()

    # Simple returns
    df["returns"] = df["close"].pct_change()

    # Log returns
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Cumulative returns
    df["cumulative_returns"] = (1 + df["returns"]).cumprod() - 1

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
