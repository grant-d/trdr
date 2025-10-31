"""
Hull Moving Average (HMA) Calculator

The Hull Moving Average is designed to reduce lag and improve smoothing.
Formula: HMA(n) = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
Where WMA is the Weighted Moving Average
"""

import numpy as np
import pandas as pd


def weighted_moving_average(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Weighted Moving Average (WMA)

    Args:
        data: Price series
        period: Number of periods

    Returns:
        WMA series
    """
    weights = np.arange(1, period + 1)

    def wma(x):
        if len(x) < period:
            return np.nan
        return np.dot(x[-period:], weights) / weights.sum()

    return data.rolling(window=period).apply(wma, raw=True)


def hull_moving_average(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Hull Moving Average (HMA)

    Args:
        data: Price series (typically close prices)
        period: HMA period

    Returns:
        HMA series
    """
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))

    # Calculate WMA with half period
    wma_half = weighted_moving_average(data, half_period)

    # Calculate WMA with full period
    wma_full = weighted_moving_average(data, period)

    # Calculate raw HMA: 2 * WMA(n/2) - WMA(n)
    raw_hma = 2 * wma_half - wma_full

    # Apply WMA to raw HMA with sqrt(period)
    hma = weighted_moving_average(raw_hma, sqrt_period)

    return hma


def is_hma_trending_up(hma: pd.Series, lookback: int = 3) -> pd.Series:
    """
    Determine if HMA is trending up

    Args:
        hma: HMA series
        lookback: Number of periods to check for uptrend

    Returns:
        Boolean series indicating uptrend
    """
    # Check if current HMA is greater than previous values
    uptrend = pd.Series(False, index=hma.index)

    for i in range(lookback):
        uptrend &= hma > hma.shift(i + 1)

    # Alternative: check if slope is positive
    # This is more lenient - just checks if recent trend is up
    slope = hma - hma.shift(lookback)
    uptrend = slope > 0

    return uptrend


def calculate_hma_signals(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculate HMA and trend signals

    Args:
        df: DataFrame with OHLC data
        period: HMA period

    Returns:
        DataFrame with HMA and trend signal added
    """
    df = df.copy()
    col_name = f'HMA_{period}'
    trend_col = f'HMA_{period}_UP'

    df[col_name] = hull_moving_average(df['Close'], period)
    df[trend_col] = is_hma_trending_up(df[col_name])

    return df


if __name__ == '__main__':
    # Test the HMA calculation
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    prices = 100 + np.cumsum(np.random.randn(200) * 2)

    df = pd.DataFrame({'Close': prices}, index=dates)

    # Calculate HMA 50
    df['HMA_50'] = hull_moving_average(df['Close'], 50)
    df['HMA_50_UP'] = is_hma_trending_up(df['HMA_50'])

    print(df.tail(10))
