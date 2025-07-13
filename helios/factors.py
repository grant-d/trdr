"""
Factor calculation functions for Helios Trader
Includes MACD, RSI, Standard Deviation and Market State Score calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'close' prices
    fast : int
        Fast EMA period
    slow : int
        Slow EMA period
    signal : int
        Signal line EMA period
    
    Returns:
    --------
    Dict[str, pd.Series]
        Dictionary with 'macd', 'signal', 'histogram', and 'histogram_normalized' series
    """
    close = df['close']
    
    # Calculate EMAs
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    # MACD line
    macd = ema_fast - ema_slow
    
    # Signal line
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    # MACD histogram
    histogram = macd - signal_line
    
    # Normalize histogram to -100 to 100 range
    # Use percentile-based normalization for robustness
    hist_abs = histogram.abs()
    hist_95th = hist_abs.rolling(window=252, min_periods=50).quantile(0.95)
    hist_95th = hist_95th.fillna(hist_abs.expanding(min_periods=50).quantile(0.95))
    
    # Avoid division by zero
    valid_mask = hist_95th > 1e-10
    histogram_normalized = histogram.copy()
    histogram_normalized[valid_mask] = (histogram[valid_mask] / hist_95th[valid_mask]) * 100
    histogram_normalized = np.clip(histogram_normalized, -100, 100)
    
    return {
        'macd': macd,
        'signal': signal_line,
        'histogram': histogram,
        'histogram_normalized': histogram_normalized
    }


def calculate_rsi(df: pd.DataFrame, period: int = 14, normalized: bool = False) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'close' prices
    period : int
        RSI period
    normalized : bool
        If True, return normalized RSI (-100 to 100)
    
    Returns:
    --------
    pd.Series
        RSI values (0-100 or -100 to 100 if normalized)
    """
    close = df['close']
    delta = close.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    if normalized:
        # Normalize to -100 to 100 range (50 becomes 0)
        rsi = (rsi - 50) * 2
    
    return rsi


def calculate_stddev(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate rolling standard deviation
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'close' prices
    period : int
        Rolling window period
    
    Returns:
    --------
    pd.Series
        Standard deviation values
    """
    return df['close'].rolling(window=period).std()


def calculate_trend_factor(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Calculate trend factor using linear regression slope
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'close' prices
    lookback : int
        Lookback period for regression
    
    Returns:
    --------
    pd.Series
        Normalized trend factor values
    """
    close = df['close'].values
    trend_factor = np.zeros(len(close))
    
    for i in range(lookback, len(close)):
        y = close[i-lookback:i]
        x = np.arange(lookback)
        
        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by price level
        trend_factor[i] = slope / close[i] * 100
    
    return pd.Series(trend_factor, index=df.index)


def calculate_volatility_factor(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Calculate volatility factor using ATR
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
    lookback : int
        Lookback period for ATR
    
    Returns:
    --------
    pd.Series
        Normalized volatility factor values
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=lookback).mean()
    
    # Normalize by price level
    volatility_factor = atr / df['close'] * 100
    
    return volatility_factor


def calculate_exhaustion_factor(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Calculate exhaustion/mean reversion factor
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
    lookback : int
        Lookback period
    
    Returns:
    --------
    pd.Series
        Normalized exhaustion factor values (-100 to 100)
    """
    close = df['close']
    sma = close.rolling(window=lookback).mean()
    
    # Calculate ATR for normalization
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=lookback).mean()
    
    # Distance from SMA normalized by ATR
    valid_mask = (atr.notna()) & (atr.abs() > 1e-9)
    exhaustion_factor = pd.Series(index=df.index, dtype=float)
    exhaustion_factor[valid_mask] = (close[valid_mask] - sma[valid_mask]) / atr[valid_mask]
    
    # Scale to -100 to 100 range (assuming -10 to 10 ATR units covers most cases)
    scaling_factor = 100 / 10
    exhaustion_factor = np.clip(exhaustion_factor * scaling_factor, -100, 100)
    
    return exhaustion_factor


def calculate_mss(df: pd.DataFrame, lookback: int = 20, 
                  weights: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calculate Market State Score (MSS) and classify regimes
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
    lookback : int
        Lookback period for calculations
    weights : Optional[Dict[str, float]]
        Custom weights for factors. If None, uses equal weights
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        DataFrame with factors and MSS, Series with regime classifications
    """
    # Default weights if not provided
    if weights is None:
        weights = {
            'trend': 1/3,
            'volatility': 1/3,
            'exhaustion': 1/3
        }
    
    # Calculate factors
    trend = calculate_trend_factor(df, lookback)
    volatility = calculate_volatility_factor(df, lookback)
    exhaustion = calculate_exhaustion_factor(df, lookback)
    
    # Add ATR for reference (used in trading)
    from data_processing import calculate_atr
    atr = calculate_atr(df, lookback)
    
    # Factors are already normalized to appropriate ranges
    # Trend: roughly -100 to 100 based on percentage slope
    # Volatility: 0 to 100 (percentage of price)
    # Exhaustion: -100 to 100 (clipped)
    
    # For MSS calculation, normalize volatility to -100 to 100
    # High volatility is generally negative for trend following
    volatility_norm = 100 - (volatility * 2)  # Convert 0-100 to 100 to -100
    volatility_norm = np.clip(volatility_norm, -100, 100)
    
    # Calculate MSS as weighted sum
    mss = (weights['trend'] * trend + 
           weights['volatility'] * volatility_norm + 
           weights['exhaustion'] * exhaustion)
    
    # Classify regimes based on MSS
    # Using thresholds from the new implementation
    regimes = pd.Series(index=df.index, dtype='object')
    regimes[mss > 50] = 'Strong Bull'
    regimes[(mss > 20) & (mss <= 50)] = 'Weak Bull'
    regimes[(mss >= -20) & (mss <= 20)] = 'Neutral'
    regimes[(mss > -50) & (mss < -20)] = 'Weak Bear'
    regimes[mss <= -50] = 'Strong Bear'
    
    # Create results DataFrame
    results = pd.DataFrame({
        'trend': trend,
        'volatility': volatility,
        'exhaustion': exhaustion,
        'volatility_norm': volatility_norm,
        'mss': mss,
        'regime': regimes,
        'atr': atr
    }, index=df.index)
    
    return results, regimes


def calculate_dynamic_weights(regime: str, base_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Calculate dynamic factor weights based on market regime
    
    Parameters:
    -----------
    regime : str
        Current market regime
    base_weights : Optional[Dict[str, float]]
        Base weights to modify. If None, uses defaults
    
    Returns:
    --------
    Dict[str, float]
        Adjusted weights for each factor
    """
    if base_weights is None:
        base_weights = {
            'trend': 0.4,
            'volatility': 0.3,
            'exhaustion': 0.3
        }
    
    # Regime-specific adjustments
    adjustments = {
        'Strong Bull': {'trend': 1.2, 'volatility': 0.8, 'exhaustion': 0.8},
        'Weak Bull': {'trend': 1.1, 'volatility': 0.9, 'exhaustion': 1.0},
        'Neutral': {'trend': 0.9, 'volatility': 1.1, 'exhaustion': 1.1},
        'Weak Bear': {'trend': 0.8, 'volatility': 1.1, 'exhaustion': 1.2},
        'Strong Bear': {'trend': 0.7, 'volatility': 1.2, 'exhaustion': 1.3}
    }
    
    # Apply adjustments
    adj = adjustments.get(regime, {'trend': 1.0, 'volatility': 1.0, 'exhaustion': 1.0})
    weights = {
        'trend': base_weights['trend'] * adj['trend'],
        'volatility': base_weights['volatility'] * adj['volatility'],
        'exhaustion': base_weights['exhaustion'] * adj['exhaustion']
    }
    
    # Normalize to sum to 1
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    return weights