"""
Dynamic MSS calculation with regime-based weights
Extension to the factors module for advanced strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from factors import (
    calculate_trend_factor,
    calculate_volatility_factor,
    calculate_exhaustion_factor
)


def get_dynamic_weights(regime: str) -> Dict[str, float]:
    """
    Get dynamic factor weights based on market regime
    
    Parameters:
    -----------
    regime : str
        Current market regime
    
    Returns:
    --------
    Dict[str, float]
        Weights for each factor
    """
    # Dynamic weights from the new implementation
    regime_weights = {
        'Strong Bull': {
            'trend': 0.5,
            'volatility': 0.2,
            'exhaustion': 0.3
        },
        'Weak Bull': {
            'trend': 0.4,
            'volatility': 0.3,
            'exhaustion': 0.3
        },
        'Neutral': {
            'trend': 0.3,
            'volatility': 0.4,
            'exhaustion': 0.3
        },
        'Weak Bear': {
            'trend': 0.3,
            'volatility': 0.3,
            'exhaustion': 0.4
        },
        'Strong Bear': {
            'trend': 0.2,
            'volatility': 0.3,
            'exhaustion': 0.5
        }
    }
    
    return regime_weights.get(regime, regime_weights['Neutral'])


def calculate_dynamic_mss(df: pd.DataFrame, lookback: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calculate Market State Score with dynamic regime-based weights
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
    lookback : int
        Lookback period for calculations
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        DataFrame with factors and dynamic MSS, Series with regime classifications
    """
    # Calculate factors
    trend = calculate_trend_factor(df, lookback)
    volatility = calculate_volatility_factor(df, lookback)
    exhaustion = calculate_exhaustion_factor(df, lookback)
    
    # Normalize volatility to -100 to 100
    volatility_norm = 100 - (volatility * 2)
    volatility_norm = np.clip(volatility_norm, -100, 100)
    
    # First pass: Calculate static MSS to determine initial regimes
    static_weights = {'trend': 0.4, 'volatility': 0.3, 'exhaustion': 0.3}
    mss_static = (static_weights['trend'] * trend + 
                  static_weights['volatility'] * volatility_norm + 
                  static_weights['exhaustion'] * exhaustion)
    
    # Classify initial regimes
    regimes_static = pd.Series(index=df.index, dtype='object')
    regimes_static[mss_static > 50] = 'Strong Bull'
    regimes_static[(mss_static > 20) & (mss_static <= 50)] = 'Weak Bull'
    regimes_static[(mss_static >= -20) & (mss_static <= 20)] = 'Neutral'
    regimes_static[(mss_static > -50) & (mss_static < -20)] = 'Weak Bear'
    regimes_static[mss_static <= -50] = 'Strong Bear'
    
    # Second pass: Calculate dynamic MSS using regime-specific weights
    mss_dynamic = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if pd.isna(trend.iloc[i]) or pd.isna(regimes_static.iloc[i]):
            continue
            
        regime = regimes_static.iloc[i]
        weights = get_dynamic_weights(regime)
        
        mss_dynamic.iloc[i] = (
            weights['trend'] * trend.iloc[i] + 
            weights['volatility'] * volatility_norm.iloc[i] + 
            weights['exhaustion'] * exhaustion.iloc[i]
        )
    
    # Final regime classification based on dynamic MSS
    regimes = pd.Series(index=df.index, dtype='object')
    regimes[mss_dynamic > 50] = 'Strong Bull'
    regimes[(mss_dynamic > 20) & (mss_dynamic <= 50)] = 'Weak Bull'
    regimes[(mss_dynamic >= -20) & (mss_dynamic <= 20)] = 'Neutral'
    regimes[(mss_dynamic > -50) & (mss_dynamic < -20)] = 'Weak Bear'
    regimes[mss_dynamic <= -50] = 'Strong Bear'
    
    # Create results DataFrame
    from data_processing import calculate_atr
    atr = calculate_atr(df, lookback)
    
    results = pd.DataFrame({
        'trend': trend,
        'volatility': volatility,
        'exhaustion': exhaustion,
        'volatility_norm': volatility_norm,
        'mss_static': mss_static,
        'mss': mss_dynamic,
        'regime_static': regimes_static,
        'regime': regimes,
        'atr': atr
    }, index=df.index)
    
    return results, regimes