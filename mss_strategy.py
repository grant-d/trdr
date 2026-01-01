"""
Market State Score (MSS) Strategy Implementation
Based on Helios Trader PRD - combines trend, volatility, and exhaustion factors
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

from strategy_optimization_framework import (
    Strategy, StrategyParameters, Bar, OrderSide
)
from indicators import (
    calculate_trend_factor, calculate_volatility_factor, 
    calculate_exhaustion_factor, calculate_atr
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class MSSStrategyParameters(StrategyParameters):
    """Parameters for MSS strategy - these will be optimized by GA"""
    # Lookback periods for factor calculations
    trend_lookback: int
    volatility_lookback: int
    exhaustion_lookback: int
    
    # Weights for combining factors into MSS
    trend_weight: float
    volatility_weight: float
    exhaustion_weight: float
    
    # MSS thresholds for trading decisions
    strong_bull_threshold: float  # Above this = enter/hold long
    weak_bull_threshold: float    # Between weak and strong = hold only
    neutral_upper: float          # Between neutral bounds = exit all
    neutral_lower: float          
    weak_bear_threshold: float    # Between weak and neutral = hold shorts only
    strong_bear_threshold: float  # Below this = enter/hold short
    
    # Risk management
    atr_multiplier_strong: float  # ATR multiplier for stop loss in strong trends
    atr_multiplier_weak: float    # ATR multiplier for stop loss in weak trends
    
    # Position sizing
    position_size: float = 0.03
    
    def __post_init__(self):
        # Ensure valid parameters
        self.trend_lookback = max(5, int(self.trend_lookback))
        self.volatility_lookback = max(5, int(self.volatility_lookback))
        self.exhaustion_lookback = max(5, int(self.exhaustion_lookback))
        
        # Normalize weights to sum to 1
        total_weight = self.trend_weight + self.volatility_weight + self.exhaustion_weight
        if total_weight > 0:
            self.trend_weight /= total_weight
            self.volatility_weight /= total_weight
            self.exhaustion_weight /= total_weight
        
        # Ensure thresholds are in correct order
        self.strong_bull_threshold = max(30.0, self.strong_bull_threshold)
        self.weak_bull_threshold = min(self.strong_bull_threshold - 10, self.weak_bull_threshold)
        self.neutral_upper = min(self.weak_bull_threshold - 10, self.neutral_upper)
        self.neutral_lower = max(self.weak_bear_threshold + 10, self.neutral_lower)
        self.weak_bear_threshold = max(self.strong_bear_threshold + 10, self.weak_bear_threshold)
        self.strong_bear_threshold = min(-30.0, self.strong_bear_threshold)
        
        # Ensure positive ATR multipliers
        self.atr_multiplier_strong = max(0.5, self.atr_multiplier_strong)
        self.atr_multiplier_weak = max(0.5, self.atr_multiplier_weak)


class MSSStrategy(Strategy):
    """Market State Score based trading strategy"""
    
    def __init__(self):
        # Define ranges for GA optimization
        # Lookback periods
        self.min_lookback = 10
        self.max_lookback = 50
        
        # Factor weights (will be normalized)
        self.min_weight = 0.1
        self.max_weight = 0.6
        
        # MSS thresholds
        self.min_strong_bull = 40.0
        self.max_strong_bull = 80.0
        self.min_weak_bull = 10.0
        self.max_weak_bull = 40.0
        self.min_neutral_upper = -10.0
        self.max_neutral_upper = 20.0
        self.min_neutral_lower = -20.0
        self.max_neutral_lower = 10.0
        self.min_weak_bear = -40.0
        self.max_weak_bear = -10.0
        self.min_strong_bear = -80.0
        self.max_strong_bear = -40.0
        
        # ATR multipliers
        self.min_atr_mult = 0.5
        self.max_atr_mult = 3.0
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds for optimization"""
        return {
            # Lookback periods
            'trend_lookback': (self.min_lookback, self.max_lookback),
            'volatility_lookback': (self.min_lookback, self.max_lookback),
            'exhaustion_lookback': (self.min_lookback, self.max_lookback),
            
            # Factor weights (will be normalized)
            'trend_weight': (self.min_weight, self.max_weight),
            'volatility_weight': (self.min_weight, self.max_weight),
            'exhaustion_weight': (self.min_weight, self.max_weight),
            
            # MSS thresholds
            'strong_bull_threshold': (self.min_strong_bull, self.max_strong_bull),
            'weak_bull_threshold': (self.min_weak_bull, self.max_weak_bull),
            'neutral_upper': (self.min_neutral_upper, self.max_neutral_upper),
            'neutral_lower': (self.min_neutral_lower, self.max_neutral_lower),
            'weak_bear_threshold': (self.min_weak_bear, self.max_weak_bear),
            'strong_bear_threshold': (self.min_strong_bear, self.max_strong_bear),
            
            # Risk management
            'atr_multiplier_strong': (self.min_atr_mult, self.max_atr_mult),
            'atr_multiplier_weak': (self.min_atr_mult, self.max_atr_mult)
        }
    
    def create_parameters(self, values: List[float]) -> MSSStrategyParameters:
        """Create parameter object from optimization values"""
        if len(values) < 14:
            raise ValueError(f"Expected 14 values, got {len(values)}")
        
        # Get threshold values and sort them to ensure valid ordering
        thresholds = sorted([values[6], values[7], values[8], values[9], values[10], values[11]], reverse=True)
        
        # Assign thresholds in proper order (highest to lowest)
        # strong_bull > weak_bull > neutral_upper > neutral_lower > weak_bear > strong_bear
        strong_bull = thresholds[0]
        weak_bull = thresholds[1]
        neutral_upper = thresholds[2]
        neutral_lower = thresholds[3]
        weak_bear = thresholds[4]
        strong_bear = thresholds[5]
        
        # Ensure minimum separation between thresholds
        min_separation = 5.0
        if weak_bull > strong_bull - min_separation:
            weak_bull = strong_bull - min_separation
        if neutral_upper > weak_bull - min_separation:
            neutral_upper = weak_bull - min_separation
        if neutral_lower > neutral_upper - min_separation:
            neutral_lower = neutral_upper - min_separation
        if weak_bear > neutral_lower - min_separation:
            weak_bear = neutral_lower - min_separation
        if strong_bear > weak_bear - min_separation:
            strong_bear = weak_bear - min_separation
        
        return MSSStrategyParameters(
            # Lookback periods
            trend_lookback=int(values[0]),
            volatility_lookback=int(values[1]),
            exhaustion_lookback=int(values[2]),
            
            # Factor weights
            trend_weight=values[3],
            volatility_weight=values[4],
            exhaustion_weight=values[5],
            
            # MSS thresholds - properly ordered
            strong_bull_threshold=strong_bull,
            weak_bull_threshold=weak_bull,
            neutral_upper=neutral_upper,
            neutral_lower=neutral_lower,
            weak_bear_threshold=weak_bear,
            strong_bear_threshold=strong_bear,
            
            # Risk management
            atr_multiplier_strong=values[12],
            atr_multiplier_weak=values[13]
        )
    
    def calculate_mss(self, df: pd.DataFrame, params: MSSStrategyParameters) -> pd.Series:
        """Calculate Market State Score"""
        # Calculate factors with their specific lookbacks
        trend = calculate_trend_factor(df, params.trend_lookback)
        volatility = calculate_volatility_factor(df, params.volatility_lookback)
        exhaustion = calculate_exhaustion_factor(df, params.exhaustion_lookback)
        
        # Normalize volatility to -100 to 100 (high volatility is negative)
        volatility_norm = 100 - (volatility * 2)
        volatility_norm = np.clip(volatility_norm, -100, 100)
        
        # Calculate MSS as weighted sum
        mss = (
            params.trend_weight * trend +
            params.volatility_weight * volatility_norm +
            params.exhaustion_weight * exhaustion
        )
        
        return mss
    
    def classify_market_state(self, mss: float, params: MSSStrategyParameters) -> str:
        """Classify market state based on MSS value"""
        if mss > params.strong_bull_threshold:
            return "STRONG_BULL"
        elif mss > params.weak_bull_threshold:
            return "WEAK_BULL"
        elif mss > params.neutral_upper:
            return "NEUTRAL_UPPER"
        elif mss > params.neutral_lower:
            return "NEUTRAL_LOWER"
        elif mss > params.weak_bear_threshold:
            return "WEAK_BEAR"
        else:
            return "STRONG_BEAR"
    
    def generate_signals(self, bars: List[Bar], params: MSSStrategyParameters) -> List[Tuple[datetime, OrderSide, float]]:
        """Generate trading signals from MSS"""
        # Need enough bars for all lookback periods
        min_bars = max(params.trend_lookback, params.volatility_lookback, params.exhaustion_lookback)
        if len(bars) < min_bars + 1:
            return []
        
        # Convert bars to DataFrame
        data = []
        use_transformed = False
        
        # Check if we have transformed features in hybrid mode
        if hasattr(bars[0], 'features') and bars[0].features and 'close_fd' in bars[0].features:
            use_transformed = True
            trade_date = bars[-1].timestamp.strftime('%Y-%m-%d %H:%M:%S') if bars else "Unknown"
            logger.info(f"{trade_date} - Using transformed data for MSS calculations")
            
        for bar in bars:
            if use_transformed and bar.features:
                # Use transformed data for MSS calculations
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.features.get('open_fd', bar.open),
                    'high': bar.features.get('high_fd', bar.high),
                    'low': bar.features.get('low_fd', bar.low),
                    'close': bar.features.get('close_fd', bar.close),
                    'volume': bar.features.get('volume_lr', bar.volume)
                })
            else:
                # Use raw data
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
        df = pd.DataFrame(data)
        
        # Calculate MSS
        mss = self.calculate_mss(df, params)
        
        # Calculate ATR for stop loss calculations
        atr_lookback = max(params.volatility_lookback, 14)
        atr = calculate_atr(df, atr_lookback)
        
        # Generate signals based on MSS and market state
        signals = []
        position = 0  # 1 = long, -1 = short, 0 = flat
        
        for idx in range(min_bars, len(df)):
            if pd.isna(mss.iloc[idx]):
                continue
            
            timestamp = df.iloc[idx]['timestamp']
            current_mss = mss.iloc[idx]
            market_state = self.classify_market_state(current_mss, params)
            
            # Trading logic based on Helios Trader Action Matrix
            if market_state == "STRONG_BULL":
                # Enter/Hold Long
                if position <= 0:
                    if position < 0:  # Close short first
                        signals.append((timestamp, OrderSide.BUY, params.position_size))
                    signals.append((timestamp, OrderSide.BUY, params.position_size))
                    position = 1
                    
            elif market_state == "WEAK_BULL":
                # Hold Longs ONLY - no new entries
                if position < 0:  # Exit shorts
                    signals.append((timestamp, OrderSide.BUY, params.position_size))
                    position = 0
                    
            elif market_state in ["NEUTRAL_UPPER", "NEUTRAL_LOWER"]:
                # EXIT ALL POSITIONS
                if position > 0:
                    signals.append((timestamp, OrderSide.SELL, params.position_size))
                elif position < 0:
                    signals.append((timestamp, OrderSide.BUY, params.position_size))
                position = 0
                
            elif market_state == "WEAK_BEAR":
                # Hold Shorts ONLY - no new entries
                if position > 0:  # Exit longs
                    signals.append((timestamp, OrderSide.SELL, params.position_size))
                    position = 0
                    
            elif market_state == "STRONG_BEAR":
                # Enter/Hold Short
                if position >= 0:
                    if position > 0:  # Close long first
                        signals.append((timestamp, OrderSide.SELL, params.position_size))
                    signals.append((timestamp, OrderSide.SELL, params.position_size))
                    position = -1
        
        return signals


class SimpleMSSStrategy(MSSStrategy):
    """Simplified MSS strategy with fewer parameters for easier optimization"""
    
    def __init__(self):
        super().__init__()
        # Use single lookback for all factors
        self.min_lookback = 10
        self.max_lookback = 50
        
        # Fixed symmetric thresholds
        self.min_threshold = 20.0
        self.max_threshold = 60.0
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds for optimization"""
        return {
            # Single lookback for all factors
            'lookback': (self.min_lookback, self.max_lookback),
            
            # Factor weights (will be normalized)
            'trend_weight': (self.min_weight, self.max_weight),
            'volatility_weight': (self.min_weight, self.max_weight),
            'exhaustion_weight': (self.min_weight, self.max_weight),
            
            # MSS thresholds - all separate
            'strong_bull_threshold': (self.min_strong_bull, self.max_strong_bull),
            'weak_bull_threshold': (self.min_weak_bull, self.max_weak_bull),
            'neutral_upper': (self.min_neutral_upper, self.max_neutral_upper),
            'neutral_lower': (self.min_neutral_lower, self.max_neutral_lower),
            'weak_bear_threshold': (self.min_weak_bear, self.max_weak_bear),
            'strong_bear_threshold': (self.min_strong_bear, self.max_strong_bear),
            
            # ATR multiplier
            'atr_multiplier': (self.min_atr_mult, self.max_atr_mult)
        }
    
    def create_parameters(self, values: List[float]) -> MSSStrategyParameters:
        """Create parameter object from optimization values"""
        if len(values) < 11:
            raise ValueError(f"Expected 11 values, got {len(values)}")
        
        lookback = int(values[0])
        
        # Get threshold values and sort them to ensure valid ordering
        thresholds = sorted([values[4], values[5], values[6], values[7], values[8], values[9]], reverse=True)
        
        # Assign thresholds in proper order (highest to lowest)
        # strong_bull > weak_bull > neutral_upper > neutral_lower > weak_bear > strong_bear
        strong_bull = thresholds[0]
        weak_bull = thresholds[1]
        neutral_upper = thresholds[2]
        neutral_lower = thresholds[3]
        weak_bear = thresholds[4]
        strong_bear = thresholds[5]
        
        # Ensure minimum separation between thresholds
        min_separation = 5.0
        if weak_bull > strong_bull - min_separation:
            weak_bull = strong_bull - min_separation
        if neutral_upper > weak_bull - min_separation:
            neutral_upper = weak_bull - min_separation
        if neutral_lower > neutral_upper - min_separation:
            neutral_lower = neutral_upper - min_separation
        if weak_bear > neutral_lower - min_separation:
            weak_bear = neutral_lower - min_separation
        if strong_bear > weak_bear - min_separation:
            strong_bear = weak_bear - min_separation
            
        return MSSStrategyParameters(
            # Use same lookback for all
            trend_lookback=lookback,
            volatility_lookback=lookback,
            exhaustion_lookback=lookback,
            
            # Factor weights
            trend_weight=values[1],
            volatility_weight=values[2],
            exhaustion_weight=values[3],
            
            # MSS thresholds - properly ordered
            strong_bull_threshold=strong_bull,
            weak_bull_threshold=weak_bull,
            neutral_upper=neutral_upper,
            neutral_lower=neutral_lower,
            weak_bear_threshold=weak_bear,
            strong_bear_threshold=strong_bear,
            
            # Same ATR multiplier for both
            atr_multiplier_strong=values[10],
            atr_multiplier_weak=values[10] * 0.5  # Half for weak trends
        )