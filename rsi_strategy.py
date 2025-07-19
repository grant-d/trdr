"""
RSI (Relative Strength Index) Strategy Implementation
Buy when RSI is oversold, sell when overbought
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

from strategy_optimization_framework import Strategy, StrategyParameters, Bar, OrderSide
import logging

logger = logging.getLogger(__name__)


@dataclass
class RSIStrategyParameters(StrategyParameters):
    """Parameters for RSI strategy"""

    rsi_period: int
    oversold_threshold: float  # Buy when RSI < this (e.g., 30)
    overbought_threshold: float  # Sell when RSI > this (e.g., 70)
    position_size: float = 0.95  # Use 95% of available capital

    def __post_init__(self):
        # Ensure valid parameters
        self.rsi_period = max(2, int(self.rsi_period))
        self.oversold_threshold = max(10.0, min(40.0, self.oversold_threshold))
        self.overbought_threshold = max(60.0, min(90.0, self.overbought_threshold))


class RSIStrategy(Strategy):
    """RSI-based trading strategy"""

    def __init__(
        self,
        min_period: int = 5,
        max_period: int = 30,
        min_oversold: float = 20.0,
        max_oversold: float = 40.0,
        min_overbought: float = 60.0,
        max_overbought: float = 80.0,
    ):
        self.min_period = min_period
        self.max_period = max_period
        self.min_oversold = min_oversold
        self.max_oversold = max_oversold
        self.min_overbought = min_overbought
        self.max_overbought = max_overbought

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds for optimization"""
        return {
            "rsi_period": (self.min_period, self.max_period),
            "oversold_threshold": (self.min_oversold, self.max_oversold),
            "overbought_threshold": (self.min_overbought, self.max_overbought),
        }

    def create_parameters(self, values: List[float]) -> RSIStrategyParameters:
        """Create parameter object from optimization values"""
        if len(values) < 3:
            raise ValueError(f"Expected 3 values, got {len(values)}")

        return RSIStrategyParameters(
            rsi_period=int(values[0]),
            oversold_threshold=values[1],
            overbought_threshold=values[2],
        )

    def calculate_rsi(self, closes: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        # Calculate price changes
        delta = closes.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(
        self, bars: List[Bar], params: RSIStrategyParameters
    ) -> List[Tuple[datetime, OrderSide, float]]:
        """Generate trading signals from RSI"""
        if len(bars) < params.rsi_period + 1:
            return []

        # Convert bars to DataFrame
        data = []
        use_transformed = False
        
        # Check if we have transformed features in hybrid mode
        if hasattr(bars[0], 'features') and bars[0].features and 'close_fd' in bars[0].features:
            use_transformed = True
            logger.info("Using transformed data for RSI calculation")
            
        for bar in bars:
            if use_transformed and bar.features and bar.features.get('close_fd') is not None:
                # Use transformed close for RSI calculation
                data.append({"timestamp": bar.timestamp, "close": bar.features['close_fd']})
            else:
                # Use raw close
                data.append({"timestamp": bar.timestamp, "close": bar.close})
        df = pd.DataFrame(data)

        # Calculate RSI
        df["rsi"] = self.calculate_rsi(df["close"], params.rsi_period)

        # Generate signals
        signals = []
        position = 0  # Track position: 1 = long, 0 = flat

        for idx, row in df.iterrows():
            if pd.isna(row["rsi"]):
                continue

            timestamp = row["timestamp"]
            rsi = row["rsi"]

            # Entry signal - buy when oversold
            if rsi < params.oversold_threshold and position == 0:
                signals.append((timestamp, OrderSide.BUY, params.position_size))
                position = 1

            # Exit signal - sell when overbought
            elif rsi > params.overbought_threshold and position == 1:
                signals.append((timestamp, OrderSide.SELL, params.position_size))
                position = 0

        return signals


class BollingerBandsStrategy(Strategy):
    """Bollinger Bands mean reversion strategy"""

    def __init__(
        self,
        min_period: int = 10,
        max_period: int = 50,
        min_std_dev: float = 1.5,
        max_std_dev: float = 3.0,
    ):
        self.min_period = min_period
        self.max_period = max_period
        self.min_std_dev = min_std_dev
        self.max_std_dev = max_std_dev

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds for optimization"""
        return {
            "bb_period": (self.min_period, self.max_period),
            "std_dev_multiplier": (self.min_std_dev, self.max_std_dev),
        }

    def create_parameters(self, values: List[float]) -> "BBStrategyParameters":
        """Create parameter object from optimization values"""
        if len(values) < 2:
            raise ValueError(f"Expected 2 values, got {len(values)}")

        return BBStrategyParameters(
            bb_period=int(values[0]), std_dev_multiplier=values[1]
        )

    def generate_signals(
        self, bars: List[Bar], params: "BBStrategyParameters"
    ) -> List[Tuple[datetime, OrderSide, float]]:
        """Generate trading signals from Bollinger Bands"""
        if len(bars) < params.bb_period:
            return []

        # Convert bars to DataFrame
        data = []
        for bar in bars:
            data.append({"timestamp": bar.timestamp, "close": bar.close})
        df = pd.DataFrame(data)

        # Calculate Bollinger Bands
        df["sma"] = df["close"].rolling(window=params.bb_period).mean()
        df["std"] = df["close"].rolling(window=params.bb_period).std()
        df["upper_band"] = df["sma"] + (df["std"] * params.std_dev_multiplier)
        df["lower_band"] = df["sma"] - (df["std"] * params.std_dev_multiplier)

        # Generate signals
        signals = []
        position = 0  # Track position: 1 = long, 0 = flat

        for idx, row in df.iterrows():
            if pd.isna(row["upper_band"]):
                continue

            timestamp = row["timestamp"]
            close = row["close"]

            # Buy when price touches lower band
            if close <= row["lower_band"] and position == 0:
                signals.append((timestamp, OrderSide.BUY, params.position_size))
                position = 1

            # Sell when price touches upper band or returns to SMA
            elif position == 1 and (close >= row["upper_band"] or close >= row["sma"]):
                signals.append((timestamp, OrderSide.SELL, params.position_size))
                position = 0

        return signals


@dataclass
class BBStrategyParameters(StrategyParameters):
    """Parameters for Bollinger Bands strategy"""

    bb_period: int
    std_dev_multiplier: float
    position_size: float = 0.95

    def __post_init__(self):
        self.bb_period = max(2, int(self.bb_period))
        self.std_dev_multiplier = max(0.5, min(5.0, self.std_dev_multiplier))
