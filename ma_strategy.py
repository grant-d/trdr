"""
Moving Average Strategy Implementation
Simple MA crossover strategy as an example
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
class MAStrategyParameters(StrategyParameters):
    """Parameters for MA crossover strategy"""

    fast_period: int
    slow_period: int
    position_size: float = 0.03  # Use 95% of available capital

    def __post_init__(self):
        # Ensure fast < slow
        if self.fast_period >= self.slow_period:
            self.fast_period, self.slow_period = min(
                self.fast_period, self.slow_period
            ), max(self.fast_period, self.slow_period)
        # Ensure minimum periods
        self.fast_period = max(2, int(self.fast_period))
        self.slow_period = max(self.fast_period + 1, int(self.slow_period))


class MovingAverageStrategy(Strategy):
    """Simple MA crossover strategy"""

    def __init__(
        self,
        min_fast: int = 5,
        max_fast: int = 50,
        min_slow: int = 20,
        max_slow: int = 200,
    ):
        self.min_fast = min_fast
        self.max_fast = max_fast
        self.min_slow = min_slow
        self.max_slow = max_slow

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds for optimization"""
        return {
            "fast_period": (self.min_fast, self.max_fast),
            "slow_period": (self.min_slow, self.max_slow),
        }

    def create_parameters(self, values: List[float]) -> MAStrategyParameters:
        """Create parameter object from optimization values"""
        # Ensure we have the right values
        if len(values) < 2:
            raise ValueError(f"Expected 2 values, got {len(values)}")

        return MAStrategyParameters(
            fast_period=int(values[0]), slow_period=int(values[1])
        )

    def generate_signals(
        self, bars: List[Bar], params: MAStrategyParameters
    ) -> List[Tuple[datetime, OrderSide, float]]:
        """Generate trading signals from MA crossover"""
        if len(bars) < params.slow_period:
            return []

        # Convert bars to DataFrame for easier calculation
        data = []
        use_transformed = False
        
        # Check if we have transformed features in hybrid mode
        if hasattr(bars[0], 'features') and bars[0].features and 'close_fd' in bars[0].features:
            use_transformed = True
            trade_date = bars[-1].timestamp.strftime('%Y-%m-%d %H:%M:%S') if bars else "Unknown"
            logger.info(f"{trade_date} - Using transformed data for MA signal generation")
            
        for bar in bars:
            if use_transformed and bar.features and bar.features.get('close_fd') is not None:
                # Use transformed close for signal generation
                data.append({"timestamp": bar.timestamp, "close": bar.features['close_fd']})
            else:
                # Use raw close
                data.append({"timestamp": bar.timestamp, "close": bar.close})
        df = pd.DataFrame(data)

        # Calculate moving averages
        df["ma_fast"] = df["close"].rolling(window=params.fast_period).mean()
        df["ma_slow"] = df["close"].rolling(window=params.slow_period).mean()

        # Generate signals
        df["signal"] = 0
        df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1
        df.loc[df["ma_fast"] < df["ma_slow"], "signal"] = -1

        # Find signal changes
        df["signal_change"] = df["signal"].diff()

        # Generate orders
        signals = []
        position = 0  # Track position: 1 = long, 0 = flat

        for idx, row in df.iterrows():
            if pd.isna(row["signal_change"]) or row["signal_change"] == 0:
                continue

            timestamp = row["timestamp"]

            # Entry signals
            if row["signal_change"] > 0 and position == 0:  # Buy signal
                # Note: quantity will be calculated by the evaluator based on available capital
                signals.append((timestamp, OrderSide.BUY, params.position_size))
                position = 1
            elif row["signal_change"] < 0 and position == 1:  # Sell signal
                # Note: quantity will be calculated by the evaluator based on current position
                signals.append((timestamp, OrderSide.SELL, params.position_size))
                position = 0

        return signals


class DualMAStrategy(Strategy):
    """Dual MA strategy with entry and exit MAs"""

    def __init__(
        self,
        min_entry_fast: int = 5,
        max_entry_fast: int = 20,
        min_entry_slow: int = 20,
        max_entry_slow: int = 50,
        min_exit_period: int = 5,
        max_exit_period: int = 30,
    ):
        self.min_entry_fast = min_entry_fast
        self.max_entry_fast = max_entry_fast
        self.min_entry_slow = min_entry_slow
        self.max_entry_slow = max_entry_slow
        self.min_exit_period = min_exit_period
        self.max_exit_period = max_exit_period

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds for optimization"""
        return {
            "entry_fast": (self.min_entry_fast, self.max_entry_fast),
            "entry_slow": (self.min_entry_slow, self.max_entry_slow),
            "exit_period": (self.min_exit_period, self.max_exit_period),
        }

    def create_parameters(self, values: List[float]) -> "DualMAParameters":
        """Create parameter object from optimization values"""
        return DualMAParameters(
            entry_fast=int(values[0]),
            entry_slow=int(values[1]),
            exit_period=int(values[2]),
        )

    def generate_signals(
        self, bars: List[Bar], params: "DualMAParameters"
    ) -> List[Tuple[datetime, OrderSide, float]]:
        """Generate trading signals"""
        max_period = max(params.entry_slow, params.exit_period)
        if len(bars) < max_period:
            return []

        # Convert to DataFrame
        data = []
        for bar in bars:
            data.append({"timestamp": bar.timestamp, "close": bar.close})
        df = pd.DataFrame(data)

        # Calculate MAs
        df["entry_fast"] = df["close"].rolling(window=params.entry_fast).mean()
        df["entry_slow"] = df["close"].rolling(window=params.entry_slow).mean()
        df["exit_ma"] = df["close"].rolling(window=params.exit_period).mean()

        # Generate signals
        signals = []
        position = 0

        for idx, row in df.iterrows():
            if (
                pd.isna(row["entry_fast"])
                or pd.isna(row["entry_slow"])
                or pd.isna(row["exit_ma"])
            ):
                continue

            timestamp = row["timestamp"]
            close = row["close"]

            # Entry signal
            if position == 0 and row["entry_fast"] > row["entry_slow"]:
                signals.append((timestamp, OrderSide.BUY, params.position_size))
                position = 1

            # Exit signal
            elif position == 1 and close < row["exit_ma"]:
                signals.append((timestamp, OrderSide.SELL, params.position_size))
                position = 0

        return signals


@dataclass
class DualMAParameters(StrategyParameters):
    """Parameters for dual MA strategy"""

    entry_fast: int
    entry_slow: int
    exit_period: int
    position_size: float = 0.95

    def __post_init__(self):
        self.entry_fast = max(2, int(self.entry_fast))
        self.entry_slow = max(self.entry_fast + 1, int(self.entry_slow))
        self.exit_period = max(2, int(self.exit_period))
