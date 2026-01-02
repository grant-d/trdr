"""MACD crossover strategy - TEMPLATE for creating new strategies.

To create a new strategy:
1. Copy this folder to src/trdr/strategy/your_strategy/
2. Rename MACDConfig → YourConfig, MACDStrategy → YourStrategy
3. Update __init__.py exports
4. Implement your signal logic in generate_signal()
5. Add tests in test_strategy.py
6. Register in strategy/__init__.py
"""

from dataclasses import dataclass

import numpy as np

from ...data.market import Bar
from ..types import Position, Signal, SignalAction
from ..base_strategy import BaseStrategy, StrategyConfig


# =============================================================================
# STEP 1: Define your config
# =============================================================================
# Extend StrategyConfig which provides: symbol, timeframe
# Add strategy-specific parameters with defaults


@dataclass
class MACDConfig(StrategyConfig):
    """Configuration for MACD crossover strategy.

    Args:
        symbol: Trading symbol (inherited from StrategyConfig)
        timeframe: Bar timeframe (inherited from StrategyConfig)
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
        stop_loss_pct: Stop loss as percentage of entry price
    """

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    stop_loss_pct: float = 0.02  # 2% stop


# =============================================================================
# STEP 2: Define helper functions (optional)
# =============================================================================
# Keep indicator calculations as module-level functions for reusability


def _ema(values: list[float], period: int) -> list[float]:
    """Calculate Exponential Moving Average."""
    if len(values) < period:
        return [0.0] * len(values)

    alpha = 2 / (period + 1)
    ema = [0.0] * len(values)

    # Start with SMA for first period
    ema[period - 1] = np.mean(values[:period])

    # Calculate EMA for rest
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

    return ema


# =============================================================================
# STEP 3: Implement your strategy class
# =============================================================================
# Extend BaseStrategy and implement generate_signal()


class MACDStrategy(BaseStrategy):
    """Simple MACD crossover strategy.

    Entry: MACD line crosses above signal line
    Exit: MACD line crosses below signal line OR stop loss hit

    This is a simple example - real strategies should have:
    - More sophisticated entry/exit logic
    - Multiple confirmation signals
    - Dynamic position sizing based on confidence
    """

    def __init__(self, config: MACDConfig):
        """Initialize with config.

        Always call super().__init__(config) first.
        Store typed config for IDE autocomplete.
        """
        super().__init__(config)
        self.config: MACDConfig = config  # Type hint for autocomplete

    @property
    def name(self) -> str:
        """Strategy name for logging/display."""
        return "MACD"

    def generate_signal(
        self,
        bars: list[Bar],
        position: Position | None,
    ) -> Signal:
        """Generate trading signal from bars.

        Args:
            bars: Historical OHLCV bars (oldest first, newest last)
            position: Current position or None if flat

        Returns:
            Signal with action (BUY/CLOSE/HOLD), price, confidence, reason

        Signal generation pattern:
        1. Check minimum data requirements
        2. If in position: check exits (stop loss, take profit, exit signal)
        3. If flat: check entry conditions
        4. Default to HOLD
        """
        # ---------------------------------------------------------------------
        # 1. Check minimum data
        # ---------------------------------------------------------------------
        min_bars = self.config.slow_period + self.config.signal_period + 1
        if len(bars) < min_bars:
            return Signal(
                action=SignalAction.HOLD,
                price=bars[-1].close if bars else 0,
                confidence=0.0,
                reason="Insufficient data",
            )

        # ---------------------------------------------------------------------
        # 2. Calculate indicators
        # ---------------------------------------------------------------------
        closes = [b.close for b in bars]
        current_price = closes[-1]

        fast_ema = _ema(closes, self.config.fast_period)
        slow_ema = _ema(closes, self.config.slow_period)

        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        signal_line = _ema(macd_line, self.config.signal_period)

        macd_current = macd_line[-1]
        macd_prev = macd_line[-2]
        signal_current = signal_line[-1]
        signal_prev = signal_line[-2]

        # ---------------------------------------------------------------------
        # 3. Check exits first (if in position)
        # ---------------------------------------------------------------------
        if position and position.side == "long":
            # Stop loss - always check first
            if current_price <= position.stop_loss:
                return Signal(
                    action=SignalAction.CLOSE,
                    price=current_price,
                    confidence=1.0,
                    reason=f"Stop loss hit at {position.stop_loss:.2f}",
                )

            # Exit signal: MACD crosses below signal line
            if macd_current < signal_current and macd_prev >= signal_prev:
                return Signal(
                    action=SignalAction.CLOSE,
                    price=current_price,
                    confidence=0.8,
                    reason="MACD crossed below signal",
                )

            # No exit signal - hold
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=0.5,
                reason="Holding position",
            )

        # ---------------------------------------------------------------------
        # 4. Check entries (if no position)
        # ---------------------------------------------------------------------
        if not position:
            # Entry signal: MACD crosses above signal line
            if macd_current > signal_current and macd_prev <= signal_prev:
                stop_loss = current_price * (1 - self.config.stop_loss_pct)
                return Signal(
                    action=SignalAction.BUY,
                    price=current_price,
                    confidence=0.7,
                    reason="MACD crossed above signal",
                    stop_loss=stop_loss,
                    # take_profit=current_price * 1.05,  # Optional
                )

        # ---------------------------------------------------------------------
        # 5. Default: no action
        # ---------------------------------------------------------------------
        return Signal(
            action=SignalAction.HOLD,
            price=current_price,
            confidence=0.0,
            reason="No signal",
        )
