"""MACD crossover strategy - TEMPLATE for creating new strategies.

To create a new strategy:
1. Copy this folder to src/trdr/strategy/your_strategy/
2. Rename MACDConfig → YourConfig, MACDStrategy → YourStrategy
3. Update __init__.py exports
4. Implement your signal logic in generate_signal()
5. Add tests in test_strategy.py
6. Register in strategy/__init__.py

RuntimeContext available via self.context in generate_signal():
    self.context.drawdown         # Current drawdown %
    self.context.win_rate         # Live win rate
    self.context.equity           # Current portfolio value
    self.context.total_trades     # Completed trade count
    self.context.current_bar      # Current Bar object
    See STRATEGY_API.md for full list.

Multi-Timeframe (MTF) Example:
    Strategies can request multiple timeframes via get_data_requirements().
    Primary feed = trading timeframe. Informative feeds = context/filters.
    See get_data_requirements() below for MTF pattern.
"""

from dataclasses import dataclass

from ...data.market import Bar
from ...indicators import ema, ema_series
from ..base_strategy import BaseStrategy, StrategyConfig
from ..types import DataRequirement, Position, Signal, SignalAction


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
        htf_timeframe: Higher timeframe for trend filter (None = disable MTF)
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
        stop_loss_pct: Stop loss as percentage of entry price
    """

    htf_timeframe: str | None = None  # e.g., "4h" when trading on "1h"
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    stop_loss_pct: float = 0.02  # 2% stop


# =============================================================================
# STEP 2: Implement your strategy class
# =============================================================================
# Extend BaseStrategy and implement generate_signal()


class MACDStrategy(BaseStrategy):
    """Simple MACD crossover strategy with optional MTF trend filter.

    Entry: MACD line crosses above signal line (+ HTF trend filter if enabled)
    Exit: MACD line crosses below signal line OR stop loss hit

    This is a simple example - real strategies should have:
    - More sophisticated entry/exit logic
    - Multiple confirmation signals
    - Dynamic position sizing based on confidence

    RuntimeContext (self.context) enables adaptive behavior:
    - Pause trading during high drawdown
    - Scale position size based on win rate
    - Access live portfolio metrics (equity, P&L, ratios)

    Multi-Timeframe (MTF) Example:
    - Set htf_timeframe in config (e.g., "4h" when trading "1h")
    - HTF EMA trend filter: only take longs when HTF trend is up
    - Demonstrates how to use informative feeds for context
    """

    def __init__(self, config: MACDConfig):
        """Initialize with config.

        Always call super().__init__(config, name) first.
        Store typed config for IDE autocomplete.
        """
        super().__init__(config, name="MACD")  # Custom name, or omit for class name
        self.config: MACDConfig = config  # Type hint for autocomplete

    def get_data_requirements(self) -> list[DataRequirement]:
        """Declare data feeds for this strategy.

        Returns:
            List with primary feed, plus optional HTF informative feed.

        MTF Pattern:
        - Primary feed: trading timeframe (determines bar iteration)
        - Informative feeds: higher timeframes for trend context
        - Informative bars are forward-filled to align with primary
        """
        reqs = [
            DataRequirement(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                lookback=500,
                role="primary",
            ),
        ]

        # Add higher timeframe feed if configured
        if self.config.htf_timeframe:
            reqs.append(
                DataRequirement(
                    symbol=self.config.symbol,
                    timeframe=self.config.htf_timeframe,
                    lookback=200,  # Fewer bars needed for HTF
                    role="informative",
                )
            )

        return reqs

    def generate_signal(
        self,
        bars: dict[str, list[Bar]],
        position: Position | None,
    ) -> Signal:
        """Generate trading signal from bars.

        Args:
            bars: Dict of bars keyed by "symbol:timeframe"
            position: Current position or None if flat

        Returns:
            Signal with action (BUY/CLOSE/HOLD), price, confidence, reason

        Signal generation pattern:
        1. Extract primary bars (and HTF bars if configured)
        2. Check minimum data requirements
        3. If in position: check exits (stop loss, take profit, exit signal)
        4. If flat: check entry conditions (with optional HTF filter)
        5. Default to HOLD
        """
        # ---------------------------------------------------------------------
        # 1. Extract bars from dict
        # ---------------------------------------------------------------------
        primary_key = f"{self.config.symbol}:{self.config.timeframe}"
        primary_bars = bars[primary_key]

        # Get HTF bars if configured (for trend filter)
        htf_bars = None
        if self.config.htf_timeframe:
            htf_key = f"{self.config.symbol}:{self.config.htf_timeframe}"
            htf_bars = bars.get(htf_key)

        # ---------------------------------------------------------------------
        # 2. Check minimum data
        # ---------------------------------------------------------------------
        min_bars = self.config.slow_period + self.config.signal_period + 1
        if len(primary_bars) < min_bars:
            return Signal(
                action=SignalAction.HOLD,
                price=primary_bars[-1].close if primary_bars else 0,
                confidence=0.0,
                reason="Insufficient data",
            )

        # ---------------------------------------------------------------------
        # 2b. RuntimeContext: Adaptive behavior based on portfolio state
        # ---------------------------------------------------------------------
        # Pause trading during high drawdown (optional)
        if self.context.drawdown > 0.15:
            return Signal(
                action=SignalAction.HOLD,
                price=primary_bars[-1].close,
                confidence=0.0,
                reason=f"Paused: drawdown {self.context.drawdown:.1%} > 15%",
            )

        # ---------------------------------------------------------------------
        # 3. Calculate indicators on PRIMARY timeframe
        # ---------------------------------------------------------------------
        closes = [b.close for b in primary_bars]
        current_price = closes[-1]

        fast_ema = ema_series(closes, self.config.fast_period)
        slow_ema = ema_series(closes, self.config.slow_period)

        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        signal_line = ema_series(macd_line, self.config.signal_period)

        macd_current = macd_line[-1]
        macd_prev = macd_line[-2]
        signal_current = signal_line[-1]
        signal_prev = signal_line[-2]

        # ---------------------------------------------------------------------
        # 3b. MTF: Calculate HTF trend filter (if configured)
        # ---------------------------------------------------------------------
        # HTF trend: only take longs when HTF EMA is trending up
        htf_trend_bullish = True  # Default: no filter
        if htf_bars and len(htf_bars) >= 50:
            # Use most recent HTF bar (already aligned to primary)
            htf_ema_val = ema(htf_bars, period=20)
            htf_current_price = htf_bars[-1].close if htf_bars[-1] else current_price
            htf_trend_bullish = htf_current_price > htf_ema_val

        # ---------------------------------------------------------------------
        # 4. Check exits first (if in position)
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
        # 5. Check entries (if no position)
        # ---------------------------------------------------------------------
        if not position:
            # Entry signal: MACD crosses above signal line
            macd_cross_up = macd_current > signal_current and macd_prev <= signal_prev

            # MTF filter: only enter if HTF trend is bullish (or no MTF configured)
            if macd_cross_up and htf_trend_bullish:
                stop_loss = current_price * (1 - self.config.stop_loss_pct)

                # RuntimeContext: Scale position based on live win rate
                # Reduce size after losing streak, increase after winning
                if self.context.total_trades >= 5:
                    size = 0.5 if self.context.win_rate < 0.4 else 1.0
                else:
                    size = 1.0  # Full size until enough trades for stats

                reason = "MACD crossed above signal"
                if self.config.htf_timeframe:
                    reason += f" (HTF {self.config.htf_timeframe} trend up)"

                return Signal(
                    action=SignalAction.BUY,
                    price=current_price,
                    confidence=0.7,
                    reason=reason,
                    stop_loss=stop_loss,
                    # take_profit=current_price * 1.05,  # Optional
                    position_size_pct=size,
                )

            # MACD crossed but HTF filter blocked
            if macd_cross_up and not htf_trend_bullish:
                return Signal(
                    action=SignalAction.HOLD,
                    price=current_price,
                    confidence=0.3,
                    reason=f"MACD cross blocked by HTF {self.config.htf_timeframe} downtrend",
                )

        # ---------------------------------------------------------------------
        # 6. Default: no action
        # ---------------------------------------------------------------------
        return Signal(
            action=SignalAction.HOLD,
            price=current_price,
            confidence=0.0,
            reason="No signal",
        )
