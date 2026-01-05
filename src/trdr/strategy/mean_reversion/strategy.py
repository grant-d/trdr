"""Adaptive Regime Strategy.

Regime-aware strategy that switches between momentum and mean reversion.

Key insight: Mean reversion fails in trending markets. This strategy:
1. Detects market regime (trending vs ranging) using ADX and slope
2. Trends: Follow momentum (buy strength, not weakness)
3. Ranges: Mean revert (buy oversold, sell overbought)

Research basis:
- BTC daily returns show AR(1) coefficient of -0.1203 (reversal tendency)
- Mean reversion strategies achieved Sharpe ~2.3 post-2021 IN RANGING MARKETS
- Trend-following essential during strong trends
- Turn-of-month calendar effects documented in equity markets
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from ...data import Bar
from ...indicators import atr, ema
from ..base_strategy import BaseStrategy, StrategyConfig
from ..types import DataRequirement, Position, Signal, SignalAction

if TYPE_CHECKING:
    pass


@dataclass
class MeanReversionConfig(StrategyConfig):
    """Configuration for Adaptive Regime strategy.

    Tunable parameters for regime detection, entry/exit, and risk management.
    """

    # Regime detection
    trend_ema_fast: int = 10
    trend_ema_slow: int = 30
    trend_slope_threshold: float = 0.025  # 2.5% slope = trending (conservative)
    regime_lookback: int = 20

    # Breakout/momentum settings - ITERATION 1 OPTIMAL (FINAL)
    breakout_period: int = 10  # Proven optimal for AAPL
    volume_multiplier: float = 1.2  # Proven optimal - quality filter
    trend_ema: int = 50  # Not used
    enable_trend_filter: bool = False  # Disabled
    gap_filter_pct: float = 0.10  # Disabled (raised to 10% = almost never triggers)
    lookback_period: int = 20  # For stats
    zscore_entry: float = 2.0  # Legacy (not used in breakout mode)
    zscore_exit: float = 0.0  # Legacy

    # Momentum settings (trending regime)
    momentum_ema: int = 10
    momentum_threshold: float = 0.01  # 1% above EMA = momentum entry

    # Consecutive down days for mean reversion entry
    consecutive_down_days: int = 4  # Legacy (not used in breakout mode)

    # Calendar effects: disable to reduce noise
    use_calendar: bool = False
    calendar_days_before_month_end: int = 2
    calendar_days_after_month_start: int = 2

    # Risk management - ITERATION 1 OPTIMAL
    atr_period: int = 14
    stop_loss_atr_mult: float = 2.0  # Proven optimal - not too tight
    trailing_stop_atr_mult: float = 3.0  # Proven optimal - let winners run
    take_profit_atr_mult: float = 0.0  # No fixed target - trail only
    max_holding_days: int = 60  # Proven optimal - let trends develop

    # Position sizing - PROVEN OPTIMAL (Iteration 8)
    base_position_pct: float = 1.0  # Max capital, no leverage (was 0.98)

    # Debug
    debug: bool = False

    # Calculated internally
    _min_bars: int = field(init=False)

    def __post_init__(self) -> None:
        """Calculate derived values."""
        self._min_bars = max(self.trend_ema_slow, self.lookback_period, self.atr_period) + 10


class MeanReversionStrategy(BaseStrategy):
    """Adaptive regime strategy for daily BTC.

    Detects market regime and switches between:
    - Trending: Momentum entries (buy strength, ride the wave)
    - Ranging: Mean reversion (buy oversold, sell overbought)

    Designed for higher trade frequency and win rate by adapting to conditions.
    """

    def __init__(
        self,
        config: MeanReversionConfig,
        name: str | None = None,
    ):
        """Initialize strategy.

        Args:
            config: Strategy configuration
            name: Optional friendly name
        """
        super().__init__(config, name or "AdaptiveRegime")
        self.config: MeanReversionConfig = config
        self._entry_bar_idx: int | None = None
        self._entry_zscore: float = 0.0
        self._current_regime: str = "unknown"

    def reset(self) -> None:
        """Reset strategy state for new backtest run."""
        self._entry_bar_idx = None
        self._entry_zscore = 0.0
        self._current_regime = "unknown"

    def get_data_requirements(self) -> list[DataRequirement]:
        """Declare data feeds for this strategy."""
        return [
            DataRequirement(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                lookback=self.config.lookback,
                role="primary",
            ),
        ]

    def generate_signal(
        self,
        bars: dict[str, list[Bar]],
        position: Position | None,
    ) -> Signal:
        """Generate trading signal based on adaptive regime logic.

        Args:
            bars: Dict of bars keyed by "symbol:timeframe"
            position: Current open position or None

        Returns:
            Signal with action, stops, and targets
        """
        # Extract primary bars from dict
        primary_key = f"{self.config.symbol}:{self.config.timeframe}"
        bars = bars[primary_key]

        if len(bars) < self.config._min_bars:
            return Signal(
                action=SignalAction.HOLD,
                price=bars[-1].close if bars else 0.0,
                confidence=0.0,
                reason="warmup",
            )

        current_bar = bars[-1]
        current_price = current_bar.close

        # Calculate ATR for stops
        current_atr = atr(bars, self.config.atr_period)

        # Track bar index for holding period
        bar_idx = len(bars) - 1

        if position is not None:
            return self._handle_exit_breakout(bars, position, current_atr, bar_idx)
        else:
            return self._handle_entry_breakout(bars, current_atr, bar_idx)

    def _detect_regime(self, bars: list[Bar]) -> tuple[str, int]:
        """Detect market regime: trending or ranging.

        Returns:
            Tuple of (regime: "trending" or "ranging", trend_direction: 1 or -1 or 0)
        """
        # Calculate EMAs (ema() expects Bar objects)
        ema_fast = ema(bars, self.config.trend_ema_fast)
        ema_slow = ema(bars, self.config.trend_ema_slow)

        # Calculate slope of slow EMA (trend strength)
        lookback = min(self.config.regime_lookback, len(bars) - 1)
        if lookback < 5:
            return "ranging", 0

        ema_slow_start = ema(bars[:-lookback], self.config.trend_ema_slow)
        slope = (ema_slow - ema_slow_start) / ema_slow_start if ema_slow_start > 0 else 0

        # Determine trend direction
        if ema_fast > ema_slow:
            trend_direction = 1  # Uptrend
        elif ema_fast < ema_slow:
            trend_direction = -1  # Downtrend
        else:
            trend_direction = 0

        # Determine regime based on slope magnitude
        if abs(slope) > self.config.trend_slope_threshold:
            return "trending", trend_direction
        else:
            return "ranging", trend_direction

    def _handle_entry_breakout(
        self,
        bars: list[Bar],
        current_atr: float,
        bar_idx: int,
    ) -> Signal:
        """Check breakout entry conditions.

        Buy on strength: price breaking above recent highs with volume confirmation.
        """
        current_bar = bars[-1]
        current_price = current_bar.close

        # Calculate N-day high
        lookback_bars = bars[-self.config.breakout_period :]
        highest_close = max(b.close for b in lookback_bars[:-1])  # Exclude current

        # Breakout condition: current close > N-day high
        is_breakout = current_price > highest_close

        if not is_breakout:
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=0.0,
                reason="no_breakout",
            )

        # Volume confirmation
        recent_volumes = [b.volume for b in bars[-20:] if b.volume > 0]
        if not recent_volumes:
            avg_volume = current_bar.volume
        else:
            avg_volume = sum(recent_volumes) / len(recent_volumes)

        volume_confirmed = current_bar.volume > (avg_volume * self.config.volume_multiplier)

        if not volume_confirmed:
            return Signal(
                action=SignalAction.HOLD,
                price=current_price,
                confidence=0.0,
                reason="low_volume",
            )

        # Strong breakout with volume - enter
        stop_loss = current_price - (current_atr * self.config.stop_loss_atr_mult)

        # Use swing low if it provides a looser stop (more room to breathe)
        recent_lows = [b.low for b in bars[-10:]]
        swing_low = min(recent_lows)
        swing_stop = swing_low - (current_atr * 0.5)
        # Take the higher (looser) of the two stops to give trades room
        stop_loss = max(stop_loss, swing_stop)

        self._entry_bar_idx = bar_idx
        self._entry_zscore = 0.0

        breakout_pct = ((current_price - highest_close) / highest_close) * 100
        volume_ratio = current_bar.volume / avg_volume if avg_volume > 0 else 1.0

        return Signal(
            action=SignalAction.BUY,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=None,  # Trail only
            position_size_pct=self.config.base_position_pct,
            confidence=0.8,
            reason=f"breakout: +{breakout_pct:.1f}%, vol×{volume_ratio:.1f}",
        )

    def _handle_exit_breakout(
        self,
        bars: list[Bar],
        position: Position,
        current_atr: float,
        bar_idx: int,
    ) -> Signal:
        """Check exit conditions for breakout trades.

        Use trailing stop only - let winners run, cut losers quickly.
        """
        current_price = bars[-1].close

        # Exit 1: Max holding period
        if self._entry_bar_idx is not None:
            bars_held = bar_idx - self._entry_bar_idx
            if bars_held >= self.config.max_holding_days:
                return Signal(
                    action=SignalAction.CLOSE,
                    price=current_price,
                    confidence=0.5,
                    reason=f"time_stop: {bars_held}d",
                )

        # Trailing stop logic
        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100

        # Tight trail if profitable
        if current_price > position.entry_price:
            trail_distance = current_atr * self.config.trailing_stop_atr_mult
            new_stop = current_price - trail_distance

            # Ensure stop is above entry (protect profits)
            if new_stop > position.entry_price * 1.005:  # 0.5% profit minimum
                return Signal(
                    action=SignalAction.HOLD,
                    price=current_price,
                    stop_loss=new_stop,
                    confidence=0.6,
                    reason=f"trail: +{pnl_pct:.1f}%",
                )

        # Default: hold with initial stop
        return Signal(
            action=SignalAction.HOLD,
            price=current_price,
            confidence=0.5,
            reason=f"hold: {pnl_pct:+.1f}%",
        )

    def _count_consecutive_down_days(self, bars: list[Bar]) -> int:
        """Count consecutive down days from most recent bar."""
        count = 0
        for i in range(-1, -min(10, len(bars)), -1):
            if bars[i].close < bars[i - 1].close:
                count += 1
            else:
                break
        return count

    def _is_turn_of_month(self, bar: Bar) -> bool:
        """Check if current bar is in turn-of-month window.

        Turn-of-month effect: excess returns in last N days of month
        and first N days of next month.
        """
        try:
            ts = bar.timestamp
            if isinstance(ts, str):
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                dt = ts

            day = dt.day

            # First N days of month
            if day <= self.config.calendar_days_after_month_start:
                return True

            # Last N days of month (approximate: day >= 28)
            # More precise would require knowing month length
            if day >= 29 - self.config.calendar_days_before_month_end:
                return True

            return False
        except (ValueError, AttributeError):
            return False

    def on_trade_complete(self, pnl: float, reason: str) -> None:
        """Reset entry tracking after trade completes."""
        self._entry_bar_idx = None
        self._entry_zscore = 0.0
