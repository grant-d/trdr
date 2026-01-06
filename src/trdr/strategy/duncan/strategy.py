"""Duncan Trailer v2 trend-following strategy.

Strategy follows trend based on adaptive trailing stops:
- Entry: When trail trend flips (1 = long, -1 = short)
- Exit: When price hits trailing stop OR trend reverses
- Stop loss: Dynamic based on ATR and RSI conditions

The Duncan Trailer adapts based on:
1. Distance from entry (exponential tightening)
2. RSI extremes (tighter at overbought/oversold)
3. Combined RSI indicators (RSI, RVI, Laguerre RSI)
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from ...data import Bar
from ...indicators.ema import EmaIndicator
from ...indicators.laguerre_rsi import LaguerreRsiIndicator
from ...indicators.rsi import RsiIndicator
from ...indicators.rvi import RviIndicator
from ...indicators.wilder import WilderEmaIndicator
from ..base_strategy import BaseStrategy, StrategyConfig
from ..types import DataRequirement, Position, Signal, SignalAction

_EF = 0.695  # Exponential factor for trail tightening
_ATR_WEIGHT = 0.236


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _atr_weighted(atr_1: float, atr_2: float) -> float:
    return (atr_1 * _ATR_WEIGHT + atr_2 * (1.0 - _ATR_WEIGHT)) / 2.0


def _atr_norm(values: deque[float]) -> float:
    return float(np.median(values) + np.std(values))


def _buy_trail_factor(
    price: float,
    origin_price: float | None,
    min_trail_factor: float,
    max_trail_factor: float,
    grid_size: float,
    grid_count: float = 3.0,
    invert: bool = False,
) -> float | None:
    if origin_price is None:
        return None

    trail_size = grid_count * grid_size
    if trail_size == 0:
        return None
    trail_floor = max(origin_price, price) - trail_size

    factor = (price - trail_floor) / trail_size
    factor = 1 - _clamp(factor, 0, 1)

    if invert:
        factor = 2 - np.exp(_EF * factor)
    else:
        factor = np.exp(_EF * factor) - 1
    factor = _clamp(factor, 0, 1)

    return min_trail_factor + factor * (max_trail_factor - min_trail_factor)


def _sell_trail_factor(
    price: float,
    origin_price: float | None,
    min_trail_factor: float,
    max_trail_factor: float,
    grid_size: float,
    grid_count: float = 3.0,
    invert: bool = False,
) -> float | None:
    if origin_price is None:
        return None

    trail_size = grid_count * grid_size
    if trail_size == 0:
        return None
    trail_ceil = min(price, origin_price) + trail_size

    factor = (trail_ceil - price) / trail_size
    factor = 1 - _clamp(factor, 0, 1)

    if invert:
        factor = 2 - np.exp(_EF * factor)
    else:
        factor = np.exp(_EF * factor) - 1
    factor = _clamp(factor, 0, 1)

    return min_trail_factor + factor * (max_trail_factor - min_trail_factor)


class _TrSmaIndicator:
    """Streaming SMA of True Range."""

    def __init__(self, period: int) -> None:
        self.period = max(1, period)
        self._values: deque[float] = deque(maxlen=self.period)
        self._sum = 0.0

    def update(self, tr: float) -> float:
        if len(self._values) == self.period:
            self._sum -= self._values[0]
        self._values.append(tr)
        self._sum += tr
        if len(self._values) < self.period:
            return 0.0
        return self._sum / self.period


# TODO: PineScript uses virtual trails; consider disabling real stops for parity.


@dataclass(frozen=True)
class DuncanConfig(StrategyConfig):
    """Configuration for Duncan Trailer v2 strategy.

    Args:
        symbol: Trading symbol (inherited from StrategyConfig)
        timeframe: Bar timeframe (inherited from StrategyConfig)
        atr_period: ATR calculation period
        multiplier: ATR multiplier (defaults to atr_period if None)
        rsi_weight: How much RSI tightens trail (0-1, default 0.62)
        min_trail_factor: Minimum trail factor (0-1, default 0.25)
        max_trail_factor: Maximum trail factor (0-1, default 0.95)
        grid_count: Grid size multiplier for exponential scaling
        close_on_reversal: Close position on trend reversal (vs wait for stop)
    """

    atr_period: int = 15
    multiplier: float | None = 20.0
    atr_norm: bool = True
    rsi_weight: float = 0.62
    lrsi_alpha: float = 0.2
    min_trail_factor: float = 0.50
    max_trail_factor: float = 1.00
    grid_count: float = 3.0
    close_on_reversal: bool = False
    min_trend_bars: int = 1
    min_entry_gap_bars: int = 0
    rsi_entry_long: float = 60.0
    rsi_entry_short: float = 100.0
    enable_short: bool = False
    min_hold_bars: int = 1
    trend_ema_period: int = 1
    trend_flip_buffer_atr: float = 0.2
    use_trailing_stop: bool = True
    exit_profit_atr_min: float = 0.5


class DuncanStrategy(BaseStrategy):
    """Duncan Trailer v2 trend-following strategy.

    Follows trend reversals using Duncan Trailer v2 adaptive stops:
    - Long when trend = 1, stop at up_trail
    - Short when trend = -1, stop at down_trail

    Trailing stops adapt based on:
    - Distance from entry (exponential tightening)
    - RSI proximity to extremes (tighter at <30/>70)
    - Combined momentum indicators (RSI + RVI + Laguerre RSI)

    Exit options:
    - close_on_reversal=True: Close immediately on trend flip
    - close_on_reversal=False: Wait for stop loss hit
    """

    def __init__(self, config: DuncanConfig):
        """Initialize with config."""
        super().__init__(config, name="Duncan")
        self.config: DuncanConfig = config
        # Track trend for entry signals (separate from internal _prev_trend).
        self._entry_prev_trend: int = 0
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset streaming indicator state."""
        self._rsi_calc = RsiIndicator(self.config.atr_period)
        self._rvi_calc = RviIndicator(self.config.atr_period, mode="ema")
        self._lrsi_calc = LaguerreRsiIndicator(alpha=self.config.lrsi_alpha)
        self._atr_wilder = WilderEmaIndicator(self.config.atr_period)
        self._tr_sma = _TrSmaIndicator(self.config.atr_period)
        self._trend_ema = EmaIndicator(self.config.trend_ema_period)
        self._atr_recent: deque[float] = deque(maxlen=3)
        self._last_index = 0
        self._trend = 1
        self._prev_trend = 1  # Used internally by _update_trailer for origin level tracking
        self._up_trail: float | None = None
        self._down_trail: float | None = None
        self._origin_up: float | None = None
        self._origin_down: float | None = None
        self._prev_up_base: float | None = None
        self._prev_down_base: float | None = None
        self._entry_price_long: float | None = None
        self._entry_price_short: float | None = None
        self._prev_close: float | None = None
        self._combined_rsi: float = 50.0
        self._trend_bars: int = 0
        self._last_entry_index: int | None = None
        self._trend_ema_value: float = 0.0
        self._prev_trend_ema_value: float = 0.0
        self._atr_value: float = 0.0

    def reset(self) -> None:
        """Reset strategy state for a new run."""
        self._entry_prev_trend = 0  # Force initial entry signal
        self._reset_state()

    def _update_trailer(self, bars: list[Bar]) -> tuple[float, float, int]:
        """Update trailing stop state with new bars."""
        if not bars:
            return (0.0, 0.0, 1)

        if self._last_index > len(bars):
            self._reset_state()

        multiplier = self.config.multiplier
        if multiplier is None:
            multiplier = float(self.config.atr_period)

        for idx in range(self._last_index, len(bars)):
            bar = bars[idx]
            prev_close = self._prev_close if self._prev_close is not None else bar.close
            prev_trend = self._trend

            # ATR blend + norm (Pine: atr = (atr_1*x + atr_2*(1-x))/2; norm via median+stdev)
            tr = max(
                bar.high - bar.low,
                abs(bar.high - prev_close),
                abs(bar.low - prev_close),
            )
            atr_1 = self._atr_wilder.update(tr)
            atr_2 = self._tr_sma.update(tr)
            atr_weighted = _atr_weighted(atr_1, atr_2)
            if atr_weighted > 0:
                self._atr_recent.append(atr_weighted)
            if self.config.atr_norm and len(self._atr_recent) >= 3:
                atr_val = _atr_norm(self._atr_recent)
            else:
                atr_val = atr_weighted
            self._atr_value = atr_val

            # Combined RSI (Pine: geometric mean, projected to [3,97])
            rsi_val = self._rsi_calc.update(bar)
            rvi_val = self._rvi_calc.update(bar)
            lrsi_val = self._lrsi_calc.update(bar)
            combined = (rsi_val * rvi_val * lrsi_val) ** (1 / 3)
            combined = (combined - 3) / 0.94
            combined = _clamp(combined, 0, 100)
            self._combined_rsi = combined
            self._prev_trend_ema_value = self._trend_ema_value
            self._trend_ema_value = self._trend_ema.update(bar)

            rsi_factor = 1.0
            if self.config.rsi_weight > 0.0:
                rsi_fac = (combined - 50) / 50
                rsi_fac = abs(rsi_fac)
                rsi_fac = 1 - rsi_fac
                rsi_factor = (1 - self.config.rsi_weight) + self.config.rsi_weight * rsi_fac

            src = (bar.high + bar.low + bar.close) / 3.0
            prc = bar.close

            # Base up/down for trend
            up_base = src - multiplier * atr_val
            down_base = src + multiplier * atr_val

            if self._prev_up_base is not None:
                if prev_close > self._prev_up_base:
                    up_base = max(up_base, self._prev_up_base)
            if self._prev_down_base is not None:
                if prev_close < self._prev_down_base:
                    down_base = min(down_base, self._prev_down_base)

            # Origin levels for trail factors
            if self._trend == 1 and (self._origin_up is None or self._trend != self._prev_trend):
                self._origin_up = up_base
            if self._trend == -1 and (self._origin_down is None or self._trend != self._prev_trend):
                self._origin_down = down_base

            # Trail factor scaling
            up_factor = _buy_trail_factor(
                src,
                self._origin_up,
                self.config.min_trail_factor,
                self.config.max_trail_factor,
                atr_val,
                grid_count=multiplier,
            )
            down_factor = _sell_trail_factor(
                src,
                self._origin_down,
                self.config.min_trail_factor,
                self.config.max_trail_factor,
                atr_val,
                grid_count=multiplier,
            )

            if up_factor is None:
                up_factor = self.config.min_trail_factor
            if down_factor is None:
                down_factor = self.config.min_trail_factor

            up = src - multiplier * atr_val * up_factor * rsi_factor
            down = src + multiplier * atr_val * down_factor * rsi_factor

            if self._up_trail is not None:
                up = max(up, self._up_trail) if prev_close > self._up_trail else up
            if self._down_trail is not None:
                down = min(down, self._down_trail) if prev_close < self._down_trail else down

            # Trend update (check against previous trail values, like PineScript's up_1/dn_1)
            flip_buffer = self.config.trend_flip_buffer_atr * atr_val
            if self._trend == -1 and self._down_trail is not None:
                if prc > self._down_trail + flip_buffer:
                    self._trend = 1
            elif self._trend == 1 and self._up_trail is not None:
                if prc < self._up_trail - flip_buffer:
                    self._trend = -1

            if self._trend == prev_trend:
                self._trend_bars += 1
            else:
                self._trend_bars = 1

            self._prev_trend = self._trend
            self._prev_up_base = up_base
            self._prev_down_base = down_base
            self._up_trail = up
            self._down_trail = down
            self._prev_close = bar.close

        self._last_index = len(bars)
        return (
            float(self._up_trail),
            float(self._down_trail),
            int(self._trend),
        )

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
        """Generate trading signal from bars.

        Args:
            bars: Dict of bars keyed by "symbol:timeframe"
            position: Current position or None if flat

        Returns:
            Signal with action (BUY/SELL/CLOSE/HOLD)
        """
        # Extract primary bars
        primary_key = f"{self.config.symbol}:{self.config.timeframe}"
        primary_bars = bars[primary_key]
        # Check minimum data
        min_bars = self.config.atr_period + 10
        if len(primary_bars) < min_bars:
            return Signal(
                action=SignalAction.HOLD,
                price=primary_bars[-1].close if primary_bars else 0,
                confidence=0.0,
                reason="Insufficient data",
            )

        current_price = primary_bars[-1].close

        # Calculate duncan trailer (streaming)
        up_trail, down_trail, trend = self._update_trailer(primary_bars)
        atr_val = self._atr_value
        bar_idx = len(primary_bars) - 1
        # Detect trend change for entry signals (use separate tracker)
        trend_changed = trend != self._entry_prev_trend
        self._entry_prev_trend = trend

        # Check exits first (if in position)
        if position:
            if position.side == "long":
                hold_bars = (
                    bar_idx - self._last_entry_index
                    if self._last_entry_index is not None
                    else self.config.min_hold_bars
                )
                if self.config.use_trailing_stop:
                    # Stop loss hit
                    if current_price <= up_trail:
                        return Signal(
                            action=SignalAction.CLOSE,
                            price=current_price,
                            confidence=1.0,
                            reason=f"Long stop hit at {up_trail:.2f}",
                        )

                # Trend reversal
                if trend == -1 and self.config.close_on_reversal:
                    if hold_bars >= self.config.min_hold_bars:
                        if atr_val <= 0:
                            return Signal(
                                action=SignalAction.CLOSE,
                                price=current_price,
                                confidence=0.9,
                                reason="Trend reversed to downtrend",
                            )
                        pnl_atr = (current_price - position.entry_price) / atr_val
                        if pnl_atr >= self.config.exit_profit_atr_min:
                            return Signal(
                                action=SignalAction.CLOSE,
                                price=current_price,
                                confidence=0.9,
                                reason=f"Trend reversed at {pnl_atr:.1f} ATR",
                            )

                # Holding
                return Signal(
                    action=SignalAction.HOLD,
                    price=current_price,
                    confidence=0.5,
                    reason=f"Holding long, trail at {up_trail:.2f}",
                )

            elif position.side == "short":
                # Stop loss hit
                if current_price >= down_trail:
                    return Signal(
                        action=SignalAction.CLOSE,
                        price=current_price,
                        confidence=1.0,
                        reason=f"Short stop hit at {down_trail:.2f}",
                    )

                # Trend reversal
                if trend == 1 and self.config.close_on_reversal:
                    return Signal(
                        action=SignalAction.CLOSE,
                        price=current_price,
                        confidence=0.9,
                        reason="Trend reversed to uptrend",
                    )

                # Holding
                return Signal(
                    action=SignalAction.HOLD,
                    price=current_price,
                    confidence=0.5,
                    reason=f"Holding short, trail at {down_trail:.2f}",
                )

        # Check entries (if no position)
        if not position:
            if self._last_entry_index is not None:
                if bar_idx - self._last_entry_index < self.config.min_entry_gap_bars:
                    return Signal(
                        action=SignalAction.HOLD,
                        price=current_price,
                        confidence=0.0,
                        reason="Entry cooldown",
                    )

            # Long entry: trend flipped to uptrend
            if (
                trend == 1
                and trend_changed
                and self._trend_bars >= self.config.min_trend_bars
                and self._combined_rsi >= self.config.rsi_entry_long
                and current_price >= self._trend_ema_value
            ):
                # Virtual trails: don't set stop_loss, let trend reversal handle exit
                self._last_entry_index = bar_idx
                return Signal(
                    action=SignalAction.BUY,
                    price=current_price,
                    confidence=0.7,
                    reason="Uptrend detected",
                )

            # Short entry: trend flipped to downtrend
            if (
                self.config.enable_short
                and trend == -1
                and trend_changed
                and self._trend_bars >= self.config.min_trend_bars
                and self._combined_rsi <= self.config.rsi_entry_short
            ):
                # Virtual trails: don't set stop_loss, let trend reversal handle exit
                self._last_entry_index = bar_idx
                return Signal(
                    action=SignalAction.SELL,
                    price=current_price,
                    confidence=0.7,
                    reason="Downtrend detected",
                )

        # Default: no action
        return Signal(
            action=SignalAction.HOLD,
            price=current_price,
            confidence=0.0,
            reason="No signal",
        )
