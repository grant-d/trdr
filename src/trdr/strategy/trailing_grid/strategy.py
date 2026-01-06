"""Trailing Grid Bot strategy with DCA.

Flow:
1. Wait for downtrend (price falling)
2. Set trailing buy above price - triggers on reversal
3. Once bought, track target sell price
4. When price hits target, set trailing stop below
5. Trailing stop follows price up, sells on reversal
6. Repeat

DCA:
- Up to 3 entries (initial + 2 DCA)
- Each DCA at grid_width below previous entry
- Each DCA uses trailing buy
- Sell target ensures 1 grid_width profit across all positions
"""

from dataclasses import dataclass

import numpy as np

from ...data import Bar
from ..base_strategy import BaseStrategy, StrategyConfig
from ..types import DataRequirement, Position, Signal, SignalAction


@dataclass(frozen=True)
class TrailingGridConfig(StrategyConfig):
    """Configuration for Trailing Grid strategy.

    Args:
        grid_width_pct: Grid width as percentage (e.g., 0.02 = 2%)
        trail_pct: Trailing stop/buy distance as percentage
        max_dca: Maximum DCA entries (including initial)
        downtrend_bars: Bars to confirm downtrend
        stop_loss_multiplier: Stop loss distance as multiple of grid_width_pct
        sell_target_multiplier: Sell target as multiple of grid_width_pct
    """

    grid_width_pct: float = 0.0345  # 2.5% grid (optimal)
    trail_pct: float = 0.0129  # 2.0% trail distance (optimal)
    max_dca: int = 2  # Maximum DCA entries
    downtrend_bars: int = 4  # Bars to confirm downtrend
    stop_loss_multiplier: float = 3.0643  # Stop loss as multiple of grid_width (optimal)
    sell_target_multiplier: float = 0.5268  # Sell target as multiple of grid_width


class TrailingGridStrategy(BaseStrategy):
    """Grid bot with trailing stops for buy and sell.

    State machine:
    - WAIT_DOWN: Waiting for downtrend to start
    - TRAIL_BUY: Trailing buy active, waiting for reversal
    - IN_POSITION: Holding, waiting for price to hit sell target
    - TRAIL_SELL: Trailing stop active, following price up
    """

    def __init__(self, config: TrailingGridConfig):
        super().__init__(config, name="TrailingGrid")
        self.config: TrailingGridConfig = config

        # State
        self.state = "WAIT_DOWN"
        self.trail_buy_price: float | None = None  # Trailing buy trigger
        self.trail_sell_price: float | None = None  # Trailing stop trigger
        self.peak_price: float | None = None  # Track peak for trailing stop
        self.entries: list[dict] = []  # List of {price, qty} for DCA tracking
        self.last_high: float | None = None  # For downtrend detection
        self.volume_threshold: float = 1.2  # Volume must be 1.2x average for entry

    def _avg_volume(self, bars: list[Bar], lookback: int = 20) -> float:
        """Calculate average volume over lookback period."""
        if len(bars) < lookback:
            return 0.0
        return float(np.mean([b.volume for b in bars[-lookback:]]))

    def get_data_requirements(self) -> list[DataRequirement]:
        return [
            DataRequirement(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                lookback=self.config.lookback,
                role="primary",
            ),
        ]

    def _avg_entry_price(self) -> float:
        """Calculate average entry price across all DCA entries."""
        if not self.entries:
            return 0.0
        total_cost = sum(e["price"] * e["qty"] for e in self.entries)
        total_qty = sum(e["qty"] for e in self.entries)
        return total_cost / total_qty if total_qty > 0 else 0.0

    def _total_qty(self) -> float:
        """Total quantity held across all entries."""
        return sum(e["qty"] for e in self.entries)

    def _sell_target(self) -> float:
        """Calculate sell target to achieve grid width profit."""
        avg = self._avg_entry_price()
        if avg <= 0:
            return 0.0
        return avg * (1 + self.config.grid_width_pct * self.config.sell_target_multiplier)

    def _is_downtrend(self, bars: list[Bar]) -> bool:
        """Check if price is in downtrend (consecutive lower highs)."""
        if len(bars) < self.config.downtrend_bars + 1:
            return False
        recent = bars[-(self.config.downtrend_bars + 1) :]
        for i in range(1, len(recent)):
            if recent[i].high >= recent[i - 1].high:
                return False
        return True

    def _is_uptrend_pullback(self, bars: list[Bar]) -> bool:
        """Check for uptrend with pullback (rising structure + current dip).

        Detects when price is in uptrend structure but has pulled back,
        creating a buy-the-dip opportunity.
        """
        if len(bars) < 10:
            return False

        # Check if recent trend is up: current price > price 5 bars ago
        lookback = 5
        past_close = bars[-lookback - 1].close
        current = bars[-1]

        # Must be in uptrend (at least 2% up over lookback period)
        trend_pct = (current.close - past_close) / past_close
        if trend_pct < 0.02:
            return False

        # Current bar should be red (pullback candle)
        if current.close >= current.open:
            return False  # Not a pullback candle

        # Pullback size: 0.5% to 3% from recent high
        recent_high = max(b.high for b in bars[-5:])
        pullback_pct = (recent_high - current.close) / recent_high

        return 0.005 <= pullback_pct <= 0.03

    def _next_dca_level(self) -> float | None:
        """Get price level for next DCA entry."""
        if len(self.entries) >= self.config.max_dca:
            return None
        if not self.entries:
            return None  # No position yet
        # Next DCA at grid_width below lowest entry
        lowest = min(e["price"] for e in self.entries)
        return lowest * (1 - self.config.grid_width_pct)

    def generate_signal(
        self,
        bars: dict[str, list[Bar]],
        position: Position | None,
    ) -> Signal:
        """Generate trading signal based on trailing grid logic."""
        primary_key = f"{self.config.symbol}:{self.config.timeframe}"
        primary_bars = bars[primary_key]

        if len(primary_bars) < 20:
            return Signal(
                action=SignalAction.HOLD,
                price=primary_bars[-1].close if primary_bars else 0,
                confidence=0.0,
                reason="Insufficient data",
            )

        current_bar = primary_bars[-1]
        price = current_bar.close
        high = current_bar.high
        low = current_bar.low

        # Sync state with position
        if position is None or position.side == "none":
            if self.entries:
                # Position closed externally, reset
                self.entries = []
                self.state = "WAIT_DOWN"
                self.trail_buy_price = None
                self.trail_sell_price = None
                self.peak_price = None

        # State machine
        if self.state == "WAIT_DOWN":
            # Wait for downtrend OR uptrend pullback to start trailing buy
            if self._is_downtrend(primary_bars):
                # Start trailing buy above current price
                self.trail_buy_price = price * (1 + self.config.trail_pct)
                self.state = "TRAIL_BUY"
                return Signal(
                    action=SignalAction.HOLD,
                    price=price,
                    confidence=0.3,
                    reason=f"Downtrend detected, trail buy at {self.trail_buy_price:.2f}",
                )

            if self._is_uptrend_pullback(primary_bars):
                # Uptrend pullback - set tighter trail for continuation
                self.trail_buy_price = price * (1 + self.config.trail_pct * 0.75)
                self.state = "TRAIL_BUY"
                return Signal(
                    action=SignalAction.HOLD,
                    price=price,
                    confidence=0.35,
                    reason=f"Uptrend pullback, trail buy at {self.trail_buy_price:.2f}",
                )

            return Signal(
                action=SignalAction.HOLD,
                price=price,
                confidence=0.0,
                reason="Waiting for entry signal",
            )

        elif self.state == "TRAIL_BUY":
            # Update trailing buy - follows price down
            new_trail = price * (1 + self.config.trail_pct)
            if self.trail_buy_price is None or new_trail < self.trail_buy_price:
                self.trail_buy_price = new_trail

            # Check if price reversed up and hit trailing buy
            if high >= self.trail_buy_price:
                # BUY triggered
                entry_price = self.trail_buy_price
                self.entries.append({"price": entry_price, "qty": 1.0})
                self.state = "IN_POSITION"
                self.trail_buy_price = None
                self.peak_price = price

                return Signal(
                    action=SignalAction.BUY,
                    price=entry_price,
                    confidence=0.7,
                    reason=f"Trail buy triggered at {entry_price:.2f}",
                    stop_loss=entry_price
                    * (1 - self.config.grid_width_pct * self.config.stop_loss_multiplier),
                    take_profit=self._sell_target(),
                )

            return Signal(
                action=SignalAction.HOLD,
                price=price,
                confidence=0.4,
                reason=f"Trailing buy at {self.trail_buy_price:.2f}, price {price:.2f}",
            )

        elif self.state == "IN_POSITION":
            # Track peak and check for DCA opportunity
            if self.peak_price is None or price > self.peak_price:
                self.peak_price = price

            # Check if price hit sell target -> activate trailing stop
            target = self._sell_target()
            if price >= target:
                self.trail_sell_price = price * (1 - self.config.trail_pct)
                self.state = "TRAIL_SELL"
                return Signal(
                    action=SignalAction.HOLD,
                    price=price,
                    confidence=0.6,
                    reason=f"Target hit, trail stop at {self.trail_sell_price:.2f}",
                )

            # Check for DCA opportunity
            next_dca = self._next_dca_level()
            if next_dca and price <= next_dca:
                # Set trailing buy for DCA
                self.trail_buy_price = price * (1 + self.config.trail_pct)
                # Don't change state - stay IN_POSITION but track DCA trail

            # Check if DCA trailing buy triggered
            if self.trail_buy_price and high >= self.trail_buy_price:
                entry_price = self.trail_buy_price
                self.entries.append({"price": entry_price, "qty": 1.0})
                self.trail_buy_price = None

                return Signal(
                    action=SignalAction.BUY,
                    price=entry_price,
                    confidence=0.65,
                    reason=f"DCA #{len(self.entries)} at {entry_price:.2f}",
                    stop_loss=self._avg_entry_price()
                    * (1 - self.config.grid_width_pct * self.config.stop_loss_multiplier),
                    take_profit=self._sell_target(),
                )

            # Update DCA trailing buy if active
            if self.trail_buy_price:
                new_trail = price * (1 + self.config.trail_pct)
                if new_trail < self.trail_buy_price:
                    self.trail_buy_price = new_trail

            return Signal(
                action=SignalAction.HOLD,
                price=price,
                confidence=0.5,
                reason=f"Holding {len(self.entries)} entries, target {target:.2f}",
            )

        elif self.state == "TRAIL_SELL":
            # Update trailing stop - follows price up
            if price > self.peak_price:
                self.peak_price = price
                self.trail_sell_price = price * (1 - self.config.trail_pct)

            # Check if trailing stop hit
            if low <= self.trail_sell_price:
                # SELL all
                sell_price = self.trail_sell_price
                avg_entry = self._avg_entry_price()
                profit_pct = (sell_price / avg_entry - 1) * 100 if avg_entry > 0 else 0

                # Reset for next cycle
                self.entries = []
                self.state = "WAIT_DOWN"
                self.trail_sell_price = None
                self.peak_price = None

                return Signal(
                    action=SignalAction.CLOSE,
                    price=sell_price,
                    confidence=0.8,
                    reason=f"Trail stop hit at {sell_price:.2f}, profit {profit_pct:.1f}%",
                )

            return Signal(
                action=SignalAction.HOLD,
                price=price,
                confidence=0.7,
                reason=f"Trail stop at {self.trail_sell_price:.2f}, peak {self.peak_price:.2f}",
            )

        return Signal(
            action=SignalAction.HOLD,
            price=price,
            confidence=0.0,
            reason="Unknown state",
        )
