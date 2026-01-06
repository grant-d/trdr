"""Circuit breaker for live trading safety."""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Callable

from ..config import RiskLimits

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    """Get current UTC time."""
    return datetime.now(UTC)


class BreakerState(Enum):
    """Circuit breaker state."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Trading halted
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class TradingStats:
    """Running trading statistics for circuit breaker.

    Args:
        start_equity: Initial equity at session start
        high_water_mark: Highest equity reached
        current_equity: Current equity
        daily_pnl: P&L since session start
        consecutive_losses: Count of consecutive losing trades
        trades_this_hour: Number of trades in last hour
        last_trade_times: Timestamps of recent trades
    """

    start_equity: float = 0.0
    high_water_mark: float = 0.0
    current_equity: float = 0.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    trades_this_hour: int = 0
    last_trade_times: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from high water mark as percentage."""
        if self.high_water_mark <= 0:
            return 0.0
        return (self.high_water_mark - self.current_equity) / self.high_water_mark * 100

    @property
    def daily_loss_pct(self) -> float:
        """Daily loss as percentage of starting equity."""
        if self.start_equity <= 0:
            return 0.0
        return -self.daily_pnl / self.start_equity * 100 if self.daily_pnl < 0 else 0.0

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade.

        Args:
            pnl: Trade P&L (positive for profit, negative for loss)
        """
        self.daily_pnl += pnl
        now = datetime.now(UTC)
        self.last_trade_times.append(now)

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Update trades in last hour
        cutoff = now - timedelta(hours=1)
        self.trades_this_hour = sum(1 for t in self.last_trade_times if t > cutoff)

    def update_equity(self, equity: float) -> None:
        """Update current equity.

        Args:
            equity: Current account equity
        """
        self.current_equity = equity
        if equity > self.high_water_mark:
            self.high_water_mark = equity

    def reset_daily(self, equity: float) -> None:
        """Reset daily statistics.

        Args:
            equity: Current equity to use as new starting point
        """
        self.start_equity = equity
        self.high_water_mark = equity
        self.current_equity = equity
        self.daily_pnl = 0.0
        self.consecutive_losses = 0


@dataclass
class BreakerTrip:
    """Record of circuit breaker trip.

    Args:
        reason: Why the breaker tripped
        value: Value that triggered the trip
        limit: Limit that was exceeded
        timestamp: When the trip occurred
    """

    reason: str
    value: float
    limit: float
    timestamp: datetime = field(default_factory=_now_utc)


class CircuitBreaker:
    """Circuit breaker for trading risk management.

    Monitors trading activity and halts trading when risk limits are exceeded.
    """

    def __init__(
        self,
        limits: RiskLimits,
        on_trip: Callable[[BreakerTrip], None] | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            limits: Risk limits configuration
            on_trip: Optional callback when breaker trips
        """
        self._limits = limits
        self._on_trip = on_trip
        self._state = BreakerState.CLOSED
        self._stats = TradingStats()
        self._trip_history: list[BreakerTrip] = []
        self._last_check = datetime.now(UTC)

    @property
    def state(self) -> BreakerState:
        """Get current breaker state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if breaker is open (trading halted)."""
        return self._state == BreakerState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if breaker is closed (normal operation)."""
        return self._state == BreakerState.CLOSED

    @property
    def stats(self) -> TradingStats:
        """Get current trading statistics."""
        return self._stats

    @property
    def trip_history(self) -> list[BreakerTrip]:
        """Get history of breaker trips."""
        return self._trip_history.copy()

    def initialize(self, equity: float) -> None:
        """Initialize breaker with starting equity.

        Args:
            equity: Starting account equity
        """
        self._stats = TradingStats(
            start_equity=equity,
            high_water_mark=equity,
            current_equity=equity,
        )
        self._state = BreakerState.CLOSED
        logger.info(f"Circuit breaker initialized with equity ${equity:.2f}")

    def update_equity(self, equity: float) -> None:
        """Update current equity and check limits.

        Args:
            equity: Current account equity
        """
        self._stats.update_equity(equity)
        self._check_limits()

    def record_trade(self, pnl: float) -> None:
        """Record completed trade and check limits.

        Args:
            pnl: Trade P&L
        """
        self._stats.record_trade(pnl)
        self._check_limits()

    def check_can_trade(self) -> tuple[bool, str | None]:
        """Check if trading is allowed.

        Returns:
            Tuple of (can_trade, reason_if_blocked)
        """
        if self._state == BreakerState.OPEN:
            return False, "Circuit breaker is open"

        # Check rate limit
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=1)
        recent_trades = sum(1 for t in self._stats.last_trade_times if t > cutoff)

        if recent_trades >= self._limits.max_orders_per_hour:
            return False, f"Rate limit: {recent_trades} orders in last hour"

        return True, None

    def _check_limits(self) -> None:
        """Check all risk limits and trip if exceeded."""
        if self._state == BreakerState.OPEN:
            return

        # Check drawdown
        if self._stats.drawdown_pct >= self._limits.max_drawdown_pct:
            self._trip(
                reason="max_drawdown",
                value=self._stats.drawdown_pct,
                limit=self._limits.max_drawdown_pct,
            )
            return

        # Check daily loss
        if self._stats.daily_loss_pct >= self._limits.max_daily_loss_pct:
            self._trip(
                reason="max_daily_loss",
                value=self._stats.daily_loss_pct,
                limit=self._limits.max_daily_loss_pct,
            )
            return

        # Check consecutive losses
        if self._stats.consecutive_losses >= self._limits.max_consecutive_losses:
            self._trip(
                reason="max_consecutive_losses",
                value=float(self._stats.consecutive_losses),
                limit=float(self._limits.max_consecutive_losses),
            )
            return

    def _trip(self, reason: str, value: float, limit: float) -> None:
        """Trip the circuit breaker.

        Args:
            reason: Reason for trip
            value: Value that caused trip
            limit: Limit that was exceeded
        """
        trip = BreakerTrip(reason=reason, value=value, limit=limit)
        self._trip_history.append(trip)
        self._state = BreakerState.OPEN

        logger.warning(
            f"CIRCUIT BREAKER TRIPPED: {reason} " f"(value={value:.2f}, limit={limit:.2f})"
        )

        if self._on_trip:
            try:
                self._on_trip(trip)
            except Exception as e:
                logger.exception(f"Trip callback error: {e}")

    def reset(self) -> None:
        """Reset circuit breaker to closed state.

        Call this manually after reviewing the trip and confirming
        it's safe to resume trading.
        """
        if self._state != BreakerState.OPEN:
            return

        self._state = BreakerState.CLOSED
        logger.info("Circuit breaker reset to CLOSED")

    def reset_daily(self, equity: float) -> None:
        """Reset daily statistics (e.g., at start of trading day).

        Args:
            equity: Current equity
        """
        self._stats.reset_daily(equity)
        logger.info(f"Circuit breaker daily reset with equity ${equity:.2f}")

    def check_position_size(
        self,
        position_value: float,
        equity: float | None = None,
    ) -> tuple[bool, str | None]:
        """Check if position size is within limits.

        Args:
            position_value: Proposed position value in dollars
            equity: Current equity (uses stored if not provided)

        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        eq = equity or self._stats.current_equity
        if eq <= 0:
            return False, "No equity"

        # Check position percentage
        position_pct = (position_value / eq) * 100
        if position_pct > self._limits.max_position_pct:
            return False, (
                f"Position {position_pct:.1f}% exceeds limit "
                f"{self._limits.max_position_pct:.1f}%"
            )

        # Check absolute position value
        if self._limits.max_position_value:
            if position_value > self._limits.max_position_value:
                return False, (
                    f"Position ${position_value:.2f} exceeds limit "
                    f"${self._limits.max_position_value:.2f}"
                )

        return True, None

    def get_status(self) -> dict:
        """Get current circuit breaker status.

        Returns:
            Dict with status information
        """
        return {
            "state": self._state.value,
            "equity": self._stats.current_equity,
            "high_water_mark": self._stats.high_water_mark,
            "drawdown_pct": self._stats.drawdown_pct,
            "daily_pnl": self._stats.daily_pnl,
            "daily_loss_pct": self._stats.daily_loss_pct,
            "consecutive_losses": self._stats.consecutive_losses,
            "trades_this_hour": self._stats.trades_this_hour,
            "limits": {
                "max_drawdown_pct": self._limits.max_drawdown_pct,
                "max_daily_loss_pct": self._limits.max_daily_loss_pct,
                "max_consecutive_losses": self._limits.max_consecutive_losses,
                "max_position_pct": self._limits.max_position_pct,
                "max_orders_per_hour": self._limits.max_orders_per_hour,
            },
        }
