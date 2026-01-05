"""Order types and order management for backtesting."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from ..data import Bar

if TYPE_CHECKING:
    pass


class OrderType(Enum):
    """Order types supported by the engine."""

    MARKET = "market"
    STOP = "stop"  # Becomes market when stop_price touched (either direction)
    LIMIT = "limit"  # Entry at specified price or better
    STOP_LIMIT = "stop_limit"  # Becomes limit when stop_price touched (either direction)
    TRAILING_STOP = "trailing_stop"  # Follows price, becomes market on reversal
    TRAILING_STOP_LIMIT = "trailing_stop_limit"  # Follows price, becomes limit on reversal


@dataclass
class Order:
    """Single order to be executed by the engine.

    Args:
        symbol: Symbol object
        side: "buy" or "sell"
        order_type: Type of order
        quantity: Number of units
        stop_price: Trigger price for stop/SL/TP orders
        limit_price: Price for limit orders
        trail_percent: Trail % for TSL (e.g., 0.02 = 2%)
        trail_amount: Trail $ for TSL (alternative to %)
    """

    symbol: str
    side: Literal["buy", "sell"]
    order_type: OrderType
    quantity: float
    stop_price: float | None = None
    limit_price: float | None = None
    trail_percent: float | None = None
    trail_amount: float | None = None
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    status: Literal["pending", "filled", "cancelled"] = "pending"
    triggered: bool = False  # For stop-limit: True after stop_price hit
    created_at: str = ""
    filled_at: str | None = None
    fill_price: float | None = None

    def __post_init__(self) -> None:
        """Validate order parameters."""
        if self.order_type == OrderType.STOP:
            if self.stop_price is None:
                raise ValueError("STOP requires stop_price")
        if self.order_type == OrderType.TRAILING_STOP:
            if self.trail_percent is None and self.trail_amount is None:
                raise ValueError("TRAILING_STOP requires trail_percent or trail_amount")
        if self.order_type == OrderType.TRAILING_STOP_LIMIT:
            if self.trail_percent is None and self.trail_amount is None:
                raise ValueError("TRAILING_STOP_LIMIT requires trail_percent or trail_amount")
            if self.limit_price is None:
                raise ValueError("TRAILING_STOP_LIMIT requires limit_price")
        if self.order_type == OrderType.LIMIT:
            if self.limit_price is None:
                raise ValueError("LIMIT requires limit_price")
        if self.order_type == OrderType.STOP_LIMIT:
            if self.stop_price is None or self.limit_price is None:
                raise ValueError("STOP_LIMIT requires both stop_price and limit_price")


@dataclass
class Fill:
    """Executed order fill.

    Args:
        order_id: ID of filled order
        price: Fill price
        quantity: Filled quantity
        timestamp: Fill time
        side: "buy" or "sell"
        order_type: Type of order that was filled
    """

    order_id: str
    price: float
    quantity: float
    timestamp: str
    side: Literal["buy", "sell"]
    order_type: OrderType


class OrderManager:
    """Manages pending orders and executes fills.

    Processes orders each bar:
    1. Update trailing stops based on price movement
    2. Check stop triggers against bar OHLC
    3. Execute market orders at bar open
    """

    def __init__(self) -> None:
        """Initialize order manager."""
        self._pending: list[Order] = []
        self._filled: list[Fill] = []

    @property
    def pending_orders(self) -> list[Order]:
        """Get list of pending orders."""
        return self._pending.copy()

    @property
    def fills(self) -> list[Fill]:
        """Get list of all fills."""
        return self._filled.copy()

    def submit(self, order: Order) -> str:
        """Submit order for execution.

        Args:
            order: Order to submit

        Returns:
            Order ID
        """
        self._pending.append(order)
        return order.id

    def cancel(self, order_id: str) -> bool:
        """Cancel pending order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancelled, False if not found
        """
        for i, order in enumerate(self._pending):
            if order.id == order_id:
                order.status = "cancelled"
                self._pending.pop(i)
                return True
        return False

    def cancel_all(self) -> int:
        """Cancel all pending orders.

        Returns:
            Number of orders cancelled
        """
        count = len(self._pending)
        for order in self._pending:
            order.status = "cancelled"
        self._pending.clear()
        return count

    def update_trailing_stops(self, bar: Bar) -> None:
        """Update trailing stop prices based on bar movement.

        For long positions (sell TSL): stop moves up with highs
        For short positions (buy TSL): stop moves down with lows

        Args:
            bar: Current bar
        """
        for order in self._pending:
            if order.order_type not in (OrderType.TRAILING_STOP, OrderType.TRAILING_STOP_LIMIT):
                continue

            if order.side == "sell":  # Long position TSL
                # Calculate new stop based on bar high
                if order.trail_amount:
                    new_stop = bar.high - order.trail_amount
                else:
                    new_stop = bar.high * (1 - order.trail_percent)

                # Only move stop up, never down
                if order.stop_price is None or new_stop > order.stop_price:
                    order.stop_price = new_stop

            else:  # Short position TSL (buy to cover)
                # Calculate new stop based on bar low
                if order.trail_amount:
                    new_stop = bar.low + order.trail_amount
                else:
                    new_stop = bar.low * (1 + order.trail_percent)

                # Only move stop down, never up
                if order.stop_price is None or new_stop < order.stop_price:
                    order.stop_price = new_stop

    def process_bar(self, bar: Bar, slippage: float = 0.0) -> list[Fill]:
        """Process all pending orders against current bar.

        Order of operations:
        1. Market orders fill at open
        2. Stop/SL/TSL check against OHLC

        Args:
            bar: Current bar
            slippage: Slippage to apply (added to buys, subtracted from sells)

        Returns:
            List of fills generated this bar
        """
        fills: list[Fill] = []
        remaining: list[Order] = []

        for order in self._pending:
            fill = self._try_fill(order, bar, slippage)
            if fill:
                fills.append(fill)
                self._filled.append(fill)
            else:
                remaining.append(order)

        self._pending = remaining
        return fills

    def _try_fill(self, order: Order, bar: Bar, slippage: float) -> Fill | None:
        """Try to fill order against bar.

        Args:
            order: Order to fill
            bar: Current bar
            slippage: Slippage amount

        Returns:
            Fill if triggered, None otherwise
        """
        fill_price: float | None = None

        if order.order_type == OrderType.MARKET:
            # Market orders fill at open
            fill_price = bar.open

        elif order.order_type in (OrderType.STOP, OrderType.TRAILING_STOP):
            # Stop triggers when bar range includes stop_price
            if order.stop_price is None:
                return None

            if bar.low <= order.stop_price <= bar.high:
                fill_price = order.stop_price

        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill at limit price or better (no slippage)
            if order.limit_price is None:
                return None

            if order.side == "buy":
                # Buy limit triggers when price drops to/below limit
                if bar.low <= order.limit_price:
                    # Fill at limit price (guaranteed) or open if gapped below
                    fill_price = min(order.limit_price, bar.open)
            else:
                # Sell limit triggers when price rises to/above limit
                if bar.high >= order.limit_price:
                    fill_price = max(order.limit_price, bar.open)

            # Limit orders have no slippage - fill at price and return early
            if fill_price is not None:
                order.status = "filled"
                order.filled_at = bar.timestamp
                order.fill_price = fill_price
                return Fill(
                    order_id=order.id,
                    price=fill_price,
                    quantity=order.quantity,
                    timestamp=bar.timestamp,
                    side=order.side,
                    order_type=order.order_type,
                )
            return None

        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop-limit: dormant until stop triggered, then acts as limit
            if order.stop_price is None or order.limit_price is None:
                return None

            # Phase 1: Check if stop is triggered (bar range touches stop)
            if not order.triggered:
                if bar.low <= order.stop_price <= bar.high:
                    order.triggered = True
                else:
                    return None

            # Phase 2: Triggered - act as limit order (no slippage)
            if order.side == "buy":
                # Buy limit fills when price drops to/below limit
                if bar.low <= order.limit_price:
                    fill_price = min(order.limit_price, bar.open)
            else:
                # Sell limit fills when price rises to/above limit
                if bar.high >= order.limit_price:
                    fill_price = max(order.limit_price, bar.open)

            if fill_price is not None:
                order.status = "filled"
                order.filled_at = bar.timestamp
                order.fill_price = fill_price
                return Fill(
                    order_id=order.id,
                    price=fill_price,
                    quantity=order.quantity,
                    timestamp=bar.timestamp,
                    side=order.side,
                    order_type=order.order_type,
                )
            return None

        elif order.order_type == OrderType.TRAILING_STOP_LIMIT:
            # Trailing stop-limit: trails like TSL, then acts as limit when triggered
            if order.stop_price is None or order.limit_price is None:
                return None

            # Phase 1: Check if trailing stop is triggered (bar range touches stop)
            if not order.triggered:
                if bar.low <= order.stop_price <= bar.high:
                    order.triggered = True
                else:
                    return None

            # Phase 2: Triggered - act as limit order (no slippage)
            if order.side == "buy":
                if bar.low <= order.limit_price:
                    fill_price = min(order.limit_price, bar.open)
            else:
                if bar.high >= order.limit_price:
                    fill_price = max(order.limit_price, bar.open)

            if fill_price is not None:
                order.status = "filled"
                order.filled_at = bar.timestamp
                order.fill_price = fill_price
                return Fill(
                    order_id=order.id,
                    price=fill_price,
                    quantity=order.quantity,
                    timestamp=bar.timestamp,
                    side=order.side,
                    order_type=order.order_type,
                )
            return None

        if fill_price is None:
            return None

        # Apply slippage (for non-limit orders)
        if order.side == "buy":
            fill_price += slippage
        else:
            fill_price -= slippage

        order.status = "filled"
        order.filled_at = bar.timestamp
        order.fill_price = fill_price

        return Fill(
            order_id=order.id,
            price=fill_price,
            quantity=order.quantity,
            timestamp=bar.timestamp,
            side=order.side,
            order_type=order.order_type,
        )
