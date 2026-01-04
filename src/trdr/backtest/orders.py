"""Order types and order management for backtesting."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
from uuid import uuid4

from ..data.market import Bar


class OrderType(Enum):
    """Order types supported by the engine."""

    MARKET = "market"
    STOP = "stop"  # Trigger buy/sell at price
    STOP_LOSS = "stop_loss"  # Exit on price breach
    TRAILING_STOP = "trailing_stop"  # Follows price, exits on reversal


@dataclass
class Order:
    """Single order to be executed by the engine.

    Args:
        symbol: Asset symbol
        side: "buy" or "sell"
        order_type: Type of order
        quantity: Number of units
        stop_price: Trigger price for stop/SL orders
        trail_percent: Trail % for TSL (e.g., 0.02 = 2%)
        trail_amount: Trail $ for TSL (alternative to %)
    """

    symbol: str
    side: Literal["buy", "sell"]
    order_type: OrderType
    quantity: float
    stop_price: float | None = None
    trail_percent: float | None = None
    trail_amount: float | None = None
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    status: Literal["pending", "filled", "cancelled"] = "pending"
    created_at: str = ""
    filled_at: str | None = None
    fill_price: float | None = None

    def __post_init__(self) -> None:
        """Validate order parameters."""
        if self.order_type in (OrderType.STOP, OrderType.STOP_LOSS):
            if self.stop_price is None:
                raise ValueError(f"{self.order_type} requires stop_price")
        if self.order_type == OrderType.TRAILING_STOP:
            if self.trail_percent is None and self.trail_amount is None:
                raise ValueError("TRAILING_STOP requires trail_percent or trail_amount")


@dataclass
class Fill:
    """Executed order fill.

    Args:
        order_id: ID of filled order
        price: Fill price
        quantity: Filled quantity
        timestamp: Fill time
        side: "buy" or "sell"
    """

    order_id: str
    price: float
    quantity: float
    timestamp: str
    side: Literal["buy", "sell"]


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
            if order.order_type != OrderType.TRAILING_STOP:
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

        elif order.order_type in (OrderType.STOP, OrderType.STOP_LOSS, OrderType.TRAILING_STOP):
            # Stop orders trigger when price crosses stop_price
            if order.stop_price is None:
                return None

            if order.side == "sell":
                # Sell stop triggers when price drops to/below stop
                if bar.low <= order.stop_price:
                    # Fill at stop price or open if gapped through
                    fill_price = min(order.stop_price, bar.open)
            else:
                # Buy stop triggers when price rises to/above stop
                if bar.high >= order.stop_price:
                    fill_price = max(order.stop_price, bar.open)

        if fill_price is None:
            return None

        # Apply slippage
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
        )
