"""Portfolio and position tracking for backtesting."""

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .orders import Order, OrderType

if TYPE_CHECKING:
    from ..core import Symbol


@dataclass
class PositionEntry:
    """Single entry in a position.

    Tracks individual buys for accurate P&L calculation.

    Args:
        price: Entry price
        quantity: Number of units
        timestamp: Entry time
    """

    price: float
    quantity: float
    timestamp: str


@dataclass
class Position:
    """Position in a single symbol with multiple entries.

    Supports pyramiding (multiple buys) and partial exits.

    Args:
        symbol: Asset symbol
        side: "long" or "short"
        entries: List of individual entries
    """

    symbol: str
    side: Literal["long", "short"]
    entries: deque[PositionEntry] = field(default_factory=deque)

    @property
    def total_quantity(self) -> float:
        """Total position size across all entries."""
        return sum(e.quantity for e in self.entries)

    @property
    def avg_price(self) -> float:
        """Volume-weighted average entry price."""
        if not self.entries:
            return 0.0
        total_cost = sum(e.price * e.quantity for e in self.entries)
        total_qty = self.total_quantity
        return total_cost / total_qty if total_qty > 0 else 0.0

    @property
    def total_cost(self) -> float:
        """Total cost basis of position."""
        return sum(e.price * e.quantity for e in self.entries)

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L (positive = profit)
        """
        if self.side == "long":
            return (current_price - self.avg_price) * self.total_quantity
        else:
            return (self.avg_price - current_price) * self.total_quantity

    def add_entry(self, price: float, quantity: float, timestamp: str) -> None:
        """Add new entry to position.

        Args:
            price: Entry price
            quantity: Units to add
            timestamp: Entry time
        """
        self.entries.append(PositionEntry(price=price, quantity=quantity, timestamp=timestamp))

    def reduce(self, quantity: float) -> float:
        """Reduce position by quantity (FIFO).

        Args:
            quantity: Units to remove

        Returns:
            Average price of removed units (for P&L calculation)
        """
        if quantity > self.total_quantity:
            quantity = self.total_quantity

        removed_cost = 0.0
        removed_qty = 0.0
        remaining = quantity

        while remaining > 0 and self.entries:
            entry = self.entries[0]
            if entry.quantity <= remaining:
                # Remove entire entry
                removed_cost += entry.price * entry.quantity
                removed_qty += entry.quantity
                remaining -= entry.quantity
                self.entries.popleft()
            else:
                # Partial removal
                removed_cost += entry.price * remaining
                removed_qty += remaining
                entry.quantity -= remaining
                remaining = 0

        return removed_cost / removed_qty if removed_qty > 0 else 0.0


@dataclass
class Portfolio:
    """Portfolio tracking cash and positions.

    Args:
        cash: Available cash
        positions: Dict of symbol -> Position
    """

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def equity(self, prices: dict[str, float]) -> float:
        """Calculate total equity (cash + positions MTM).

        Args:
            prices: Current prices by symbol

        Returns:
            Total equity value
        """
        mtm = 0.0
        for symbol, position in self.positions.items():
            if symbol in prices:
                mtm += position.total_cost + position.unrealized_pnl(prices[symbol])
        return self.cash + mtm

    def buying_power(self) -> float:
        """Get available buying power.

        Returns:
            Cash available for new positions
        """
        return max(0.0, self.cash)

    def get_position(self, symbol: "Symbol") -> Position | None:
        """Get position for symbol.

        Args:
            symbol: Asset symbol

        Returns:
            Position or None if no position
        """
        pos = self.positions.get(str(symbol))  # eg, crypto:BTC/USD
        if pos and pos.total_quantity > 0:
            return pos
        return None

    def open_position(
        self,
        symbol: "Symbol",
        side: Literal["long", "short"],
        price: float,
        quantity: float,
        timestamp: str,
        cost: float = 0.0,
    ) -> None:
        """Open or add to position.

        Args:
            symbol: Asset symbol
            side: "long" or "short"
            price: Entry price
            quantity: Units to buy
            timestamp: Entry time
            cost: Transaction cost
        """
        total_cost = price * quantity + cost
        if total_cost > self.cash:
            # Reduce quantity to fit available cash
            quantity = (self.cash - cost) / price
            total_cost = price * quantity + cost

        if quantity <= 0:
            return

        self.cash -= total_cost

        symbol_str = str(symbol)  # eg, crypto:BTC/USD
        if symbol_str not in self.positions:
            self.positions[symbol_str] = Position(symbol=symbol_str, side=side)

        self.positions[symbol_str].add_entry(price, quantity, timestamp)

    def close_position(
        self,
        symbol: "Symbol",
        price: float,
        quantity: float | None = None,
        cost: float = 0.0,
    ) -> float:
        """Close or reduce position.

        Args:
            symbol: Asset symbol
            price: Exit price
            quantity: Units to sell (None = close all)
            cost: Transaction cost

        Returns:
            Realized P&L
        """
        position = self.positions.get(str(symbol))  # eg, crypto:BTC/USD
        if not position:
            return 0.0

        if quantity is None:
            quantity = position.total_quantity

        quantity = min(quantity, position.total_quantity)
        if quantity <= 0:
            return 0.0

        avg_entry = position.reduce(quantity)
        proceeds = price * quantity - cost

        if position.side == "long":
            pnl = (price - avg_entry) * quantity - cost
        else:
            pnl = (avg_entry - price) * quantity - cost

        self.cash += proceeds

        # Clean up empty positions
        if position.total_quantity <= 0:
            del self.positions[str(symbol)]  # eg, crypto:BTC/USD

        return pnl

    def liquidate(self, prices: dict[str, float], cost_pct: float = 0.0) -> list[Order]:
        """Generate market sell orders for all positions.

        Args:
            prices: Current prices (for cost calculation)
            cost_pct: Transaction cost as decimal

        Returns:
            List of market orders to close all positions
        """
        orders = []
        for symbol, position in list(self.positions.items()):
            if position.total_quantity <= 0:
                continue

            orders.append(
                Order(
                    symbol=symbol,
                    side="sell" if position.side == "long" else "buy",
                    order_type=OrderType.MARKET,
                    quantity=position.total_quantity,
                )
            )
        return orders
