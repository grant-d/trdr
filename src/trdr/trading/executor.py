"""Order execution via Alpaca API."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from ..core.config import AlpacaConfig


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order details."""

    id: str
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    order_type: str
    status: OrderStatus
    filled_qty: float
    filled_price: float | None
    submitted_at: str
    filled_at: str | None = None


@dataclass
class PositionInfo:
    """Position from exchange."""

    symbol: str
    qty: float
    side: str  # "long" or "short"
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float
    current_price: float


@dataclass
class AccountInfo:
    """Account summary."""

    equity: float
    cash: float
    buying_power: float
    portfolio_value: float


class OrderExecutor:
    """Executes orders via Alpaca API."""

    def __init__(self, config: AlpacaConfig):
        """Initialize executor.

        Args:
            config: Alpaca API configuration
        """
        self.config = config
        self._client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.is_paper,
        )

    async def get_account(self) -> AccountInfo:
        """Get account information.

        Returns:
            Account summary with equity, cash, buying power
        """
        account = self._client.get_account()
        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
        )

    async def get_position(self, symbol: str) -> PositionInfo | None:
        """Get position for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position info or None if no position
        """
        try:
            pos = self._client.get_open_position(symbol)
            return PositionInfo(
                symbol=symbol,
                qty=float(pos.qty),
                side="long" if float(pos.qty) > 0 else "short",
                avg_entry_price=float(pos.avg_entry_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                current_price=float(pos.current_price),
            )
        except Exception:
            return None

    async def get_all_positions(self) -> list[PositionInfo]:
        """Get all open positions.

        Returns:
            List of position info
        """
        positions = self._client.get_all_positions()
        return [
            PositionInfo(
                symbol=pos.symbol,
                qty=float(pos.qty),
                side="long" if float(pos.qty) > 0 else "short",
                avg_entry_price=float(pos.avg_entry_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                current_price=float(pos.current_price),
            )
            for pos in positions
        ]

    async def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
    ) -> Order:
        """Submit a market order.

        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: "buy" or "sell"

        Returns:
            Order details
        """
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=Decimal(str(qty)),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

        order = self._client.submit_order(request)

        return Order(
            id=str(order.id),
            symbol=order.symbol,
            side=side,
            qty=float(order.qty),
            order_type="market",
            status=OrderStatus.PENDING,
            filled_qty=float(order.filled_qty) if order.filled_qty else 0,
            filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            submitted_at=order.submitted_at.isoformat() if order.submitted_at else "",
            filled_at=order.filled_at.isoformat() if order.filled_at else None,
        )

    async def get_order(self, order_id: str) -> Order:
        """Get order status.

        Args:
            order_id: Order ID

        Returns:
            Order details with current status
        """
        order = self._client.get_order_by_id(order_id)

        status_map = {
            "new": OrderStatus.PENDING,
            "pending_new": OrderStatus.PENDING,
            "accepted": OrderStatus.OPEN,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }

        return Order(
            id=str(order.id),
            symbol=order.symbol,
            side="buy" if order.side == OrderSide.BUY else "sell",
            qty=float(order.qty),
            order_type=order.type.value if order.type else "market",
            status=status_map.get(order.status.value, OrderStatus.PENDING),
            filled_qty=float(order.filled_qty) if order.filled_qty else 0,
            filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            submitted_at=order.submitted_at.isoformat() if order.submitted_at else "",
            filled_at=order.filled_at.isoformat() if order.filled_at else None,
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID

        Returns:
            True if cancelled successfully
        """
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    async def close_position(self, symbol: str) -> Order | None:
        """Close entire position for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Closing order or None if no position
        """
        try:
            order = self._client.close_position(symbol)
            return Order(
                id=str(order.id),
                symbol=order.symbol,
                side="sell" if order.side == OrderSide.SELL else "buy",
                qty=float(order.qty),
                order_type="market",
                status=OrderStatus.PENDING,
                filled_qty=float(order.filled_qty) if order.filled_qty else 0,
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                submitted_at=order.submitted_at.isoformat() if order.submitted_at else "",
            )
        except Exception:
            return None
