"""Shared types for exchange interfaces.

All types prefixed with Hydra to avoid conflicts with alpaca-py types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4


class HydraOrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class HydraOrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class HydraOrderStatus(Enum):
    """Order status."""

    NEW = "new"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    PENDING_CANCEL = "pending_cancel"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class HydraBar:
    """OHLCV bar data.

    Args:
        timestamp: Bar timestamp (ISO format)
        open: Open price
        high: High price
        low: Low price
        close: Close price
        volume: Volume
    """

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class HydraAccountInfo:
    """Account information from exchange.

    Args:
        equity: Total account equity
        cash: Available cash
        buying_power: Available buying power
        currency: Account currency (e.g., "USD")
    """

    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"


@dataclass
class HydraPositionInfo:
    """Position information from exchange.

    Args:
        symbol: Trading symbol
        side: Position side ("long" or "short")
        qty: Position quantity
        avg_entry_price: Average entry price
        market_value: Current market value
        unrealized_pnl: Unrealized P&L
        qty_available: Quantity available to sell
    """

    symbol: str
    side: Literal["long", "short"]
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float
    qty_available: float


@dataclass
class HydraOrderRequest:
    """Order submission request.

    Args:
        symbol: Trading symbol
        side: Order side (buy/sell)
        qty: Order quantity
        order_type: Order type
        limit_price: Limit price (for limit/stop-limit orders)
        stop_price: Stop price (for stop/stop-limit orders)
        take_profit: Take profit price (for bracket orders)
        stop_loss: Stop loss price (for bracket orders)
        time_in_force: Order duration (GTC, DAY, IOC, FOK)
        client_order_id: Optional client-generated order ID
    """

    symbol: str
    side: HydraOrderSide
    qty: float
    order_type: HydraOrderType = HydraOrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    time_in_force: str = "gtc"
    client_order_id: str = field(default_factory=lambda: str(uuid4())[:12])


@dataclass
class HydraOrderResponse:
    """Order response from exchange.

    Args:
        order_id: Exchange order ID
        client_order_id: Client order ID
        symbol: Trading symbol
        side: Order side
        qty: Order quantity
        filled_qty: Filled quantity
        filled_avg_price: Average fill price
        status: Order status
        order_type: Order type
        submitted_at: Submission timestamp
        filled_at: Fill timestamp (if filled)
    """

    order_id: str
    client_order_id: str
    symbol: str
    side: HydraOrderSide
    qty: float
    filled_qty: float
    filled_avg_price: float | None
    status: HydraOrderStatus
    order_type: HydraOrderType
    submitted_at: datetime
    filled_at: datetime | None = None
    limit_price: float | None = None
    stop_price: float | None = None


@dataclass
class HydraFill:
    """Order fill event.

    Args:
        order_id: Exchange order ID
        symbol: Trading symbol
        side: Order side
        qty: Filled quantity
        price: Fill price
        timestamp: Fill timestamp
    """

    order_id: str
    symbol: str
    side: HydraOrderSide
    qty: float
    price: float
    timestamp: datetime
