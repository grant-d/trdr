"""Order types for live trading."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from ..exchange.types import (
    HydraOrderRequest,
    HydraOrderResponse,
    HydraOrderSide,
    HydraOrderStatus,
)


def _now_utc() -> datetime:
    """Get current UTC time."""
    return datetime.now(UTC)


class OrderState(Enum):
    """Internal order state for tracking."""

    PENDING_SUBMIT = "pending_submit"
    SUBMITTED = "submitted"
    PENDING_CANCEL = "pending_cancel"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class LiveOrder:
    """Live order with tracking metadata.

    Wraps exchange order with internal state tracking.

    Args:
        request: Original order request
        state: Internal tracking state
        exchange_response: Response from exchange
        created_at: When order was created locally
        submitted_at: When order was submitted to exchange
        completed_at: When order reached terminal state
        cancel_requested: Whether cancel was requested
        error: Error message if failed
        reason: Optional reason for the order (e.g., "entry", "stop_loss")
    """

    request: HydraOrderRequest
    state: OrderState = OrderState.PENDING_SUBMIT
    exchange_response: HydraOrderResponse | None = None
    created_at: datetime = field(default_factory=_now_utc)
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    cancel_requested: bool = False
    error: str | None = None
    reason: str | None = None

    @property
    def order_id(self) -> str | None:
        """Get exchange order ID."""
        return self.exchange_response.order_id if self.exchange_response else None

    @property
    def client_order_id(self) -> str:
        """Get client order ID."""
        return self.request.client_order_id

    @property
    def symbol(self) -> str:
        """Get symbol."""
        return self.request.symbol

    @property
    def side(self) -> HydraOrderSide:
        """Get order side."""
        return self.request.side

    @property
    def qty(self) -> float:
        """Get order quantity."""
        return self.request.qty

    @property
    def filled_qty(self) -> float:
        """Get filled quantity."""
        if self.exchange_response:
            return self.exchange_response.filled_qty
        return 0.0

    @property
    def filled_avg_price(self) -> float | None:
        """Get average fill price."""
        if self.exchange_response:
            return self.exchange_response.filled_avg_price
        return None

    @property
    def status(self) -> HydraOrderStatus | None:
        """Get exchange order status."""
        if self.exchange_response:
            return self.exchange_response.status
        return None

    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        if self.state in (OrderState.COMPLETE, OrderState.FAILED):
            return True
        if self.exchange_response:
            terminal_statuses = {
                HydraOrderStatus.FILLED,
                HydraOrderStatus.CANCELED,
                HydraOrderStatus.REJECTED,
                HydraOrderStatus.EXPIRED,
            }
            return self.exchange_response.status in terminal_statuses
        return False

    @property
    def is_filled(self) -> bool:
        """Check if order is filled."""
        if self.exchange_response:
            return self.exchange_response.status == HydraOrderStatus.FILLED
        return False

    @property
    def is_active(self) -> bool:
        """Check if order is still active (pending/partial fill)."""
        if self.is_terminal:
            return False
        if self.exchange_response:
            active_statuses = {
                HydraOrderStatus.NEW,
                HydraOrderStatus.ACCEPTED,
                HydraOrderStatus.PENDING_NEW,
                HydraOrderStatus.PARTIALLY_FILLED,
            }
            return self.exchange_response.status in active_statuses
        return self.state == OrderState.SUBMITTED

    def update_from_response(self, response: HydraOrderResponse) -> None:
        """Update order from exchange response.

        Args:
            response: Exchange order response
        """
        self.exchange_response = response

        if response.status in {
            HydraOrderStatus.FILLED,
            HydraOrderStatus.CANCELED,
            HydraOrderStatus.REJECTED,
            HydraOrderStatus.EXPIRED,
        }:
            self.state = OrderState.COMPLETE
            self.completed_at = datetime.now(UTC)
