"""Order manager for live trading."""

import logging
from datetime import UTC, datetime
from typing import Callable

from ..exchange.base import (
    ExchangeInterface,
    InsufficientFundsError,
    OrderRejectedError,
    RateLimitError,
)
from ..exchange.types import (
    HydraFill,
    HydraOrderRequest,
    HydraOrderResponse,
    HydraOrderStatus,
)
from .retry import RetryableError, RetryPolicy, retry_async
from .types import LiveOrder, OrderState

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order lifecycle for live trading.

    Handles order submission, tracking, cancellation, and fill detection.
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        retry_policy: RetryPolicy | None = None,
        audit_log_path: str | None = None,
    ):
        """Initialize order manager.

        Args:
            exchange: Exchange interface for order operations
            retry_policy: Retry configuration (default provided)
            audit_log_path: Path for audit logging
        """
        self._exchange = exchange
        self._retry_policy = retry_policy or RetryPolicy()
        self._audit_log_path = audit_log_path

        # Order tracking
        self._orders: dict[str, LiveOrder] = {}  # client_order_id -> order
        self._order_id_map: dict[str, str] = {}  # exchange_order_id -> client_order_id

        # Callbacks
        self._fill_callbacks: list[Callable[[LiveOrder, HydraFill], None]] = []
        self._order_callbacks: list[Callable[[LiveOrder], None]] = []

    @property
    def orders(self) -> dict[str, LiveOrder]:
        """Get all tracked orders."""
        return self._orders.copy()

    @property
    def active_orders(self) -> list[LiveOrder]:
        """Get all active (non-terminal) orders."""
        return [o for o in self._orders.values() if o.is_active]

    @property
    def pending_orders(self) -> list[LiveOrder]:
        """Get pending orders (not yet submitted)."""
        return [o for o in self._orders.values() if o.state == OrderState.PENDING_SUBMIT]

    def get_order(self, client_order_id: str) -> LiveOrder | None:
        """Get order by client order ID."""
        return self._orders.get(client_order_id)

    def get_order_by_exchange_id(self, order_id: str) -> LiveOrder | None:
        """Get order by exchange order ID."""
        client_id = self._order_id_map.get(order_id)
        if client_id:
            return self._orders.get(client_id)
        return None

    async def submit_order(
        self,
        request: HydraOrderRequest,
        reason: str | None = None,
    ) -> LiveOrder:
        """Submit order to exchange.

        Args:
            request: Order request
            reason: Optional reason tag (e.g., "entry", "stop_loss")

        Returns:
            LiveOrder with tracking state
        """
        # Create tracked order
        order = LiveOrder(request=request, reason=reason)
        self._orders[request.client_order_id] = order

        self._audit_log(
            f"SUBMIT {request.side.value.upper()} {request.qty} {request.symbol} "
            f"@ {request.order_type.value} {request.limit_price or 'market'}"
        )

        try:
            # Submit with retry
            result = await retry_async(
                operation=lambda: self._exchange.submit_order(request),
                policy=self._retry_policy,
                retryable_exceptions=(RateLimitError, RetryableError),
                operation_name=f"submit_order({request.symbol})",
            )

            if result.success and result.result:
                order.exchange_response = result.result
                order.state = OrderState.SUBMITTED
                order.submitted_at = datetime.now(UTC)
                self._order_id_map[result.result.order_id] = request.client_order_id

                self._audit_log(
                    f"SUBMITTED order_id={result.result.order_id} "
                    f"status={result.result.status.value}"
                )
                self._notify_order_update(order)
            else:
                order.state = OrderState.FAILED
                order.error = str(result.last_error)
                order.completed_at = datetime.now(UTC)
                self._audit_log(f"FAILED: {result.last_error}")

        except InsufficientFundsError as e:
            order.state = OrderState.FAILED
            order.error = f"Insufficient funds: {e}"
            order.completed_at = datetime.now(UTC)
            self._audit_log(f"REJECTED (insufficient funds): {e}")

        except OrderRejectedError as e:
            order.state = OrderState.FAILED
            order.error = f"Rejected: {e}"
            order.completed_at = datetime.now(UTC)
            self._audit_log(f"REJECTED: {e}")

        except Exception as e:
            order.state = OrderState.FAILED
            order.error = str(e)
            order.completed_at = datetime.now(UTC)
            self._audit_log(f"ERROR: {e}")
            logger.exception(f"Failed to submit order: {e}")

        return order

    async def cancel_order(self, client_order_id: str) -> bool:
        """Cancel order by client order ID.

        Args:
            client_order_id: Client order ID

        Returns:
            True if cancel succeeded or order already terminal
        """
        order = self._orders.get(client_order_id)
        if not order:
            logger.warning(f"Order not found: {client_order_id}")
            return False

        if order.is_terminal:
            return True

        if not order.order_id:
            # Not yet submitted, just mark as failed
            order.state = OrderState.FAILED
            order.completed_at = datetime.now(UTC)
            return True

        order.cancel_requested = True
        order.state = OrderState.PENDING_CANCEL

        self._audit_log(f"CANCEL order_id={order.order_id}")

        try:
            result = await retry_async(
                operation=lambda: self._exchange.cancel_order(order.order_id),
                policy=self._retry_policy,
                retryable_exceptions=(RateLimitError, RetryableError),
                operation_name=f"cancel_order({order.order_id})",
            )

            if result.success:
                self._audit_log(f"CANCELLED order_id={order.order_id}")
                # Refresh order status
                await self._refresh_order(order)
                return True
            else:
                self._audit_log(f"CANCEL FAILED: {result.last_error}")
                return False

        except Exception as e:
            self._audit_log(f"CANCEL ERROR: {e}")
            logger.exception(f"Failed to cancel order: {e}")
            return False

    async def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all active orders.

        Args:
            symbol: Optional filter by symbol

        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        for order in self.active_orders:
            if symbol and order.symbol != symbol:
                continue
            if await self.cancel_order(order.client_order_id):
                cancelled += 1
        return cancelled

    async def refresh_orders(self) -> None:
        """Refresh status of all active orders."""
        for order in self.active_orders:
            await self._refresh_order(order)

    async def _refresh_order(self, order: LiveOrder) -> None:
        """Refresh single order status from exchange."""
        if not order.order_id:
            return

        try:
            response = await self._exchange.get_order(order.order_id)
            if response:
                old_status = order.status
                order.update_from_response(response)

                if old_status != response.status:
                    self._notify_order_update(order)
                    self._audit_log(
                        f"STATUS order_id={order.order_id} "
                        f"{old_status.value if old_status else 'None'} -> "
                        f"{response.status.value}"
                    )

                    # Check for fill
                    if response.status == HydraOrderStatus.FILLED:
                        self._handle_fill(order, response)

        except Exception as e:
            logger.warning(f"Failed to refresh order {order.order_id}: {e}")

    def _handle_fill(self, order: LiveOrder, response: HydraOrderResponse) -> None:
        """Handle order fill event."""
        fill = HydraFill(
            order_id=response.order_id,
            symbol=response.symbol,
            side=response.side,
            qty=response.filled_qty,
            price=response.filled_avg_price or 0.0,
            timestamp=response.filled_at or datetime.now(UTC),
        )

        self._audit_log(f"FILL order_id={order.order_id} {fill.qty} @ ${fill.price:.2f}")

        for callback in self._fill_callbacks:
            try:
                callback(order, fill)
            except Exception as e:
                logger.exception(f"Fill callback error: {e}")

    def on_fill(self, callback: Callable[[LiveOrder, HydraFill], None]) -> None:
        """Register fill callback.

        Args:
            callback: Function called on each fill
        """
        self._fill_callbacks.append(callback)

    def on_order_update(self, callback: Callable[[LiveOrder], None]) -> None:
        """Register order update callback.

        Args:
            callback: Function called on order state changes
        """
        self._order_callbacks.append(callback)

    def _notify_order_update(self, order: LiveOrder) -> None:
        """Notify callbacks of order update."""
        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.exception(f"Order callback error: {e}")

    def _audit_log(self, message: str) -> None:
        """Write to audit log."""
        timestamp = datetime.now(UTC).isoformat()
        log_line = f"{timestamp} {message}"
        logger.info(log_line)

        if self._audit_log_path:
            try:
                with open(self._audit_log_path, "a") as f:
                    f.write(log_line + "\n")
            except Exception as e:
                logger.warning(f"Failed to write audit log: {e}")

    async def sync_with_exchange(self) -> None:
        """Sync order state with exchange.

        Fetches open orders from exchange and reconciles with local state.
        """
        try:
            open_orders = await self._exchange.get_open_orders()

            # Create set of known exchange IDs
            known_ids = set(self._order_id_map.keys())

            for response in open_orders:
                if response.order_id in known_ids:
                    # Update existing order
                    order = self.get_order_by_exchange_id(response.order_id)
                    if order:
                        order.update_from_response(response)
                else:
                    # Unknown order - create tracking entry
                    logger.info(f"Found untracked order: {response.order_id}")
                    request = HydraOrderRequest(
                        symbol=response.symbol,
                        side=response.side,
                        qty=response.qty,
                        order_type=response.order_type,
                        limit_price=response.limit_price,
                        stop_price=response.stop_price,
                        client_order_id=response.client_order_id,
                    )
                    order = LiveOrder(
                        request=request,
                        state=OrderState.SUBMITTED,
                        exchange_response=response,
                    )
                    self._orders[response.client_order_id] = order
                    self._order_id_map[response.order_id] = response.client_order_id

        except Exception as e:
            logger.exception(f"Failed to sync with exchange: {e}")

    def clear_completed(self) -> int:
        """Remove completed orders from tracking.

        Returns:
            Number of orders removed
        """
        to_remove = [client_id for client_id, order in self._orders.items() if order.is_terminal]

        for client_id in to_remove:
            order = self._orders.pop(client_id)
            if order.order_id and order.order_id in self._order_id_map:
                del self._order_id_map[order.order_id]

        return len(to_remove)
