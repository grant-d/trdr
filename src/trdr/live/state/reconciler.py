"""State reconciliation for live trading startup."""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..exchange.base import ExchangeInterface
from ..exchange.types import HydraAccountInfo, HydraOrderResponse, HydraPositionInfo
from ..orders.manager import OrderManager

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    """Get current UTC time."""
    return datetime.now(UTC)


@dataclass
class ReconciliationResult:
    """Result of state reconciliation.

    Args:
        success: Whether reconciliation succeeded
        account: Current account state
        positions: Current positions by symbol
        open_orders: Current open orders
        warnings: Non-fatal warnings
        errors: Fatal errors
        timestamp: When reconciliation was performed
    """

    success: bool
    account: HydraAccountInfo | None = None
    positions: dict[str, HydraPositionInfo] = field(default_factory=dict)
    open_orders: list[HydraOrderResponse] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_now_utc)

    @property
    def has_position(self) -> bool:
        """Check if any positions exist."""
        return len(self.positions) > 0

    @property
    def has_open_orders(self) -> bool:
        """Check if any open orders exist."""
        return len(self.open_orders) > 0

    def get_position(self, symbol: str) -> HydraPositionInfo | None:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def position_summary(self) -> str:
        """Get human-readable position summary."""
        if not self.positions:
            return "No positions"

        lines = []
        for sym, pos in self.positions.items():
            pnl_pct = (pos.unrealized_pnl / pos.market_value * 100) if pos.market_value else 0
            lines.append(
                f"  {sym}: {pos.qty} @ ${pos.avg_entry_price:.2f} "
                f"(${pos.unrealized_pnl:+.2f} / {pnl_pct:+.1f}%)"
            )
        return "\n".join(lines)


class StateReconciler:
    """Reconciles local state with exchange state on startup.

    Ensures the trading system has accurate state before processing signals.
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        order_manager: OrderManager | None = None,
    ):
        """Initialize reconciler.

        Args:
            exchange: Exchange interface
            order_manager: Optional order manager to sync
        """
        self._exchange = exchange
        self._order_manager = order_manager

    async def reconcile(self, symbol: str | None = None) -> ReconciliationResult:
        """Perform full state reconciliation.

        Args:
            symbol: Optional filter to specific symbol

        Returns:
            ReconciliationResult with current state
        """
        result = ReconciliationResult(success=True)
        logger.info("Starting state reconciliation...")

        # Get account info
        try:
            result.account = await self._exchange.get_account()
            logger.info(
                f"Account: equity=${result.account.equity:.2f}, cash=${result.account.cash:.2f}"
            )
        except Exception as e:
            result.errors.append(f"Failed to get account: {e}")
            result.success = False
            logger.error(f"Failed to get account: {e}")
            return result

        # Get positions
        try:
            all_positions = await self._exchange.get_positions()
            if symbol:
                if symbol in all_positions:
                    result.positions = {symbol: all_positions[symbol]}
            else:
                result.positions = all_positions

            if result.positions:
                logger.info(f"Found {len(result.positions)} position(s)")
                for sym, pos in result.positions.items():
                    logger.info(
                        f"  {sym}: {pos.qty} @ ${pos.avg_entry_price:.2f} "
                        f"(P&L: ${pos.unrealized_pnl:+.2f})"
                    )
            else:
                logger.info("No open positions")

        except Exception as e:
            result.warnings.append(f"Failed to get positions: {e}")
            logger.warning(f"Failed to get positions: {e}")

        # Get open orders
        try:
            result.open_orders = await self._exchange.get_open_orders(symbol)
            if result.open_orders:
                logger.info(f"Found {len(result.open_orders)} open order(s)")
                for order in result.open_orders:
                    logger.info(
                        f"  {order.side.value} {order.qty} {order.symbol} "
                        f"@ {order.order_type.value} - {order.status.value}"
                    )
            else:
                logger.info("No open orders")

        except Exception as e:
            result.warnings.append(f"Failed to get open orders: {e}")
            logger.warning(f"Failed to get open orders: {e}")

        # Sync order manager if provided
        if self._order_manager:
            try:
                await self._order_manager.sync_with_exchange()
                logger.info("Order manager synced")
            except Exception as e:
                result.warnings.append(f"Failed to sync order manager: {e}")
                logger.warning(f"Failed to sync order manager: {e}")

        logger.info(
            f"Reconciliation complete: "
            f"{len(result.positions)} positions, "
            f"{len(result.open_orders)} open orders"
        )

        return result

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all open orders.

        Args:
            symbol: Optional filter to specific symbol

        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        open_orders = await self._exchange.get_open_orders(symbol)

        for order in open_orders:
            try:
                if await self._exchange.cancel_order(order.order_id):
                    cancelled += 1
                    logger.info(f"Cancelled order {order.order_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel order {order.order_id}: {e}")

        return cancelled

    async def close_position(self, symbol: str) -> bool:
        """Close position for symbol with market order.

        Args:
            symbol: Symbol to close

        Returns:
            True if position closed or no position
        """
        from ..exchange.types import HydraOrderRequest, HydraOrderSide, HydraOrderType

        position = await self._exchange.get_position(symbol)
        if not position or position.qty <= 0:
            return True

        # Determine side (opposite of position)
        side = HydraOrderSide.SELL if position.side == "long" else HydraOrderSide.BUY

        try:
            order = HydraOrderRequest(
                symbol=symbol,
                side=side,
                qty=position.qty,
                order_type=HydraOrderType.MARKET,
            )
            response = await self._exchange.submit_order(order)
            logger.info(
                f"Close order submitted: {response.order_id} {side.value} {position.qty} {symbol}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return False

    async def ensure_clean_state(
        self,
        symbol: str,
        cancel_orders: bool = True,
        close_position: bool = False,
    ) -> ReconciliationResult:
        """Ensure clean trading state before starting.

        Args:
            symbol: Symbol to clean up
            cancel_orders: Cancel open orders for symbol
            close_position: Close position for symbol

        Returns:
            ReconciliationResult after cleanup
        """
        if cancel_orders:
            cancelled = await self.cancel_all_orders(symbol)
            logger.info(f"Cancelled {cancelled} orders for {symbol}")

        if close_position:
            closed = await self.close_position(symbol)
            if not closed:
                logger.warning(f"Failed to close position for {symbol}")

        return await self.reconcile(symbol)
