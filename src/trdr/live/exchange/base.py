"""Base exchange interface."""

from abc import ABC, abstractmethod
from typing import Callable

from .types import (
    HydraAccountInfo,
    HydraBar,
    HydraFill,
    HydraOrderRequest,
    HydraOrderResponse,
    HydraPositionInfo,
)


class ExchangeInterface(ABC):
    """Abstract base class for exchange implementations.

    Defines unified interface for paper (backtest) and live exchanges.
    All methods are async for compatibility with real exchange APIs.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to exchange.

        Initializes API clients and WebSocket streams.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from exchange.

        Closes WebSocket streams and cleans up resources.
        """
        ...

    @abstractmethod
    async def get_account(self) -> HydraAccountInfo:
        """Get account information.

        Returns:
            HydraAccountInfo with equity, cash, buying power.
        """
        ...

    @abstractmethod
    async def get_positions(self) -> dict[str, HydraPositionInfo]:
        """Get all open positions.

        Returns:
            Dict mapping symbol to HydraPositionInfo.
        """
        ...

    @abstractmethod
    async def get_position(self, symbol: str) -> HydraPositionInfo | None:
        """Get position for specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            HydraPositionInfo or None if no position.
        """
        ...

    @abstractmethod
    async def submit_order(self, order: HydraOrderRequest) -> HydraOrderResponse:
        """Submit order to exchange.

        Args:
            order: Order request

        Returns:
            HydraOrderResponse with order ID and initial status.

        Raises:
            OrderRejectedError: If order is rejected
            ExchangeError: For other exchange errors
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order.

        Args:
            order_id: Exchange order ID

        Returns:
            True if cancelled, False if not found or already filled.
        """
        ...

    @abstractmethod
    async def get_order(self, order_id: str) -> HydraOrderResponse | None:
        """Get order status.

        Args:
            order_id: Exchange order ID

        Returns:
            HydraOrderResponse or None if not found.
        """
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[HydraOrderResponse]:
        """Get all open orders.

        Args:
            symbol: Optional filter by symbol

        Returns:
            List of open orders.
        """
        ...

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[HydraBar]:
        """Get historical bars.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")
            limit: Number of bars to fetch

        Returns:
            List of bars, oldest first.
        """
        ...

    @abstractmethod
    async def get_latest_bar(self, symbol: str) -> HydraBar | None:
        """Get latest bar for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest HydraBar or None.
        """
        ...

    @abstractmethod
    async def get_latest_price(self, symbol: str) -> float | None:
        """Get latest price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest price or None.
        """
        ...

    def subscribe_fills(self, callback: Callable[[HydraFill], None]) -> None:
        """Subscribe to fill events.

        Args:
            callback: Function called on each fill event.
        """
        pass

    def unsubscribe_fills(self) -> None:
        """Unsubscribe from fill events."""
        pass


class ExchangeError(Exception):
    """Base exception for exchange errors."""

    pass


class OrderRejectedError(ExchangeError):
    """Order was rejected by exchange."""

    def __init__(self, message: str, reason: str | None = None):
        super().__init__(message)
        self.reason = reason


class InsufficientFundsError(OrderRejectedError):
    """Insufficient funds for order."""

    pass


class RateLimitError(ExchangeError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class HydraConnectionError(ExchangeError):
    """Connection to exchange failed."""

    pass


class AuthenticationError(ExchangeError):
    """Authentication failed."""

    pass
