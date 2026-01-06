"""Live trading module.

Provides infrastructure for running strategies against real exchanges.
"""

from .config import AlpacaCredentials, LiveConfig, RiskLimits
from .exchange import (
    AlpacaExchange,
    AuthenticationError,
    ExchangeError,
    ExchangeInterface,
    HydraAccountInfo,
    HydraBar,
    HydraConnectionError,
    HydraFill,
    HydraOrderRequest,
    HydraOrderResponse,
    HydraOrderSide,
    HydraOrderStatus,
    HydraOrderType,
    HydraPositionInfo,
    InsufficientFundsError,
    OrderRejectedError,
    RateLimitError,
)
from .harness import HarnessState, LiveHarness
from .orders import (
    LiveOrder,
    NonRetryableError,
    OrderManager,
    OrderState,
    RetryableError,
    RetryPolicy,
)
from .safety import CircuitBreaker
from .state import (
    LiveContextBuilder,
    LiveRuntimeContext,
    ReconciliationResult,
    StateReconciler,
)

__all__ = [
    # Config
    "AlpacaCredentials",
    "LiveConfig",
    "RiskLimits",
    # Exchange
    "AlpacaExchange",
    "AuthenticationError",
    "ExchangeError",
    "ExchangeInterface",
    "HydraAccountInfo",
    "HydraBar",
    "HydraConnectionError",
    "HydraFill",
    "HydraOrderRequest",
    "HydraOrderResponse",
    "HydraOrderSide",
    "HydraOrderStatus",
    "HydraOrderType",
    "HydraPositionInfo",
    "InsufficientFundsError",
    "OrderRejectedError",
    "RateLimitError",
    # Harness
    "HarnessState",
    "LiveHarness",
    # Orders
    "LiveOrder",
    "NonRetryableError",
    "OrderManager",
    "OrderState",
    "RetryableError",
    "RetryPolicy",
    # Safety
    "CircuitBreaker",
    # State
    "LiveContextBuilder",
    "LiveRuntimeContext",
    "ReconciliationResult",
    "StateReconciler",
]
