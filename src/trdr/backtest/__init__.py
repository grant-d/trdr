"""Backtesting framework for strategy validation."""

from .align import align_feeds
from .calendar import filter_trading_bars, get_trading_days_in_year, is_trading_day
from .orders import Fill, Order, OrderManager, OrderType
from .paper_exchange import PaperExchange, RuntimeContext
from .portfolio import Portfolio, Position, PositionEntry
from .types import PaperExchangeConfig, PaperExchangeResult, Trade
from .walk_forward import (
    Fold,
    WalkForwardConfig,
    WalkForwardResult,
    generate_folds,
    run_walk_forward,
)

__all__ = [
    # Paper exchange
    "PaperExchange",
    "PaperExchangeConfig",
    "PaperExchangeResult",
    "RuntimeContext",
    "Trade",
    # Walk forward
    "Fold",
    "WalkForwardConfig",
    "WalkForwardResult",
    "generate_folds",
    "run_walk_forward",
    # Orders
    "Order",
    "OrderType",
    "OrderManager",
    "Fill",
    # Portfolio
    "Portfolio",
    "Position",
    "PositionEntry",
    # Calendar
    "is_trading_day",
    "filter_trading_bars",
    "get_trading_days_in_year",
    # Alignment
    "align_feeds",
]
