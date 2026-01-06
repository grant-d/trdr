"""Live context builder for strategy runtime."""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..exchange.base import ExchangeInterface
from ..exchange.types import HydraAccountInfo, HydraBar
from ..orders.manager import OrderManager
from ..safety.circuit_breaker import CircuitBreaker

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class LivePosition:
    """Live position for strategy context.

    Compatible with backtest Position interface.
    """

    symbol: str
    quantity: float
    avg_entry_price: float
    side: str  # "long" or "short"
    unrealized_pnl: float = 0.0
    market_value: float = 0.0

    @property
    def entry_price(self) -> float:
        """Alias for avg_entry_price."""
        return self.avg_entry_price


@dataclass
class LiveTrade:
    """Completed trade record.

    Compatible with backtest Trade interface.
    """

    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    entry_time: datetime
    exit_time: datetime
    reason: str = ""


@dataclass
class LiveRuntimeContext:
    """Runtime context for live trading.

    Provides strategy with access to live trading state.
    Designed to be compatible with backtest RuntimeContext interface.

    Args:
        account: Current account state
        positions: Current positions by symbol
        pending_orders: Pending orders
        trades: Completed trades this session
        equity_curve: Equity history
        current_bar: Latest bar
        symbol: Trading symbol
        bar_index: Number of bars processed
        strategy_name: Strategy identifier
        is_live: Whether trading live (vs paper)
    """

    account: HydraAccountInfo
    positions: dict[str, LivePosition] = field(default_factory=dict)
    pending_orders: list = field(default_factory=list)
    trades: list[LiveTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    current_bar: HydraBar | None = None
    symbol: str = ""
    bar_index: int = 0
    total_bars: int = 0
    strategy_name: str = ""
    is_live: bool = False
    start_time: str = ""

    @property
    def equity(self) -> float:
        """Current account equity."""
        return self.account.equity

    @property
    def cash(self) -> float:
        """Available cash."""
        return self.account.cash

    @property
    def buying_power(self) -> float:
        """Buying power."""
        return self.account.buying_power

    @property
    def bars_remaining(self) -> int:
        """Bars remaining (always 0 for live)."""
        return 0

    # Trade metrics (computed on demand)
    @property
    def total_trades(self) -> int:
        """Total completed trades."""
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        """Number of winning trades."""
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return sum(1 for t in self.trades if t.pnl < 0)

    @property
    def win_rate(self) -> float:
        """Win rate (0-1)."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_pnl(self) -> float:
        """Total P&L from completed trades."""
        return sum(t.pnl for t in self.trades)

    @property
    def drawdown(self) -> float:
        """Current drawdown from high water mark."""
        if not self.equity_curve:
            return 0.0
        hwm = max(self.equity_curve)
        if hwm <= 0:
            return 0.0
        return (hwm - self.equity) / hwm

    @property
    def sharpe(self) -> float:
        """Sharpe ratio (requires sufficient trades)."""
        if len(self.trades) < 10:
            return 0.0
        import statistics

        returns = [t.pnl / t.entry_price / t.quantity for t in self.trades]
        if len(returns) < 2:
            return 0.0
        try:
            mean_ret = statistics.mean(returns)
            std_ret = statistics.stdev(returns)
            if std_ret == 0:
                return 0.0
            return mean_ret / std_ret
        except Exception:
            return 0.0


class LiveContextBuilder:
    """Builds RuntimeContext from live trading state.

    Assembles context from exchange, order manager, and circuit breaker state.
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        order_manager: OrderManager,
        circuit_breaker: CircuitBreaker,
        symbol: str,
        strategy_name: str = "",
    ):
        """Initialize context builder.

        Args:
            exchange: Exchange interface
            order_manager: Order manager
            circuit_breaker: Circuit breaker
            symbol: Trading symbol
            strategy_name: Strategy name
        """
        self._exchange = exchange
        self._order_manager = order_manager
        self._circuit_breaker = circuit_breaker
        self._symbol = symbol
        self._strategy_name = strategy_name
        self._trades: list[LiveTrade] = []
        self._equity_curve: list[float] = []
        self._bar_count = 0
        self._start_time = datetime.now(UTC).isoformat()

    def record_trade(self, trade: LiveTrade) -> None:
        """Record a completed trade.

        Args:
            trade: Completed trade
        """
        self._trades.append(trade)
        self._circuit_breaker.record_trade(trade.pnl)

    def record_equity(self, equity: float) -> None:
        """Record equity snapshot.

        Args:
            equity: Current equity
        """
        self._equity_curve.append(equity)
        self._circuit_breaker.update_equity(equity)

    async def build(
        self,
        current_bar: HydraBar | None = None,
    ) -> LiveRuntimeContext:
        """Build runtime context from current state.

        Args:
            current_bar: Current bar (optional, will fetch if not provided)

        Returns:
            LiveRuntimeContext for strategy
        """
        # Get account
        account = await self._exchange.get_account()

        # Get positions
        positions = {}
        exchange_positions = await self._exchange.get_positions()
        for sym, pos in exchange_positions.items():
            positions[sym] = LivePosition(
                symbol=sym,
                quantity=pos.qty,
                avg_entry_price=pos.avg_entry_price,
                side=pos.side,
                unrealized_pnl=pos.unrealized_pnl,
                market_value=pos.market_value,
            )

        # Get pending orders
        pending = self._order_manager.active_orders

        # Get current bar if not provided
        if current_bar is None:
            current_bar = await self._exchange.get_latest_bar(self._symbol)

        self._bar_count += 1

        # Record equity
        self.record_equity(account.equity)

        return LiveRuntimeContext(
            account=account,
            positions=positions,
            pending_orders=pending,
            trades=self._trades.copy(),
            equity_curve=self._equity_curve.copy(),
            current_bar=current_bar,
            symbol=self._symbol,
            bar_index=self._bar_count,
            total_bars=self._bar_count,
            strategy_name=self._strategy_name,
            is_live=not getattr(self._exchange, "is_paper", True),
            start_time=self._start_time,
        )

    def reset(self) -> None:
        """Reset context builder state."""
        self._trades = []
        self._equity_curve = []
        self._bar_count = 0
        self._start_time = datetime.now(UTC).isoformat()
