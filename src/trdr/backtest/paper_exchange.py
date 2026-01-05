"""Paper exchange engine with advanced order types and position management.

Simple API for LLM strategy writers:
- Strategy implements generate_signal(bars, position) -> Signal
- Engine handles orders, portfolio, calendar internally

Example:
    config = PaperExchangeConfig(symbol="crypto:ETH/USD")
    engine = PaperExchange(config, strategy)
    result = engine.run(bars)
"""

from typing import TYPE_CHECKING

from ..core import Symbol
from ..data import Bar
from ..strategy.types import Position as StrategyPosition
from ..strategy.types import Signal, SignalAction
from .calendar import filter_trading_bars
from .metrics import TradeMetrics
from .orders import Order, OrderManager, OrderType
from .portfolio import Portfolio
from .types import EntryPlan, PaperExchangeConfig, PaperExchangeResult, Trade

if TYPE_CHECKING:
    from ..strategy import BaseStrategy


def _validate_order_direction(
    side: str,
    order_type: OrderType,
    price: float,
    stop_price: float | None,
    limit_price: float | None,
) -> str | None:
    """Validate order direction matches exchange semantics.

    Returns error message if invalid, None if valid.
    """
    if order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
        if stop_price is None:
            return None  # Will fail later validation
        if side == "buy" and stop_price <= price:
            return f"Buy stop must be above price ({stop_price} <= {price})"
        if side == "sell" and stop_price >= price:
            return f"Sell stop must be below price ({stop_price} >= {price})"

    if order_type == OrderType.LIMIT:
        if limit_price is None:
            return None
        if side == "buy" and limit_price >= price:
            return f"Buy limit must be below price ({limit_price} >= {price})"
        if side == "sell" and limit_price <= price:
            return f"Sell limit must be above price ({limit_price} <= {price})"

    return None


class RuntimeContext:
    """Live portfolio state available to strategy during generate_signal().

    Provides access to portfolio, orders, trades, and computed metrics.
    All stats are computed on-demand from current state via TradeMetrics.

    Example:
        def generate_signal(self, bars, position):
            if self.context.drawdown > 0.1:
                return Signal(action=SignalAction.HOLD, ...)  # pause during drawdown
            if self.context.win_rate < 0.4 and self.context.total_trades > 10:
                # reduce size after poor performance
                return Signal(..., position_size_pct=0.5)
    """

    def __init__(
        self,
        portfolio: Portfolio,
        order_manager: OrderManager,
        trades: list[Trade],
        equity_curve: list[float],
        config: PaperExchangeConfig,
        current_bar: Bar,
        bar_index: int,
        total_bars: int,
        start_time: str,
        strategy_name: str = "",
    ):
        self._portfolio = portfolio
        self._orders = order_manager
        self._trades = trades
        self._equity_curve = equity_curve
        self._config = config
        self._current_bar = current_bar
        self._bar_index = bar_index
        self._total_bars = total_bars
        self._start_time = start_time
        self._strategy_name = strategy_name
        self._metrics = TradeMetrics(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=config.initial_capital,
            asset_type=config.asset_type,
            start_time=start_time,
            end_time=current_bar.timestamp,
        )

    # Run params
    @property
    def strategy_name(self) -> str:
        """Strategy name."""
        return self._strategy_name

    @property
    def symbol(self) -> Symbol:
        """Trading symbol."""
        return self._config.symbol

    @property
    def current_bar(self) -> Bar:
        """Current bar being processed."""
        return self._current_bar

    @property
    def bar_index(self) -> int:
        """Current bar index (0-based)."""
        return self._bar_index

    @property
    def total_bars(self) -> int:
        """Total bars in run."""
        return self._total_bars

    @property
    def bars_remaining(self) -> int:
        """Bars remaining in run."""
        return self._total_bars - self._bar_index - 1

    # Portfolio state
    @property
    def positions(self) -> dict:
        """Open positions by symbol."""
        return self._portfolio.positions.copy()

    @property
    def pending_orders(self) -> list[Order]:
        """Pending stop/limit orders."""
        return self._orders.pending_orders

    @property
    def trades(self) -> list[Trade]:
        """Completed trades."""
        return self._trades

    @property
    def equity(self) -> float:
        """Current portfolio equity."""
        return self._portfolio.equity({str(self._config.symbol): self._current_bar.close})

    @property
    def cash(self) -> float:
        """Available cash."""
        return self._portfolio.cash

    @property
    def initial_capital(self) -> float:
        """Starting capital."""
        return self._config.initial_capital

    # Delegate computed stats to TradeMetrics
    @property
    def total_trades(self) -> int:
        return self._metrics.total_trades

    @property
    def winning_trades(self) -> int:
        return self._metrics.winning_trades

    @property
    def losing_trades(self) -> int:
        return self._metrics.losing_trades

    @property
    def win_rate(self) -> float:
        return self._metrics.win_rate

    @property
    def total_pnl(self) -> float:
        return self._metrics.total_pnl

    @property
    def profit_factor(self) -> float:
        return self._metrics.profit_factor

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak as decimal."""
        return self._metrics.current_drawdown(self.equity)

    @property
    def max_drawdown(self) -> float:
        return self._metrics.max_drawdown

    @property
    def total_return(self) -> float:
        """Total return based on current equity."""
        return self._metrics.total_return_from_equity(self.equity)

    @property
    def avg_trade_pnl(self) -> float:
        return self._metrics.avg_trade_pnl

    @property
    def avg_win(self) -> float:
        return self._metrics.avg_win

    @property
    def avg_loss(self) -> float:
        return self._metrics.avg_loss

    @property
    def largest_win(self) -> float:
        return self._metrics.largest_win

    @property
    def largest_loss(self) -> float:
        return self._metrics.largest_loss

    @property
    def expectancy(self) -> float:
        return self._metrics.expectancy

    @property
    def max_consecutive_wins(self) -> int:
        return self._metrics.max_consecutive_wins

    @property
    def max_consecutive_losses(self) -> int:
        return self._metrics.max_consecutive_losses

    @property
    def total_costs(self) -> float:
        return self._metrics.total_costs

    @property
    def sharpe_ratio(self) -> float | None:
        return self._metrics.sharpe_ratio

    @property
    def sortino_ratio(self) -> float | None:
        return self._metrics.sortino_ratio

    @property
    def cagr(self) -> float | None:
        """CAGR from start to current bar."""
        return self._metrics.cagr_live(self.equity, self._current_bar.timestamp)

    @property
    def calmar_ratio(self) -> float | None:
        """Calmar ratio using live CAGR."""
        cagr_val = self.cagr
        max_dd = self.max_drawdown
        if cagr_val is None or max_dd == 0:
            return float("inf") if cagr_val and cagr_val > 0 else None
        return cagr_val / max_dd


class PaperExchange:
    """Paper exchange engine with orders, portfolio, and calendar.

    Simple interface: strategy.generate_signal() returns Signal,
    engine handles everything else.

    Args:
        config: Exchange configuration
        strategy: Strategy instance
    """

    def __init__(self, config: PaperExchangeConfig, strategy: "BaseStrategy"):
        """Initialize paper exchange."""
        self.config = config
        self.strategy = strategy

    def run(self, bars: dict[str, list[Bar]] | list[Bar]) -> PaperExchangeResult:
        """Run strategy over bar data.

        Args:
            bars: Dict of bars keyed by "symbol:timeframe" (e.g., "crypto:ETH/USD:15m")
                  OR single list of bars (auto-wrapped using primary_feed key)

        Returns:
            PaperExchangeResult with trades and metrics
        """
        # Auto-wrap single feed if list provided
        if isinstance(bars, list):
            primary_feed = self.config.primary_feed
            bars = {str(primary_feed): bars}

        # Extract primary bars using primary_feed
        primary_feed_key = str(self.config.primary_feed)
        primary_bars = bars[primary_feed_key]

        # Filter primary bars to trading days
        filtered_bars = filter_trading_bars(primary_bars, self.config.asset_type)

        if len(filtered_bars) < self.config.warmup_bars + 1:
            return PaperExchangeResult(
                trades=[],
                config=self.config,
                start_time=filtered_bars[0].timestamp if filtered_bars else "",
                end_time=filtered_bars[-1].timestamp if filtered_bars else "",
            )

        # Initialize components
        portfolio = Portfolio(cash=self.config.initial_capital)
        order_manager = OrderManager()
        trades: list[Trade] = []
        equity_curve: list[float] = []

        # Extract symbol string from primary feed
        symbol_str = str(self.config.primary_feed.symbol)

        # Track open trade info
        entry_time: str = ""
        entry_reason: str = ""
        entry_cost: float = 0.0
        entry_stop_loss: float | None = None
        entry_take_profit: float | None = None
        # OCO: Map order_id -> EntryPlan to submit exits only after entry fills
        pending_entry_plans: dict[str, EntryPlan] = {}

        for i in range(self.config.warmup_bars, len(filtered_bars)):
            bar = filtered_bars[i]
            slippage = bar.close * self.config.slippage_pct

            # 1. Update trailing stops
            order_manager.update_trailing_stops(bar)

            # 2. Process fills
            fills = order_manager.process_bar(bar, slippage)
            position_closed = False
            for fill in fills:
                cost = fill.price * fill.quantity * self.config.transaction_cost_pct

                if fill.side == "buy":
                    # Opening or adding to position
                    portfolio.open_position(
                        symbol=self.config.symbol,
                        side="long",
                        price=fill.price,
                        quantity=fill.quantity,
                        timestamp=fill.timestamp,
                        cost=cost,
                    )
                    # OCO: Check if this fill has a pending entry plan (limit/stop-limit)
                    # If yes, NOW submit the exit orders since entry actually filled
                    plan = pending_entry_plans.pop(fill.order_id, None)
                    if plan:
                        self._submit_exit_orders(
                            qty=plan.quantity,
                            stop_loss=plan.stop_loss,
                            take_profit=plan.take_profit,
                            trailing_stop=plan.trailing_stop,
                            order_manager=order_manager,
                            price=fill.price,
                            timestamp=fill.timestamp,
                        )
                        if not entry_time:
                            entry_time = fill.timestamp
                            entry_cost = cost
                            entry_reason = plan.reason
                            entry_stop_loss = plan.stop_loss
                            entry_take_profit = plan.take_profit
                    elif not entry_time:
                        # Fallback for direct order_manager usage (no plan)
                        entry_time = fill.timestamp
                        entry_cost = cost
                        entry_reason = "direct_order"
                else:
                    # Closing position
                    position = portfolio.get_position(self.config.symbol)
                    if position:
                        avg_entry = position.avg_price
                        qty = min(fill.quantity, position.total_quantity)
                        pnl = portfolio.close_position(
                            symbol=self.config.symbol,
                            price=fill.price,
                            quantity=qty,
                            cost=cost,
                        )

                        # Capture order reason (fallback to order type)
                        exit_reason = fill.reason or str(fill.order_type.value)

                        trade = Trade(
                            entry_time=entry_time,
                            exit_time=fill.timestamp,
                            entry_price=avg_entry,
                            exit_price=fill.price,
                            quantity=qty,
                            side="long",
                            gross_pnl=pnl + cost + entry_cost,
                            costs=cost + entry_cost,
                            net_pnl=pnl,
                            entry_reason=entry_reason,
                            exit_reason=exit_reason,
                            stop_loss=entry_stop_loss,
                            take_profit=entry_take_profit,
                        )
                        trades.append(trade)
                        self.strategy.on_trade_complete(trade.net_pnl, trade.exit_reason)
                        entry_time = ""
                        entry_cost = 0.0
                        entry_stop_loss = None
                        entry_take_profit = None
                        position_closed = True

            # Pseudo-OCO: cancel remaining orders when position closes
            if position_closed:
                order_manager.cancel_all()
                # Clear any pending entry plans (entry never filled or was cancelled)
                pending_entry_plans.clear()

            # 3. Get current position for strategy
            pos = portfolio.get_position(self.config.symbol)
            strategy_position = None
            if pos:
                strategy_position = StrategyPosition(
                    symbol=pos.symbol,
                    side=pos.side,
                    size=pos.total_quantity,
                    entry_price=pos.avg_price,
                    stop_loss=0.0,
                    take_profit=None,
                )

            # 4. Generate signal (skip on last bar)
            if i < len(filtered_bars) - 1:
                # Build dict of visible bars for all feeds
                # Primary uses filtered bars; informative feeds are pre-aligned
                visible_bars: dict[str, list[Bar]] = {}
                for key, feed_bars in bars.items():
                    if key == primary_feed_key:
                        visible_bars[key] = filtered_bars[: i + 1]
                    else:
                        # Informative feeds aligned to primary, slice same length
                        visible_bars[key] = feed_bars[: i + 1]

                # Set runtime context for strategy
                self.strategy.context = RuntimeContext(
                    portfolio=portfolio,
                    order_manager=order_manager,
                    trades=trades,
                    equity_curve=equity_curve,
                    config=self.config,
                    current_bar=bar,
                    bar_index=i - self.config.warmup_bars,
                    total_bars=len(filtered_bars) - self.config.warmup_bars,
                    start_time=filtered_bars[self.config.warmup_bars].timestamp,
                    strategy_name=self.strategy.name,
                )

                signal = self.strategy.generate_signal(visible_bars, strategy_position)

                # 5. Convert signal to orders
                # OCO: _process_signal returns entry plans instead of submitting exits
                entry_plan = self._process_signal(
                    signal, portfolio, order_manager, bar, strategy_position
                )
                if entry_plan:
                    # Store the plan - exits will be submitted when entry fills
                    pending_entry_plans.update(entry_plan)
                if signal.action == SignalAction.CLOSE:
                    # Manual close: abandon any pending entry plans
                    pending_entry_plans.clear()

            # 6. Record equity
            equity_curve.append(portfolio.equity({symbol_str: bar.close}))

        # Force close any open position at end
        final_bar = filtered_bars[-1]
        position = portfolio.get_position(self.config.symbol)
        if position:
            # Save position info before closing (close_position clears entries)
            pos_qty = position.total_quantity
            pos_avg = position.avg_price
            cost = final_bar.close * pos_qty * self.config.transaction_cost_pct
            pnl = portfolio.close_position(
                symbol=self.config.symbol,
                price=final_bar.close,
                cost=cost,
            )
            trades.append(
                Trade(
                    entry_time=entry_time,
                    exit_time=final_bar.timestamp,
                    entry_price=pos_avg,
                    exit_price=final_bar.close,
                    quantity=pos_qty,
                    side="long",
                    gross_pnl=pnl + cost + entry_cost,
                    costs=cost + entry_cost,
                    net_pnl=pnl,
                    entry_reason=entry_reason,
                    exit_reason="end_of_data",
                    stop_loss=entry_stop_loss,
                    take_profit=entry_take_profit,
                )
            )
            if equity_curve:
                equity_curve[-1] = portfolio.equity({symbol_str: final_bar.close})

        return PaperExchangeResult(
            trades=trades,
            config=self.config,
            start_time=filtered_bars[self.config.warmup_bars].timestamp,
            end_time=filtered_bars[-1].timestamp,
            equity_curve=equity_curve,
        )

    def _process_signal(
        self,
        signal: Signal,
        portfolio: Portfolio,
        order_manager: OrderManager,
        bar: Bar,
        position: StrategyPosition | None,
    ) -> dict[str, EntryPlan]:
        """Convert signal to orders.

        Args:
            signal: Strategy signal
            portfolio: Current portfolio
            order_manager: Order manager
            bar: Current bar
            position: Current position

        Returns:
            Mapping of entry order IDs to EntryPlan for delayed exit orders.
        """
        if signal.action == SignalAction.HOLD:
            return {}

        if signal.action == SignalAction.BUY:
            # Calculate quantity
            if signal.quantity:
                qty = signal.quantity
            else:
                size_pct = signal.position_size_pct * self.config.default_position_pct
                qty = (portfolio.buying_power() * size_pct) / bar.close

            if qty <= 0:
                return {}

            # Submit entry order (stop-limit, limit, or market)
            if signal.stop_price and signal.limit_price:
                # Stop-limit: triggered at stop_price, fills at limit_price
                err = _validate_order_direction(
                    "buy",
                    OrderType.STOP_LIMIT,
                    bar.close,
                    signal.stop_price,
                    signal.limit_price,
                )
                if err:
                    raise ValueError(f"Invalid buy stop-limit: {err}")
                order_id = order_manager.submit(
                    Order(
                        symbol=self.config.symbol,
                        side="buy",
                        order_type=OrderType.STOP_LIMIT,
                        quantity=qty,
                        stop_price=signal.stop_price,
                        limit_price=signal.limit_price,
                        created_at=bar.timestamp,
                    )
                )
            elif signal.limit_price:
                err = _validate_order_direction(
                    "buy",
                    OrderType.LIMIT,
                    bar.close,
                    None,
                    signal.limit_price,
                )
                if err:
                    raise ValueError(f"Invalid buy limit: {err}")
                order_id = order_manager.submit(
                    Order(
                        symbol=self.config.symbol,
                        side="buy",
                        order_type=OrderType.LIMIT,
                        quantity=qty,
                        limit_price=signal.limit_price,
                        created_at=bar.timestamp,
                    )
                )
            else:
                order_id = order_manager.submit(
                    Order(
                        symbol=self.config.symbol,
                        side="buy",
                        order_type=OrderType.MARKET,
                        quantity=qty,
                        created_at=bar.timestamp,
                    )
                )

            # Validate exit parameters against signal price
            if signal.stop_loss:
                err = _validate_order_direction(
                    "sell",
                    OrderType.STOP,
                    bar.close,
                    signal.stop_loss,
                    None,
                )
                if err:
                    raise ValueError(f"Invalid stop loss: {err}")

            if signal.take_profit:
                err = _validate_order_direction(
                    "sell",
                    OrderType.LIMIT,
                    bar.close,
                    None,
                    signal.take_profit,
                )
                if err:
                    raise ValueError(f"Invalid take profit: {err}")

            return {
                order_id: EntryPlan(
                    quantity=qty,
                    reason=signal.reason,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    trailing_stop=signal.trailing_stop,
                )
            }

        elif signal.action == SignalAction.CLOSE and position:
            # Cancel any pending stop orders
            order_manager.cancel_all()

            # Submit market sell
            qty = signal.quantity if signal.quantity else position.size
            order_manager.submit(
                Order(
                    symbol=self.config.symbol,
                    side="sell",
                    order_type=OrderType.MARKET,
                    quantity=qty,
                    created_at=bar.timestamp,
                )
            )

        elif signal.action == SignalAction.SELL and position:
            # Partial sell
            qty = signal.quantity if signal.quantity else position.size
            order_manager.submit(
                Order(
                    symbol=self.config.symbol,
                    side="sell",
                    order_type=OrderType.MARKET,
                    quantity=qty,
                    created_at=bar.timestamp,
                )
            )

        return {}

    def _submit_exit_orders(
        self,
        qty: float,
        stop_loss: float | None,
        take_profit: float | None,
        trailing_stop: float | None,
        order_manager: OrderManager,
        price: float,
        timestamp: str,
    ) -> None:
        """Submit exit orders after entry fills."""
        if stop_loss:
            # Gap-through detection: Entry filled at/below stop loss
            # Submit immediate market exit instead of raising
            if price <= stop_loss:
                order_manager.submit(
                    Order(
                        symbol=self.config.symbol,
                        side="sell",
                        order_type=OrderType.MARKET,
                        quantity=qty,
                        created_at=timestamp,
                        reason="stop_loss_gap",
                    )
                )
                return
            else:
                err = _validate_order_direction(
                    "sell",
                    OrderType.STOP,
                    price,
                    stop_loss,
                    None,
                )
                if err:
                    raise ValueError(f"Invalid stop loss: {err}")
                order_manager.submit(
                    Order(
                        symbol=self.config.symbol,
                        side="sell",
                        order_type=OrderType.STOP,
                        quantity=qty,
                        stop_price=stop_loss,
                        created_at=timestamp,
                    )
                )

        if trailing_stop:
            if trailing_stop < 1:
                order_manager.submit(
                    Order(
                        symbol=self.config.symbol,
                        side="sell",
                        order_type=OrderType.TRAILING_STOP,
                        quantity=qty,
                        trail_percent=trailing_stop,
                        created_at=timestamp,
                    )
                )
            else:
                order_manager.submit(
                    Order(
                        symbol=self.config.symbol,
                        side="sell",
                        order_type=OrderType.TRAILING_STOP,
                        quantity=qty,
                        trail_amount=trailing_stop,
                        created_at=timestamp,
                    )
                )

        if take_profit:
            # Gap-through detection: Entry filled at/above take profit
            # Submit immediate market exit instead of raising
            if price >= take_profit:
                order_manager.submit(
                    Order(
                        symbol=self.config.symbol,
                        side="sell",
                        order_type=OrderType.MARKET,
                        quantity=qty,
                        created_at=timestamp,
                        reason="take_profit_gap",
                    )
                )
                return
            else:
                err = _validate_order_direction(
                    "sell",
                    OrderType.LIMIT,
                    price,
                    None,
                    take_profit,
                )
                if err:
                    raise ValueError(f"Invalid take profit: {err}")
                order_manager.submit(
                    Order(
                        symbol=self.config.symbol,
                        side="sell",
                        order_type=OrderType.LIMIT,
                        quantity=qty,
                        limit_price=take_profit,
                    created_at=timestamp,
                )
            )
