"""Paper exchange engine with advanced order types and position management.

Simple API for LLM strategy writers:
- Strategy implements generate_signal(bars, position) -> Signal
- Engine handles orders, portfolio, calendar internally

Example:
    config = PaperExchangeConfig(symbol="crypto:ETH/USD")
    engine = PaperExchange(config, strategy)
    result = engine.run(bars)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from ..data import Bar
from ..core import Symbol
from ..strategy.types import Position as StrategyPosition
from ..strategy.types import Signal, SignalAction
from .calendar import filter_trading_bars
from .metrics import TradeMetrics
from .orders import Order, OrderManager, OrderType
from .portfolio import Portfolio

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


@dataclass(frozen=True)
class PaperExchangeConfig:
    """Configuration for paper exchange.

    Args:
        symbol: Asset symbol (e.g., "crypto:ETH/USD", "stock:AAPL")
        primary_feed: Key for primary data feed (e.g., "crypto:ETH/USD:15m")
        warmup_bars: Bars to skip before generating signals
        transaction_cost_pct: Cost per trade as decimal (0.0025 = 0.25%)
        slippage_pct: Slippage as % of price (0.001 = 0.1%)
        default_position_pct: Default position size as % of equity
        initial_capital: Starting capital
    """

    symbol: str
    primary_feed: str = ""
    warmup_bars: int = 65
    transaction_cost_pct: float = 0.0025
    slippage_pct: float = 0.001
    default_position_pct: float = 1.0
    initial_capital: float = 10_000.0

    @property
    def asset_type(self) -> str:
        """Get asset type from symbol."""
        return Symbol.parse(self.symbol).asset_type


@dataclass
class Trade:
    """Completed trade record.

    Args:
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        entry_price: Average entry price
        exit_price: Exit price
        quantity: Position size
        side: "long" or "short"
        gross_pnl: P&L before costs
        costs: Total transaction costs
        net_pnl: P&L after costs
        entry_reason: Why we entered
        exit_reason: Why we exited
        stop_loss: Stop loss price at entry
        take_profit: Take profit price at entry
    """

    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    gross_pnl: float
    costs: float
    net_pnl: float
    entry_reason: str
    exit_reason: str
    stop_loss: float | None = None
    take_profit: float | None = None

    @property
    def duration_hours(self) -> float:
        """Trade duration in hours."""
        entry = datetime.fromisoformat(self.entry_time.replace("Z", "+00:00"))
        exit_dt = datetime.fromisoformat(self.exit_time.replace("Z", "+00:00"))
        return (exit_dt - entry).total_seconds() / 3600

    @property
    def is_winner(self) -> bool:
        """True if trade was profitable."""
        return self.net_pnl > 0


@dataclass
class PaperExchangeResult:
    """Results from paper exchange run.

    Args:
        trades: Completed trades
        config: Configuration used
        start_time: First bar timestamp
        end_time: Last bar timestamp
        equity_curve: Equity at each bar
    """

    trades: list[Trade]
    config: PaperExchangeConfig
    start_time: str
    end_time: str
    equity_curve: list[float] = field(default_factory=list)
    _metrics: TradeMetrics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Create metrics calculator."""
        object.__setattr__(self, "_metrics", TradeMetrics(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_capital=self.config.initial_capital,
            asset_type=self.config.asset_type,
            start_time=self.start_time,
            end_time=self.end_time,
        ))

    # Delegate all metrics to TradeMetrics
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
    def max_drawdown(self) -> float:
        return self._metrics.max_drawdown

    @property
    def sortino_ratio(self) -> float | None:
        return self._metrics.sortino_ratio

    @property
    def sharpe_ratio(self) -> float | None:
        return self._metrics.sharpe_ratio

    @property
    def calmar_ratio(self) -> float | None:
        return self._metrics.calmar_ratio

    @property
    def total_return(self) -> float:
        return self._metrics.total_return

    @property
    def cagr(self) -> float | None:
        return self._metrics.cagr

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
    def avg_trade_duration_hours(self) -> float:
        return self._metrics.avg_trade_duration_hours

    @property
    def max_consecutive_wins(self) -> int:
        return self._metrics.max_consecutive_wins

    @property
    def max_consecutive_losses(self) -> int:
        return self._metrics.max_consecutive_losses

    @property
    def expectancy(self) -> float:
        return self._metrics.expectancy

    @property
    def trades_per_year(self) -> float:
        return self._metrics.trades_per_year

    @property
    def total_costs(self) -> float:
        return self._metrics.total_costs

    def print_trades(self) -> None:
        """Print detailed trade log to stdout.

        Shows entry/exit times, prices, SL/TP levels, P&L, and reasons.
        Useful for debugging strategy behavior.
        """
        if not self.trades:
            print("No trades")
            return

        print(f"\n{'='*80}")
        print(f"TRADE LOG ({len(self.trades)} trades)")
        print(f"{'='*80}")

        for i, t in enumerate(self.trades, 1):
            # Parse dates for cleaner display
            entry_dt = t.entry_time[:16].replace("T", " ")
            exit_dt = t.exit_time[:16].replace("T", " ")

            result = "WIN" if t.is_winner else "LOSS"
            pnl_sign = "+" if t.net_pnl >= 0 else ""

            print(f"\n#{i} [{result}] {pnl_sign}${t.net_pnl:.2f}")
            print(f"  Entry: {entry_dt} @ ${t.entry_price:.2f}")
            print(f"  Exit:  {exit_dt} @ ${t.exit_price:.2f}")

            if t.stop_loss is not None or t.take_profit is not None:
                sl_str = f"${t.stop_loss:.2f}" if t.stop_loss else "—"
                tp_str = f"${t.take_profit:.2f}" if t.take_profit else "—"
                print(f"  SL: {sl_str}  |  TP: {tp_str}")

            print(f"  Reason: {t.entry_reason}")
            print(f"  Exit:   {t.exit_reason}")
            print(f"  Duration: {t.duration_hours:.1f}h  |  Qty: {t.quantity:.4f}")

        print(f"\n{'='*80}")
        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]
        print(f"Summary: {len(winners)}W / {len(losers)}L  |  WR: {self.win_rate:.1%}")
        print(f"{'='*80}\n")


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
    def symbol(self) -> str:
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
        return self._portfolio.equity({self._config.symbol: self._current_bar.close})

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

    def run(self, bars: dict[str, list[Bar]]) -> PaperExchangeResult:
        """Run strategy over bar data.

        Args:
            bars: Dict of bars keyed by "symbol:timeframe" (e.g., "crypto:ETH/USD:15m")

        Returns:
            PaperExchangeResult with trades and metrics
        """
        # Extract primary bars using primary_feed
        primary_feed = self.config.primary_feed
        if not primary_feed:
            # Fallback: use first key (for backwards compat with single-feed)
            primary_feed = next(iter(bars.keys()))
        primary_bars = bars[primary_feed]

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

        # Track open trade info
        entry_time: str = ""
        entry_reason: str = ""
        entry_cost: float = 0.0
        entry_stop_loss: float | None = None
        entry_take_profit: float | None = None

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
                    if not entry_time:
                        entry_time = fill.timestamp
                        entry_cost = cost
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

                        # Find the order that triggered this fill
                        exit_reason = "signal"
                        for order in order_manager.fills:
                            if order.order_id == fill.order_id:
                                if hasattr(order, "order_type"):
                                    exit_reason = str(order.order_type)

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
                    if key == primary_feed:
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
                self._process_signal(signal, portfolio, order_manager, bar, strategy_position)
                if signal.action == SignalAction.BUY and not strategy_position:
                    entry_reason = signal.reason
                    entry_stop_loss = signal.stop_loss
                    entry_take_profit = signal.take_profit

            # 6. Record equity
            prices = {self.config.symbol: bar.close}
            equity_curve.append(portfolio.equity(prices))

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
            trades.append(Trade(
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
            ))
            if equity_curve:
                equity_curve[-1] = portfolio.equity({self.config.symbol: final_bar.close})

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
    ) -> None:
        """Convert signal to orders.

        Args:
            signal: Strategy signal
            portfolio: Current portfolio
            order_manager: Order manager
            bar: Current bar
            position: Current position
        """
        if signal.action == SignalAction.HOLD:
            return

        if signal.action == SignalAction.BUY:
            # Calculate quantity
            if signal.quantity:
                qty = signal.quantity
            else:
                size_pct = signal.position_size_pct * self.config.default_position_pct
                qty = (portfolio.buying_power() * size_pct) / bar.close

            if qty <= 0:
                return

            # Submit entry order (stop-limit, limit, or market)
            if signal.stop_price and signal.limit_price:
                # Stop-limit: triggered at stop_price, fills at limit_price
                err = _validate_order_direction(
                    "buy", OrderType.STOP_LIMIT, bar.close,
                    signal.stop_price, signal.limit_price,
                )
                if err:
                    raise ValueError(f"Invalid buy stop-limit: {err}")
                order_manager.submit(Order(
                    symbol=self.config.symbol,
                    side="buy",
                    order_type=OrderType.STOP_LIMIT,
                    quantity=qty,
                    stop_price=signal.stop_price,
                    limit_price=signal.limit_price,
                    created_at=bar.timestamp,
                ))
            elif signal.limit_price:
                err = _validate_order_direction(
                    "buy", OrderType.LIMIT, bar.close,
                    None, signal.limit_price,
                )
                if err:
                    raise ValueError(f"Invalid buy limit: {err}")
                order_manager.submit(Order(
                    symbol=self.config.symbol,
                    side="buy",
                    order_type=OrderType.LIMIT,
                    quantity=qty,
                    limit_price=signal.limit_price,
                    created_at=bar.timestamp,
                ))
            else:
                order_manager.submit(Order(
                    symbol=self.config.symbol,
                    side="buy",
                    order_type=OrderType.MARKET,
                    quantity=qty,
                    created_at=bar.timestamp,
                ))

            # Submit stop loss if specified (sell stop below price)
            if signal.stop_loss:
                err = _validate_order_direction(
                    "sell", OrderType.STOP, bar.close,
                    signal.stop_loss, None,
                )
                if err:
                    raise ValueError(f"Invalid stop loss: {err}")
                order_manager.submit(Order(
                    symbol=self.config.symbol,
                    side="sell",
                    order_type=OrderType.STOP,
                    quantity=qty,
                    stop_price=signal.stop_loss,
                    created_at=bar.timestamp,
                ))

            # Submit trailing stop if specified
            if signal.trailing_stop:
                # Determine if it's a percent or absolute amount
                if signal.trailing_stop < 1:
                    # Treat as percent
                    order_manager.submit(Order(
                        symbol=self.config.symbol,
                        side="sell",
                        order_type=OrderType.TRAILING_STOP,
                        quantity=qty,
                        trail_percent=signal.trailing_stop,
                        created_at=bar.timestamp,
                    ))
                else:
                    # Treat as absolute amount
                    order_manager.submit(Order(
                        symbol=self.config.symbol,
                        side="sell",
                        order_type=OrderType.TRAILING_STOP,
                        quantity=qty,
                        trail_amount=signal.trailing_stop,
                        created_at=bar.timestamp,
                    ))

            # Submit take profit if specified (sell limit above price)
            if signal.take_profit:
                err = _validate_order_direction(
                    "sell", OrderType.LIMIT, bar.close,
                    None, signal.take_profit,
                )
                if err:
                    raise ValueError(f"Invalid take profit: {err}")
                order_manager.submit(Order(
                    symbol=self.config.symbol,
                    side="sell",
                    order_type=OrderType.LIMIT,
                    quantity=qty,
                    limit_price=signal.take_profit,
                    created_at=bar.timestamp,
                ))

        elif signal.action == SignalAction.CLOSE and position:
            # Cancel any pending stop orders
            order_manager.cancel_all()

            # Submit market sell
            qty = signal.quantity if signal.quantity else position.size
            order_manager.submit(Order(
                symbol=self.config.symbol,
                side="sell",
                order_type=OrderType.MARKET,
                quantity=qty,
                created_at=bar.timestamp,
            ))

        elif signal.action == SignalAction.SELL and position:
            # Partial sell
            qty = signal.quantity if signal.quantity else position.size
            order_manager.submit(Order(
                symbol=self.config.symbol,
                side="sell",
                order_type=OrderType.MARKET,
                quantity=qty,
                created_at=bar.timestamp,
            ))
