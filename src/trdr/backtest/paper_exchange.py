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

import numpy as np

from ..data.market import Bar, Symbol
from ..strategy.types import Position as StrategyPosition, Signal, SignalAction
from .calendar import filter_trading_bars, get_trading_days_in_year
from .orders import Order, OrderManager, OrderType
from .portfolio import Portfolio

if TYPE_CHECKING:
    from ..strategy import BaseStrategy


@dataclass(frozen=True)
class PaperExchangeConfig:
    """Configuration for paper exchange.

    Args:
        symbol: Asset symbol (e.g., "crypto:ETH/USD", "stock:AAPL")
        warmup_bars: Bars to skip before generating signals
        transaction_cost_pct: Cost per trade as decimal (0.0025 = 0.25%)
        slippage_pct: Slippage as % of price (0.001 = 0.1%)
        default_position_pct: Default position size as % of equity
        initial_capital: Starting capital
    """

    symbol: str
    warmup_bars: int = 65
    transaction_cost_pct: float = 0.0025
    slippage_pct: float = 0.001
    default_position_pct: float = 1.0
    initial_capital: float = 10000.0

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

    @property
    def total_trades(self) -> int:
        """Number of completed trades."""
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        """Number of winning trades."""
        return sum(1 for t in self.trades if t.is_winner)

    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return self.total_trades - self.winning_trades

    @property
    def win_rate(self) -> float:
        """Win rate as decimal."""
        if not self.trades:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_pnl(self) -> float:
        """Total net P&L."""
        return sum(t.net_pnl for t in self.trades)

    @property
    def profit_factor(self) -> float:
        """Net profits / net losses. >1 is profitable."""
        profits = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        losses = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        if losses == 0:
            return float("inf") if profits > 0 else 0.0
        return profits / losses

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as decimal."""
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 1.0
            max_dd = max(max_dd, min(dd, 1.0))
        return max_dd

    @property
    def sortino_ratio(self) -> float | None:
        """Annualized Sortino ratio."""
        if len(self.trades) < 2:
            return None
        returns = [t.net_pnl / (t.entry_price * t.quantity) for t in self.trades]
        downside = [r for r in returns if r < 0]
        if not downside:
            return float("inf") if np.mean(returns) > 0 else None
        downside_std = np.std(downside)
        if downside_std == 0:
            return None
        days = get_trading_days_in_year(self.config.asset_type)
        return float(np.mean(returns) / downside_std * np.sqrt(days))


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

    def run(self, bars: list[Bar]) -> PaperExchangeResult:
        """Run strategy over bar data.

        Args:
            bars: Historical bars (oldest first)

        Returns:
            PaperExchangeResult with trades and metrics
        """
        # Filter to trading days
        filtered_bars = filter_trading_bars(bars, self.config.asset_type)

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

        for i in range(self.config.warmup_bars, len(filtered_bars)):
            bar = filtered_bars[i]
            slippage = bar.close * self.config.slippage_pct

            # 1. Update trailing stops
            order_manager.update_trailing_stops(bar)

            # 2. Process fills
            fills = order_manager.process_bar(bar, slippage)
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

                        trades.append(Trade(
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
                        ))
                        entry_time = ""
                        entry_cost = 0.0

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
                visible_bars = filtered_bars[: i + 1]
                signal = self.strategy.generate_signal(visible_bars, strategy_position)

                # 5. Convert signal to orders
                self._process_signal(signal, portfolio, order_manager, bar, strategy_position)
                if signal.action == SignalAction.BUY and not strategy_position:
                    entry_reason = signal.reason

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

            # Submit market buy
            order_manager.submit(Order(
                symbol=self.config.symbol,
                side="buy",
                order_type=OrderType.MARKET,
                quantity=qty,
                created_at=bar.timestamp,
            ))

            # Submit stop loss if specified
            if signal.stop_loss:
                order_manager.submit(Order(
                    symbol=self.config.symbol,
                    side="sell",
                    order_type=OrderType.STOP_LOSS,
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
