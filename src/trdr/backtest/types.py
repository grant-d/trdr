"""Data types for backtest engine."""

from dataclasses import dataclass, field
from datetime import datetime

from ..core import Feed, Symbol
from .metrics import TradeMetrics


@dataclass(frozen=True)
class PaperExchangeConfig:
    """Configuration for paper exchange.

    Args:
        primary_feed: Primary data feed (symbol + timeframe)
        warmup_bars: Bars to skip before generating signals
        transaction_cost_pct: Cost per trade as decimal (0.0025 = 0.25%)
        slippage_pct: Slippage as % of price (0.001 = 0.1%)
        default_position_pct: Default position size as % of equity
        initial_capital: Starting capital
    """

    primary_feed: Feed
    warmup_bars: int = 65
    transaction_cost_pct: float = 0.0025
    slippage_pct: float = 0.001
    default_position_pct: float = 1.0
    initial_capital: float = 10_000.0

    @property
    def symbol(self) -> Symbol:
        """Get symbol from primary feed."""
        return self.primary_feed.symbol

    @property
    def asset_type(self) -> str:
        """Get asset type from symbol."""
        return self.symbol.asset_type


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
        object.__setattr__(
            self,
            "_metrics",
            TradeMetrics(
                trades=self.trades,
                equity_curve=self.equity_curve,
                initial_capital=self.config.initial_capital,
                asset_type=self.config.asset_type,
                start_time=self.start_time,
                end_time=self.end_time,
            ),
        )

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

        print(f"\n{'=' * 80}")
        print(f"TRADE LOG ({len(self.trades)} trades)")
        print(f"{'=' * 80}")

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

        print(f"\n{'=' * 80}")
        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]
        print(f"Summary: {len(winners)}W / {len(losers)}L  |  WR: {self.win_rate:.1%}")
        print(f"{'=' * 80}\n")


@dataclass(frozen=True)
class EntryPlan:
    """Exit order settings to apply after entry fills.

    OCO (One-Cancels-Other) behavior: Exit orders (SL/TP) should only be
    submitted AFTER the entry order fills. Without this, a limit entry at 98
    with stop at 95 would submit both orders immediately - if price drops to 96,
    the stop triggers before the limit entry fills, creating phantom exits.

    This class stores exit parameters until the entry order actually fills.
    """

    quantity: float
    reason: str
    stop_loss: float | None
    take_profit: float | None
    trailing_stop: float | None
