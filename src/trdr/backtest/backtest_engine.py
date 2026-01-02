"""Backtesting engine with bar-by-bar replay."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ..data.market import Bar
from ..strategy.types import Position, Signal, SignalAction
from ..strategy.volume_area_breakout.strategy import calculate_atr

if TYPE_CHECKING:
    from ..strategy import BaseStrategy


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for backtesting.

    Engine-level config only. Strategy params belong in StrategyConfig.

    Args:
        symbol: Asset symbol (e.g., "crypto:BTC/USD", "AAPL")
        warmup_bars: Bars to skip before generating signals (default 65 = 50 VP + 15 ATR)
        transaction_cost_pct: Cost per trade as decimal (0.0025 = 0.25%)
        slippage_atr: Slippage as fraction of ATR (0.01 = 1% of ATR per fill)
        position_size: Fixed position size per trade
        initial_capital: Starting capital for equity curve (default 10000)
    """

    symbol: str
    warmup_bars: int = 65
    transaction_cost_pct: float = 0.0
    slippage_atr: float = 0.0
    position_size: float = 1.0
    initial_capital: float = 10000.0

    @classmethod
    def for_crypto(
        cls,
        symbol: str,
        transaction_cost_pct: float = 0.0025,
        **kwargs,
    ) -> "BacktestConfig":
        """Create config for crypto with default 0.25% transaction cost."""
        return cls(symbol=symbol, transaction_cost_pct=transaction_cost_pct, **kwargs)

    @classmethod
    def for_stock(cls, symbol: str, **kwargs) -> "BacktestConfig":
        """Create config for stocks with 0% transaction cost."""
        return cls(symbol=symbol, transaction_cost_pct=0.0, **kwargs)


@dataclass
class Trade:
    """Single completed trade.

    Args:
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        entry_price: Entry price
        exit_price: Exit price
        size: Position size
        side: "long" or "short"
        gross_pnl: P&L before costs
        costs: Transaction costs
        net_pnl: P&L after costs
        entry_reason: Why we entered
        exit_reason: Why we exited
    """

    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: float
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
class BacktestResult:
    """Backtest results with metrics.

    Args:
        trades: List of completed trades
        config: Config used for backtest
        start_time: First bar timestamp
        end_time: Last bar timestamp
    """

    trades: list[Trade]
    config: BacktestConfig
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
    def total_costs(self) -> float:
        """Total transaction costs."""
        return sum(t.costs for t in self.trades)

    @property
    def gross_pnl(self) -> float:
        """Total P&L before costs."""
        return sum(t.gross_pnl for t in self.trades)

    @property
    def profit_factor(self) -> float:
        """Net profits / net losses. >1 is profitable."""
        net_profits = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        net_losses = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        if net_losses == 0:
            return float("inf") if net_profits > 0 else 0.0
        return net_profits / net_losses

    @property
    def avg_trade_duration_hours(self) -> float:
        """Average trade duration in hours."""
        if not self.trades:
            return 0.0
        return sum(t.duration_hours for t in self.trades) / len(self.trades)

    @property
    def max_consecutive_losses(self) -> int:
        """Maximum consecutive losing trades."""
        if not self.trades:
            return 0
        max_streak = 0
        current_streak = 0
        for trade in self.trades:
            if not trade.is_winner:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as decimal, capped at 1.0 (100%).

        Computed relative to running peak equity. If equity goes negative,
        drawdown caps at 100% (total loss of capital).
        """
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 1.0
            max_dd = max(max_dd, min(dd, 1.0))  # Cap at 100%
        return max_dd

    @property
    def max_drawdown_abs(self) -> float:
        """Maximum drawdown in absolute dollars.

        This metric remains meaningful even when equity goes negative.
        """
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd_abs = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd_abs = peak - equity
            max_dd_abs = max(max_dd_abs, dd_abs)
        return max_dd_abs

    @property
    def sharpe_ratio(self) -> float | None:
        """Annualized Sharpe ratio (assuming hourly bars, 0% risk-free rate)."""
        if len(self.trades) < 2:
            return None
        returns = [t.net_pnl / (t.entry_price * t.size) for t in self.trades]
        if np.std(returns) == 0:
            return None
        # Annualize: sqrt(8760) for hourly (365 * 24 hours/year)
        return float(np.mean(returns) / np.std(returns) * np.sqrt(8760))

    @property
    def sortino_ratio(self) -> float | None:
        """Annualized Sortino ratio (downside deviation only)."""
        if len(self.trades) < 2:
            return None
        returns = [t.net_pnl / (t.entry_price * t.size) for t in self.trades]
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return float("inf") if np.mean(returns) > 0 else None
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return None
        return float(np.mean(returns) / downside_std * np.sqrt(8760))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": {
                "symbol": self.config.symbol,
                "warmup_bars": self.config.warmup_bars,
                "transaction_cost_pct": self.config.transaction_cost_pct,
                "position_size": self.config.position_size,
                "initial_capital": self.config.initial_capital,
            },
            "period": {
                "start": self.start_time,
                "end": self.end_time,
            },
            "metrics": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": round(self.win_rate, 4),
                "total_pnl": round(self.total_pnl, 2),
                "gross_pnl": round(self.gross_pnl, 2),
                "total_costs": round(self.total_costs, 2),
                "profit_factor": (
                    round(self.profit_factor, 4) if self.profit_factor != float("inf") else "inf"
                ),
                "avg_trade_duration_hours": round(self.avg_trade_duration_hours, 2),
                "max_consecutive_losses": self.max_consecutive_losses,
                "max_drawdown": round(self.max_drawdown, 4),
                "max_drawdown_abs": round(self.max_drawdown_abs, 2),
                "sharpe_ratio": round(self.sharpe_ratio, 4) if self.sharpe_ratio else None,
                "sortino_ratio": round(self.sortino_ratio, 4) if self.sortino_ratio else None,
            },
            "trades": [
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": round(t.entry_price, 2),
                    "exit_price": round(t.exit_price, 2),
                    "size": t.size,
                    "side": t.side,
                    "gross_pnl": round(t.gross_pnl, 2),
                    "costs": round(t.costs, 2),
                    "net_pnl": round(t.net_pnl, 2),
                    "entry_reason": t.entry_reason,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ],
        }


class BacktestEngine:
    """Bar-by-bar backtesting engine.

    Replays historical bars, calling strategy on point-in-time data only.
    No future data leakage - strategy sees bars[:current_idx+1].
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy: "BaseStrategy",
    ):
        """Initialize engine.

        Args:
            config: Backtest configuration (engine params only)
            strategy: BaseStrategy instance with its own config/params
        """
        self.config = config
        self.strategy = strategy

    def run(self, bars: list[Bar]) -> BacktestResult:
        """Run backtest on bar data.

        Signals are computed on bar close, but fills occur at NEXT bar open
        to avoid lookahead bias.
        Any open position is marked-to-market and closed at the final bar close.

        Args:
            bars: Historical bars (oldest first)

        Returns:
            BacktestResult with trades and metrics
        """
        if len(bars) < self.config.warmup_bars + 1:
            return BacktestResult(
                trades=[],
                config=self.config,
                start_time=bars[0].timestamp if bars else "",
                end_time=bars[-1].timestamp if bars else "",
            )

        trades: list[Trade] = []
        position: Position | None = None
        pending_entry: Signal | None = None
        pending_entry_time: str = ""
        pending_entry_cost: float = 0.0
        pending_buy: Signal | None = None
        pending_close: Signal | None = None
        equity = self.config.initial_capital
        equity_curve: list[float] = []

        # Start after warmup period.
        # Key invariant: signal generated at bar[i].close executes at bar[i+1].open.
        # We process bar[i] by first executing any pending order at bar[i].open,
        # then generating a new signal using bars[:i+1] for next-bar execution.
        for i in range(self.config.warmup_bars, len(bars)):
            current_bar = bars[i]

            # Calculate current ATR for slippage (using visible bars only)
            current_atr = calculate_atr(bars[: i + 1]) if self.config.slippage_atr > 0 else 0.0

            # Execute pending orders at current bar open (queued on prior bar close)
            if pending_buy and not position:
                entry_price = self._apply_slippage(current_bar.open, current_atr, is_buy=True)
                entry_cost = self._calc_cost(entry_price, self.config.position_size)

                position = Position(
                    symbol=self.config.symbol,
                    side="long",
                    size=self.config.position_size,
                    entry_price=entry_price,
                    stop_loss=pending_buy.stop_loss or 0,
                    take_profit=pending_buy.take_profit,
                )
                pending_entry = pending_buy
                pending_entry_time = current_bar.timestamp
                pending_entry_cost = entry_cost
                pending_buy = None

            elif pending_close and position:
                exit_price = self._apply_slippage(current_bar.open, current_atr, is_buy=False)
                exit_cost = self._calc_cost(exit_price, position.size)
                total_cost = pending_entry_cost + exit_cost

                if position.side == "long":
                    gross_pnl = (exit_price - position.entry_price) * position.size
                else:
                    gross_pnl = (position.entry_price - exit_price) * position.size

                net_pnl = gross_pnl - total_cost

                trade = Trade(
                    entry_time=pending_entry_time,
                    exit_time=current_bar.timestamp,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    size=position.size,
                    side=position.side,
                    gross_pnl=gross_pnl,
                    costs=total_cost,
                    net_pnl=net_pnl,
                    entry_reason=pending_entry.reason if pending_entry else "",
                    exit_reason=pending_close.reason,
                )
                trades.append(trade)
                equity += net_pnl
                position = None
                pending_entry = None
                pending_close = None

            # Point-in-time data only - no future leakage
            visible_bars = bars[: i + 1]

            # Skip signal generation on final bar - no next bar exists for execution.
            # Any open position will be force-closed as mark-to-market below.
            if i < len(bars) - 1:
                # Get signal from strategy (using current bar close)
                signal = self.strategy.generate_signal(visible_bars, position)

                # Queue signals for next-bar execution
                if signal.action == SignalAction.BUY and not position and not pending_buy:
                    pending_buy = signal

                elif signal.action == SignalAction.CLOSE and position and not pending_close:
                    pending_close = signal

            equity_curve.append(equity)

        # Force-close any open position at final bar's close as end-of-test mark-to-market.
        # Signals are not generated on the final bar, so this does not create same-bar signal fills.
        if position:
            final_bar = bars[-1]
            final_atr = calculate_atr(bars) if self.config.slippage_atr > 0 else 0.0
            exit_price = self._apply_slippage(final_bar.close, final_atr, is_buy=False)
            exit_cost = self._calc_cost(exit_price, position.size)
            total_cost = pending_entry_cost + exit_cost

            if position.side == "long":
                gross_pnl = (exit_price - position.entry_price) * position.size
            else:
                gross_pnl = (position.entry_price - exit_price) * position.size

            net_pnl = gross_pnl - total_cost

            trade = Trade(
                entry_time=pending_entry_time,
                exit_time=final_bar.timestamp,
                entry_price=position.entry_price,
                exit_price=exit_price,
                size=position.size,
                side=position.side,
                gross_pnl=gross_pnl,
                costs=total_cost,
                net_pnl=net_pnl,
                entry_reason=pending_entry.reason if pending_entry else "",
                exit_reason="backtest_end",
            )
            trades.append(trade)
            equity += net_pnl
            # Update last equity point rather than appending - the final bar's equity
            # was already recorded in the loop; we just update it with the P&L.
            if equity_curve:
                equity_curve[-1] = equity

        return BacktestResult(
            trades=trades,
            config=self.config,
            start_time=bars[self.config.warmup_bars].timestamp,
            end_time=bars[-1].timestamp,
            equity_curve=equity_curve,
        )

    def _calc_cost(self, price: float, size: float) -> float:
        """Calculate transaction cost for a trade side."""
        notional = price * size
        return notional * self.config.transaction_cost_pct

    def _apply_slippage(self, price: float, atr: float, is_buy: bool) -> float:
        """Apply slippage to fill price.

        Slippage works against the trader:
        - Buy: pay more (price + slippage)
        - Sell: receive less (price - slippage)
        """
        slippage = atr * self.config.slippage_atr
        if is_buy:
            return price + slippage
        return price - slippage
