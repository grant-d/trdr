"""Shared trade metrics calculations."""

from datetime import datetime

import numpy as np

from .calendar import get_trading_days_in_year

if False:  # TYPE_CHECKING
    from .paper_exchange import Trade


class TradeMetrics:
    """Computes trading metrics from trades and equity curve.

    Used by both RuntimeContext (live) and PaperExchangeResult (final).
    """

    def __init__(
        self,
        trades: list["Trade"],
        equity_curve: list[float],
        initial_capital: float,
        asset_type: str,
        start_time: str = "",
        end_time: str = "",
    ):
        self._trades = trades
        self._equity_curve = equity_curve
        self._initial_capital = initial_capital
        self._asset_type = asset_type
        self._start_time = start_time
        self._end_time = end_time

    @property
    def total_trades(self) -> int:
        """Number of completed trades."""
        return len(self._trades)

    @property
    def winning_trades(self) -> int:
        """Number of winning trades."""
        return sum(1 for t in self._trades if t.is_winner)

    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return self.total_trades - self.winning_trades

    @property
    def win_rate(self) -> float:
        """Win rate as decimal."""
        if not self._trades:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_pnl(self) -> float:
        """Total net P&L."""
        return sum(t.net_pnl for t in self._trades)

    @property
    def profit_factor(self) -> float:
        """Net profits / net losses. >1 is profitable."""
        profits = sum(t.net_pnl for t in self._trades if t.net_pnl > 0)
        losses = abs(sum(t.net_pnl for t in self._trades if t.net_pnl < 0))
        if losses == 0:
            return float("inf") if profits > 0 else 0.0
        return profits / losses

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as decimal."""
        if not self._equity_curve:
            return 0.0
        peak = self._equity_curve[0]
        max_dd = 0.0
        for equity in self._equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, min((peak - equity) / peak, 1.0))
        return max_dd

    def current_drawdown(self, current_equity: float) -> float:
        """Current drawdown from peak as decimal."""
        if not self._equity_curve:
            return 0.0
        peak = max(self._equity_curve)
        if peak <= 0:
            return 0.0
        # Clamp to [0, 1] - drawdown is 0 at new highs, max 1.0 (100%)
        dd = (peak - current_equity) / peak
        return max(0.0, min(dd, 1.0))

    @property
    def total_return(self) -> float:
        """Total return as decimal (0.10 = 10%)."""
        if not self._equity_curve:
            return 0.0
        final = self._equity_curve[-1]
        if self._initial_capital <= 0:
            return 0.0
        return (final - self._initial_capital) / self._initial_capital

    def total_return_from_equity(self, current_equity: float) -> float:
        """Total return from given equity value."""
        if self._initial_capital <= 0:
            return 0.0
        return (current_equity - self._initial_capital) / self._initial_capital

    @property
    def avg_trade_pnl(self) -> float:
        """Average P&L per trade."""
        if not self._trades:
            return 0.0
        return self.total_pnl / self.total_trades

    @property
    def avg_win(self) -> float:
        """Average winning trade P&L."""
        winners = [t.net_pnl for t in self._trades if t.net_pnl > 0]
        return float(np.mean(winners)) if winners else 0.0

    @property
    def avg_loss(self) -> float:
        """Average losing trade P&L (negative value)."""
        losers = [t.net_pnl for t in self._trades if t.net_pnl < 0]
        return float(np.mean(losers)) if losers else 0.0

    @property
    def largest_win(self) -> float:
        """Largest single winning trade."""
        winners = [t.net_pnl for t in self._trades if t.net_pnl > 0]
        return max(winners) if winners else 0.0

    @property
    def largest_loss(self) -> float:
        """Largest single losing trade (negative value)."""
        losers = [t.net_pnl for t in self._trades if t.net_pnl < 0]
        return min(losers) if losers else 0.0

    @property
    def avg_trade_duration_hours(self) -> float:
        """Average trade duration in hours."""
        if not self._trades:
            return 0.0
        return float(np.mean([t.duration_hours for t in self._trades]))

    @property
    def max_consecutive_wins(self) -> int:
        """Maximum consecutive winning trades."""
        if not self._trades:
            return 0
        max_streak, current = 0, 0
        for t in self._trades:
            current = (current + 1) if t.is_winner else 0
            max_streak = max(max_streak, current)
        return max_streak

    @property
    def max_consecutive_losses(self) -> int:
        """Maximum consecutive losing trades."""
        if not self._trades:
            return 0
        max_streak, current = 0, 0
        for t in self._trades:
            current = (current + 1) if not t.is_winner else 0
            max_streak = max(max_streak, current)
        return max_streak

    @property
    def expectancy(self) -> float:
        """Expected value per trade: (WR * avg_win) + ((1-WR) * avg_loss)."""
        if not self._trades:
            return 0.0
        wr = self.win_rate
        return (wr * self.avg_win) + ((1 - wr) * self.avg_loss)

    @property
    def total_costs(self) -> float:
        """Total transaction costs across all trades."""
        return sum(t.costs for t in self._trades)

    @property
    def sharpe_ratio(self) -> float | None:
        """Annualized Sharpe ratio (assumes 0 risk-free rate)."""
        if len(self._trades) < 2:
            return None
        returns = [t.net_pnl / (t.entry_price * t.quantity) for t in self._trades]
        std = np.std(returns)
        if std == 0:
            return float("inf") if np.mean(returns) > 0 else None
        days = get_trading_days_in_year(self._asset_type)
        return float(np.mean(returns) / std * np.sqrt(days))

    @property
    def sortino_ratio(self) -> float | None:
        """Annualized Sortino ratio."""
        if len(self._trades) < 2:
            return None
        returns = [t.net_pnl / (t.entry_price * t.quantity) for t in self._trades]
        downside = [r for r in returns if r < 0]
        if not downside:
            return float("inf") if np.mean(returns) > 0 else None
        downside_std = np.std(downside)
        if downside_std == 0:
            return None
        days = get_trading_days_in_year(self._asset_type)
        return float(np.mean(returns) / downside_std * np.sqrt(days))

    @property
    def cagr(self) -> float | None:
        """Compound Annual Growth Rate."""
        if not self._equity_curve or not self._start_time or not self._end_time:
            return None
        initial = self._initial_capital
        final = self._equity_curve[-1]
        if initial <= 0 or final <= 0:
            return None

        start = datetime.fromisoformat(self._start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(self._end_time.replace("Z", "+00:00"))
        years = (end - start).total_seconds() / (365.25 * 24 * 3600)
        if years <= 0:
            return None

        return float((final / initial) ** (1 / years) - 1)

    def cagr_live(self, current_equity: float, current_time: str) -> float | None:
        """CAGR computed from start to current time."""
        if not self._start_time or not current_time:
            return None
        initial = self._initial_capital
        if initial <= 0 or current_equity <= 0:
            return None

        start = datetime.fromisoformat(self._start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(current_time.replace("Z", "+00:00"))
        years = (end - start).total_seconds() / (365.25 * 24 * 3600)
        if years <= 0:
            return None

        return float((current_equity / initial) ** (1 / years) - 1)

    @property
    def calmar_ratio(self) -> float | None:
        """Calmar ratio: CAGR / max drawdown."""
        cagr_val = self.cagr
        max_dd = self.max_drawdown
        if cagr_val is None or max_dd == 0:
            return float("inf") if cagr_val and cagr_val > 0 else None
        return cagr_val / max_dd

    @property
    def trades_per_year(self) -> float:
        """Annualized trade frequency."""
        if not self._trades or not self._start_time or not self._end_time:
            return 0.0
        start = datetime.fromisoformat(self._start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(self._end_time.replace("Z", "+00:00"))
        years = (end - start).total_seconds() / (365.25 * 24 * 3600)
        if years <= 0:
            return 0.0
        return self.total_trades / years
