#!/usr/bin/env python3
"""
Strategy Evaluator for backtesting and position management.
Bridges strategy signals with performance metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from financial_metrics import FinancialMetrics


@dataclass
class Position:
    """Represents a trading position."""

    entry_time: pd.Timestamp
    entry_price: float
    size: float  # Positive for long, negative for short
    stop_loss: float
    take_profit: float
    entry_signal_confidence: float = 1.0


@dataclass
class Trade:
    """Completed trade with entry and exit."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    exit_reason: str  # 'stop_loss', 'take_profit', 'signal', 'end_of_data'


class StrategyEvaluator:
    """
    Evaluates trading strategy performance through backtesting.
    Handles position management, trade execution, and metrics calculation.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        position_size_pct: float = 0.02,  # 2% of capital per trade
        max_positions: int = 1,
        commission_pct: float = 0.001,  # 0.1% commission
        slippage_pct: float = 0.0005,  # 0.05% slippage
    ):
        """
        Initialize evaluator with trading parameters.

        Args:
            initial_capital: Starting capital
            position_size_pct: Percentage of capital per position
            max_positions: Maximum concurrent positions
            commission_pct: Commission percentage
            slippage_pct: Slippage percentage
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        # State tracking
        self.capital = initial_capital
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def backtest(
        self,
        df: pd.DataFrame,
        signals: Union[List[int], np.ndarray],
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.02,
        signal_confidence: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data with signals.

        Args:
            df: DataFrame with price data (must have 'close' or 'close_fd')
            signals: List of signals (1=buy, -1=sell, 0=hold)
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            signal_confidence: Optional confidence scores for each signal

        Returns:
            Dictionary with backtest results and metrics
        """
        # Reset state
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []

        # Get price column
        price_col = "close_fd" if "close_fd" in df.columns else "close"
        prices = df[price_col].values

        # Handle timestamps - ensure we have a proper array
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        elif "timestamp" in df.columns:
            timestamps = df["timestamp"].values
        else:
            # Create numeric index if no timestamp available
            timestamps = np.arange(len(df))

        # Convert signals to numpy array
        signals = np.array(signals)

        # Default confidence to 1.0 if not provided
        if signal_confidence is None:
            signal_confidence = np.ones(len(signals))

        # Track returns for metrics
        daily_returns = []

        # Process each bar
        for i in range(len(df)):
            current_price = prices[i]
            current_time = timestamps[i]
            signal = signals[i] if i < len(signals) else 0
            confidence = signal_confidence[i] if i < len(signal_confidence) else 1.0

            # Check existing positions for exit conditions
            self._check_exits(current_price, current_time)

            # Process new signals
            if signal != 0 and len(self.positions) < self.max_positions:
                self._enter_position(
                    current_time,
                    current_price,
                    signal,
                    confidence,
                    stop_loss_pct,
                    take_profit_pct,
                )

            # Update equity
            equity = self._calculate_equity(current_price)
            self.equity_curve.append(equity)

            # Calculate daily returns
            if i > 0:
                daily_return = (equity - self.equity_curve[-2]) / self.equity_curve[-2]
                daily_returns.append(daily_return)

        # Close any remaining positions
        if len(self.positions) > 0:
            self._close_all_positions(prices[-1], timestamps[-1], "end_of_data")

        # Calculate metrics
        results = self._calculate_results(daily_returns)

        return results

    def _enter_position(
        self,
        entry_time: pd.Timestamp,
        entry_price: float,
        signal: int,
        confidence: float,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> None:
        """Enter a new position."""
        # Apply slippage
        if signal > 0:  # Buy
            entry_price *= 1 + self.slippage_pct
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:  # Sell
            entry_price *= 1 - self.slippage_pct
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)

        # Calculate position size
        position_value = self.capital * self.position_size_pct * confidence
        size = position_value / entry_price * signal

        # Apply commission
        commission = abs(position_value) * self.commission_pct
        self.capital -= commission

        # Create position
        position = Position(
            entry_time=entry_time,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_signal_confidence=confidence,
        )

        self.positions.append(position)

    def _check_exits(self, current_price: float, current_time: pd.Timestamp) -> None:
        """Check and execute position exits."""
        positions_to_close = []

        for i, pos in enumerate(self.positions):
            exit_price = None
            exit_reason = None

            if pos.size > 0:  # Long position
                if current_price <= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                elif current_price >= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "take_profit"
            else:  # Short position
                if current_price >= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                elif current_price <= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "take_profit"

            if exit_price is not None:
                self._close_position(i, exit_price, current_time, exit_reason)
                positions_to_close.append(i)

        # Remove closed positions
        for i in reversed(positions_to_close):
            self.positions.pop(i)

    def _close_position(
        self,
        position_index: int,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
    ) -> None:
        """Close a position and record the trade."""
        pos = self.positions[position_index]

        # Apply slippage
        if pos.size > 0:  # Closing long
            exit_price *= 1 - self.slippage_pct
        else:  # Closing short
            exit_price *= 1 + self.slippage_pct

        # Calculate P&L
        if pos.size > 0:
            pnl = (exit_price - pos.entry_price) * abs(pos.size)
            return_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl = (pos.entry_price - exit_price) * abs(pos.size)
            return_pct = (pos.entry_price - exit_price) / pos.entry_price

        # Apply commission
        commission = abs(pos.size * exit_price) * self.commission_pct
        pnl -= commission

        # Update capital
        self.capital += pnl

        # Record trade
        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            pnl=pnl,
            return_pct=return_pct,
            exit_reason=exit_reason,
        )

        self.trades.append(trade)

    def _close_all_positions(
        self, exit_price: float, exit_time: pd.Timestamp, exit_reason: str
    ) -> None:
        """Close all open positions."""
        for i in range(len(self.positions)):
            self._close_position(0, exit_price, exit_time, exit_reason)
            self.positions.pop(0)

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current account equity."""
        equity = self.capital

        # Add unrealized P&L
        for pos in self.positions:
            if pos.size > 0:
                unrealized_pnl = (current_price - pos.entry_price) * abs(pos.size)
            else:
                unrealized_pnl = (pos.entry_price - current_price) * abs(pos.size)
            equity += unrealized_pnl

        return equity

    def _calculate_results(self, daily_returns: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        # Calculate metrics using FinancialMetrics
        fm = FinancialMetrics()
        returns_series = pd.Series(daily_returns)
        metrics = fm.calculate_all_metrics(returns_series)

        # Add trade-specific metrics
        if total_trades > 0:
            metrics["total_trades"] = total_trades
            metrics["winning_trades"] = len(winning_trades)
            metrics["losing_trades"] = len(losing_trades)
            metrics["win_rate"] = len(winning_trades) / total_trades * 100

            if winning_trades:
                metrics["avg_win"] = (
                    np.mean([t.return_pct for t in winning_trades]) * 100
                )
            else:
                metrics["avg_win"] = 0

            if losing_trades:
                metrics["avg_loss"] = (
                    np.mean([t.return_pct for t in losing_trades]) * 100
                )
            else:
                metrics["avg_loss"] = 0

            # Exit reason breakdown
            exit_reasons = {}
            for trade in self.trades:
                exit_reasons[trade.exit_reason] = (
                    exit_reasons.get(trade.exit_reason, 0) + 1
                )
            metrics["exit_reasons"] = exit_reasons

        # Final equity
        metrics["final_capital"] = self.capital
        metrics["total_return"] = (
            (self.capital - self.initial_capital) / self.initial_capital * 100
        )

        # Create results dictionary
        results = {
            "metrics": metrics,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "daily_returns": daily_returns,
        }

        return results

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of backtest results."""
        metrics = results["metrics"]

        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)

        print(
            f"\nCapital: ${self.initial_capital:,.2f} → ${metrics['final_capital']:,.2f}"
        )
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe', 0):.3f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")

        print(f"\nTotal Trades: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"Avg Win: {metrics.get('avg_win', 0):.2f}%")
        print(f"Avg Loss: {metrics.get('avg_loss', 0):.2f}%")

        if "exit_reasons" in metrics:
            print("\nExit Reasons:")
            for reason, count in metrics["exit_reasons"].items():
                print(f"  {reason}: {count}")


def example_usage():
    """Example of using the StrategyEvaluator."""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")
    prices = 100 + np.cumsum(np.random.normal(0, 0.5, 1000))

    df = pd.DataFrame({"timestamp": dates, "close": prices})

    # Generate random signals
    signals = np.random.choice([-1, 0, 0, 0, 1], size=len(df))

    # Create evaluator
    evaluator = StrategyEvaluator(
        initial_capital=100000, position_size_pct=0.02, commission_pct=0.001
    )

    # Run backtest
    results = evaluator.backtest(df, signals, stop_loss_pct=0.01, take_profit_pct=0.02)

    # Print summary
    evaluator.print_summary(results)


if __name__ == "__main__":
    example_usage()
