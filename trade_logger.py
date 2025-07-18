#!/usr/bin/env python3
"""
Comprehensive trade logging and position tracking system.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class Trade:
    """Individual trade record."""

    trade_id: int
    timestamp: datetime
    symbol: str
    side: str  # 'buy', 'sell'
    quantity: float
    price: float
    value: float
    position_before: float
    position_after: float
    cash_before: float
    cash_after: float
    signal_strength: float
    reason: str  # 'entry', 'exit', 'stop_loss'


@dataclass
class Position:
    """Current position state."""

    timestamp: datetime
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    cash: float
    total_equity: float


class TradeLogger:
    """
    Comprehensive trade logging and portfolio tracking system.
    """

    def __init__(self, initial_cash: float = 100000, commission: float = 0.001):
        self.initial_cash = initial_cash
        self.commission = commission  # 0.1% per trade

        # Current state
        self.cash = initial_cash
        self.position = 0.0
        self.avg_price = 0.0
        self.trade_id_counter = 1

        # Logs
        self.trades: List[Trade] = []
        self.positions: List[Position] = []
        self.equity_curve: List[Dict] = []

    def execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        signal: float,
        current_price: float,
        signal_strength: float = 1.0,
        reason: str = "signal",
    ) -> Optional[Trade]:
        """
        Execute a trade based on signal.

        Args:
            timestamp: Current time
            symbol: Trading symbol
            signal: Trade signal (-1, 0, 1)
            current_price: Current market price
            signal_strength: Strength of the signal (0-1)
            reason: Reason for trade

        Returns:
            Trade object if trade was executed, None otherwise
        """
        if signal == 0:
            return None

        # Determine trade size (for now, use fixed percentage of equity)
        total_equity = self.cash + (self.position * current_price)
        max_position_size = total_equity * 0.95  # Use 95% of equity

        if signal > 0:  # Buy signal
            if self.position >= 0:  # Going long or adding to long
                available_cash = self.cash * 0.95  # Leave some cash buffer
                quantity = available_cash / current_price

                if quantity * current_price < 10:  # Minimum trade size
                    return None

                side = "buy"

            else:  # Covering short position
                quantity = abs(self.position)  # Cover entire short position
                side = "buy"

        else:  # Sell signal
            if self.position <= 0:  # Going short or adding to short
                max_short_value = max_position_size
                quantity = max_short_value / current_price
                side = "sell"

            else:  # Selling long position
                quantity = self.position  # Sell entire long position
                side = "sell"

        # Calculate trade value and commission
        trade_value = quantity * current_price
        commission_cost = trade_value * self.commission

        # Record state before trade
        position_before = self.position
        cash_before = self.cash

        # Execute trade
        if side == "buy":
            # Update position and cash
            if self.position < 0:  # Covering short
                # Realize P&L from short position
                pnl = (self.avg_price - current_price) * abs(self.position)
                self.cash += pnl
                self.position += quantity
                if abs(self.position) < 1e-6:  # Essentially zero
                    self.position = 0
                    self.avg_price = 0
                else:
                    self.avg_price = current_price
            else:  # Buying long
                new_total_quantity = self.position + quantity
                if new_total_quantity > 0:
                    # Update average price
                    total_cost = (self.position * self.avg_price) + (
                        quantity * current_price
                    )
                    self.avg_price = total_cost / new_total_quantity
                self.position = new_total_quantity

            self.cash -= trade_value + commission_cost

        else:  # sell
            if self.position > 0:  # Selling long
                # Realize P&L from long position
                pnl = (current_price - self.avg_price) * quantity
                self.cash += trade_value + pnl - commission_cost
                self.position -= quantity
                if abs(self.position) < 1e-6:
                    self.position = 0
                    self.avg_price = 0
            else:  # Going short
                self.position -= quantity
                self.avg_price = current_price
                self.cash += trade_value - commission_cost

        # Create trade record
        trade = Trade(
            trade_id=self.trade_id_counter,
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=current_price,
            value=trade_value,
            position_before=position_before,
            position_after=self.position,
            cash_before=cash_before,
            cash_after=self.cash,
            signal_strength=signal_strength,
            reason=reason,
        )

        self.trades.append(trade)
        self.trade_id_counter += 1

        return trade

    def update_position(self, timestamp: datetime, symbol: str, current_price: float):
        """Update current position state."""
        market_value = self.position * current_price
        unrealized_pnl = 0.0

        if self.position != 0:
            if self.position > 0:  # Long position
                unrealized_pnl = (current_price - self.avg_price) * self.position
            else:  # Short position
                unrealized_pnl = (self.avg_price - current_price) * abs(self.position)

        total_equity = self.cash + market_value + unrealized_pnl

        position = Position(
            timestamp=timestamp,
            symbol=symbol,
            quantity=self.position,
            avg_price=self.avg_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            cash=self.cash,
            total_equity=total_equity,
        )

        self.positions.append(position)

        # Update equity curve
        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "cash": self.cash,
                "position_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "total_equity": total_equity,
                "position_quantity": self.position,
                "position_avg_price": self.avg_price,
            }
        )

    def get_trade_statistics(self) -> Dict:
        """Calculate comprehensive trade statistics."""
        if not self.trades:
            return {"total_trades": 0}

        trades_df = pd.DataFrame([asdict(trade) for trade in self.trades])

        # Basic stats
        total_trades = len(trades_df)
        total_volume = trades_df["value"].sum()
        total_commission = total_volume * self.commission

        # P&L calculation (simplified)
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 1:
            total_return = (
                equity_df["total_equity"].iloc[-1] / equity_df["total_equity"].iloc[0]
            ) - 1
            max_equity = equity_df["total_equity"].cummax()
            drawdown = (equity_df["total_equity"] - max_equity) / max_equity
            max_drawdown = drawdown.min()
        else:
            total_return = 0
            max_drawdown = 0

        # Trade analysis
        buy_trades = trades_df[trades_df["side"] == "buy"]
        sell_trades = trades_df[trades_df["side"] == "sell"]

        return {
            "total_trades": total_trades,
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total_volume": total_volume,
            "total_commission": total_commission,
            "total_return": total_return * 100,
            "max_drawdown": max_drawdown * 100,
            "avg_trade_size": trades_df["value"].mean(),
            "largest_trade": trades_df["value"].max(),
            "current_cash": self.cash,
            "current_position": self.position,
            "current_equity": (
                equity_df["total_equity"].iloc[-1]
                if len(equity_df) > 0
                else self.initial_cash
            ),
        }

    def save_logs(self, filename_prefix: str = "trade_log"):
        """Save all logs to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save trades
        if self.trades:
            trades_df = pd.DataFrame([asdict(trade) for trade in self.trades])
            trades_df.to_csv(f"{filename_prefix}_trades_{timestamp}.csv", index=False)

        # Save positions
        if self.positions:
            positions_df = pd.DataFrame([asdict(pos) for pos in self.positions])
            positions_df.to_csv(
                f"{filename_prefix}_positions_{timestamp}.csv", index=False
            )

        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(f"{filename_prefix}_equity_{timestamp}.csv", index=False)

        # Save summary statistics
        stats = self.get_trade_statistics()
        with open(f"{filename_prefix}_stats_{timestamp}.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

        print(f"Trade logs saved with timestamp: {timestamp}")

        return {
            "trades_file": f"{filename_prefix}_trades_{timestamp}.csv",
            "positions_file": f"{filename_prefix}_positions_{timestamp}.csv",
            "equity_file": f"{filename_prefix}_equity_{timestamp}.csv",
            "stats_file": f"{filename_prefix}_stats_{timestamp}.json",
        }

    def get_current_state(self) -> Dict:
        """Get current portfolio state."""
        return {
            "cash": self.cash,
            "position_quantity": self.position,
            "position_avg_price": self.avg_price,
            "total_trades": len(self.trades),
            "current_equity": (
                self.equity_curve[-1]["total_equity"]
                if self.equity_curve
                else self.initial_cash
            ),
        }


def example_usage():
    """Example of how to use TradeLogger."""
    logger = TradeLogger(initial_cash=100000)

    # Simulate some trades
    timestamps = pd.date_range("2024-01-01", periods=100, freq="1H")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    signals = np.random.choice([-1, 0, 1], 100, p=[0.1, 0.8, 0.1])

    for i, (ts, price, signal) in enumerate(zip(timestamps, prices, signals)):
        # Execute trade if signal
        if signal != 0:
            trade = logger.execute_trade(
                ts, "BTC/USD", signal, price, reason="test_signal"
            )
            if trade:
                print(
                    f"Trade {trade.trade_id}: {trade.side} {trade.quantity:.2f} @ ${trade.price:.2f}"
                )

        # Update position
        logger.update_position(ts, "BTC/USD", price)

    # Print statistics
    stats = logger.get_trade_statistics()
    print(f"\nTrade Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save logs
    files = logger.save_logs("example")
    print(f"\nSaved files: {files}")


if __name__ == "__main__":
    example_usage()
