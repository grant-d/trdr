"""
Options Position Management

Handles individual option positions and spreads
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd


@dataclass
class OptionPosition:
    """Represents a single option position"""
    option_type: str  # 'put' or 'call'
    strike: float
    expiration: datetime
    entry_date: datetime
    entry_price: float
    quantity: int  # Positive for long, negative for short
    underlying_entry_price: float

    def __repr__(self):
        position_type = "LONG" if self.quantity > 0 else "SHORT"
        return f"{position_type} {abs(self.quantity)} {self.option_type.upper()} ${self.strike} exp {self.expiration.date()}"


@dataclass
class BullPutSpread:
    """
    Bull Put Spread: Sell higher strike put, buy lower strike put
    This is a credit spread - profitable in sideways to bullish markets
    """
    short_put: OptionPosition  # Higher strike (sold)
    long_put: OptionPosition   # Lower strike (bought for protection)
    entry_date: datetime
    net_credit: float  # Premium received

    def __repr__(self):
        return (f"Bull Put Spread: Short ${self.short_put.strike} / Long ${self.long_put.strike} "
                f"Credit: ${self.net_credit:.2f}")

    def days_to_expiration(self, current_date: datetime) -> int:
        """Calculate days remaining until expiration"""
        return (self.short_put.expiration - current_date).days

    def is_expired(self, current_date: datetime) -> bool:
        """Check if the spread has expired"""
        return current_date >= self.short_put.expiration

    def calculate_pnl(self, current_short_price: float, current_long_price: float) -> float:
        """
        Calculate current P&L

        Args:
            current_short_price: Current price of short put
            current_long_price: Current price of long put

        Returns:
            P&L (positive is profit)
        """
        # Short put P&L: entry_price - current_price (we want it to decrease)
        short_pnl = (self.short_put.entry_price - current_short_price) * abs(self.short_put.quantity) * 100

        # Long put P&L: current_price - entry_price (we want it to increase if needed)
        long_pnl = (current_long_price - self.long_put.entry_price) * self.long_put.quantity * 100

        total_pnl = short_pnl + long_pnl
        return total_pnl

    def calculate_expiration_pnl(self, underlying_price: float) -> float:
        """
        Calculate P&L at expiration

        Args:
            underlying_price: Price of underlying at expiration

        Returns:
            P&L at expiration
        """
        # Short put value at expiration
        short_value = max(0, self.short_put.strike - underlying_price)

        # Long put value at expiration
        long_value = max(0, self.long_put.strike - underlying_price)

        # P&L calculation
        # We sold the short put, so we lose if it's in the money
        short_pnl = (self.short_put.entry_price - short_value) * abs(self.short_put.quantity) * 100

        # We bought the long put, so we profit if it's in the money
        long_pnl = (long_value - self.long_put.entry_price) * self.long_put.quantity * 100

        total_pnl = short_pnl + long_pnl
        return total_pnl

    def max_profit(self) -> float:
        """Maximum profit (the credit received)"""
        return self.net_credit * 100

    def max_loss(self) -> float:
        """Maximum loss"""
        spread_width = self.short_put.strike - self.long_put.strike
        return (spread_width * 100) - self.max_profit()


class Portfolio:
    """Manages a portfolio of option positions"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[int, BullPutSpread] = {}
        self.position_counter = 0
        self.closed_trades: list = []
        self.equity_curve = []

    def add_position(self, spread: BullPutSpread):
        """Add a new spread position"""
        self.position_counter += 1
        self.positions[self.position_counter] = spread

        # Collect credit
        self.cash += spread.net_credit * 100

        return self.position_counter

    def close_position(self, position_id: int, current_date: datetime,
                      short_exit_price: float, long_exit_price: float,
                      reason: str = "manual"):
        """Close a position"""
        if position_id not in self.positions:
            return

        spread = self.positions[position_id]

        # Calculate P&L
        pnl = spread.calculate_pnl(short_exit_price, long_exit_price)

        # Update cash
        self.cash += pnl

        # Record closed trade
        self.closed_trades.append({
            'entry_date': spread.entry_date,
            'exit_date': current_date,
            'short_strike': spread.short_put.strike,
            'long_strike': spread.long_put.strike,
            'credit': spread.net_credit,
            'pnl': pnl,
            'return_pct': (pnl / (spread.max_loss() if spread.max_loss() != 0 else 1)) * 100,
            'reason': reason,
            'days_held': (current_date - spread.entry_date).days
        })

        # Remove position
        del self.positions[position_id]

    def update_equity(self, current_date: datetime, current_positions_value: float):
        """Update equity curve"""
        total_equity = self.cash + current_positions_value

        self.equity_curve.append({
            'date': current_date,
            'cash': self.cash,
            'positions_value': current_positions_value,
            'total_equity': total_equity,
            'num_positions': len(self.positions)
        })

    def get_statistics(self) -> Dict:
        """Calculate portfolio statistics"""
        if not self.closed_trades:
            return {}

        df = pd.DataFrame(self.closed_trades)

        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])

        total_pnl = df['pnl'].sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_days_held': df['days_held'].mean(),
            'final_equity': self.cash,
            'return_pct': ((self.cash - self.initial_capital) / self.initial_capital) * 100
        }
