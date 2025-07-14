"""
Trading strategy implementation for Helios Trader
Includes action matrix, position sizing, and trade execution logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """Trading actions"""

    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class Position:
    """Position information"""

    shares: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None

    @property
    def is_open(self) -> bool:
        return self.shares != 0.0


@dataclass
class Trade:
    """Trade record"""

    timestamp: pd.Timestamp
    action: str
    shares: float
    price: float
    position_before: float
    position_after: float
    cash_before: float
    cash_after: float
    portfolio_value: float

    def __repr__(self):
        return (
            f"Trade({self.timestamp}, {self.action}, "
            f"shares={self.shares:.0f}, price=${self.price:.2f})"
        )


class TradingStrategy:
    """
    Enhanced trading strategy with fractional positions and dynamic sizing
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_pct: float = 0.95,
        min_position_pct: float = 0.1,
        allow_shorts: bool = False,
    ):
        """
        Initialize trading strategy

        Parameters:
        -----------
        initial_capital : float
            Starting capital
        max_position_pct : float
            Maximum position size as percentage of portfolio
        min_position_pct : float
            Minimum position size as percentage of portfolio
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.allow_shorts = allow_shorts
        self.position = Position()
        self.trades: List[Trade] = []

        # Action matrix based on regime
        self.action_matrix = {
            "Strong Bull": {
                "MACD_Bull": Action.STRONG_BUY,
                "MACD_Bear": Action.HOLD,
                "RSI_Oversold": Action.STRONG_BUY,
                "RSI_Overbought": Action.BUY,
                "RSI_Neutral": Action.BUY,
            },
            "Weak Bull": {
                "MACD_Bull": Action.BUY,
                "MACD_Bear": Action.HOLD,
                "RSI_Oversold": Action.BUY,
                "RSI_Overbought": Action.HOLD,
                "RSI_Neutral": Action.HOLD,
            },
            "Neutral": {
                "MACD_Bull": Action.BUY,
                "MACD_Bear": Action.SELL,
                "RSI_Oversold": Action.BUY,
                "RSI_Overbought": Action.SELL,
                "RSI_Neutral": Action.HOLD,
            },
            "Weak Bear": {
                "MACD_Bull": Action.HOLD,
                "MACD_Bear": Action.SELL,
                "RSI_Oversold": Action.HOLD,
                "RSI_Overbought": Action.SELL,
                "RSI_Neutral": Action.HOLD,
            },
            "Strong Bear": {
                "MACD_Bull": Action.SELL,
                "MACD_Bear": Action.STRONG_SELL,
                "RSI_Oversold": Action.SELL,
                "RSI_Overbought": Action.STRONG_SELL,
                "RSI_Neutral": Action.STRONG_SELL,
            },
        }

    def get_signal_state(self, macd_hist: float, rsi: float) -> Dict[str, str]:
        """
        Determine signal states for MACD and RSI

        Parameters:
        -----------
        macd_hist : float
            MACD histogram value
        rsi : float
            RSI value

        Returns:
        --------
        Dict[str, str]
            Signal states
        """
        # MACD state
        macd_state = "MACD_Bull" if macd_hist > 0 else "MACD_Bear"

        # RSI state
        if rsi < 30:
            rsi_state = "RSI_Oversold"
        elif rsi > 70:
            rsi_state = "RSI_Overbought"
        else:
            rsi_state = "RSI_Neutral"

        return {"macd": macd_state, "rsi": rsi_state}

    def get_action(self, regime: str, macd_hist: float, rsi: float) -> Action:
        """
        Get trading action based on regime and indicators

        Parameters:
        -----------
        regime : str
            Current market regime
        macd_hist : float
            MACD histogram value
        rsi : float
            RSI value

        Returns:
        --------
        Action
            Trading action to take
        """
        signals = self.get_signal_state(macd_hist, rsi)

        # Get base action from matrix
        regime_actions = self.action_matrix.get(regime, self.action_matrix["Neutral"])
        macd_action = regime_actions.get(signals["macd"], Action.HOLD)
        rsi_action = regime_actions.get(signals["rsi"], Action.HOLD)

        # Combine signals (average the action values)
        combined_value = (macd_action.value + rsi_action.value) / 2

        # Map to discrete action
        if combined_value >= 1.5:
            return Action.STRONG_BUY
        elif combined_value >= 0.5:
            return Action.BUY
        elif combined_value <= -1.5:
            return Action.STRONG_SELL if self.allow_shorts else Action.HOLD
        elif combined_value <= -0.5:
            return Action.SELL if self.allow_shorts else Action.HOLD
        else:
            return Action.HOLD

    def calculate_position_size(
        self, action: Action, current_price: float, volatility: float, mss: float
    ) -> float:
        """
        Calculate position size based on action strength and market conditions

        Parameters:
        -----------
        action : Action
            Trading action
        current_price : float
            Current asset price
        volatility : float
            Current volatility (normalized)
        mss : float
            Market State Score

        Returns:
        --------
        float
            Number of shares to trade
        """
        portfolio_value = self.cash + self.position.shares * current_price

        # Base position sizing based on action strength
        action_weights = {
            Action.STRONG_BUY: 1.0,
            Action.BUY: 0.7,
            Action.HOLD: 0.0,
            Action.SELL: -0.7,
            Action.STRONG_SELL: -1.0,
        }

        base_weight = action_weights[action]

        # Adjust for volatility (reduce position size in high volatility)
        volatility_adj = 1.0 - min(volatility, 0.5)

        # Adjust for MSS confidence
        mss_adj = min(abs(mss), 1.0)

        # Calculate target position value
        target_position_pct = (
            base_weight * volatility_adj * mss_adj * self.max_position_pct
        )
        target_position_value = portfolio_value * target_position_pct

        # Calculate shares to trade
        current_position_value = self.position.shares * current_price
        position_change_value = target_position_value - current_position_value
        shares_to_trade = position_change_value / current_price

        # Apply minimum position size constraint
        if (
            abs(shares_to_trade * current_price)
            < portfolio_value * self.min_position_pct
        ):
            shares_to_trade = 0

        return shares_to_trade

    def execute_trade(
        self, timestamp: pd.Timestamp, action: Action, shares: float, price: float
    ) -> Optional[Trade]:
        """
        Execute a trade

        Parameters:
        -----------
        timestamp : pd.Timestamp
            Trade timestamp
        action : Action
            Trading action
        shares : float
            Number of shares (positive for buy, negative for sell)
        price : float
            Execution price

        Returns:
        --------
        Optional[Trade]
            Trade record if executed, None otherwise
        """
        if shares == 0:
            return None

        # Check if we have enough cash for buy
        if shares > 0 and shares * price > self.cash:
            # Adjust shares to available cash
            shares = self.cash / price

        # Check if we have enough shares for sell
        if shares < 0 and abs(shares) > self.position.shares:
            # Adjust to close position
            shares = -self.position.shares

        if shares == 0:
            return None

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            action=action.name,
            shares=shares,
            price=price,
            position_before=self.position.shares,
            position_after=self.position.shares + shares,
            cash_before=self.cash,
            cash_after=self.cash - shares * price,
            portfolio_value=self.cash
            - shares * price
            + (self.position.shares + shares) * price,
        )

        # Update position and cash
        self.position.shares += shares
        self.cash -= shares * price

        if self.position.shares == 0:
            self.position.entry_price = 0
            self.position.entry_time = None
        elif shares > 0:  # Buy
            if self.position.entry_price == 0:
                self.position.entry_price = price
                self.position.entry_time = timestamp
            else:
                # Update average entry price
                total_value = (
                    self.position.entry_price * (self.position.shares - shares)
                    + price * shares
                )
                self.position.entry_price = total_value / self.position.shares

        self.trades.append(trade)
        return trade

    def run_backtest(self, df: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest on historical data

        Parameters:
        -----------
        df : pd.DataFrame
            OHLC data
        factors_df : pd.DataFrame
            DataFrame with calculated factors and indicators

        Returns:
        --------
        pd.DataFrame
            Results DataFrame with trades and performance metrics
        """
        results = []

        for i in range(len(df)):
            timestamp = pd.Timestamp(df.index[i])
            close_price = df.at[df.index[i], "close"]

            # Skip if no factor data
            try:
                mss_value = factors_df.at[factors_df.index[i], "mss"]
                if np.isnan(mss_value):
                    continue
            except (KeyError, IndexError):
                continue

            # Get current market conditions
            regime = factors_df.at[factors_df.index[i], "regime"]
            mss = mss_value  # Already a float from .at accessor
            volatility = factors_df.at[factors_df.index[i], "volatility_norm"]
            
            # Get optional indicators with defaults
            try:
                macd_hist = factors_df.at[factors_df.index[i], "macd_hist"]
            except KeyError:
                macd_hist = 0.0
                
            try:
                rsi = factors_df.at[factors_df.index[i], "rsi"]
            except KeyError:
                rsi = 50.0

            # Get trading action
            action = self.get_action(regime, macd_hist, rsi)

            # Calculate position size
            shares_to_trade = self.calculate_position_size(
                action, close_price, volatility, mss
            )

            # Execute trade
            self.execute_trade(timestamp, action, shares_to_trade, close_price)

            # Record results
            portfolio_value = self.cash + self.position.shares * close_price

            results.append(
                {
                    "timestamp": timestamp,
                    "close": close_price,
                    "regime": regime,
                    "mss": mss,
                    "action": action.name,
                    "shares_traded": shares_to_trade,
                    "position": self.position.shares,
                    "cash": self.cash,
                    "portfolio_value": portfolio_value,
                    "returns": (portfolio_value / self.initial_capital - 1) * 100,
                }
            )

        return pd.DataFrame(results)

    def get_trade_summary(self) -> Dict:
        """
        Get summary statistics of trades

        Returns:
        --------
        Dict
            Trade statistics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
            }

        # Group trades by position cycles
        position_cycles = []
        cycle_trades = []

        for trade in self.trades:
            cycle_trades.append(trade)
            if trade.position_after == 0:  # Position closed
                if len(cycle_trades) > 1:  # At least entry and exit
                    entry_price = cycle_trades[0].price
                    exit_price = cycle_trades[-1].price
                    total_shares = sum(t.shares for t in cycle_trades[:-1])
                    pnl = total_shares * (exit_price - entry_price)
                    position_cycles.append(
                        {
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "shares": total_shares,
                            "pnl": pnl,
                            "return": (exit_price / entry_price - 1) * 100,
                        }
                    )
                cycle_trades = []

        if not position_cycles:
            return {
                "total_trades": len(self.trades),
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
            }

        # Calculate statistics
        winning_trades = [c for c in position_cycles if c["pnl"] > 0]
        losing_trades = [c for c in position_cycles if c["pnl"] < 0]

        avg_win = (
            np.mean([c["return"] for c in winning_trades]) if winning_trades else 0
        )
        avg_loss = np.mean([c["return"] for c in losing_trades]) if losing_trades else 0

        total_profit = sum(c["pnl"] for c in winning_trades)
        total_loss = abs(sum(c["pnl"] for c in losing_trades))

        return {
            "total_trades": len(self.trades),
            "position_cycles": len(position_cycles),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(position_cycles) * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": total_profit / total_loss if total_loss > 0 else np.inf,
            "total_pnl": sum(c["pnl"] for c in position_cycles),
        }
