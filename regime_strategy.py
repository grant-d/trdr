"""
Enhanced trading strategy implementation for Helios Trader
Matches the old notebook implementation with:
- Gradual position entry/exit
- Regime-specific stop-loss multipliers
- Dynamic position sizing based on MSS magnitude
- Support for parameterized thresholds
"""

from matplotlib.pylab import f
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from trade import Position, Trade
from pydantic import Field
from strategy_parameters import BaseStrategyParameters, MinMaxRange


class RegimeStrategyParameters(BaseStrategyParameters):
    """
    Parameters for regime-based trading strategy.

    The strategy classifies market into regimes based on MSS thresholds and
    adjusts position sizing, stop-losses, and entry/exit behavior accordingly.
    """

    # Position sizing parameters
    max_position_fraction: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=0.5, max_value=1.0),
        description="Maximum position size as fraction of capital",
    )

    entry_step_size: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=0.1, max_value=0.5),
        description="Gradual position entry/exit step size",
    )

    # Stop-loss multipliers (in ATR units)
    stop_loss_multiplier_strong: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=1.5, max_value=3.0),
        description="Stop-loss ATR multiplier for strong regimes",
    )

    stop_loss_multiplier_weak: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=0.5, max_value=1.5),
        description="Stop-loss ATR multiplier for weak regimes",
    )

    # Regime thresholds (MSS values)
    strong_bull_threshold: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=50.0, max_value=70.0),
        description="MSS threshold for strong bull regime",
    )

    weak_bull_threshold: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=20.0, max_value=49.0),
        description="MSS threshold for weak bull regime",
    )

    neutral_upper: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=1.0, max_value=19.0),
        description="Upper bound for neutral regime",
    )

    neutral_lower: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=-19.0, max_value=-1.0),
        description="Lower bound for neutral regime",
    )

    weak_bear_threshold: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=-49.0, max_value=-20.0),
        description="MSS threshold for weak bear regime",
    )

    strong_bear_threshold: MinMaxRange = Field(
        default_factory=lambda: MinMaxRange(min_value=-70.0, max_value=-50.0),
        description="MSS threshold for strong bear regime",
    )

    def validate_constraints(self) -> bool:
        """
        Ensure regime thresholds are properly ordered.

        Returns:
            True if all constraints are satisfied
        """
        # Sample values to check constraints
        strong_bull = self.strong_bull_threshold.sample()
        weak_bull = self.weak_bull_threshold.sample()
        neutral_up = self.neutral_upper.sample()
        neutral_low = self.neutral_lower.sample()
        weak_bear = self.weak_bear_threshold.sample()
        strong_bear = self.strong_bear_threshold.sample()

        # Check ordering: strong_bear < weak_bear <= neutral_lower < neutral_upper <= weak_bull < strong_bull
        return (
            strong_bear
            < weak_bear
            <= neutral_low
            < neutral_up
            <= weak_bull
            < strong_bull
        )

class RegimeStrategy:
    """
    Enhanced trading strategy
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_fraction: float = 1.0,
        entry_step_size: float = 0.2,
        stop_loss_multiplier_strong: float = 2.0,
        stop_loss_multiplier_weak: float = 1.0,
        strong_bull_threshold: float = 50.0,
        weak_bull_threshold: float = 20.0,
        neutral_upper: float = 20.0,
        neutral_lower: float = -20.0,
        weak_bear_threshold: float = -20.0,
        strong_bear_threshold: float = -50.0,
        allow_shorts: bool = False,
    ):
        """
        Initialize enhanced trading strategy

        Parameters match the old notebook's GA parameters
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_fraction = max_position_fraction
        self.entry_step_size = entry_step_size
        self.allow_shorts = allow_shorts

        # Stop-loss multipliers
        self.stop_loss_multipliers = {
            "Strong Bull": stop_loss_multiplier_strong,
            "Weak Bull": stop_loss_multiplier_weak,
            "Neutral": 0.0,  # No stop-loss in neutral
            "Weak Bear": stop_loss_multiplier_weak,
            "Strong Bear": stop_loss_multiplier_strong,
        }

        # Regime thresholds
        self.thresholds = {
            "strong_bull": strong_bull_threshold,
            "weak_bull": weak_bull_threshold,
            "neutral_upper": neutral_upper,
            "neutral_lower": neutral_lower,
            "weak_bear": weak_bear_threshold,
            "strong_bear": strong_bear_threshold,
        }

        self.position = Position()
        self.trades: List[Trade] = []
        
        # Trading filters to prevent overtrading
        self.min_position_change_pct = 0.02  # Minimum 2% position change to trade
        self.last_trade_bar = -10  # Track last trade to enforce cooldown
        self.trade_cooldown_bars = 2  # Wait at least 2 bars between trades
        self.last_mss = None  # Track MSS for change detection
        self.mss_change_threshold = 1.0  # Minimum MSS change to consider trading

    def classify_regime(self, mss: float) -> str:
        """
        Classify regime based on MSS and thresholds
        """
        if mss > self.thresholds["strong_bull"]:
            return "Strong Bull"
        elif mss > self.thresholds["weak_bull"]:
            return "Weak Bull"
        elif (
            mss >= self.thresholds["neutral_lower"]
            and mss <= self.thresholds["neutral_upper"]
        ):
            return "Neutral"
        elif mss > self.thresholds["strong_bear"]:
            return "Weak Bear"
        else:
            return "Strong Bear"

    def get_target_position_fraction(self, regime: str, mss: float) -> float:
        """
        Calculate target position fraction based on regime and MSS magnitude
        Matches the old notebook's gradual position sizing logic
        """
        if regime == "Strong Bull":
            # Scale from strong_bull_threshold to 100
            if (100 - self.thresholds["strong_bull"]) > 0:
                normalized_mss = (mss - self.thresholds["strong_bull"]) / (
                    100 - self.thresholds["strong_bull"]
                )
            else:
                normalized_mss = 0
            return self.max_position_fraction * np.clip(normalized_mss, 0, 1)

        elif regime == "Weak Bull":
            # Scale position based on MSS within weak bull range
            # Weak bull goes from weak_bull_threshold to strong_bull_threshold
            if (self.thresholds["strong_bull"] - self.thresholds["weak_bull"]) > 0:
                normalized_mss = (mss - self.thresholds["weak_bull"]) / (
                    self.thresholds["strong_bull"] - self.thresholds["weak_bull"]
                )
            else:
                normalized_mss = 0
            # Use reduced position size for weak regimes
            return (
                self.max_position_fraction * 0.7 * np.clip(normalized_mss, 0, 1)
            )  # 70% max in weak regime

        elif regime == "Neutral":
            return 0.0  # Flat in neutral

        elif regime == "Weak Bear":
            if not self.allow_shorts:
                return 0.0  # No shorts allowed
            # Scale position based on MSS within weak bear range
            # Weak bear goes from strong_bear_threshold to weak_bear_threshold
            if (self.thresholds["weak_bear"] - self.thresholds["strong_bear"]) > 0:
                normalized_mss = (mss - self.thresholds["strong_bear"]) / (
                    self.thresholds["weak_bear"] - self.thresholds["strong_bear"]
                )
            else:
                normalized_mss = 0
            # Use reduced position size for weak regimes (invert for short positions)
            return (
                -self.max_position_fraction * 0.7 * np.clip(1 - normalized_mss, 0, 1)
            )  # 70% max in weak regime

        elif regime == "Strong Bear":
            if not self.allow_shorts:
                return 0.0  # No shorts allowed
            # Scale from -100 to strong_bear_threshold
            if (self.thresholds["strong_bear"] - (-100)) > 0:
                normalized_mss = (mss - (-100)) / (
                    self.thresholds["strong_bear"] - (-100)
                )
            else:
                normalized_mss = 0
            return -self.max_position_fraction * np.clip(1 - normalized_mss, 0, 1)

        return 0.0

    def calculate_stop_loss(
        self, entry_price: float, regime: str, atr: float, is_long: bool
    ) -> float:
        """
        Calculate stop-loss level based on regime and ATR
        """
        multiplier = self.stop_loss_multipliers.get(regime, 0.0)
        if multiplier == 0:
            return 0.0  # No stop-loss

        stop_distance = multiplier * atr

        if is_long:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def execute_trade(
        self,
        timestamp: pd.Timestamp,
        units_to_trade: float,
        price: float,
        stop_loss: float,
        reason: str = "Signal",
    ) -> Optional[Trade]:
        """
        Execute a trade with enhanced tracking
        """
        if abs(units_to_trade) < 1e-9:
            return None
            
        # Prevent shorts when not allowed
        if not self.allow_shorts and units_to_trade < 0:
            # Can't sell more than current position
            max_sell = self.position.units
            if abs(units_to_trade) > max_sell:
                units_to_trade = -max_sell  # Limit to current position
                if abs(units_to_trade) < 1e-9:
                    return None  # Nothing to sell

        # Calculate P&L for closing trades
        pnl = 0.0
        if units_to_trade < 0 and self.position.units > 0:  # Selling long position
            units_to_sell = min(abs(units_to_trade), self.position.units)
            avg_entry = self.position.avg_price
            pnl = units_to_sell * (price - avg_entry)
        elif units_to_trade > 0 and self.position.units < 0:  # Buying to cover short
            units_to_cover = min(units_to_trade, abs(self.position.units))
            avg_entry = self.position.avg_price
            pnl = units_to_cover * (avg_entry - price)

        # Update position
        old_units = self.position.units
        old_cost_basis = self.position.cost_basis

        if units_to_trade > 0:  # Buying
            self.position.cost_basis += units_to_trade * price
            self.position.units += units_to_trade
        else:  # Selling
            # Adjust cost basis proportionally
            if abs(self.position.units) > 1e-9:
                ratio = abs(units_to_trade) / abs(self.position.units)
                self.position.cost_basis *= 1 - ratio
            self.position.units += units_to_trade

        # Update cash
        old_cash = self.cash
        self.cash -= units_to_trade * price
        self.cash += pnl  # Add realized P&L

        # Update position tracking
        if abs(self.position.units) < 1e-9:  # Position closed
            self.position = Position()
        else:
            self.position.entry_price = self.position.avg_price
            self.position.stop_loss = stop_loss
            if units_to_trade > 0 and old_units <= 0:  # New long position
                self.position.entry_time = timestamp
                self.position.peak_price = price
            elif units_to_trade < 0 and old_units >= 0:  # New short position
                self.position.entry_time = timestamp
                self.position.peak_price = price

        # Calculate portfolio value
        portfolio_value = self.cash + self.position.units * price

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            action="Buy" if units_to_trade > 0 else "Sell",
            units=units_to_trade,
            price=price,
            units_before=old_units,
            units_after=self.position.units,
            cost_basis_before=old_cost_basis,
            cost_basis_after=self.position.cost_basis,
            pnl=pnl,
            cash_before=old_cash,
            cash_after=self.cash,
            portfolio_value=portfolio_value,
            stop_loss=stop_loss,
            reason=reason,
        )

        self.trades.append(trade)
        return trade

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop-loss is hit
        """
        if not self.position.is_open or self.position.stop_loss == 0:
            return False

        if self.position.units > 0:  # Long position
            return current_price <= self.position.stop_loss
        else:  # Short position
            return current_price >= self.position.stop_loss

    def update_trailing_stop(self, current_price: float, atr: float, regime: str):
        """
        Update trailing stop-loss for profitable positions
        """
        if not self.position.is_open:
            return

        multiplier = self.stop_loss_multipliers.get(regime, 0.0)
        if multiplier == 0:
            return

        if self.position.units > 0:  # Long position
            if current_price > self.position.peak_price:
                self.position.peak_price = current_price
                # Update stop-loss
                new_stop = current_price - (multiplier * atr)
                self.position.stop_loss = max(self.position.stop_loss, new_stop)
        else:  # Short position
            if current_price < self.position.peak_price:
                self.position.peak_price = current_price
                # Update stop-loss
                new_stop = current_price + (multiplier * atr)
                self.position.stop_loss = min(self.position.stop_loss, new_stop)

    def run_backtest(self, df: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run enhanced backtest matching old notebook implementation
        """
        results = []

        for i in range(len(df)):
            # Use timestamp from data if available, otherwise use index
            if 'timestamp' in df.columns:
                timestamp = pd.to_datetime(df.at[df.index[i], 'timestamp'])
            else:
                timestamp = pd.Timestamp(df.index[i])
            current_price = df.at[df.index[i], "close"]

            # Skip if no factor data
            try:
                mss_value = factors_df.at[factors_df.index[i], "mss"]
                if np.isnan(mss_value):
                    continue
                regime = factors_df.at[factors_df.index[i], "regime"]
                atr = factors_df.at[factors_df.index[i], "atr"]
            except (KeyError, IndexError):
                continue

            # Check stop-loss first
            if self.check_stop_loss(current_price):
                # Close position at stop-loss
                units_to_close = -self.position.units
                self.execute_trade(
                    timestamp, units_to_close, current_price, 0.0, "Stop Loss"
                )
                self.last_trade_bar = i  # Update last trade bar for cooldown
                self.last_mss = mss_value  # Update MSS tracking
                # Skip rest of iteration after stop-loss to avoid immediate re-entry
                # Record state
                portfolio_value = self.cash + self.position.units * current_price
                results.append({
                    "timestamp": timestamp,
                    "close": current_price,
                    "portfolio_value": portfolio_value,
                    "cash": self.cash,
                    "position_units": self.position.units,
                    "position_value": self.position.units * current_price,
                    "trade": None,
                    "mss": mss_value,
                    "regime": regime,
                })
                continue

            # Update trailing stop
            self.update_trailing_stop(current_price, atr, regime)

            # Calculate target position
            target_position_fraction = self.get_target_position_fraction(
                regime, mss_value
            )

            # Calculate current position fraction
            portfolio_value = self.cash + self.position.units * current_price
            current_position_value = self.position.units * current_price
            current_position_fraction = current_position_value / self.initial_capital

            # Calculate position change (gradual entry/exit)
            position_fraction_change = (
                target_position_fraction - current_position_fraction
            )

            # Limit change to step size
            max_step = self.entry_step_size * self.max_position_fraction
            position_fraction_change = np.clip(
                position_fraction_change, -max_step, max_step
            )

            # Convert to units
            capital_to_trade = position_fraction_change * self.initial_capital
            units_to_trade = (
                capital_to_trade / current_price if current_price > 0 else 0.0
            )

            # Calculate stop-loss for new position
            if abs(units_to_trade) > 1e-9:
                is_long = (self.position.units + units_to_trade) > 0
                stop_loss = self.calculate_stop_loss(
                    current_price, regime, atr, is_long
                )
            else:
                stop_loss = self.position.stop_loss

            # Apply trading filters to prevent overtrading
            should_trade = abs(units_to_trade) > 1e-9
            
            if should_trade:
                # Filter 1: Minimum position change (2% of current position or initial)
                if self.position.units != 0:
                    position_change_pct = abs(units_to_trade / self.position.units)
                    should_trade = position_change_pct >= self.min_position_change_pct
                else:
                    # For initial positions, require at least 2% of max position
                    min_units = self.max_position_fraction * self.initial_capital * 0.02 / current_price
                    should_trade = abs(units_to_trade) >= min_units
            
            if should_trade:
                # Filter 2: Trade cooldown (wait N bars between trades)
                bars_since_last_trade = i - self.last_trade_bar
                should_trade = bars_since_last_trade >= self.trade_cooldown_bars
            
            if should_trade and self.last_mss is not None:
                # Filter 3: Significant MSS change required
                mss_change = abs(mss_value - self.last_mss)
                # Relax threshold for strong regimes
                if regime in ['Strong Bull', 'Strong Bear']:
                    threshold = self.mss_change_threshold * 0.5  # More sensitive in strong regimes
                else:
                    threshold = self.mss_change_threshold
                should_trade = mss_change >= threshold
            
            # Execute trade if it passes all filters
            if should_trade:
                reason = f"{regime} Signal"
                self.execute_trade(
                    timestamp, units_to_trade, current_price, stop_loss, reason
                )
                self.last_trade_bar = i  # Update last trade bar
            
            # Always update last MSS
            self.last_mss = mss_value

            # Record results
            portfolio_value = self.cash + self.position.units * current_price

            results.append(
                {
                    "timestamp": timestamp,
                    "close": current_price,
                    "regime": regime,
                    "mss": mss_value,
                    "atr": atr,
                    "position_units": self.position.units,
                    "position_fraction": self.position.units
                    * current_price
                    / self.initial_capital,
                    "target_fraction": target_position_fraction,
                    "cash": self.cash,
                    "portfolio_value": portfolio_value,
                    "returns": (portfolio_value / self.initial_capital - 1) * 100,
                    "stop_loss": (
                        self.position.stop_loss if self.position.is_open else 0.0
                    ),
                }
            )

        return pd.DataFrame(results)

    def get_trade_summary(self) -> Dict:
        """
        Get enhanced trade summary statistics
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
                "total_pnl": 0,
                "stop_losses": 0,
            }

        # Analyze trades
        trades_with_pnl = [t for t in self.trades if t.pnl != 0]
        winning_trades = [t for t in trades_with_pnl if t.pnl > 0]
        losing_trades = [t for t in trades_with_pnl if t.pnl < 0]
        stop_losses = [t for t in self.trades if t.reason == "Stop Loss"]

        total_pnl = sum(t.pnl for t in trades_with_pnl)
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))

        # Calculate returns
        winning_returns = []
        losing_returns = []

        for t in winning_trades:
            if abs(t.units) > 0 and abs(t.units_before) > 1e-9:
                # Calculate entry price from cost basis
                entry_price = abs(t.cost_basis_before / t.units_before)
                if entry_price > 1e-9:  # Avoid division by zero
                    ret = (t.price / entry_price - 1) * 100
                    if not np.isnan(ret) and not np.isinf(ret):
                        winning_returns.append(ret)

        for t in losing_trades:
            if abs(t.units) > 0 and abs(t.units_before) > 1e-9:
                # Calculate entry price from cost basis
                entry_price = abs(t.cost_basis_before / t.units_before)
                if entry_price > 1e-9:  # Avoid division by zero
                    ret = (t.price / entry_price - 1) * 100
                    if not np.isnan(ret) and not np.isinf(ret):
                        losing_returns.append(ret)

        return {
            "total_trades": len(self.trades),
            "trades_with_pnl": len(trades_with_pnl),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (
                len(winning_trades) / len(trades_with_pnl) * 100
                if trades_with_pnl
                else 0
            ),
            "avg_win": np.mean(winning_returns) if winning_returns else 0,
            "avg_loss": np.mean(losing_returns) if losing_returns else 0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else np.inf,
            "total_pnl": total_pnl,
            "stop_losses": len(stop_losses),
        }
