"""
Trading Context implementation for Helios Trader
Manages isolated strategy instances with persistent state
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


@dataclass
class TradingContextState:
    """Persistent state for a trading context"""

    context_id: str
    instrument: str
    timeframe: str
    experiment_name: str
    created_at: str
    last_processed_timestamp: Optional[str] = None
    last_processed_index: int = 0
    position_shares: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[str] = None
    stop_loss_price: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    cash: float = 100000.0
    initial_capital: float = 100000.0
    is_active: bool = True
    parameters: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    trade_history: list = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = self._default_parameters()
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.trade_history is None:
            self.trade_history = []

    def _default_parameters(self) -> Dict[str, Any]:
        """Default strategy parameters"""
        return {
            "lookback": 20,
            "max_position_pct": 0.95,
            "min_position_pct": 0.1,
            "weights": {"trend": 0.4, "volatility": 0.3, "exhaustion": 0.3},
            "stop_loss_atr_multiplier": 2.0,
            "dollar_bar_threshold": 1000000,
        }


class TradingContext:
    """
    Manages a single trading strategy instance with persistent state
    """

    def __init__(self, context_id: str, state_dir: str = "./state"):
        """
        Initialize a trading context

        Parameters:
        -----------
        context_id : str
            Unique identifier in format: instrument_exchange_timeframe_experiment
        state_dir : str
            Directory for state persistence
        """
        self.context_id = context_id
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / f"{context_id}.json"

        # Parse context ID
        parts = context_id.split("_")
        if len(parts) < 4:
            raise ValueError(f"Invalid context_id format: {context_id}")

        self.instrument = parts[0]
        self.exchange = parts[1]
        self.timeframe = parts[2]
        self.experiment_name = "_".join(parts[3:])

        # Load or create state
        self.state = self._load_state()

        # Initialize strategy components
        self._init_strategy()

    def _load_state(self) -> TradingContextState:
        """Load state from file or create new"""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                data = json.load(f)
                return TradingContextState(**data)
        else:
            # Create new state
            return TradingContextState(
                context_id=self.context_id,
                instrument=self.instrument,
                timeframe=self.timeframe,
                experiment_name=self.experiment_name,
                created_at=datetime.now().isoformat(),
            )

    def save_state(self):
        """Persist state to file atomically"""
        # Write to temp file first
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(asdict(self.state), f, indent=2)

        # Atomic rename
        temp_file.replace(self.state_file)

    def _init_strategy(self):
        """Initialize strategy components based on state"""
        from strategy_enhanced import EnhancedTradingStrategy
        from strategy import Position

        # Create strategy instance
        self.strategy = EnhancedTradingStrategy(
            initial_capital=self.state.initial_capital,
            max_position_fraction=self.state.parameters.get("max_position_pct", 0.95),
            allow_shorts=self.state.parameters.get("allow_shorts", False)
        )

        # Restore position state
        self.strategy.cash = self.state.cash
        self.strategy.position = Position(
            shares=self.state.position_shares,
            entry_price=self.state.entry_price,
            entry_time=(
                pd.Timestamp(self.state.entry_time) if self.state.entry_time else None
            ),
        )

        # Restore trade history
        self.strategy.trades = []  # Would need to deserialize properly

    def update_parameters(self, new_params: Dict[str, Any]):
        """Update strategy parameters"""
        self.state.parameters.update(new_params)
        self.save_state()

    def process_bar(
        self, bar_data: pd.Series, factors_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Process a new bar of data

        Parameters:
        -----------
        bar_data : pd.Series
            OHLCV data for the bar
        factors_data : pd.Series
            Calculated factors and indicators

        Returns:
        --------
        Dict[str, Any]
            Processing results including trade actions
        """
        timestamp = bar_data.name
        close_price = bar_data["close"]

        # Get regime and indicators
        regime = factors_data["regime"]
        mss = factors_data["mss"]
        volatility = factors_data["volatility_norm"]
        macd_hist = factors_data.get("macd_hist", 0)
        rsi = factors_data.get("rsi", 50)
        atr = factors_data.get(
            "atr", close_price * 0.02
        )  # Default 2% if not calculated

        # Get trading action
        action = self.strategy.get_action(regime, macd_hist, rsi)

        # Calculate position size
        shares_to_trade = self.strategy.calculate_position_size(
            action, close_price, volatility, mss
        )

        # Execute trade
        trade = self.strategy.execute_trade(
            timestamp, action, shares_to_trade, close_price
        )

        # Update stop loss based on regime
        self._update_stop_loss(close_price, atr, regime)

        # Update state
        self.state.last_processed_timestamp = timestamp.isoformat()
        self.state.position_shares = self.strategy.position.shares
        self.state.entry_price = self.strategy.position.entry_price
        self.state.entry_time = (
            self.strategy.position.entry_time.isoformat()
            if self.strategy.position.entry_time
            else None
        )
        self.state.cash = self.strategy.cash

        # Calculate portfolio value
        portfolio_value = self.state.cash + self.state.position_shares * close_price

        # Update performance metrics
        self._update_performance_metrics(portfolio_value)

        # Save state
        self.save_state()

        return {
            "timestamp": timestamp,
            "action": action.name,
            "shares_traded": shares_to_trade,
            "position": self.state.position_shares,
            "portfolio_value": portfolio_value,
            "stop_loss": self.state.stop_loss_price,
            "trade": trade,
        }

    def _update_stop_loss(self, current_price: float, atr: float, regime: str):
        """Update stop loss based on regime and ATR"""
        if self.state.position_shares == 0:
            self.state.stop_loss_price = None
            return

        # Determine ATR multiplier based on regime
        atr_multipliers = {
            "Strong Bull": 2.0,
            "Weak Bull": 1.0,
            "Neutral": 0.5,  # Tighter stop in neutral
            "Weak Bear": 1.0,
            "Strong Bear": 2.0,
        }

        multiplier = atr_multipliers.get(regime, 1.5)
        stop_distance = atr * multiplier

        if self.state.position_shares > 0:  # Long position
            new_stop = current_price - stop_distance
            # Trailing stop - only move up
            if (
                self.state.stop_loss_price is None
                or new_stop > self.state.stop_loss_price
            ):
                self.state.stop_loss_price = new_stop
        else:  # Short position
            new_stop = current_price + stop_distance
            # Trailing stop - only move down
            if (
                self.state.stop_loss_price is None
                or new_stop < self.state.stop_loss_price
            ):
                self.state.stop_loss_price = new_stop

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop loss is hit

        Returns:
        --------
        bool
            True if stop loss is triggered
        """
        if self.state.stop_loss_price is None or self.state.position_shares == 0:
            return False

        if self.state.position_shares > 0:  # Long position
            return current_price <= self.state.stop_loss_price
        else:  # Short position
            return current_price >= self.state.stop_loss_price

    def _update_performance_metrics(self, portfolio_value: float):
        """Update running performance metrics"""
        returns = (portfolio_value / self.state.initial_capital - 1) * 100

        self.state.performance_metrics.update(
            {
                "portfolio_value": portfolio_value,
                "total_return_pct": returns,
                "total_trades": len(self.strategy.trades),
                "last_updated": datetime.now().isoformat(),
            }
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current context status"""
        return {
            "context_id": self.context_id,
            "is_active": self.state.is_active,
            "position": self.state.position_shares,
            "portfolio_value": self.state.performance_metrics.get(
                "portfolio_value", self.state.initial_capital
            ),
            "total_return_pct": self.state.performance_metrics.get(
                "total_return_pct", 0
            ),
            "last_processed": self.state.last_processed_timestamp,
            "stop_loss": self.state.stop_loss_price,
        }

    def pause(self):
        """Pause the trading context"""
        self.state.is_active = False
        self.save_state()

    def resume(self):
        """Resume the trading context"""
        self.state.is_active = True
        self.save_state()

    def close_all_positions(self, current_price: float):
        """Close all open positions"""
        if self.state.position_shares != 0:
            # Close position
            timestamp = datetime.now()
            shares_to_close = -self.state.position_shares

            from strategy import Action

            action = Action.SELL if self.state.position_shares > 0 else Action.BUY

            self.strategy.execute_trade(
                timestamp, action, shares_to_close, current_price
            )

            # Update state
            self.state.position_shares = 0
            self.state.entry_price = 0
            self.state.entry_time = None
            self.state.stop_loss_price = None
            self.state.cash = self.strategy.cash

            self.save_state()


class TradingContextManager:
    """
    Manages multiple trading contexts
    """

    def __init__(self, state_dir: str = "./state"):
        """Initialize the context manager"""
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.contexts: Dict[str, TradingContext] = {}

        # Load existing contexts
        self._load_contexts()

    def _load_contexts(self):
        """Load all existing contexts from state files"""
        for state_file in self.state_dir.glob("*.json"):
            if state_file.stem != "manager":  # Skip manager state file
                try:
                    context = TradingContext(state_file.stem, str(self.state_dir))
                    self.contexts[context.context_id] = context
                except Exception as e:
                    print(f"Error loading context {state_file.stem}: {e}")

    def create_context(
        self,
        instrument: str,
        exchange: str,
        timeframe: str,
        experiment_name: str = "default",
        initial_capital: float = 100000,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> TradingContext:
        """Create a new trading context"""
        context_id = f"{instrument}_{exchange}_{timeframe}_{experiment_name}"

        if context_id in self.contexts:
            raise ValueError(f"Context {context_id} already exists")

        # Create context
        context = TradingContext(context_id, str(self.state_dir))

        # Update initial parameters
        if parameters:
            context.update_parameters(parameters)

        context.state.initial_capital = initial_capital
        context.state.cash = initial_capital
        context.save_state()

        self.contexts[context_id] = context
        return context

    def get_context(self, context_id: str) -> Optional[TradingContext]:
        """Get a trading context by ID"""
        return self.contexts.get(context_id)

    def list_contexts(self) -> list:
        """List all contexts with their status"""
        return [ctx.get_status() for ctx in self.contexts.values()]

    def get_active_contexts(self) -> list:
        """Get all active trading contexts"""
        return [ctx for ctx in self.contexts.values() if ctx.state.is_active]

    def pause_context(self, context_id: str):
        """Pause a trading context"""
        if context_id in self.contexts:
            self.contexts[context_id].pause()

    def resume_context(self, context_id: str):
        """Resume a trading context"""
        if context_id in self.contexts:
            self.contexts[context_id].resume()

    def delete_context(self, context_id: str):
        """Delete a trading context"""
        if context_id in self.contexts:
            # Close positions first
            context = self.contexts[context_id]
            if context.state.position_shares != 0:
                raise ValueError("Cannot delete context with open positions")

            # Remove state file
            context.state_file.unlink()

            # Remove from manager
            del self.contexts[context_id]
