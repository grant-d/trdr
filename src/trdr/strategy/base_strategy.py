"""Base strategy interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..data.market import Bar
from .types import Position, Signal


@dataclass
class StrategyConfig:
    """Base configuration for strategies.

    Subclass to add strategy-specific parameters.
    """

    symbol: str
    timeframe: str = "1h"


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    Strategies encapsulate signal generation logic. The engine calls
    generate_signal() on each bar and executes the returned signal.

    Example:
        @dataclass
        class MyConfig(StrategyConfig):
            fast_period: int = 10
            slow_period: int = 50

        class MyStrategy(BaseStrategy):
            def __init__(self, config: MyConfig):
                super().__init__(config)
                self.config = config

            def generate_signal(self, bars, position):
                # Use self.config.fast_period, etc.
                ...
    """

    def __init__(self, config: StrategyConfig):
        """Initialize strategy with configuration.

        Args:
            config: Strategy configuration with symbol, timeframe, and params
        """
        self.config = config

    @property
    def name(self) -> str:
        """Strategy name. Override for custom name."""
        return self.__class__.__name__

    @abstractmethod
    def generate_signal(
        self,
        bars: list[Bar],
        position: Position | None,
    ) -> Signal:
        """Generate trading signal for current bar.

        Called once per bar with point-in-time data (no lookahead).

        Args:
            bars: Historical bars up to current bar (oldest first)
            position: Current open position or None

        Returns:
            Signal with action (BUY/SELL/HOLD/CLOSE), stops, targets
        """
        pass

    def on_trade_complete(self, pnl: float, reason: str) -> None:
        """Called when a trade completes. Override to adapt parameters.

        Args:
            pnl: Net P&L of completed trade
            reason: Exit reason (e.g., "stop_loss", "take_profit")
        """
        pass

    def reset(self) -> None:
        """Reset strategy state. Called before each backtest run."""
        pass
