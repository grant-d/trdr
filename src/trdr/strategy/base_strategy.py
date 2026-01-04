"""Base strategy interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.duration import Duration
from ..core.timeframe import Timeframe
from ..data import Bar
from .types import DataRequirement, Position, Signal

if TYPE_CHECKING:
    from ..backtest.paper_exchange import RuntimeContext


@dataclass
class StrategyConfig:
    """Base configuration for strategies.

    Subclass to add strategy-specific parameters.

    Args:
        symbol: Trading symbol (e.g., "crypto:ETH/USD", "stock:AAPL")
        timeframe: Timeframe (e.g., Timeframe.parse("15m"))
        lookback: Duration (e.g., Duration.parse("1M"))

    Example:
        @dataclass
        class MyConfig(StrategyConfig):
            fast_period: int = 12
            slow_period: int = 26

        config = MyConfig(
            symbol="crypto:ETH/USD",
            timeframe=Timeframe.parse("15m"),
            lookback=Duration.parse("1M"),
        )
    """

    symbol: str
    timeframe: Timeframe
    lookback: Duration

    @property
    def lookback_bars(self) -> int:
        """Lookback as bar count."""
        return self.lookback.to_bars(self.timeframe, self.symbol)


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    Strategies encapsulate signal generation logic. The engine calls
    generate_signal() on each bar and executes the returned signal.

    Attributes:
        config: Strategy configuration
        context: Live portfolio state (set by engine before generate_signal)
        name: Strategy name (custom or class name)

    Example:
        class MyStrategy(BaseStrategy):
            def get_data_requirements(self) -> list[DataRequirement]:
                return [
                    DataRequirement(
                        self.config.symbol,
                        self.config.timeframe,
                        self.config.lookback_bars,  # Use resolved bar count
                        role="primary",
                    ),
                ]

            def generate_signal(self, bars, position):
                primary = bars[f"{self.config.symbol}:{self.config.timeframe}"]
                ...
    """

    context: "RuntimeContext"

    def __init__(self, config: StrategyConfig, name: str | None = None):
        """Initialize strategy with configuration.

        Args:
            config: Strategy configuration with symbol, timeframe, and params
            name: Optional friendly name (defaults to class name)
        """
        self.config = config
        self._name = name

    @property
    def name(self) -> str:
        """Strategy name. Returns custom name if set, else class name."""
        return self._name if self._name else self.__class__.__name__

    @abstractmethod
    def get_data_requirements(self) -> list[DataRequirement]:
        """Declare all data feeds this strategy needs.

        Returns:
            List of DataRequirement. Exactly one must have role="primary".
            The primary feed determines bar iteration and trading symbol.

        Example:
            return [
                DataRequirement(
                    self.config.symbol,
                    self.config.timeframe,
                    self.config.lookback_bars,
                    role="primary",
                ),
            ]
        """
        ...

    @abstractmethod
    def generate_signal(
        self,
        bars: dict[str, list[Bar]],
        position: Position | None,
    ) -> Signal:
        """Generate trading signal for current bar.

        Called once per bar with point-in-time data (no lookahead).

        Args:
            bars: Dict of bars keyed by "symbol:timeframe" (e.g., "crypto:ETH/USD:15m")
            position: Current open position or None

        Returns:
            Signal with action (BUY/SELL/HOLD/CLOSE), stops, targets
        """
        ...

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
