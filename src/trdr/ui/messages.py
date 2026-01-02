"""TUI message types for inter-component communication."""

from dataclasses import dataclass

from textual.message import Message


@dataclass
class MarketState:
    """Current market state for display."""

    symbol: str
    price: float
    change_pct: float
    poc: float
    vah: float
    val: float
    volume: int


@dataclass
class PositionState:
    """Current position for display."""

    side: str  # "LONG", "SHORT", "NONE"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: float | None
    take_profit: float | None


@dataclass
class PerformanceState:
    """Performance metrics for display."""

    total_pnl: float
    daily_pnl: float
    win_rate: float
    total_trades: int
    winning_trades: int


class MarketUpdate(Message):
    """Message for market data update."""

    def __init__(self, state: MarketState):
        self.state = state
        super().__init__()


class PositionUpdate(Message):
    """Message for position update."""

    def __init__(self, state: PositionState):
        self.state = state
        super().__init__()


class PerformanceUpdate(Message):
    """Message for performance update."""

    def __init__(self, state: PerformanceState):
        self.state = state
        super().__init__()


class LogMessage(Message):
    """Message for log entry."""

    def __init__(self, text: str, level: str = "info"):
        self.text = text
        self.level = level
        super().__init__()


class StatusUpdate(Message):
    """Message for bot status update."""

    def __init__(self, status: str):
        self.status = status
        super().__init__()
