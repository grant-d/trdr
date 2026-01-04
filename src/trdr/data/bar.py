"""Bar OHLCV data type."""

from dataclasses import asdict, dataclass


@dataclass
class Bar:
    """Single OHLCV bar."""

    timestamp: str  # ISO format
    open: float
    high: float
    low: float
    close: float
    volume: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Bar":
        """Create Bar from dictionary."""
        return cls(**data)
