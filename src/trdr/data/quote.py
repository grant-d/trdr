"""Quote price data type."""

from dataclasses import dataclass


@dataclass
class Quote:
    """Current price quote."""

    symbol: str
    price: float
    bid: float
    ask: float
    timestamp: str
