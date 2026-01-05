"""Quote price data type."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class Quote:
    """Current price quote."""

    symbol: str
    price: float
    bid: float
    ask: float
    timestamp: str
