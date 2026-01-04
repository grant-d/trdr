"""Duration parsing for lookback periods."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .timeframe import Timeframe


@dataclass(frozen=True)
class Duration:
    """Duration for bar counts (for lookback or other purposes).

    Separate from Timeframe - this represents how far back to look,
    not bar interval size.

    Units (case-insensitive except M):
        h = hours
        d = days
        w = weeks
        M = months (capital M to distinguish)
        y = years

    Examples:
        Duration.parse("30d")  # 30 days
        Duration.parse("2w")   # 2 weeks
        Duration.parse("3M")   # 3 months
        Duration.parse("1y")   # 1 year
        Duration(30, "d")      # Direct construction
    """

    amount: int
    unit: str  # Normalized: "h", "d", "w", "M", "y"

    @classmethod
    def parse(cls, s: str) -> "Duration":
        """Parse duration string like '30d', '2w', '3M', '1y'."""
        s = s.strip()
        match = re.match(r"^(\d+)([hdwMy])$", s)
        if not match:
            # Try case-insensitive for convenience (except M)
            match = re.match(r"^(\d+)([hdwy])$", s.lower())
            if not match:
                raise ValueError(f"Invalid duration: {s}. Use Nh/d/w/M/y")

        amount = int(match.group(1))
        unit = match.group(2)

        if amount <= 0:
            raise ValueError(f"Duration amount must be positive: {s}")

        # Normalize unit
        if unit in ("h", "H"):
            return cls(amount, "h")
        if unit in ("d", "D"):
            return cls(amount, "d")
        if unit in ("w", "W"):
            return cls(amount, "w")
        if unit == "M":
            return cls(amount, "M")
        if unit in ("y", "Y"):
            return cls(amount, "y")

        raise ValueError(f"Unknown duration unit: {unit}")

    @property
    def hours(self) -> int:
        """Duration in hours."""
        if self.unit == "h":
            return self.amount
        if self.unit == "d":
            return self.amount * 24
        if self.unit == "w":
            return self.amount * 7 * 24
        if self.unit == "M":
            return self.amount * 30 * 24
        if self.unit == "y":
            return self.amount * 365 * 24
        return 0

    def to_bars(self, timeframe: "Timeframe", symbol: str) -> int:
        """Convert to bar count for given timeframe and symbol.

        Args:
            timeframe: Timeframe
            symbol: Trading symbol (e.g., "crypto:BTC/USD", "stock:AAPL")

        Returns:
            Number of bars

        Examples:
            Duration.parse("30d").to_bars(Timeframe.parse("15m"), "crypto:BTC/USD")
            Duration.parse("30d").to_bars(Timeframe.parse("15m"), "stock:AAPL")
        """
        tf = timeframe
        tf_minutes = tf.seconds // 60
        is_crypto = symbol.lower().startswith("crypto:")

        # Convert to calendar days first
        if self.unit == "h":
            calendar_days = self.amount / 24
        elif self.unit == "d":
            calendar_days = self.amount
        elif self.unit == "w":
            calendar_days = self.amount * 7
        elif self.unit == "M":
            calendar_days = self.amount * 30
        elif self.unit == "y":
            calendar_days = self.amount * 365
        else:
            calendar_days = 0

        if tf.unit == "d":
            # Daily bars
            if is_crypto:
                return int(calendar_days) // tf.amount
            else:
                trading_days = int(calendar_days * 252 / 365)
                return trading_days // tf.amount
        elif tf.unit in ("w", "mo"):
            if tf.unit == "w":
                return int(calendar_days) // (7 * tf.amount)
            else:
                return int(calendar_days) // (30 * tf.amount)
        else:
            # Intraday (minutes/hours)
            if is_crypto:
                bars_per_day = 1440 // tf_minutes
            else:
                bars_per_day = 390 // tf_minutes
                calendar_days = calendar_days * 252 / 365

            return int(calendar_days * bars_per_day)

    def __eq__(self, other: object) -> bool:
        """Compare by total hours (7d == 1w, 24h == 1d)."""
        if isinstance(other, Duration):
            return self.hours == other.hours
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on total hours."""
        return hash(self.hours)

    def __str__(self) -> str:
        return f"{self.amount}{self.unit}"


def parse_duration(s: str | int) -> int | Duration:
    """Parse duration - returns int if already int, Duration if string."""
    if isinstance(s, int):
        return s
    return Duration.parse(s)
