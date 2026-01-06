"""Timeframe domain type."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Timeframe:
    """Bar interval (e.g., 15m, 1h, 1d).

    Core domain type. Exchange-specific translation handled by TimeframeAdapter.
    """

    amount: int
    unit: str  # Normalized: "m", "h", "d", "w", "mo"

    @property
    def seconds(self) -> int:
        """Duration in seconds."""
        if self.unit == "m":
            return self.amount * 60
        if self.unit == "h":
            return self.amount * 3600
        if self.unit == "d":
            return self.amount * 86400
        if self.unit == "w":
            return self.amount * 7 * 86400
        if self.unit == "mo":
            return self.amount * 30 * 86400
        return 0

    def to_minutes(self) -> int:
        """Duration in minutes."""
        return self.seconds // 60

    @property
    def canonical(self) -> "Timeframe":
        """Return canonical form (60m→1h, 24h→1d, etc.)."""
        # Minutes to hours (60m→1h, 120m→2h)
        if self.unit == "m" and self.amount >= 60 and self.amount % 60 == 0:
            return Timeframe(self.amount // 60, "h").canonical
        # Hours to days (24h→1d, 48h→2d)
        if self.unit == "h" and self.amount >= 24 and self.amount % 24 == 0:
            return Timeframe(self.amount // 24, "d").canonical
        # Don't convert days→weeks (5d != 1w semantically in trading)
        return self

    def __eq__(self, other: object) -> bool:
        """Compare canonical forms (60m == 1h)."""
        if isinstance(other, Timeframe):
            return (
                self.canonical.amount == other.canonical.amount
                and self.canonical.unit == other.canonical.unit
            )
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on canonical form."""
        c = self.canonical
        return hash((c.amount, c.unit))

    @property
    def is_intraday(self) -> bool:
        """True if less than 1 day."""
        return self.canonical.unit in ("m", "h")

    def __str__(self) -> str:
        """Return canonical string representation."""
        c = self.canonical
        return f"{c.amount}{c.unit}"

    @classmethod
    def parse(cls, tf_str: str) -> "Timeframe":
        """Parse timeframe string (e.g., "15m", "1h", "1d")."""
        return parse_timeframe(tf_str)


def parse_timeframe(tf_str: str) -> Timeframe:
    """Parse timeframe string into Timeframe.

    Args:
        tf_str: Timeframe string (e.g., "15m", "60m", "3d", "2w")

    Returns:
        Timeframe object

    Raises:
        ValueError: If timeframe format is invalid
    """
    tf = tf_str.lower().strip()
    match = re.match(r"^(\d+)([a-z]+)$", tf)

    if match:
        amount = int(match.group(1))
        unit_str = match.group(2)

        if amount <= 0:
            raise ValueError(f"Timeframe amount must be positive: {tf_str}")

        # Normalize unit
        if unit_str in ("m", "min", "minute"):
            return Timeframe(amount, "m")
        if unit_str in ("h", "hour"):
            return Timeframe(amount, "h")
        if unit_str in ("d", "day"):
            return Timeframe(amount, "d")
        if unit_str in ("w", "week"):
            return Timeframe(amount, "w")
        if unit_str in ("mo", "month"):
            return Timeframe(amount, "mo")

    # Fallback for bare unit names
    if tf in ("m", "min", "minute"):
        return Timeframe(1, "m")
    if tf in ("h", "hour"):
        return Timeframe(1, "h")
    if tf in ("d", "day"):
        return Timeframe(1, "d")

    raise ValueError(f"Invalid timeframe format: {tf_str}")


def get_interval_seconds(tf: str) -> int:
    """Get interval duration in seconds for a timeframe string."""
    return parse_timeframe(tf).seconds
