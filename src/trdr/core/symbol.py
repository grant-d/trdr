"""Symbol type for asset identification."""

from dataclasses import dataclass


@dataclass
class Symbol:
    """Asset symbol with type info.

    Format: "type:symbol" (e.g., "crypto:BTC/USD", "stock:AAPL")
    Plain symbols default to stock type.
    """

    asset_type: str  # "stock" or "crypto"
    raw: str  # The actual symbol (e.g., "BTC/USD", "AAPL")

    @classmethod
    def parse(cls, symbol: str) -> "Symbol":
        """Parse symbol string. Plain symbols default to stock type."""
        if ":" in symbol:
            asset_type, raw = symbol.split(":", 1)
            return cls(asset_type=asset_type.lower(), raw=raw)
        return cls(asset_type="stock", raw=symbol)

    @property
    def is_crypto(self) -> bool:
        """Check if this is a crypto asset."""
        return self.asset_type == "crypto"

    @property
    def is_stock(self) -> bool:
        """Check if this is a stock asset."""
        return self.asset_type == "stock"

    @property
    def cache_key(self) -> str:
        """Safe string for cache filenames. Format: type:symbol (lowercase)."""
        safe_raw = self.raw.replace("/", "_").lower()
        return f"{self.asset_type}:{safe_raw}"

    def __eq__(self, other: object) -> bool:
        """Compare symbols (case-insensitive asset_type)."""
        if isinstance(other, Symbol):
            return self.asset_type.lower() == other.asset_type.lower() and self.raw == other.raw
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on normalized values."""
        return hash((self.asset_type.lower(), self.raw))

    def __str__(self) -> str:
        """Return full symbol string."""
        return f"{self.asset_type}:{self.raw}"
