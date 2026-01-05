"""Configuration dataclasses and environment loading."""

from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from .symbol import Symbol


@dataclass(frozen=True)
class AlpacaConfig:
    """Alpaca API configuration."""

    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"

    @property
    def is_paper(self) -> bool:
        """Check if using paper trading endpoint."""
        return "paper" in self.base_url


@dataclass(frozen=True)
class StrategyConfig:
    """Volume Profile strategy parameters."""

    symbol: "Symbol"
    lookback: int = 50  # Bars for volume profile calculation
    price_levels: int = 40  # Number of price buckets
    value_area_pct: float = 0.70  # 70% of volume
    atr_period: int = 14
    atr_threshold: float = 2.0  # Entry when price > 2 ATR from VA
    stop_loss_multiplier: float = 1.75  # Stop at 1.75x VA width
    position_size: float = 1.0  # Shares/units per trade


@dataclass(frozen=True)
class LoopConfig:
    """Async loop timing configuration."""

    price_interval: float = 1.0  # seconds
    trading_interval: float = 5.0
    order_interval: float = 0.5
    display_interval: float = 0.5


@dataclass(frozen=True)
class BotConfig:
    """Complete bot configuration."""

    alpaca: AlpacaConfig
    strategy: StrategyConfig
    loops: LoopConfig
    data_dir: Path = Path("data")

    @property
    def cache_dir(self) -> Path:
        """Directory for price/volume cache."""
        return self.data_dir / "cache"

    @property
    def runs_dir(self) -> Path:
        """Directory for archived run data."""
        return self.data_dir / "runs"


def load_config(env_path: Path | None = None) -> BotConfig:
    """Load configuration from environment variables.

    Args:
        env_path: Path to .env file, defaults to .env in current directory

    Returns:
        Complete bot configuration

    Raises:
        ValueError: If required environment variables are missing
    """
    load_dotenv(env_path or Path(".env"))

    api_key = environ.get("ALPACA_API_KEY")
    secret_key = environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment")

    symbol = environ.get("SYMBOL", "AAPL")
    base_url = environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    return BotConfig(
        alpaca=AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
        ),
        strategy=StrategyConfig(symbol=symbol),
        loops=LoopConfig(),
    )
