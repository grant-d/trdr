"""Configuration dataclasses and environment loading."""

from dataclasses import dataclass
from os import environ
from pathlib import Path

from dotenv import load_dotenv

from .duration import Duration
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
    loops: LoopConfig
    symbol: Symbol
    lookback: Duration
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
        Bot configuration with Alpaca credentials

    Raises:
        ValueError: If required environment variables are missing
    """
    load_dotenv(env_path or Path(".env"))

    api_key = environ.get("ALPACA_API_KEY")
    secret_key = environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment")

    symbol_str = environ.get("SYMBOL", "stock:AAPL")
    lookback_str = environ.get("LOOKBACK", "30d")
    base_url = environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    return BotConfig(
        alpaca=AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
        ),
        loops=LoopConfig(),
        symbol=Symbol.parse(symbol_str),
        lookback=Duration.parse(lookback_str),
    )
