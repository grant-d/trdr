"""Market data fetching and caching via Alpaca API."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from ..core.config import AlpacaConfig


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


@dataclass
class Quote:
    """Current price quote."""

    symbol: str
    price: float
    bid: float
    ask: float
    timestamp: str


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
        """Parse symbol string into Symbol object."""
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
        """Safe string for cache filenames."""
        return self.raw.replace("/", "_")

    def __str__(self) -> str:
        """Return full symbol string."""
        return f"{self.asset_type}:{self.raw}"


class MarketDataClient:
    """Fetches market data from Alpaca with disk caching."""

    def __init__(self, config: AlpacaConfig, cache_dir: Path):
        """Initialize market data client.

        Args:
            config: Alpaca API configuration
            cache_dir: Directory for caching bar data
        """
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Alpaca clients
        self._stock_client = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )
        self._crypto_client = CryptoHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )
        self._trading_client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.is_paper,
        )

    async def get_bars(
        self,
        symbol: str,
        lookback: int = 50,
        timeframe: TimeFrame = TimeFrame.Hour,
    ) -> list[Bar]:
        """Fetch historical bars with caching.

        Args:
            symbol: Symbol (e.g., "AAPL" for stocks, "BTC/USD" for crypto)
            lookback: Number of bars to fetch
            timeframe: Bar timeframe (default 1 hour)

        Returns:
            List of Bar objects, oldest first
        """
        sym = Symbol.parse(symbol) if isinstance(symbol, str) else symbol
        cache_file = self._cache_path(sym, timeframe)
        cached_bars = self._load_cache(cache_file)

        # Check if cache is fresh (less than 1 hour old)
        if cached_bars and len(cached_bars) >= lookback:
            last_bar_time = datetime.fromisoformat(
                cached_bars[-1].timestamp.replace("Z", "+00:00")
            )
            if datetime.now(last_bar_time.tzinfo) - last_bar_time < timedelta(hours=1):
                return cached_bars[-lookback:]

        # Fetch from Alpaca
        end = datetime.now()
        start = end - timedelta(days=lookback // 6 + 5)

        if sym.is_crypto:
            request = CryptoBarsRequest(
                symbol_or_symbols=sym.raw,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=lookback,
            )
            bars_data = self._crypto_client.get_crypto_bars(request)
        else:
            request = StockBarsRequest(
                symbol_or_symbols=sym.raw,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=lookback,
            )
            bars_data = self._stock_client.get_stock_bars(request)

        bars = []
        # Access via .data dict - BarSet's __contains__ doesn't work correctly
        bars_dict = bars_data.data if hasattr(bars_data, "data") else bars_data
        if sym.raw in bars_dict:
            for bar in bars_dict[sym.raw]:
                bars.append(
                    Bar(
                        timestamp=bar.timestamp.isoformat(),
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=int(bar.volume),
                    )
                )

        # Save to cache
        self._save_cache(cache_file, bars)

        return bars[-lookback:] if len(bars) > lookback else bars

    async def get_current_price(self, symbol: str) -> Quote:
        """Get current price quote.

        Args:
            symbol: Symbol (e.g., "AAPL" for stocks, "BTC/USD" for crypto)

        Returns:
            Current quote with bid/ask
        """
        sym = Symbol.parse(symbol) if isinstance(symbol, str) else symbol

        if sym.is_crypto:
            request = CryptoLatestQuoteRequest(symbol_or_symbols=sym.raw)
            quotes = self._crypto_client.get_crypto_latest_quote(request)
        else:
            request = StockLatestQuoteRequest(symbol_or_symbols=sym.raw)
            quotes = self._stock_client.get_stock_latest_quote(request)

        if sym.raw in quotes:
            quote = quotes[sym.raw]
            return Quote(
                symbol=str(sym),
                price=(float(quote.bid_price) + float(quote.ask_price)) / 2,
                bid=float(quote.bid_price),
                ask=float(quote.ask_price),
                timestamp=quote.timestamp.isoformat(),
            )

        raise ValueError(f"No quote available for {sym.raw}")

    def _cache_path(self, symbol: Symbol, timeframe: TimeFrame) -> Path:
        """Get cache file path for symbol and timeframe."""
        return self.cache_dir / f"{symbol.cache_key}_{timeframe.value}.json"

    def _load_cache(self, cache_file: Path) -> list[Bar]:
        """Load bars from cache file."""
        if not cache_file.exists():
            return []

        try:
            with open(cache_file) as f:
                data = json.load(f)
            return [Bar.from_dict(b) for b in data]
        except (json.JSONDecodeError, KeyError):
            return []

    def _save_cache(self, cache_file: Path, bars: list[Bar]) -> None:
        """Save bars to cache file."""
        with open(cache_file, "w") as f:
            json.dump([b.to_dict() for b in bars], f)
