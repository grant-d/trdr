"""Alpaca market data client with caching."""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestBarRequest,
    CryptoLatestQuoteRequest,
    StockBarsRequest,
    StockLatestBarRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from ..core.config import AlpacaConfig
from ..core.symbol import Symbol
from ..core.timeframe import Timeframe
from .bar import Bar
from .quote import Quote

# Max historical data available from Alpaca IEX feed (free tier).
# IEX data starts from ~2020, Alpaca limit is ~7 years.
# See: https://docs.alpaca.markets/docs/about-market-data-api
#      https://alpaca.markets/data
MAX_HISTORY_DAYS = 365 * 7  # 7 years


class AlpacaDataClient:
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
        symbol: Symbol,
        lookback: int,
        timeframe: Timeframe,
        force_refresh: bool = False,
    ) -> list[Bar]:
        """Fetch historical bars with caching.

        Supports arbitrary timeframes (60m, 3d, 2w) via automatic aggregation.
        When timeframe exceeds Alpaca limits, fetches base bars and aggregates.

        Args:
            symbol: Symbol object (e.g., Symbol.parse("stock:AAPL"), Symbol.parse("crypto:BTC/USD"))
            lookback: Number of bars to fetch
            timeframe: Bar timeframe
            force_refresh: If True, bypass cache freshness check

        Returns:
            List of Bar objects, oldest first
        """
        from .aggregator import BarAggregator
        from .timeframe_adapter import TimeframeAdapter

        adapter = TimeframeAdapter(timeframe, symbol)

        if adapter.needs_aggregation:
            # Fetch base bars and aggregate
            bars_needed = lookback * adapter.aggregation_factor + adapter.aggregation_factor
            raw_bars = await self._fetch_bars(
                symbol, adapter.to_alpaca(), bars_needed, force_refresh=force_refresh
            )

            aggregator = BarAggregator()
            aggregated = aggregator.aggregate(raw_bars, adapter.aggregation_factor)
            return aggregated[-lookback:]

        return await self._fetch_bars(
            symbol, adapter.to_alpaca(), lookback, force_refresh=force_refresh
        )

    async def _fetch_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        lookback: int,
        force_refresh: bool = False,
    ) -> list[Bar]:
        """Internal method to fetch bars with caching.

        Args:
            symbol: Symbol string
            timeframe: Alpaca TimeFrame object
            lookback: Number of bars to fetch

        Returns:
            List of Bar objects, oldest first
        """
        sym = Symbol.parse(symbol) if isinstance(symbol, str) else symbol
        cache_file = self._cache_path(sym, timeframe)
        cached_bars = self._load_cache(cache_file)
        original_cached = list(cached_bars)

        # Check if cache is fresh (less than 1 hour old)
        if not force_refresh and cached_bars and len(cached_bars) >= lookback:
            last_bar_time = datetime.fromisoformat(cached_bars[-1].timestamp.replace("Z", "+00:00"))
            if datetime.now(last_bar_time.tzinfo) - last_bar_time < timedelta(hours=1):
                return cached_bars[-lookback:]

        # Determine fetch start: full history if insufficient, else overlap for restatement
        end = datetime.now(timezone.utc)
        restatement_overlap = 10

        # Full fetch if: no cache, insufficient bars for lookback, or tiny cache
        needs_full_fetch: bool = (
            not cached_bars
            or len(cached_bars) < lookback
            or len(cached_bars) <= restatement_overlap
        )

        if needs_full_fetch:
            start = end - timedelta(days=MAX_HISTORY_DAYS)
        else:
            # Fetch from 10 bars back to handle restatements
            overlap_ts = datetime.fromisoformat(
                cached_bars[-restatement_overlap].timestamp.replace("Z", "+00:00")
            )
            start = overlap_ts

        # Only fetch if start is before end
        new_bars = []
        if start < end:
            if sym.is_crypto:
                request = CryptoBarsRequest(
                    symbol_or_symbols=sym.raw,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                )
                bars_data = self._crypto_client.get_crypto_bars(request)
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=sym.raw,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    feed=DataFeed.IEX,  # Use free IEX feed (no SIP subscription needed)
                )
                bars_data = self._stock_client.get_stock_bars(request)

            # Access via .data dict - BarSet's __contains__ doesn't work correctly
            bars_dict = bars_data.data if hasattr(bars_data, "data") else bars_data
            if sym.raw in bars_dict:
                for bar in bars_dict[sym.raw]:
                    new_bars.append(
                        Bar(
                            timestamp=bar.timestamp.isoformat(),
                            open=float(bar.open),
                            high=float(bar.high),
                            low=float(bar.low),
                            close=float(bar.close),
                            volume=int(bar.volume),
                        )
                    )

        # Merge: existing cache + new bars
        # TODO: detect gaps in cache and fill them (currently only last 10 bars are restated)
        if new_bars:
            if needs_full_fetch:
                cached_keep = []
            else:
                cached_keep = cached_bars[:-restatement_overlap]
            all_bars = cached_keep + new_bars
            self._save_cache(cache_file, all_bars)
        else:
            all_bars = original_cached

        return all_bars[-lookback:] if len(all_bars) > lookback else all_bars

    async def get_bars_multi(
        self,
        requirements: list,  # list[DataRequirement] - avoid circular import
    ) -> dict[str, list[Bar]]:
        """Fetch bars for multiple symbol/timeframe combinations.

        Supports arbitrary timeframes via automatic aggregation.

        Args:
            requirements: List of DataRequirement specifying each feed

        Returns:
            Dict mapping "symbol:timeframe" to list of bars
        """
        result = {}
        for req in requirements:
            symbol = Symbol.parse(req.symbol) if isinstance(req.symbol, str) else req.symbol
            bars = await self.get_bars(symbol, req.lookback_bars, req.timeframe)
            result[req.key] = bars
        return result

    async def get_latest_bar(self, symbol: Symbol) -> Bar | None:
        """Get the latest bar for a symbol.

        Used by live trading to get current market data.

        Args:
            symbol: Symbol object

        Returns:
            Latest bar or None if unavailable
        """
        sym = Symbol.parse(symbol) if isinstance(symbol, str) else symbol

        if sym.is_crypto:
            request = CryptoLatestBarRequest(symbol_or_symbols=sym.raw)
            result = self._crypto_client.get_crypto_latest_bar(request)
        else:
            request = StockLatestBarRequest(symbol_or_symbols=sym.raw)
            result = self._stock_client.get_stock_latest_bar(request)

        # SDK returns dict keyed by symbol
        result_data = result.data if hasattr(result, "data") else result
        if sym.raw in result_data:
            bar_data = result_data[sym.raw]
            return Bar(
                timestamp=bar_data.timestamp.isoformat(),
                open=float(bar_data.open),
                high=float(bar_data.high),
                low=float(bar_data.low),
                close=float(bar_data.close),
                volume=int(bar_data.volume),
            )
        return None

    async def get_current_price(self, symbol: Symbol) -> Quote:
        """Get current price quote.

        Args:
            symbol: Symbol object

        Returns:
            Current quote with bid/ask
        """

        if symbol.is_crypto:
            request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol.raw)
            quotes = self._crypto_client.get_crypto_latest_quote(request)
        else:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol.raw)
            quotes = self._stock_client.get_stock_latest_quote(request)

        # SDK returns object with .data dict, but may vary by version
        quotes_data = quotes.data if hasattr(quotes, "data") else quotes
        if symbol.raw in quotes_data:
            quote = quotes_data[symbol.raw]
            return Quote(
                symbol=str(symbol),
                price=(float(quote.bid_price) + float(quote.ask_price)) / 2,
                bid=float(quote.bid_price),
                ask=float(quote.ask_price),
                timestamp=quote.timestamp.isoformat(),
            )

        raise ValueError(f"No quote available for {symbol.raw}")

    def _cache_path(self, symbol: Symbol, timeframe: TimeFrame) -> Path:
        """Get cache file path for symbol and timeframe."""
        tf_str = str(timeframe.value).lower()
        return self.cache_dir / f"{symbol.cache_key}:{tf_str}.csv"

    def _load_cache(self, cache_file: Path) -> list[Bar]:
        """Load bars from CSV cache file."""
        if not cache_file.exists():
            return []

        bars = []
        try:
            with open(cache_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bars.append(
                        Bar(
                            timestamp=row["timestamp"],
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=int(row["volume"]),
                        )
                    )
            return bars
        except (KeyError, ValueError):
            return []

    def _save_cache(self, cache_file: Path, bars: list[Bar]) -> None:
        """Save bars to CSV cache file."""
        with open(cache_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for bar in bars:
                writer.writerow([bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume])
