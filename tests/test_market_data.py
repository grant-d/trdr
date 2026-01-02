from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from trdr.data.market import Bar, MarketDataClient, Quote, Symbol


class FakeQuote:
    def __init__(self, bid_price: float, ask_price: float, timestamp: datetime) -> None:
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.timestamp = timestamp


class FakeClient:
    def __init__(self, quote) -> None:
        self._quote = quote

    def get_crypto_latest_quote(self, request):
        return SimpleNamespace(data={"BTC/USD": self._quote})

    def get_stock_latest_quote(self, request):
        return SimpleNamespace(data={"AAPL": self._quote})


def test_symbol_cache_key_format():
    """Cache key includes asset type and is lowercase."""
    crypto = Symbol.parse("crypto:BTC/USD")
    assert crypto.cache_key == "crypto:btc_usd"

    stock = Symbol.parse("AAPL")
    assert stock.cache_key == "stock:aapl"

    stock_explicit = Symbol.parse("stock:MSFT")
    assert stock_explicit.cache_key == "stock:msft"


@pytest.mark.asyncio
async def test_get_current_price_handles_data_wrapper():
    ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    quote = FakeQuote(100.0, 102.0, ts)
    client = MarketDataClient.__new__(MarketDataClient)
    client._crypto_client = FakeClient(quote)
    client._stock_client = FakeClient(quote)

    result = await MarketDataClient.get_current_price(client, Symbol.parse("crypto:BTC/USD"))
    assert isinstance(result, Quote)
    assert result.price == 101.0
    assert result.bid == 100.0
    assert result.ask == 102.0


class FakeBar:
    """Fake Alpaca bar for testing."""

    def __init__(self, timestamp: datetime, open_: float, high: float, low: float, close: float, volume: int):
        self.timestamp = timestamp
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class FakeBarsClient:
    """Fake Alpaca bars client that returns specified bars."""

    def __init__(self, bars: list[FakeBar]):
        self._bars = bars

    def get_crypto_bars(self, request):
        return SimpleNamespace(data={"BTC/USD": self._bars})


def make_bar(hour_offset: int, price: float) -> Bar:
    """Create a bar with timestamp offset from base time."""
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts = base + timedelta(hours=hour_offset)
    return Bar(
        timestamp=ts.isoformat(),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=100,
    )


def make_fake_bar(hour_offset: int, price: float) -> FakeBar:
    """Create a fake Alpaca bar for API response."""
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts = base + timedelta(hours=hour_offset)
    return FakeBar(ts, price, price + 1, price - 1, price, 100)


@pytest.mark.asyncio
async def test_cache_append_preserves_order_no_duplicates(tmp_path: Path):
    """Cache append merges correctly: no duplicates, maintains chronological order."""
    from alpaca.data.timeframe import TimeFrame

    # Create initial cache with 20 bars (hours 0-19)
    initial_bars = [make_bar(i, 100.0 + i) for i in range(20)]
    cache_file = tmp_path / "crypto:btc_usd:1hour.jsonl"
    with open(cache_file, "w") as f:
        for bar in initial_bars:
            f.write(f'{{"timestamp": "{bar.timestamp}", "open": {bar.open}, "high": {bar.high}, "low": {bar.low}, "close": {bar.close}, "volume": {bar.volume}}}\n')

    # Fake API returns bars for hours 10-25 (overlapping last 10, plus 6 new)
    # Hours 10-19 have UPDATED prices (150 instead of 110-119)
    api_bars = [make_fake_bar(i, 150.0 + i) for i in range(10, 26)]

    # Create client with fake Alpaca client
    client = MarketDataClient.__new__(MarketDataClient)
    client.cache_dir = tmp_path
    client._crypto_client = FakeBarsClient(api_bars)

    # Force cache to be stale by modifying the last bar timestamp to be old
    stale_bars = initial_bars.copy()
    old_ts = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    stale_bars[-1] = Bar(
        timestamp=old_ts.isoformat(),
        open=119.0,
        high=120.0,
        low=118.0,
        close=119.0,
        volume=100,
    )
    with open(cache_file, "w") as f:
        for bar in stale_bars:
            f.write(f'{{"timestamp": "{bar.timestamp}", "open": {bar.open}, "high": {bar.high}, "low": {bar.low}, "close": {bar.close}, "volume": {bar.volume}}}\n')

    # Call get_bars - should merge cache with API response
    result = await client.get_bars("crypto:BTC/USD", lookback=50, timeframe=TimeFrame.Hour)

    # Verify: should have bars 0-9 from cache + bars 10-25 from API = 26 bars
    assert len(result) == 26, f"Expected 26 bars, got {len(result)}"

    # Verify chronological order (timestamps increasing)
    timestamps = [bar.timestamp for bar in result]
    assert timestamps == sorted(timestamps), "Bars not in chronological order"

    # Verify no duplicate timestamps
    assert len(timestamps) == len(set(timestamps)), "Duplicate timestamps found"

    # Verify first 10 bars have original prices (100-109)
    for i in range(10):
        assert result[i].close == 100.0 + i, f"Bar {i} has wrong price"

    # Verify bars 10-25 have updated prices from API (160-175)
    for i in range(10, 26):
        assert result[i].close == 150.0 + i, f"Bar {i} has wrong price, expected {150.0 + i}, got {result[i].close}"


@pytest.mark.asyncio
async def test_cache_gap_behavior(tmp_path: Path):
    """Gaps within overlap window get filled; older gaps persist (TODO: full gap detection)."""
    from alpaca.data.timeframe import TimeFrame

    # Create cache with 20 bars but DELETE the 3rd-last (hour 17)
    all_bars = [make_bar(i, 100.0 + i) for i in range(20)]
    bars_with_gap = all_bars[:17] + all_bars[18:]  # Skip hour 17

    cache_file = tmp_path / "crypto:btc_usd:1hour.jsonl"
    with open(cache_file, "w") as f:
        for bar in bars_with_gap:
            f.write(f'{{"timestamp": "{bar.timestamp}", "open": {bar.open}, "high": {bar.high}, "low": {bar.low}, "close": {bar.close}, "volume": {bar.volume}}}\n')

    # Verify gap exists in cache
    assert len(bars_with_gap) == 19, "Should have 19 bars (20 - 1 deleted)"

    # API returns bars 10-25 (which includes the missing hour 17)
    api_bars = [make_fake_bar(i, 100.0 + i) for i in range(10, 26)]

    # Force cache to be stale
    stale_bars = bars_with_gap.copy()
    old_ts = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    stale_bars[-1] = Bar(
        timestamp=old_ts.isoformat(),
        open=119.0, high=120.0, low=118.0, close=119.0, volume=100,
    )
    with open(cache_file, "w") as f:
        for bar in stale_bars:
            f.write(f'{{"timestamp": "{bar.timestamp}", "open": {bar.open}, "high": {bar.high}, "low": {bar.low}, "close": {bar.close}, "volume": {bar.volume}}}\n')

    client = MarketDataClient.__new__(MarketDataClient)
    client.cache_dir = tmp_path
    client._crypto_client = FakeBarsClient(api_bars)

    result = await client.get_bars("crypto:BTC/USD", lookback=50, timeframe=TimeFrame.Hour)

    # With restatement overlap of 10, we keep bars 0-8 from cache (9 bars)
    # and get bars 10-25 from API (16 bars) = 25 total
    # The gap at hour 9 is NOT filled because we only fetch from overlap point
    # Hour 17 IS filled because it's in the API response range

    timestamps = [bar.timestamp for bar in result]

    # Check no duplicates
    assert len(timestamps) == len(set(timestamps)), "Duplicate timestamps found"

    # Check chronological order
    assert timestamps == sorted(timestamps), "Bars not in chronological order"

    # The result should have: hours 0-8 (from cache before overlap) + hours 10-25 (from API)
    # Hour 9 is lost because overlap starts at hour 9 (19 - 10 = 9)
    # So we expect 9 + 16 = 25 bars, missing hour 9
    assert len(result) == 25, f"Expected 25 bars, got {len(result)}"
