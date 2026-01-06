"""Tests for bar aggregation functionality."""

import pytest

from trdr.core import Symbol, Timeframe, parse_timeframe
from trdr.data import Bar, BarAggregator, TimeframeAdapter

_TEST_STOCK = Symbol.parse("stock:AAPL")
_TEST_CRYPTO = Symbol.parse("crypto:ETH/USD")


class TestBarAggregator:
    """Test BarAggregator class."""

    def _make_bars(self, n: int, base_price: float = 100.0) -> list[Bar]:
        """Create n test bars with incrementing timestamps."""
        bars = []
        for i in range(n):
            bars.append(
                Bar(
                    timestamp=f"2024-01-0{(i // 24) + 1}T{i % 24:02d}:00:00+00:00",
                    open=base_price + i,
                    high=base_price + i + 1,
                    low=base_price + i - 0.5,
                    close=base_price + i + 0.5,
                    volume=1000 + i * 100,
                )
            )
        return bars

    def test_aggregate_no_aggregation_needed(self):
        """Factor of 1 returns original bars."""
        aggregator = BarAggregator()
        bars = self._make_bars(5)

        result = aggregator.aggregate(bars, n=1)

        assert result == bars

    def test_aggregate_two_bars(self):
        """Aggregate 2 bars into 1."""
        aggregator = BarAggregator()
        bars = [
            Bar(
                timestamp="2024-01-01T00:00:00+00:00",
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000,
            ),
            Bar(
                timestamp="2024-01-01T01:00:00+00:00",
                open=103.0,
                high=108.0,
                low=101.0,
                close=106.0,
                volume=1500,
            ),
        ]

        result = aggregator.aggregate(bars, n=2)

        assert len(result) == 1
        agg = result[0]
        assert agg.open == 100.0  # First bar's open
        assert agg.high == 108.0  # Max high
        assert agg.low == 99.0  # Min low
        assert agg.close == 106.0  # Last bar's close
        assert agg.volume == 2500  # Sum of volumes
        assert agg.timestamp == "2024-01-01T01:00:00+00:00"  # Last timestamp

    def test_aggregate_three_bars(self):
        """Aggregate 3 bars simulating 3-day bars from 1-day."""
        aggregator = BarAggregator()
        bars = [
            Bar(
                timestamp="2024-01-01T00:00:00+00:00",
                open=100.0,
                high=110.0,
                low=95.0,
                close=108.0,
                volume=1000,
            ),
            Bar(
                timestamp="2024-01-02T00:00:00+00:00",
                open=108.0,
                high=115.0,
                low=105.0,
                close=112.0,
                volume=1200,
            ),
            Bar(
                timestamp="2024-01-03T00:00:00+00:00",
                open=112.0,
                high=120.0,
                low=110.0,
                close=118.0,
                volume=1500,
            ),
        ]

        result = aggregator.aggregate(bars, n=3)

        assert len(result) == 1
        agg = result[0]
        assert agg.open == 100.0
        assert agg.high == 120.0
        assert agg.low == 95.0
        assert agg.close == 118.0
        assert agg.volume == 3700
        assert agg.timestamp == "2024-01-03T00:00:00+00:00"

    def test_aggregate_incomplete_first_period_dropped(self):
        """Incomplete first period is dropped by default."""
        aggregator = BarAggregator()
        bars = self._make_bars(5)  # 5 bars, factor 3 = 1 complete group + 2 leftover

        result = aggregator.aggregate(bars, n=3, drop_incomplete=True)

        # Only 1 complete group of 3 (bars 2-4), first 2 bars dropped
        assert len(result) == 1

    def test_aggregate_incomplete_first_period_kept(self):
        """Incomplete first period kept when drop_incomplete=False."""
        aggregator = BarAggregator()
        bars = self._make_bars(5)

        result = aggregator.aggregate(bars, n=3, drop_incomplete=False)

        # 1 incomplete group (2 bars) + 1 complete group (3 bars) = 2 groups
        assert len(result) == 2

    def test_aggregate_multiple_complete_groups(self):
        """Multiple complete groups with no remainder."""
        aggregator = BarAggregator()
        bars = self._make_bars(6)

        result = aggregator.aggregate(bars, n=3)

        assert len(result) == 2

    def test_aggregate_60_minute_bars(self):
        """Simulate 60m aggregation from 1m bars."""
        aggregator = BarAggregator()
        # Create 60 1-minute bars
        bars = []
        for i in range(60):
            bars.append(
                Bar(
                    timestamp=f"2024-01-01T10:{i:02d}:00+00:00",
                    open=100.0 + i * 0.1,
                    high=100.0 + i * 0.1 + 0.2,
                    low=100.0 + i * 0.1 - 0.1,
                    close=100.0 + i * 0.1 + 0.1,
                    volume=100,
                )
            )

        result = aggregator.aggregate(bars, n=60)

        assert len(result) == 1
        agg = result[0]
        assert agg.open == 100.0  # First minute's open
        assert agg.close == pytest.approx(106.0, rel=0.01)  # Last minute's close
        assert agg.volume == 6000  # 60 * 100
        assert agg.timestamp == "2024-01-01T10:59:00+00:00"

    def test_aggregate_empty_bars(self):
        """Empty bar list returns empty."""
        aggregator = BarAggregator()

        result = aggregator.aggregate([], n=3)

        assert result == []


class TestTimeframeAdapter:
    """Test TimeframeAdapter for Alpaca translation."""

    def test_native_minute_timeframes(self):
        """Minutes 1-59 use native Alpaca - no aggregation."""
        assert not TimeframeAdapter(parse_timeframe("1m"), _TEST_STOCK).needs_aggregation
        assert not TimeframeAdapter(parse_timeframe("15m"), _TEST_STOCK).needs_aggregation
        assert not TimeframeAdapter(parse_timeframe("30m"), _TEST_STOCK).needs_aggregation
        assert not TimeframeAdapter(parse_timeframe("59m"), _TEST_STOCK).needs_aggregation

    def test_aggregated_minute_timeframes(self):
        """Non-canonical minutes need aggregation from 1m."""
        # 60m = 1h (native), 120m = 2h (native) - no aggregation
        assert not TimeframeAdapter(parse_timeframe("60m"), _TEST_STOCK).needs_aggregation  # -> 1h
        assert not TimeframeAdapter(parse_timeframe("120m"), _TEST_STOCK).needs_aggregation  # -> 2h

        # 90m cannot be expressed as hours - needs aggregation
        adapter = TimeframeAdapter(parse_timeframe("90m"), _TEST_STOCK)
        assert adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "m")
        assert adapter.aggregation_factor == 90

        # 61m needs aggregation (not divisible by 60)
        adapter = TimeframeAdapter(parse_timeframe("61m"), _TEST_STOCK)
        assert adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "m")
        assert adapter.aggregation_factor == 61

    def test_native_hour_timeframes(self):
        """Hours 1-23 use native Alpaca - no aggregation."""
        assert not TimeframeAdapter(parse_timeframe("1h"), _TEST_STOCK).needs_aggregation
        assert not TimeframeAdapter(parse_timeframe("4h"), _TEST_STOCK).needs_aggregation
        assert not TimeframeAdapter(parse_timeframe("23h"), _TEST_STOCK).needs_aggregation

    def test_aggregated_hour_timeframes(self):
        """Non-canonical hours need aggregation from 1h."""
        # 24h = 1d (native), 48h = 2d (needs aggregation from 1d)
        assert not TimeframeAdapter(parse_timeframe("24h"), _TEST_STOCK).needs_aggregation  # -> 1d

        # 48h = 2d, but 2d needs aggregation
        adapter = TimeframeAdapter(parse_timeframe("48h"), _TEST_STOCK)
        assert adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "d")
        assert adapter.aggregation_factor == 2

        # 25h needs aggregation (not divisible by 24)
        adapter = TimeframeAdapter(parse_timeframe("25h"), _TEST_STOCK)
        assert adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "h")
        assert adapter.aggregation_factor == 25

    def test_native_day_timeframe(self):
        """1d uses native Alpaca - no aggregation."""
        assert not TimeframeAdapter(parse_timeframe("1d"), _TEST_STOCK).needs_aggregation

    def test_aggregated_day_timeframes(self):
        """Days 2+ need aggregation from 1d."""
        adapter = TimeframeAdapter(parse_timeframe("2d"), _TEST_STOCK)
        assert adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "d")
        assert adapter.aggregation_factor == 2

        adapter = TimeframeAdapter(parse_timeframe("3d"), _TEST_STOCK)
        assert adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "d")
        assert adapter.aggregation_factor == 3

        adapter = TimeframeAdapter(parse_timeframe("5d"), _TEST_STOCK)
        assert adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "d")
        assert adapter.aggregation_factor == 5

    def test_native_week_timeframe(self):
        """1w uses native Alpaca - no aggregation."""
        assert not TimeframeAdapter(parse_timeframe("1w"), _TEST_STOCK).needs_aggregation
        assert not TimeframeAdapter(parse_timeframe("1week"), _TEST_STOCK).needs_aggregation

    def test_aggregated_week_timeframes(self):
        """Weeks 2+ need aggregation from 1d (5 trading days/week)."""
        adapter = TimeframeAdapter(parse_timeframe("2w"), _TEST_STOCK)
        assert adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "d")
        assert adapter.aggregation_factor == 10  # 2 weeks * 5 trading days

        adapter = TimeframeAdapter(parse_timeframe("4w"), _TEST_STOCK)
        assert adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "d")
        assert adapter.aggregation_factor == 20

    def test_native_month_timeframes(self):
        """Months 1, 2, 3, 6, 12 use native Alpaca."""
        assert not TimeframeAdapter(parse_timeframe("1mo"), _TEST_STOCK).needs_aggregation
        assert not TimeframeAdapter(parse_timeframe("2mo"), _TEST_STOCK).needs_aggregation
        assert not TimeframeAdapter(parse_timeframe("3mo"), _TEST_STOCK).needs_aggregation
        assert not TimeframeAdapter(parse_timeframe("6mo"), _TEST_STOCK).needs_aggregation
        assert not TimeframeAdapter(parse_timeframe("12mo"), _TEST_STOCK).needs_aggregation

    def test_aggregated_month_timeframes(self):
        """Non-standard months need aggregation from 1d."""
        adapter = TimeframeAdapter(parse_timeframe("4mo"), _TEST_STOCK)
        assert adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "d")
        assert adapter.aggregation_factor == 84  # 4 * 21 trading days

        adapter = TimeframeAdapter(parse_timeframe("5mo"), _TEST_STOCK)
        assert adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "d")
        assert adapter.aggregation_factor == 105

    def test_base_timeframe_native(self):
        """Native timeframes return themselves as base."""
        adapter = TimeframeAdapter(parse_timeframe("15m"), _TEST_STOCK)
        assert not adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(15, "m")

        adapter = TimeframeAdapter(parse_timeframe("4h"), _TEST_STOCK)
        assert not adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(4, "h")

        adapter = TimeframeAdapter(parse_timeframe("1d"), _TEST_STOCK)
        assert not adapter.needs_aggregation
        assert adapter.base_timeframe == Timeframe(1, "d")

    def test_base_timeframe_aggregated(self):
        """Aggregated timeframes return canonical base unit."""
        # 90m uses 1m base
        assert TimeframeAdapter(parse_timeframe("90m"), _TEST_STOCK).base_timeframe == Timeframe(
            1, "m"
        )
        # 25h uses 1h base
        assert TimeframeAdapter(parse_timeframe("25h"), _TEST_STOCK).base_timeframe == Timeframe(
            1, "h"
        )
        # 3d uses 1d base
        assert TimeframeAdapter(parse_timeframe("3d"), _TEST_STOCK).base_timeframe == Timeframe(
            1, "d"
        )
        # 2w uses 1d base (weeks aggregate from days)
        assert TimeframeAdapter(parse_timeframe("2w"), _TEST_STOCK).base_timeframe == Timeframe(
            1, "d"
        )
        # 48h -> 2d uses 1d base (canonicalizes then aggregates)
        assert TimeframeAdapter(parse_timeframe("48h"), _TEST_STOCK).base_timeframe == Timeframe(
            1, "d"
        )

    def test_aggregation_factor_native(self):
        """Native timeframes have factor 1."""
        assert TimeframeAdapter(parse_timeframe("15m"), _TEST_STOCK).aggregation_factor == 1
        assert TimeframeAdapter(parse_timeframe("1h"), _TEST_STOCK).aggregation_factor == 1
        assert TimeframeAdapter(parse_timeframe("1d"), _TEST_STOCK).aggregation_factor == 1
        assert TimeframeAdapter(parse_timeframe("1w"), _TEST_STOCK).aggregation_factor == 1

    def test_aggregation_factor_aggregated(self):
        """Aggregated timeframes use canonical form factor."""
        assert TimeframeAdapter(parse_timeframe("90m"), _TEST_STOCK).aggregation_factor == 90
        assert TimeframeAdapter(parse_timeframe("25h"), _TEST_STOCK).aggregation_factor == 25
        assert TimeframeAdapter(parse_timeframe("3d"), _TEST_STOCK).aggregation_factor == 3
        assert (
            TimeframeAdapter(parse_timeframe("2w"), _TEST_STOCK).aggregation_factor == 10
        )  # 2 * 5 trading days
        # 48h -> 2d has factor 2, not 48
        assert TimeframeAdapter(parse_timeframe("48h"), _TEST_STOCK).aggregation_factor == 2

    def test_to_alpaca(self):
        """Convert to Alpaca TimeFrame."""
        from alpaca.data.timeframe import TimeFrameUnit

        # Native - returns canonical Alpaca timeframe
        adapter = TimeframeAdapter(parse_timeframe("15m"), _TEST_STOCK)
        alpaca_tf = adapter.to_alpaca()
        assert alpaca_tf.amount == 15
        assert alpaca_tf.unit == TimeFrameUnit.Minute

        # Canonicalized
        adapter = TimeframeAdapter(parse_timeframe("60m"), _TEST_STOCK)
        alpaca_tf = adapter.to_alpaca()
        assert alpaca_tf.amount == 1
        assert alpaca_tf.unit == TimeFrameUnit.Hour

        # Needs aggregation - returns base timeframe
        adapter = TimeframeAdapter(parse_timeframe("90m"), _TEST_STOCK)
        alpaca_tf = adapter.to_alpaca()
        assert alpaca_tf.amount == 1  # Base is 1m
        assert alpaca_tf.unit == TimeFrameUnit.Minute

    def test_crypto_weekly_aggregation_24x7(self):
        """Crypto uses 7 days/week (24/7 trading) vs stock 5 days/week."""
        # Stock: 2 weeks = 10 days (5 trading days/week)
        stock_adapter = TimeframeAdapter(parse_timeframe("2w"), _TEST_STOCK)
        assert stock_adapter.aggregation_factor == 10

        # Crypto: 2 weeks = 14 days (7 days/week)
        crypto_adapter = TimeframeAdapter(parse_timeframe("2w"), _TEST_CRYPTO)
        assert crypto_adapter.aggregation_factor == 14

        # 4 weeks
        assert TimeframeAdapter(parse_timeframe("4w"), _TEST_STOCK).aggregation_factor == 20
        assert TimeframeAdapter(parse_timeframe("4w"), _TEST_CRYPTO).aggregation_factor == 28

    def test_crypto_monthly_aggregation_24x7(self):
        """Crypto uses 30 days/month vs stock 21 trading days/month."""
        # Stock: 4 months = 84 days (21 trading days/month)
        stock_adapter = TimeframeAdapter(parse_timeframe("4mo"), _TEST_STOCK)
        assert stock_adapter.aggregation_factor == 84

        # Crypto: 4 months = 120 days (30 days/month)
        crypto_adapter = TimeframeAdapter(parse_timeframe("4mo"), _TEST_CRYPTO)
        assert crypto_adapter.aggregation_factor == 120

        # 5 months
        assert TimeframeAdapter(parse_timeframe("5mo"), _TEST_STOCK).aggregation_factor == 105
        assert TimeframeAdapter(parse_timeframe("5mo"), _TEST_CRYPTO).aggregation_factor == 150

    def test_crypto_intraday_same_as_stock(self):
        """Intraday timeframes (m, h, d) work the same for crypto and stocks."""
        # Minutes
        assert TimeframeAdapter(parse_timeframe("90m"), _TEST_STOCK).aggregation_factor == 90
        assert TimeframeAdapter(parse_timeframe("90m"), _TEST_CRYPTO).aggregation_factor == 90

        # Hours
        assert TimeframeAdapter(parse_timeframe("25h"), _TEST_STOCK).aggregation_factor == 25
        assert TimeframeAdapter(parse_timeframe("25h"), _TEST_CRYPTO).aggregation_factor == 25

        # Days
        assert TimeframeAdapter(parse_timeframe("3d"), _TEST_STOCK).aggregation_factor == 3
        assert TimeframeAdapter(parse_timeframe("3d"), _TEST_CRYPTO).aggregation_factor == 3


class TestTimeframe:
    """Test Timeframe class and canonical conversion."""

    def test_canonical_minutes_to_hours(self):
        """Minutes divisible by 60 canonicalize to hours."""
        # 60m -> 1h
        tf = parse_timeframe("60m")
        assert tf.canonical.amount == 1
        assert tf.canonical.unit == "h"
        assert str(tf) == "1h"

        # 120m -> 2h
        tf = parse_timeframe("120m")
        assert tf.canonical.amount == 2
        assert tf.canonical.unit == "h"
        assert str(tf) == "2h"

        # 180m -> 3h
        tf = parse_timeframe("180m")
        assert tf.canonical.amount == 3
        assert tf.canonical.unit == "h"
        assert str(tf) == "3h"

    def test_canonical_hours_to_days(self):
        """Hours divisible by 24 canonicalize to days."""
        # 24h -> 1d
        tf = parse_timeframe("24h")
        assert tf.canonical.amount == 1
        assert tf.canonical.unit == "d"
        assert str(tf) == "1d"

        # 48h -> 2d
        tf = parse_timeframe("48h")
        assert tf.canonical.amount == 2
        assert tf.canonical.unit == "d"
        assert str(tf) == "2d"

        # 96h -> 4d
        tf = parse_timeframe("96h")
        assert tf.canonical.amount == 4
        assert tf.canonical.unit == "d"
        assert str(tf) == "4d"

    def test_canonical_chained(self):
        """Chained canonicalization: 1440m -> 24h -> 1d."""
        tf = parse_timeframe("1440m")
        assert tf.canonical.amount == 1
        assert tf.canonical.unit == "d"
        assert str(tf) == "1d"

        # 2880m -> 48h -> 2d
        tf = parse_timeframe("2880m")
        assert tf.canonical.amount == 2
        assert tf.canonical.unit == "d"
        assert str(tf) == "2d"

    def test_non_canonical_minutes(self):
        """Minutes not divisible by 60 stay as minutes."""
        tf = parse_timeframe("15m")
        assert tf.canonical.amount == 15
        assert tf.canonical.unit == "m"
        assert str(tf) == "15m"

        tf = parse_timeframe("90m")
        assert tf.canonical.amount == 90
        assert tf.canonical.unit == "m"
        assert str(tf) == "90m"

    def test_non_canonical_hours(self):
        """Hours not divisible by 24 stay as hours."""
        tf = parse_timeframe("4h")
        assert tf.canonical.amount == 4
        assert tf.canonical.unit == "h"
        assert str(tf) == "4h"

        tf = parse_timeframe("25h")
        assert tf.canonical.amount == 25
        assert tf.canonical.unit == "h"
        assert str(tf) == "25h"

    def test_seconds(self):
        """Duration in seconds."""
        assert parse_timeframe("1m").seconds == 60
        assert parse_timeframe("15m").seconds == 900
        assert parse_timeframe("1h").seconds == 3600
        assert parse_timeframe("4h").seconds == 14400
        assert parse_timeframe("1d").seconds == 86400
        assert parse_timeframe("1w").seconds == 7 * 86400

    def test_invalid_zero_amount(self):
        """Zero amount raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            parse_timeframe("0m")
        with pytest.raises(ValueError, match="must be positive"):
            parse_timeframe("0h")
        with pytest.raises(ValueError, match="must be positive"):
            parse_timeframe("0d")

    def test_invalid_format(self):
        """Invalid formats raise ValueError."""
        with pytest.raises(ValueError):
            parse_timeframe("invalid")
        with pytest.raises(ValueError):
            parse_timeframe("")
        with pytest.raises(ValueError):
            parse_timeframe("abc123")
        with pytest.raises(ValueError):
            parse_timeframe("-1m")  # Negative
        with pytest.raises(ValueError):
            parse_timeframe("1x")  # Unknown unit

    def test_equality_same_timeframe(self):
        """Same timeframes are equal."""
        assert parse_timeframe("15m") == parse_timeframe("15m")
        assert parse_timeframe("1h") == parse_timeframe("1h")
        assert parse_timeframe("1d") == parse_timeframe("1d")

    def test_equality_canonical_forms(self):
        """Canonical equivalents are equal (60m == 1h)."""
        assert parse_timeframe("60m") == parse_timeframe("1h")
        assert parse_timeframe("120m") == parse_timeframe("2h")
        assert parse_timeframe("240m") == parse_timeframe("4h")
        assert parse_timeframe("24h") == parse_timeframe("1d")
        assert parse_timeframe("48h") == parse_timeframe("2d")
        assert parse_timeframe("1440m") == parse_timeframe("1d")  # 1440m -> 24h -> 1d

    def test_equality_different_timeframes(self):
        """Different timeframes are not equal."""
        assert parse_timeframe("15m") != parse_timeframe("30m")
        assert parse_timeframe("1h") != parse_timeframe("4h")
        assert parse_timeframe("1d") != parse_timeframe("1w")

    def test_equality_with_timeframe_object(self):
        """Compare with Timeframe object directly."""
        assert parse_timeframe("15m") == Timeframe(15, "m")
        assert parse_timeframe("1h") == Timeframe(1, "h")
        assert parse_timeframe("60m") == Timeframe(1, "h")  # Canonical comparison

    def test_hash_canonical_forms(self):
        """Canonical equivalents have same hash (usable in sets/dicts)."""
        assert hash(parse_timeframe("60m")) == hash(parse_timeframe("1h"))
        assert hash(parse_timeframe("24h")) == hash(parse_timeframe("1d"))
        # Can use in set
        s = {parse_timeframe("1h"), parse_timeframe("60m")}
        assert len(s) == 1

    def test_is_intraday(self):
        """Test is_intraday property."""
        assert parse_timeframe("15m").is_intraday
        assert parse_timeframe("1h").is_intraday
        assert parse_timeframe("4h").is_intraday
        assert not parse_timeframe("1d").is_intraday
        assert not parse_timeframe("1w").is_intraday
