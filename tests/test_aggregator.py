"""Tests for bar aggregation functionality."""

import pytest

from trdr.backtest import AggregationConfig, get_aggregation_config, parse_timeframe, Timeframe
from trdr.data import Bar, BarAggregator


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


class TestGetAggregationConfig:
    """Test get_aggregation_config function."""

    def test_native_minute_timeframes(self):
        """Minutes 1-59 use native Alpaca - no aggregation."""
        assert get_aggregation_config("1m") is None
        assert get_aggregation_config("15m") is None
        assert get_aggregation_config("30m") is None
        assert get_aggregation_config("59m") is None

    def test_aggregated_minute_timeframes(self):
        """Non-canonical minutes need aggregation from 1m."""
        # 60m = 1h (native), 120m = 2h (native) - no aggregation
        assert get_aggregation_config("60m") is None  # Canonicalizes to 1h
        assert get_aggregation_config("120m") is None  # Canonicalizes to 2h

        # 90m cannot be expressed as hours - needs aggregation
        config = get_aggregation_config("90m")
        assert config is not None
        assert config.base_timeframe == "1m"
        assert config.factor == 90

        # 61m needs aggregation (not divisible by 60)
        config = get_aggregation_config("61m")
        assert config is not None
        assert config.base_timeframe == "1m"
        assert config.factor == 61

    def test_native_hour_timeframes(self):
        """Hours 1-23 use native Alpaca - no aggregation."""
        assert get_aggregation_config("1h") is None
        assert get_aggregation_config("4h") is None
        assert get_aggregation_config("23h") is None

    def test_aggregated_hour_timeframes(self):
        """Non-canonical hours need aggregation from 1h."""
        # 24h = 1d (native), 48h = 2d (needs aggregation from 1d)
        assert get_aggregation_config("24h") is None  # Canonicalizes to 1d

        # 48h = 2d, but 2d needs aggregation
        config = get_aggregation_config("48h")
        assert config is not None
        assert config.base_timeframe == "1d"
        assert config.factor == 2

        # 25h needs aggregation (not divisible by 24)
        config = get_aggregation_config("25h")
        assert config is not None
        assert config.base_timeframe == "1h"
        assert config.factor == 25

    def test_native_day_timeframe(self):
        """1d uses native Alpaca - no aggregation."""
        assert get_aggregation_config("1d") is None

    def test_aggregated_day_timeframes(self):
        """Days 2+ need aggregation from 1d."""
        config = get_aggregation_config("2d")
        assert config is not None
        assert config.base_timeframe == "1d"
        assert config.factor == 2

        config = get_aggregation_config("3d")
        assert config is not None
        assert config.base_timeframe == "1d"
        assert config.factor == 3

        config = get_aggregation_config("5d")
        assert config is not None
        assert config.base_timeframe == "1d"
        assert config.factor == 5

    def test_native_week_timeframe(self):
        """1w uses native Alpaca - no aggregation."""
        assert get_aggregation_config("1w") is None
        assert get_aggregation_config("1week") is None

    def test_aggregated_week_timeframes(self):
        """Weeks 2+ need aggregation from 1d (5 trading days/week)."""
        config = get_aggregation_config("2w")
        assert config is not None
        assert config.base_timeframe == "1d"
        assert config.factor == 10  # 2 weeks * 5 trading days

        config = get_aggregation_config("4w")
        assert config is not None
        assert config.base_timeframe == "1d"
        assert config.factor == 20

    def test_native_month_timeframes(self):
        """Months 1, 2, 3, 6, 12 use native Alpaca."""
        assert get_aggregation_config("1mo") is None
        assert get_aggregation_config("2mo") is None
        assert get_aggregation_config("3mo") is None
        assert get_aggregation_config("6mo") is None
        assert get_aggregation_config("12mo") is None

    def test_aggregated_month_timeframes(self):
        """Non-standard months need aggregation from 1d."""
        config = get_aggregation_config("4mo")
        assert config is not None
        assert config.base_timeframe == "1d"
        assert config.factor == 84  # 4 * 21 trading days

        config = get_aggregation_config("5mo")
        assert config is not None
        assert config.base_timeframe == "1d"
        assert config.factor == 105

    def test_unit_aliases(self):
        """Various unit aliases work correctly."""
        # Minute aliases - 60min canonicalizes to 1h (native)
        assert get_aggregation_config("60min") is None
        assert get_aggregation_config("60minute") is None
        # 90min needs aggregation (not divisible by 60)
        config = get_aggregation_config("90min")
        assert config is not None
        assert config.base_timeframe == "1m"
        assert config.factor == 90

        # Hour aliases - 24hour canonicalizes to 1d (native)
        assert get_aggregation_config("24hour") is None
        # 25hour needs aggregation
        config = get_aggregation_config("25hour")
        assert config is not None
        assert config.base_timeframe == "1h"
        assert config.factor == 25

        # Day aliases - 3day needs aggregation
        config = get_aggregation_config("3day")
        assert config is not None
        assert config.base_timeframe == "1d"
        assert config.factor == 3

    def test_invalid_format(self):
        """Invalid formats return None."""
        assert get_aggregation_config("invalid") is None
        assert get_aggregation_config("") is None
        assert get_aggregation_config("abc123") is None


class TestAggregationConfig:
    """Test AggregationConfig dataclass."""

    def test_dataclass_fields(self):
        """AggregationConfig has expected fields."""
        config = AggregationConfig(base_timeframe="1d", factor=3)

        assert config.base_timeframe == "1d"
        assert config.factor == 3


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

    def test_needs_aggregation_native_minutes(self):
        """Minutes 1-59 don't need aggregation."""
        assert not parse_timeframe("1m").needs_aggregation
        assert not parse_timeframe("15m").needs_aggregation
        assert not parse_timeframe("30m").needs_aggregation
        assert not parse_timeframe("59m").needs_aggregation

    def test_needs_aggregation_canonical_hours(self):
        """60m, 120m canonicalize to hours - no aggregation."""
        assert not parse_timeframe("60m").needs_aggregation  # -> 1h
        assert not parse_timeframe("120m").needs_aggregation  # -> 2h
        assert not parse_timeframe("23h").needs_aggregation

    def test_needs_aggregation_canonical_days(self):
        """24h, 48h canonicalize to days - but 2d+ needs aggregation."""
        assert not parse_timeframe("24h").needs_aggregation  # -> 1d (native)
        assert parse_timeframe("48h").needs_aggregation  # -> 2d (needs agg)
        assert parse_timeframe("96h").needs_aggregation  # -> 4d (needs agg)

    def test_needs_aggregation_non_canonical(self):
        """Non-canonical timeframes need aggregation."""
        assert parse_timeframe("90m").needs_aggregation  # Not divisible by 60
        assert parse_timeframe("25h").needs_aggregation  # Not divisible by 24
        assert parse_timeframe("3d").needs_aggregation  # > 1d

    def test_base_timeframe(self):
        """Base timeframe for aggregation."""
        # Native - returns self
        assert parse_timeframe("15m").base_timeframe == Timeframe(15, "m")
        assert parse_timeframe("4h").base_timeframe == Timeframe(4, "h")
        assert parse_timeframe("1d").base_timeframe == Timeframe(1, "d")

        # Needs aggregation - returns base unit (uses canonical form)
        assert parse_timeframe("90m").base_timeframe == Timeframe(1, "m")
        assert parse_timeframe("25h").base_timeframe == Timeframe(1, "h")
        assert parse_timeframe("3d").base_timeframe == Timeframe(1, "d")
        assert parse_timeframe("2w").base_timeframe == Timeframe(1, "d")
        # 48h→2d uses 1d base, not 1h
        assert parse_timeframe("48h").base_timeframe == Timeframe(1, "d")

    def test_aggregation_factor(self):
        """Aggregation factor calculation."""
        # Native - factor is 1
        assert parse_timeframe("15m").aggregation_factor == 1
        assert parse_timeframe("1h").aggregation_factor == 1
        assert parse_timeframe("1d").aggregation_factor == 1

        # Needs aggregation - uses canonical form
        assert parse_timeframe("90m").aggregation_factor == 90
        assert parse_timeframe("25h").aggregation_factor == 25
        assert parse_timeframe("3d").aggregation_factor == 3
        assert parse_timeframe("2w").aggregation_factor == 10  # 2 * 5 trading days
        # 48h→2d has factor 2, not 48
        assert parse_timeframe("48h").aggregation_factor == 2

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
