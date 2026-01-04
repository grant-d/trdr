"""Tests for timeframe utilities and multi-feed alignment."""

from dataclasses import dataclass, field

import pytest

from trdr.backtest import align_feeds
from trdr.core import Duration, Timeframe, get_interval_seconds, parse_timeframe
from trdr.data import Bar
from trdr.strategy.sica_runner import get_primary_requirement
from trdr.strategy.types import DataRequirement

# Test defaults
_TEST_TF = Timeframe.parse("15m")
_TEST_LOOKBACK = Duration.parse("30d")


def make_bar(timestamp: str, close: float = 100.0) -> Bar:
    """Create test bar with given timestamp."""
    return Bar(
        timestamp=timestamp,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1000,
    )


class TestGetIntervalSeconds:
    """Tests for get_interval_seconds."""

    def test_minutes(self) -> None:
        """Parse minute timeframes."""
        assert get_interval_seconds("1m") == 60
        assert get_interval_seconds("15m") == 900
        assert get_interval_seconds("30m") == 1800

    def test_hours(self) -> None:
        """Parse hour timeframes."""
        assert get_interval_seconds("1h") == 3600
        assert get_interval_seconds("4h") == 14400

    def test_days(self) -> None:
        """Parse day timeframes."""
        assert get_interval_seconds("1d") == 86400

    def test_case_insensitive(self) -> None:
        """Timeframe parsing is case insensitive."""
        assert get_interval_seconds("1H") == 3600
        assert get_interval_seconds("15M") == 900

    def test_invalid_raises(self) -> None:
        """Invalid timeframe raises ValueError."""
        with pytest.raises(ValueError):
            get_interval_seconds("invalid")


class TestParseTimeframe:
    """Tests for parse_timeframe."""

    def test_minutes(self) -> None:
        """Parse minute timeframes to Alpaca format."""
        tf = parse_timeframe("15m")
        assert tf.amount == 15

    def test_hours(self) -> None:
        """Parse hour timeframes."""
        tf = parse_timeframe("4h")
        assert tf.amount == 4

    def test_days(self) -> None:
        """Parse day timeframe."""
        tf = parse_timeframe("1d")
        assert tf.amount == 1

    def test_invalid_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_timeframe("invalid")


class TestAlignFeeds:
    """Tests for align_feeds."""

    def test_empty_informative(self) -> None:
        """Empty informative returns list of None."""
        primary = [make_bar("2024-01-01T10:00:00Z"), make_bar("2024-01-01T10:15:00Z")]
        result = align_feeds(primary, [])
        assert result == [None, None]

    def test_empty_primary(self) -> None:
        """Empty primary returns empty list."""
        info = [make_bar("2024-01-01T10:00:00Z")]
        result = align_feeds([], info)
        assert result == []

    def test_exact_match(self) -> None:
        """Timestamps match exactly."""
        primary = [make_bar("2024-01-01T10:00:00Z"), make_bar("2024-01-01T11:00:00Z")]
        info = [make_bar("2024-01-01T10:00:00Z", 50.0), make_bar("2024-01-01T11:00:00Z", 60.0)]

        result = align_feeds(primary, info)

        assert len(result) == 2
        assert result[0].close == 50.0
        assert result[1].close == 60.0

    def test_forward_fill(self) -> None:
        """Higher TF bar forward-fills to multiple lower TF bars."""
        # 15m primary bars
        primary = [
            make_bar("2024-01-01T10:00:00Z"),
            make_bar("2024-01-01T10:15:00Z"),
            make_bar("2024-01-01T10:30:00Z"),
            make_bar("2024-01-01T10:45:00Z"),
            make_bar("2024-01-01T11:00:00Z"),
        ]
        # 1h informative bars
        info = [
            make_bar("2024-01-01T10:00:00Z", 100.0),
            make_bar("2024-01-01T11:00:00Z", 110.0),
        ]

        result = align_feeds(primary, info)

        # First 4 bars get 10:00 1h bar, last gets 11:00 1h bar
        assert len(result) == 5
        assert result[0].close == 100.0
        assert result[1].close == 100.0
        assert result[2].close == 100.0
        assert result[3].close == 100.0
        assert result[4].close == 110.0

    def test_no_lookahead(self) -> None:
        """Informative bar after primary bar returns None."""
        primary = [make_bar("2024-01-01T10:00:00Z")]
        info = [make_bar("2024-01-01T11:00:00Z", 50.0)]  # After primary

        result = align_feeds(primary, info)

        assert result == [None]

    def test_warmup_period(self) -> None:
        """Primary bars before any informative bar get None."""
        primary = [
            make_bar("2024-01-01T09:00:00Z"),  # Before any info
            make_bar("2024-01-01T09:15:00Z"),  # Before any info
            make_bar("2024-01-01T10:00:00Z"),  # At info timestamp
            make_bar("2024-01-01T10:15:00Z"),  # After info timestamp
        ]
        info = [make_bar("2024-01-01T10:00:00Z", 100.0)]

        result = align_feeds(primary, info)

        assert result[0] is None
        assert result[1] is None
        assert result[2].close == 100.0
        assert result[3].close == 100.0

    def test_multi_symbol_alignment(self) -> None:
        """Different symbols align by timestamp."""
        # ETH 15m bars
        eth_bars = [
            make_bar("2024-01-01T10:00:00Z"),
            make_bar("2024-01-01T10:15:00Z"),
        ]
        # BTC 1h bars (different symbol, aligned by time)
        btc_bars = [make_bar("2024-01-01T10:00:00Z", 42000.0)]

        result = align_feeds(eth_bars, btc_bars)

        assert len(result) == 2
        assert result[0].close == 42000.0
        assert result[1].close == 42000.0


class TestGetPrimaryRequirement:
    """Tests for get_primary_requirement."""

    def test_single_primary(self) -> None:
        """Single primary requirement is returned."""
        reqs = [
            DataRequirement("crypto:ETH/USD", "15m", 3000, role="primary"),
            DataRequirement("crypto:ETH/USD", "1h", 500),
        ]
        primary = get_primary_requirement(reqs)
        assert primary.timeframe == "15m"
        assert primary.role == "primary"

    def test_no_primary_raises(self) -> None:
        """No primary requirement raises ValueError."""
        reqs = [
            DataRequirement("crypto:ETH/USD", "15m", 3000),  # informative
            DataRequirement("crypto:ETH/USD", "1h", 500),  # informative
        ]
        with pytest.raises(ValueError, match="Expected exactly 1 primary"):
            get_primary_requirement(reqs)

    def test_multiple_primary_raises(self) -> None:
        """Multiple primary requirements raises ValueError."""
        reqs = [
            DataRequirement("crypto:ETH/USD", "15m", 3000, role="primary"),
            DataRequirement("crypto:ETH/USD", "1h", 500, role="primary"),
        ]
        with pytest.raises(ValueError, match="Expected exactly 1 primary"):
            get_primary_requirement(reqs)

    def test_empty_raises(self) -> None:
        """Empty requirements raises ValueError."""
        with pytest.raises(ValueError, match="Expected exactly 1 primary"):
            get_primary_requirement([])


class TestDataRequirement:
    """Tests for DataRequirement dataclass."""

    def test_key_format(self) -> None:
        """Key is symbol:timeframe."""
        req = DataRequirement("crypto:ETH/USD", "15m", 3000)
        assert req.key == "crypto:ETH/USD:15m"

    def test_default_role(self) -> None:
        """Default role is informative."""
        req = DataRequirement("crypto:ETH/USD", "15m", 3000)
        assert req.role == "informative"

    def test_invalid_role_raises(self) -> None:
        """Invalid role raises ValueError."""
        with pytest.raises(ValueError, match="role must be"):
            DataRequirement("crypto:ETH/USD", "15m", 3000, role="invalid")


class TestMultiFeedIntegration:
    """Integration tests for multi-feed strategy execution."""

    def test_three_feed_strategy(self) -> None:
        """Strategy with 3 feeds (primary + 2 informative) executes correctly."""
        from trdr.backtest.paper_exchange import PaperExchange, PaperExchangeConfig
        from trdr.strategy.base_strategy import BaseStrategy, StrategyConfig
        from trdr.strategy.types import Position, Signal, SignalAction

        @dataclass
        class MTFConfig(StrategyConfig):
            """Multi-TF config with defaults."""

            symbol: str = "crypto:ETH/USD"
            timeframe: Timeframe = field(default_factory=lambda: _TEST_TF)
            lookback: Duration = field(default_factory=lambda: _TEST_LOOKBACK)

        class MTFStrategy(BaseStrategy):
            """Test strategy using 3 feeds."""

            def __init__(self, config: MTFConfig):
                super().__init__(config)
                self.htf_values_seen: list[float | None] = []
                self.btc_values_seen: list[float | None] = []

            def get_data_requirements(self) -> list[DataRequirement]:
                return [
                    DataRequirement("crypto:ETH/USD", Timeframe.parse("15m"), Duration.parse("7d"), role="primary"),
                    DataRequirement("crypto:ETH/USD", Timeframe.parse("1h"), Duration.parse("2d")),  # HTF same symbol
                    DataRequirement("crypto:BTC/USD", Timeframe.parse("1h"), Duration.parse("2d")),  # Cross-symbol
                ]

            def generate_signal(
                self,
                bars: dict[str, list[Bar]],
                position: Position | None,
            ) -> Signal:
                primary = bars["crypto:ETH/USD:15m"]
                htf = bars.get("crypto:ETH/USD:1h", [])
                btc = bars.get("crypto:BTC/USD:1h", [])

                # Record what we see for verification
                self.htf_values_seen.append(htf[-1].close if htf and htf[-1] else None)
                self.btc_values_seen.append(btc[-1].close if btc and btc[-1] else None)

                return Signal(
                    action=SignalAction.HOLD,
                    price=primary[-1].close,
                    confidence=0.0,
                    reason="test",
                )

        # Create test data
        # 15m ETH bars: 10:00 through 11:15 (6 bars, so 5 signals generated)
        eth_15m = [
            make_bar("2024-01-01T10:00:00Z", 100.0),
            make_bar("2024-01-01T10:15:00Z", 101.0),
            make_bar("2024-01-01T10:30:00Z", 102.0),
            make_bar("2024-01-01T10:45:00Z", 103.0),
            make_bar("2024-01-01T11:00:00Z", 104.0),
            make_bar("2024-01-01T11:15:00Z", 105.0),  # Extra bar so 11:00 gets signal
        ]
        # 1h ETH bars
        eth_1h = [
            make_bar("2024-01-01T10:00:00Z", 200.0),
            make_bar("2024-01-01T11:00:00Z", 210.0),
        ]
        # 1h BTC bars
        btc_1h = [
            make_bar("2024-01-01T10:00:00Z", 42000.0),
            make_bar("2024-01-01T11:00:00Z", 42500.0),
        ]

        # Align informative feeds to primary
        eth_1h_aligned = align_feeds(eth_15m, eth_1h)
        btc_1h_aligned = align_feeds(eth_15m, btc_1h)

        bars = {
            "crypto:ETH/USD:15m": eth_15m,
            "crypto:ETH/USD:1h": eth_1h_aligned,
            "crypto:BTC/USD:1h": btc_1h_aligned,
        }

        config = MTFConfig(symbol="crypto:ETH/USD", timeframe="15m")
        strategy = MTFStrategy(config)

        exchange_config = PaperExchangeConfig(
            symbol="crypto:ETH/USD",
            primary_feed="crypto:ETH/USD:15m",
            warmup_bars=0,  # No warmup for test
        )
        exchange = PaperExchange(exchange_config, strategy)
        result = exchange.run(bars)

        # Verify execution completed
        assert result is not None

        # Verify strategy saw the right HTF values (forward-filled)
        # First 4 bars see 10:00 HTF value (200.0), last sees 11:00 (210.0)
        assert strategy.htf_values_seen[0] == 200.0
        assert strategy.htf_values_seen[-1] == 210.0

        # Verify BTC values were available
        assert strategy.btc_values_seen[0] == 42000.0
        assert strategy.btc_values_seen[-1] == 42500.0
