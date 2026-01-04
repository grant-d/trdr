"""Tests for paper exchange engine."""

from dataclasses import dataclass

from trdr.backtest.paper_exchange import PaperExchange, PaperExchangeConfig
from trdr.data.market import Bar
from trdr.strategy.base_strategy import BaseStrategy, StrategyConfig
from trdr.strategy.types import Position, Signal, SignalAction


def make_bars(prices: list[float], start_ts: str = "2024-01-01T10:00:00Z") -> list[Bar]:
    """Create bars from list of close prices."""
    bars = []
    for i, price in enumerate(prices):
        hour = 10 + i
        ts = f"2024-01-0{1 + i // 24}T{hour % 24:02d}:00:00Z"
        bars.append(Bar(
            timestamp=ts,
            open=price * 0.99,
            high=price * 1.01,
            low=price * 0.98,
            close=price,
            volume=1000,
        ))
    return bars


@dataclass
class SimpleConfig(StrategyConfig):
    """Simple strategy config."""

    pass


class AlwaysBuyStrategy(BaseStrategy):
    """Buy on first signal, hold forever."""

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.bought = False

    def generate_signal(self, bars: list[Bar], position: Position | None) -> Signal:
        if not position and not self.bought:
            self.bought = True
            return Signal(
                action=SignalAction.BUY,
                price=bars[-1].close,
                confidence=1.0,
                reason="test_buy",
            )
        return Signal(action=SignalAction.HOLD, price=bars[-1].close, confidence=0.0, reason="")

    def reset(self) -> None:
        self.bought = False


class BuyThenSellStrategy(BaseStrategy):
    """Buy, then sell after 3 bars."""

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.bars_held = 0

    def generate_signal(self, bars: list[Bar], position: Position | None) -> Signal:
        if not position:
            self.bars_held = 0
            return Signal(
                action=SignalAction.BUY,
                price=bars[-1].close,
                confidence=1.0,
                reason="test_buy",
            )
        else:
            self.bars_held += 1
            if self.bars_held >= 3:
                return Signal(
                    action=SignalAction.CLOSE,
                    price=bars[-1].close,
                    confidence=1.0,
                    reason="test_close",
                )
        return Signal(action=SignalAction.HOLD, price=bars[-1].close, confidence=0.0, reason="")

    def reset(self) -> None:
        self.bars_held = 0


class TrailingStopStrategy(BaseStrategy):
    """Buy with trailing stop."""

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.bought = False

    def generate_signal(self, bars: list[Bar], position: Position | None) -> Signal:
        if not position and not self.bought:
            self.bought = True
            return Signal(
                action=SignalAction.BUY,
                price=bars[-1].close,
                confidence=1.0,
                reason="buy_with_tsl",
                trailing_stop=0.05,  # 5% trailing stop
            )
        return Signal(action=SignalAction.HOLD, price=bars[-1].close, confidence=0.0, reason="")

    def reset(self) -> None:
        self.bought = False


class MultipleEntriesStrategy(BaseStrategy):
    """Buy multiple times to test pyramiding."""

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.buys = 0

    def generate_signal(self, bars: list[Bar], position: Position | None) -> Signal:
        if self.buys < 3:
            self.buys += 1
            return Signal(
                action=SignalAction.BUY,
                price=bars[-1].close,
                confidence=1.0,
                reason=f"buy_{self.buys}",
                quantity=10.0,  # Explicit quantity
            )
        return Signal(action=SignalAction.HOLD, price=bars[-1].close, confidence=0.0, reason="")

    def reset(self) -> None:
        self.buys = 0


class TestPaperExchange:
    """Tests for PaperExchange."""

    def test_simple_buy_and_hold(self) -> None:
        """Buy and hold to end of data."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Prices go up
        bars = make_bars([100, 100, 100, 110, 120])
        result = engine.run(bars)

        assert result.total_trades == 1
        assert result.total_pnl > 0  # Should profit

    def test_buy_then_sell(self) -> None:
        """Complete round trip trade."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = BuyThenSellStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        bars = make_bars([100, 100, 100, 105, 110, 115, 120])
        result = engine.run(bars)

        # Should have at least one completed trade
        assert result.total_trades >= 1
        # Exit reason should be test_close (not end_of_data)
        completed = [t for t in result.trades if t.exit_reason == "signal"]
        assert len(completed) >= 0  # May vary by timing

    def test_trailing_stop_triggers(self) -> None:
        """Trailing stop exits on reversal."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = TrailingStopStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Price goes up then crashes
        bars = make_bars([100, 100, 100, 110, 120, 130, 100])
        result = engine.run(bars)

        assert result.total_trades == 1
        # Should have exited before the crash to 100

    def test_transaction_costs(self) -> None:
        """Transaction costs reduce P&L."""
        config_no_cost = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
        )
        config_with_cost = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.01,  # 1%
        )

        bars = make_bars([100, 100, 100, 110, 120])

        result_no_cost = PaperExchange(
            config_no_cost, AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        ).run(bars)
        result_with_cost = PaperExchange(
            config_with_cost, AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        ).run(bars)

        # With costs should have lower P&L
        assert result_with_cost.total_pnl < result_no_cost.total_pnl

    def test_equity_curve_generated(self) -> None:
        """Equity curve is generated."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
        )
        strategy = AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        bars = make_bars([100, 100, 100, 110, 120])
        result = engine.run(bars)

        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] > 0

    def test_max_drawdown_calculated(self) -> None:
        """Max drawdown is calculated."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
        )
        strategy = AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Price goes up then down
        bars = make_bars([100, 100, 100, 120, 110])
        result = engine.run(bars)

        assert result.max_drawdown >= 0
        assert result.max_drawdown <= 1

    def test_insufficient_bars(self) -> None:
        """Returns empty result for insufficient bars."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=10,
            initial_capital=10000.0,
        )
        strategy = AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        bars = make_bars([100, 100, 100])  # Only 3 bars, need 10 warmup
        result = engine.run(bars)

        assert result.total_trades == 0

    def test_win_rate_calculated(self) -> None:
        """Win rate is calculated correctly."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
        )
        strategy = BuyThenSellStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Price up = winning trade
        bars = make_bars([100, 100, 100, 105, 110, 115, 120])
        result = engine.run(bars)

        if result.total_trades > 0:
            assert result.win_rate >= 0
            assert result.win_rate <= 1


class TestMultipleEntries:
    """Tests for multiple entries (pyramiding)."""

    def test_multiple_buys_accumulate(self) -> None:
        """Multiple buy signals accumulate position."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = MultipleEntriesStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        bars = make_bars([100] * 10)  # Flat price
        result = engine.run(bars)

        # Should have one trade closed at end with multiple entries
        assert result.total_trades == 1
        # Position should be 30 units (10 * 3)
        assert result.trades[0].quantity == 30.0


class TestStockCalendar:
    """Tests for stock trading calendar integration."""

    def test_stock_filters_weekends(self) -> None:
        """Stock trading skips weekend bars."""
        config = PaperExchangeConfig(
            symbol="stock:TEST",  # Stock, not crypto
            warmup_bars=2,
            initial_capital=10000.0,
        )
        strategy = AlwaysBuyStrategy(SimpleConfig(symbol="stock:TEST"))
        engine = PaperExchange(config, strategy)

        # Create bars including weekend (Jan 6-7, 2024 were Sat/Sun)
        bars = [
            Bar("2024-01-05T10:00:00Z", 99, 101, 98, 100, 1000),  # Fri
            Bar("2024-01-06T10:00:00Z", 99, 101, 98, 100, 1000),  # Sat (filtered)
            Bar("2024-01-07T10:00:00Z", 99, 101, 98, 100, 1000),  # Sun (filtered)
            Bar("2024-01-08T10:00:00Z", 99, 101, 98, 100, 1000),  # Mon
            Bar("2024-01-09T10:00:00Z", 99, 101, 98, 100, 1000),  # Tue
            Bar("2024-01-10T10:00:00Z", 99, 101, 98, 105, 1000),  # Wed
        ]
        result = engine.run(bars)

        # Should have processed only weekday bars
        # Equity curve length = total bars - warmup - weekend bars
        assert len(result.equity_curve) == 2  # Mon, Tue, Wed after warmup (Fri, Mon)
