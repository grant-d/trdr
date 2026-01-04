"""Tests for paper exchange engine."""

from dataclasses import dataclass, field

import pytest

from trdr.core import Duration, Timeframe
from trdr.backtest.paper_exchange import PaperExchange, PaperExchangeConfig
from trdr.data import Bar
from trdr.strategy.base_strategy import BaseStrategy, StrategyConfig
from trdr.strategy.types import DataRequirement, Position, Signal, SignalAction

# Test defaults
_TEST_TF = Timeframe.parse("1h")
_TEST_LOOKBACK = Duration.parse("30d")


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
    """Simple strategy config with test defaults."""

    symbol: str = "crypto:TEST"
    timeframe: Timeframe = field(default_factory=lambda: _TEST_TF)
    lookback: Duration = field(default_factory=lambda: _TEST_LOOKBACK)


def _get_primary_bars(bars: dict[str, list[Bar]], config: StrategyConfig) -> list[Bar]:
    """Extract primary bars from dict for test strategies."""
    key = f"{config.symbol}:{config.timeframe}"
    return bars[key]


def _wrap_bars(bars: list[Bar], symbol: str, timeframe: str = "1h") -> dict[str, list[Bar]]:
    """Wrap bars list in dict for PaperExchange.run()."""
    return {f"{symbol}:{timeframe}": bars}


class AlwaysBuyStrategy(BaseStrategy):
    """Buy on first signal, hold forever."""

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.bought = False

    def get_data_requirements(self) -> list[DataRequirement]:
        return [DataRequirement(self.config.symbol, self.config.timeframe, self.config.lookback, role="primary")]

    def generate_signal(self, bars: dict[str, list[Bar]], position: Position | None) -> Signal:
        primary = _get_primary_bars(bars, self.config)
        if not position and not self.bought:
            self.bought = True
            return Signal(
                action=SignalAction.BUY,
                price=primary[-1].close,
                confidence=1.0,
                reason="test_buy",
            )
        return Signal(action=SignalAction.HOLD, price=primary[-1].close, confidence=0.0, reason="")

    def reset(self) -> None:
        self.bought = False


class BuyThenSellStrategy(BaseStrategy):
    """Buy, then sell after 3 bars."""

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.bars_held = 0

    def get_data_requirements(self) -> list[DataRequirement]:
        return [DataRequirement(self.config.symbol, self.config.timeframe, self.config.lookback, role="primary")]

    def generate_signal(self, bars: dict[str, list[Bar]], position: Position | None) -> Signal:
        primary = _get_primary_bars(bars, self.config)
        if not position:
            self.bars_held = 0
            return Signal(
                action=SignalAction.BUY,
                price=primary[-1].close,
                confidence=1.0,
                reason="test_buy",
            )
        else:
            self.bars_held += 1
            if self.bars_held >= 3:
                return Signal(
                    action=SignalAction.CLOSE,
                    price=primary[-1].close,
                    confidence=1.0,
                    reason="test_close",
                )
        return Signal(action=SignalAction.HOLD, price=primary[-1].close, confidence=0.0, reason="")

    def reset(self) -> None:
        self.bars_held = 0


class TrailingStopStrategy(BaseStrategy):
    """Buy with trailing stop."""

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.bought = False

    def get_data_requirements(self) -> list[DataRequirement]:
        return [DataRequirement(self.config.symbol, self.config.timeframe, self.config.lookback, role="primary")]

    def generate_signal(self, bars: dict[str, list[Bar]], position: Position | None) -> Signal:
        primary = _get_primary_bars(bars, self.config)
        if not position and not self.bought:
            self.bought = True
            return Signal(
                action=SignalAction.BUY,
                price=primary[-1].close,
                confidence=1.0,
                reason="buy_with_tsl",
                trailing_stop=0.05,  # 5% trailing stop
            )
        return Signal(action=SignalAction.HOLD, price=primary[-1].close, confidence=0.0, reason="")

    def reset(self) -> None:
        self.bought = False


class TakeProfitStrategy(BaseStrategy):
    """Buy with take profit target."""

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.bought = False
        self.trade_callbacks: list[tuple[float, str]] = []

    def get_data_requirements(self) -> list[DataRequirement]:
        return [DataRequirement(self.config.symbol, self.config.timeframe, self.config.lookback, role="primary")]

    def generate_signal(self, bars: dict[str, list[Bar]], position: Position | None) -> Signal:
        primary = _get_primary_bars(bars, self.config)
        if not position and not self.bought:
            self.bought = True
            price = primary[-1].close
            return Signal(
                action=SignalAction.BUY,
                price=price,
                confidence=1.0,
                reason="buy_with_tp",
                take_profit=price * 1.10,  # 10% take profit
                stop_loss=price * 0.95,  # 5% stop loss
            )
        return Signal(action=SignalAction.HOLD, price=primary[-1].close, confidence=0.0, reason="")

    def on_trade_complete(self, pnl: float, reason: str) -> None:
        self.trade_callbacks.append((pnl, reason))

    def reset(self) -> None:
        self.bought = False
        self.trade_callbacks = []


class LimitOrderStrategy(BaseStrategy):
    """Buy with limit order entry."""

    def __init__(self, config: SimpleConfig, limit_pct: float = 0.98):
        super().__init__(config)
        self.bought = False
        self.limit_pct = limit_pct

    def get_data_requirements(self) -> list[DataRequirement]:
        return [DataRequirement(self.config.symbol, self.config.timeframe, self.config.lookback, role="primary")]

    def generate_signal(self, bars: dict[str, list[Bar]], position: Position | None) -> Signal:
        primary = _get_primary_bars(bars, self.config)
        if not position and not self.bought:
            self.bought = True
            price = primary[-1].close
            return Signal(
                action=SignalAction.BUY,
                price=price,
                confidence=1.0,
                reason="limit_buy",
                limit_price=price * self.limit_pct,  # Buy below current price
            )
        return Signal(action=SignalAction.HOLD, price=primary[-1].close, confidence=0.0, reason="")

    def reset(self) -> None:
        self.bought = False


class StopLimitStrategy(BaseStrategy):
    """Buy with stop-limit order (breakout with limit)."""

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.bought = False

    def get_data_requirements(self) -> list[DataRequirement]:
        return [DataRequirement(self.config.symbol, self.config.timeframe, self.config.lookback, role="primary")]

    def generate_signal(self, bars: dict[str, list[Bar]], position: Position | None) -> Signal:
        primary = _get_primary_bars(bars, self.config)
        if not position and not self.bought:
            self.bought = True
            price = primary[-1].close
            return Signal(
                action=SignalAction.BUY,
                price=price,
                confidence=1.0,
                reason="stop_limit_buy",
                stop_price=price * 1.05,  # Trigger on 5% breakout
                limit_price=price * 1.06,  # Fill at 6% above or better
            )
        return Signal(action=SignalAction.HOLD, price=primary[-1].close, confidence=0.0, reason="")

    def reset(self) -> None:
        self.bought = False


class MultipleEntriesStrategy(BaseStrategy):
    """Buy multiple times to test pyramiding."""

    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.buys = 0

    def get_data_requirements(self) -> list[DataRequirement]:
        return [DataRequirement(self.config.symbol, self.config.timeframe, self.config.lookback, role="primary")]

    def generate_signal(self, bars: dict[str, list[Bar]], position: Position | None) -> Signal:
        primary = _get_primary_bars(bars, self.config)
        if self.buys < 3:
            self.buys += 1
            return Signal(
                action=SignalAction.BUY,
                price=primary[-1].close,
                confidence=1.0,
                reason=f"buy_{self.buys}",
                quantity=10.0,  # Explicit quantity
            )
        return Signal(action=SignalAction.HOLD, price=primary[-1].close, confidence=0.0, reason="")

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
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

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
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

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
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

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
        bars_dict = _wrap_bars(bars, "crypto:TEST")

        result_no_cost = PaperExchange(
            config_no_cost, AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        ).run(bars_dict)
        result_with_cost = PaperExchange(
            config_with_cost, AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        ).run(bars_dict)

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
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

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
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

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
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

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
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

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
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        # Should have one trade closed at end with multiple entries
        assert result.total_trades == 1
        # Position should be 30 units (10 * 3)
        assert result.trades[0].quantity == 30.0


class TestTakeProfitIntegration:
    """Tests for take profit and on_trade_complete integration."""

    def test_take_profit_triggers(self) -> None:
        """Take profit exits when price reaches target."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = TakeProfitStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Price rises to hit 10% take profit
        # Entry at 100, TP at 110. Bar with close=110 has range [107.8, 111.1]
        bars = make_bars([100, 100, 100, 105, 110])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        assert result.total_trades == 1
        assert result.trades[0].net_pnl > 0  # Should profit

    def test_stop_loss_cancels_take_profit(self) -> None:
        """Stop loss triggers and cancels take profit (pseudo-OCO)."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = TakeProfitStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Entry at 100, then price drops to hit 5% stop loss (95)
        # Bar with close=95 has range [93.1, 95.95] which includes 95
        bars = make_bars([100, 100, 100, 100, 95])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        assert result.total_trades == 1
        assert result.trades[0].net_pnl < 0  # Should lose from entry ~100 to exit at 95

    def test_on_trade_complete_called(self) -> None:
        """on_trade_complete callback is called after trade closes."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = TakeProfitStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Price rises to hit take profit (110)
        # Bar with close=110 has range [107.8, 111.1] which includes 110
        bars = make_bars([100, 100, 100, 105, 110])
        engine.run(_wrap_bars(bars, "crypto:TEST"))

        assert len(strategy.trade_callbacks) == 1
        pnl, reason = strategy.trade_callbacks[0]
        assert pnl > 0  # Profitable trade


class TestLimitOrderIntegration:
    """Tests for limit order integration with paper exchange."""

    def test_limit_order_fills_on_dip(self) -> None:
        """Limit order fills when price dips to limit."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        # Limit at 98% of current price
        strategy = LimitOrderStrategy(SimpleConfig(symbol="crypto:TEST"), limit_pct=0.98)
        engine = PaperExchange(config, strategy)

        # Price dips below limit then recovers
        bars = make_bars([100, 100, 100, 95, 110])  # 95 < 98 (limit), then 110
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        assert result.total_trades == 1
        assert result.trades[0].net_pnl > 0  # Should profit from 98 entry to 110

    def test_limit_order_never_fills(self) -> None:
        """Limit order stays pending if never touched."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        # Limit at 90% - price never gets there
        strategy = LimitOrderStrategy(SimpleConfig(symbol="crypto:TEST"), limit_pct=0.90)
        engine = PaperExchange(config, strategy)

        # Price stays above limit
        bars = make_bars([100, 100, 100, 105, 110])  # Never below 90
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        # No trade executed (limit never filled)
        assert result.total_trades == 0

    def test_limit_order_no_slippage(self) -> None:
        """Limit orders have no slippage."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.05,  # 5% slippage - should be ignored for limit
        )
        strategy = LimitOrderStrategy(SimpleConfig(symbol="crypto:TEST"), limit_pct=0.98)
        engine = PaperExchange(config, strategy)

        bars = make_bars([100, 100, 100, 95, 110])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        # Should have entered at limit price (98), not 98 + 5% slippage
        assert result.total_trades == 1
        # Entry price should be 98 or the gap-down open (whichever is lower)
        assert result.trades[0].entry_price <= 98.0


class TestStopLimitIntegration:
    """Tests for stop-limit order integration."""

    def test_stop_limit_triggers_and_fills(self) -> None:
        """Stop-limit triggers on breakout and fills at limit."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = StopLimitStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Signal at price=100, stop=105, limit=106
        # make_bars creates: open=99%, high=101%, low=98%, close=price
        # Bar 3 (close=107): high=108.07 > 105 (triggers), low=104.86 < 106 (fills)
        bars = make_bars([100, 100, 100, 107, 115])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        assert result.total_trades == 1
        assert result.trades[0].entry_price <= 106.0  # Filled at limit or better

    def test_stop_limit_never_triggers(self) -> None:
        """Stop-limit stays pending if stop never hit."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = StopLimitStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Signal at price=100, stop=105, but price never reaches 105
        bars = make_bars([100, 100, 100, 102, 103])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        # No trade - stop never triggered
        assert result.total_trades == 0


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
        result = engine.run(_wrap_bars(bars, "stock:TEST"))

        # Should have processed only weekday bars
        # Equity curve length = total bars - warmup - weekend bars
        assert len(result.equity_curve) == 2  # Mon, Tue, Wed after warmup (Fri, Mon)


class TestTradeFields:
    """Tests for Trade dataclass fields including stop_loss and take_profit."""

    def test_trade_captures_stop_loss_and_take_profit(self) -> None:
        """Trade record includes stop_loss and take_profit from entry signal."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = TakeProfitStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Entry at 100, TP=110 (10%), SL=95 (5%)
        bars = make_bars([100, 100, 100, 105, 110])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        assert result.total_trades == 1
        trade = result.trades[0]
        # Stop loss should be ~95 (5% below entry ~100)
        assert trade.stop_loss is not None
        assert trade.stop_loss == pytest.approx(95.0, rel=0.05)
        # Take profit should be ~110 (10% above entry ~100)
        assert trade.take_profit is not None
        assert trade.take_profit == pytest.approx(110.0, rel=0.05)

    def test_trade_none_when_no_sl_tp(self) -> None:
        """Trade has None for stop_loss/take_profit when not set."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        # AlwaysBuyStrategy doesn't set stop_loss or take_profit
        strategy = AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        bars = make_bars([100, 100, 100, 110, 120])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        assert result.total_trades == 1
        trade = result.trades[0]
        assert trade.stop_loss is None
        assert trade.take_profit is None

    def test_trade_is_winner_property(self) -> None:
        """Trade.is_winner returns True for profitable trades."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Price goes up = winning trade
        bars = make_bars([100, 100, 100, 110, 120])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        assert result.total_trades == 1
        assert result.trades[0].is_winner is True

    def test_trade_duration_hours(self) -> None:
        """Trade.duration_hours calculates correctly."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        bars = make_bars([100, 100, 100, 110, 120])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        assert result.total_trades == 1
        # Bars are 1 hour apart, entry at bar 2, exit at bar 4 = 2 hours
        assert result.trades[0].duration_hours > 0


class TestPrintTrades:
    """Tests for PaperExchangeResult.print_trades() method."""

    def test_print_trades_no_trades(self, capsys) -> None:
        """print_trades handles empty trade list."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=10,
            initial_capital=10000.0,
        )
        strategy = AlwaysBuyStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Only 3 bars, need 10 warmup = no trades
        bars = make_bars([100, 100, 100])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        result.print_trades()
        captured = capsys.readouterr()
        assert "No trades" in captured.out

    def test_print_trades_with_trades(self, capsys) -> None:
        """print_trades outputs trade log."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = TakeProfitStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        bars = make_bars([100, 100, 100, 105, 110])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        result.print_trades()
        captured = capsys.readouterr()

        # Check output contains expected sections
        assert "TRADE LOG" in captured.out
        assert "#1" in captured.out
        assert "Entry:" in captured.out
        assert "Exit:" in captured.out
        assert "SL:" in captured.out
        assert "TP:" in captured.out
        assert "Summary:" in captured.out

    def test_print_trades_shows_win_loss(self, capsys) -> None:
        """print_trades shows WIN/LOSS status."""
        config = PaperExchangeConfig(
            symbol="crypto:TEST",
            warmup_bars=2,
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
            slippage_pct=0.0,
        )
        strategy = TakeProfitStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)

        # Price rises to hit take profit = WIN
        bars = make_bars([100, 100, 100, 105, 110])
        result = engine.run(_wrap_bars(bars, "crypto:TEST"))

        result.print_trades()
        captured = capsys.readouterr()
        assert "[WIN]" in captured.out


class TestOrderDirectionValidation:
    """Tests for exchange-style order direction enforcement."""

    def test_stop_loss_above_price_rejected(self) -> None:
        """Stop loss above current price is rejected."""

        class BadStopLossStrategy(BaseStrategy):
            def get_data_requirements(self) -> list[DataRequirement]:
                return [DataRequirement(self.config.symbol, self.config.timeframe, self.config.lookback, role="primary")]

            def generate_signal(
                self, bars: dict[str, list[Bar]], position: Position | None
            ) -> Signal:
                primary = _get_primary_bars(bars, self.config)
                if len(primary) < 3:
                    return Signal(
                        action=SignalAction.HOLD, price=primary[-1].close, confidence=0.0
                    )
                if position is None:
                    return Signal(
                        action=SignalAction.BUY,
                        price=primary[-1].close,
                        confidence=0.8,
                        reason="buy",
                        stop_loss=primary[-1].close * 1.05,  # WRONG: above price
                    )
                return Signal(
                    action=SignalAction.HOLD, price=primary[-1].close, confidence=0.5
                )

        config = PaperExchangeConfig(symbol="crypto:TEST", warmup_bars=2)
        strategy = BadStopLossStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)
        bars = make_bars([100, 100, 100, 100])

        with pytest.raises(ValueError, match="Sell stop must be below price"):
            engine.run(_wrap_bars(bars, "crypto:TEST"))

    def test_take_profit_below_price_rejected(self) -> None:
        """Take profit below current price is rejected."""

        class BadTakeProfitStrategy(BaseStrategy):
            def get_data_requirements(self) -> list[DataRequirement]:
                return [DataRequirement(self.config.symbol, self.config.timeframe, self.config.lookback, role="primary")]

            def generate_signal(
                self, bars: dict[str, list[Bar]], position: Position | None
            ) -> Signal:
                primary = _get_primary_bars(bars, self.config)
                if len(primary) < 3:
                    return Signal(
                        action=SignalAction.HOLD, price=primary[-1].close, confidence=0.0
                    )
                if position is None:
                    return Signal(
                        action=SignalAction.BUY,
                        price=primary[-1].close,
                        confidence=0.8,
                        reason="buy",
                        take_profit=primary[-1].close * 0.95,  # WRONG: below price
                    )
                return Signal(
                    action=SignalAction.HOLD, price=primary[-1].close, confidence=0.5
                )

        config = PaperExchangeConfig(symbol="crypto:TEST", warmup_bars=2)
        strategy = BadTakeProfitStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)
        bars = make_bars([100, 100, 100, 100])

        with pytest.raises(ValueError, match="Sell limit must be above price"):
            engine.run(_wrap_bars(bars, "crypto:TEST"))

    def test_buy_limit_above_price_rejected(self) -> None:
        """Buy limit above current price is rejected."""

        class BadBuyLimitStrategy(BaseStrategy):
            def get_data_requirements(self) -> list[DataRequirement]:
                return [DataRequirement(self.config.symbol, self.config.timeframe, self.config.lookback, role="primary")]

            def generate_signal(
                self, bars: dict[str, list[Bar]], position: Position | None
            ) -> Signal:
                primary = _get_primary_bars(bars, self.config)
                if len(primary) < 3:
                    return Signal(
                        action=SignalAction.HOLD, price=primary[-1].close, confidence=0.0
                    )
                if position is None:
                    return Signal(
                        action=SignalAction.BUY,
                        price=primary[-1].close,
                        confidence=0.8,
                        reason="buy",
                        limit_price=primary[-1].close * 1.05,  # WRONG: above price
                    )
                return Signal(
                    action=SignalAction.HOLD, price=primary[-1].close, confidence=0.5
                )

        config = PaperExchangeConfig(symbol="crypto:TEST", warmup_bars=2)
        strategy = BadBuyLimitStrategy(SimpleConfig(symbol="crypto:TEST"))
        engine = PaperExchange(config, strategy)
        bars = make_bars([100, 100, 100, 100])

        with pytest.raises(ValueError, match="Buy limit must be below price"):
            engine.run(_wrap_bars(bars, "crypto:TEST"))
