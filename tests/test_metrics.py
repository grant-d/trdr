"""Tests for TradeMetrics and RuntimeContext."""

from dataclasses import dataclass, field

from trdr.core import Duration, Symbol, Timeframe
from trdr.backtest.metrics import TradeMetrics
from trdr.backtest.orders import OrderManager
from trdr.backtest.paper_exchange import (
    PaperExchange,
    PaperExchangeConfig,
    RuntimeContext,
    Trade,
)
from trdr.backtest.portfolio import Portfolio
from trdr.data import Bar
from trdr.strategy.base_strategy import BaseStrategy, StrategyConfig
from trdr.strategy.types import DataRequirement, Position, Signal, SignalAction

# Test defaults
_TEST_SYMBOL = Symbol.parse("crypto:TEST")
_TEST_TF = Timeframe.parse("1h")
_TEST_LOOKBACK = Duration.parse("30d")


def make_trade(
    entry_price: float = 100.0,
    exit_price: float = 110.0,
    quantity: float = 1.0,
    costs: float = 0.5,
) -> Trade:
    """Create test trade."""
    gross = (exit_price - entry_price) * quantity
    return Trade(
        entry_time="2024-01-01T10:00:00Z",
        exit_time="2024-01-01T14:00:00Z",
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        side="long",
        gross_pnl=gross,
        costs=costs,
        net_pnl=gross - costs,
        entry_reason="test",
        exit_reason="test",
    )


def make_bar(price: float = 100.0, ts: str = "2024-01-01T10:00:00Z") -> Bar:
    """Create test bar."""
    return Bar(
        timestamp=ts,
        open=price * 0.99,
        high=price * 1.01,
        low=price * 0.98,
        close=price,
        volume=1000,
    )


class TestTradeMetricsBasic:
    """Test TradeMetrics computes correct values."""

    def test_total_trades(self) -> None:
        trades = [make_trade() for _ in range(5)]
        m = TradeMetrics(trades, [], 10000.0, "crypto")
        assert m.total_trades == 5

    def test_win_rate(self) -> None:
        trades = [
            make_trade(entry_price=100, exit_price=110),  # win
            make_trade(entry_price=100, exit_price=110),  # win
            make_trade(entry_price=100, exit_price=90),  # loss
        ]
        m = TradeMetrics(trades, [], 10000.0, "crypto")
        assert m.win_rate == 2 / 3

    def test_total_pnl(self) -> None:
        trades = [
            make_trade(entry_price=100, exit_price=110, costs=0.5),  # 9.5
            make_trade(entry_price=100, exit_price=90, costs=0.5),  # -10.5
        ]
        m = TradeMetrics(trades, [], 10000.0, "crypto")
        assert m.total_pnl == -1.0  # 9.5 + (-10.5)

    def test_profit_factor(self) -> None:
        trades = [
            make_trade(entry_price=100, exit_price=120, costs=0),  # 20
            make_trade(entry_price=100, exit_price=90, costs=0),  # -10
        ]
        m = TradeMetrics(trades, [], 10000.0, "crypto")
        assert m.profit_factor == 2.0  # 20 / 10

    def test_max_drawdown(self) -> None:
        curve = [10000, 11000, 10500, 9000, 9500]  # peak 11000, trough 9000
        m = TradeMetrics([], curve, 10000.0, "crypto")
        expected = (11000 - 9000) / 11000
        assert abs(m.max_drawdown - expected) < 0.001

    def test_expectancy(self) -> None:
        trades = [
            make_trade(entry_price=100, exit_price=120, costs=0),  # 20
            make_trade(entry_price=100, exit_price=120, costs=0),  # 20
            make_trade(entry_price=100, exit_price=90, costs=0),  # -10
        ]
        m = TradeMetrics(trades, [], 10000.0, "crypto")
        # WR=2/3, avg_win=20, avg_loss=-10
        expected = (2 / 3 * 20) + (1 / 3 * -10)
        assert abs(m.expectancy - expected) < 0.001


class TestTradeMetricsEmpty:
    """Test TradeMetrics handles empty trades list."""

    def test_empty_trades_total(self) -> None:
        m = TradeMetrics([], [], 10000.0, "crypto")
        assert m.total_trades == 0

    def test_empty_trades_win_rate(self) -> None:
        m = TradeMetrics([], [], 10000.0, "crypto")
        assert m.win_rate == 0.0

    def test_empty_trades_profit_factor(self) -> None:
        m = TradeMetrics([], [], 10000.0, "crypto")
        assert m.profit_factor == 0.0

    def test_empty_equity_curve(self) -> None:
        m = TradeMetrics([], [], 10000.0, "crypto")
        assert m.max_drawdown == 0.0
        assert m.total_return == 0.0

    def test_sharpe_requires_trades(self) -> None:
        m = TradeMetrics([], [], 10000.0, "crypto")
        assert m.sharpe_ratio is None


class TestCurrentDrawdown:
    """Test current_drawdown calculation."""

    def test_at_new_high(self) -> None:
        curve = [10000, 10500, 11000]
        m = TradeMetrics([], curve, 10000.0, "crypto")
        # At new high, drawdown should be 0
        assert m.current_drawdown(12000) == 0.0

    def test_at_peak(self) -> None:
        curve = [10000, 11000, 10500]
        m = TradeMetrics([], curve, 10000.0, "crypto")
        # At peak, drawdown should be 0
        assert m.current_drawdown(11000) == 0.0

    def test_below_peak(self) -> None:
        curve = [10000, 11000, 10500]
        m = TradeMetrics([], curve, 10000.0, "crypto")
        # 10% below peak
        expected = (11000 - 9900) / 11000
        assert abs(m.current_drawdown(9900) - expected) < 0.001

    def test_clamps_to_one(self) -> None:
        curve = [10000, 11000]
        m = TradeMetrics([], curve, 10000.0, "crypto")
        # Even at 0 equity, drawdown capped at 1.0
        assert m.current_drawdown(0) == 1.0


class TestRuntimeContextRunParams:
    """Test RuntimeContext run parameters."""

    def test_symbol(self) -> None:
        config = PaperExchangeConfig(symbol=_TEST_SYMBOL)
        ctx = RuntimeContext(
            portfolio=Portfolio(cash=10000),
            order_manager=OrderManager(),
            trades=[],
            equity_curve=[10000],
            config=config,
            current_bar=make_bar(),
            bar_index=5,
            total_bars=100,
            start_time="2024-01-01T10:00:00Z",
        )
        assert ctx.symbol == _TEST_SYMBOL

    def test_bar_index(self) -> None:
        config = PaperExchangeConfig(symbol=_TEST_SYMBOL)
        ctx = RuntimeContext(
            portfolio=Portfolio(cash=10000),
            order_manager=OrderManager(),
            trades=[],
            equity_curve=[10000],
            config=config,
            current_bar=make_bar(),
            bar_index=5,
            total_bars=100,
            start_time="2024-01-01T10:00:00Z",
        )
        assert ctx.bar_index == 5
        assert ctx.total_bars == 100
        assert ctx.bars_remaining == 94


class TestRuntimeContextPortfolioState:
    """Test RuntimeContext portfolio state access."""

    def test_equity_cash_only(self) -> None:
        config = PaperExchangeConfig(symbol=_TEST_SYMBOL)
        ctx = RuntimeContext(
            portfolio=Portfolio(cash=10000),
            order_manager=OrderManager(),
            trades=[],
            equity_curve=[10000],
            config=config,
            current_bar=make_bar(price=100),
            bar_index=0,
            total_bars=10,
            start_time="2024-01-01T10:00:00Z",
        )
        assert ctx.cash == 10000
        assert ctx.equity == 10000

    def test_equity_with_position(self) -> None:
        config = PaperExchangeConfig(symbol=_TEST_SYMBOL)
        portfolio = Portfolio(cash=10000)
        # open_position deducts from cash: 10000 - (100 * 10) = 9000
        portfolio.open_position(str(_TEST_SYMBOL), "long", 100.0, 10.0, "2024-01-01T10:00:00Z")
        ctx = RuntimeContext(
            portfolio=portfolio,
            order_manager=OrderManager(),
            trades=[],
            equity_curve=[10000],
            config=config,
            current_bar=make_bar(price=110),  # 10% up
            bar_index=0,
            total_bars=10,
            start_time="2024-01-01T10:00:00Z",
        )
        assert ctx.cash == 9000  # 10000 - 1000 position cost
        # equity = cash + (qty * current_price) = 9000 + (10 * 110) = 10100
        assert ctx.equity == 10100


class TestRuntimeContextLiveMetrics:
    """Test RuntimeContext live metrics computed on demand."""

    def test_win_rate_live(self) -> None:
        trades = [
            make_trade(entry_price=100, exit_price=110),  # win
            make_trade(entry_price=100, exit_price=90),  # loss
        ]
        config = PaperExchangeConfig(symbol=_TEST_SYMBOL)
        ctx = RuntimeContext(
            portfolio=Portfolio(cash=10000),
            order_manager=OrderManager(),
            trades=trades,
            equity_curve=[10000],
            config=config,
            current_bar=make_bar(),
            bar_index=0,
            total_bars=10,
            start_time="2024-01-01T10:00:00Z",
        )
        assert ctx.win_rate == 0.5
        assert ctx.total_trades == 2

    def test_drawdown_live(self) -> None:
        config = PaperExchangeConfig(symbol=_TEST_SYMBOL)
        ctx = RuntimeContext(
            portfolio=Portfolio(cash=9000),  # 10% drawdown from 10000
            order_manager=OrderManager(),
            trades=[],
            equity_curve=[10000, 10500, 10000],  # peak 10500
            config=config,
            current_bar=make_bar(price=100),
            bar_index=0,
            total_bars=10,
            start_time="2024-01-01T10:00:00Z",
        )
        # equity = 9000, peak = 10500
        expected = (10500 - 9000) / 10500
        assert abs(ctx.drawdown - expected) < 0.001


class TestPaperExchangeResultDelegates:
    """Test PaperExchangeResult delegates to TradeMetrics."""

    def test_result_delegates_win_rate(self) -> None:
        from trdr.backtest.paper_exchange import PaperExchangeResult

        trades = [
            make_trade(entry_price=100, exit_price=110),
            make_trade(entry_price=100, exit_price=90),
        ]
        config = PaperExchangeConfig(symbol=_TEST_SYMBOL)
        result = PaperExchangeResult(
            trades=trades,
            config=config,
            start_time="2024-01-01T10:00:00Z",
            end_time="2024-01-02T10:00:00Z",
            equity_curve=[10000, 10500, 10000],
        )
        assert result.win_rate == 0.5
        assert result.total_trades == 2

    def test_result_total_costs(self) -> None:
        from trdr.backtest.paper_exchange import PaperExchangeResult

        trades = [
            make_trade(costs=5.0),
            make_trade(costs=10.0),
        ]
        config = PaperExchangeConfig(symbol=_TEST_SYMBOL)
        result = PaperExchangeResult(
            trades=trades,
            config=config,
            start_time="2024-01-01T10:00:00Z",
            end_time="2024-01-02T10:00:00Z",
            equity_curve=[10000],
        )
        assert result.total_costs == 15.0


@dataclass
class SimpleConfig(StrategyConfig):
    """Simple strategy config with test defaults."""

    symbol: Symbol = field(default_factory=lambda: _TEST_SYMBOL)
    timeframe: Timeframe = field(default_factory=lambda: _TEST_TF)
    lookback: Duration = field(default_factory=lambda: _TEST_LOOKBACK)


def _get_primary_bars(bars: dict[str, list[Bar]], config: StrategyConfig) -> list[Bar]:
    """Extract primary bars from dict for test strategies."""
    key = f"{config.symbol}:{config.timeframe}"
    return bars[key]


class ContextAwareStrategy(BaseStrategy):
    """Strategy that uses context for adaptive sizing."""

    def __init__(self, config: SimpleConfig, name: str | None = None):
        super().__init__(config, name=name)
        self.context_checks: list[tuple[float, float]] = []

    def get_data_requirements(self) -> list[DataRequirement]:
        return [
            DataRequirement(
                self.config.symbol, self.config.timeframe, self.config.lookback, role="primary"
            )
        ]

    def generate_signal(self, bars: dict[str, list[Bar]], position: Position | None) -> Signal:
        primary = _get_primary_bars(bars, self.config)
        # Record context values for testing
        self.context_checks.append((self.context.drawdown, self.context.total_return))
        return Signal(action=SignalAction.HOLD, price=primary[-1].close, confidence=0, reason="")


class TestStrategyAccessesContext:
    """Test that strategy can access self.context."""

    def test_context_available_in_generate_signal(self) -> None:
        config = PaperExchangeConfig(
            symbol=_TEST_SYMBOL,
            primary_feed=f"{_TEST_SYMBOL}:{_TEST_TF}",
            warmup_bars=2,
        )
        strategy = ContextAwareStrategy(SimpleConfig(symbol=_TEST_SYMBOL))

        # Create 5 bars
        bars = [make_bar(price=100 + i, ts=f"2024-01-01T{10+i}:00:00Z") for i in range(5)]

        engine = PaperExchange(config, strategy)
        engine.run(bars)

        # Strategy should have recorded context checks
        assert len(strategy.context_checks) > 0
        # All drawdowns should be >= 0
        for dd, _ in strategy.context_checks:
            assert dd >= 0.0


class TestStrategyName:
    """Test strategy name feature."""

    def test_default_name_is_class_name(self) -> None:
        strategy = ContextAwareStrategy(SimpleConfig(symbol=_TEST_SYMBOL))
        assert strategy.name == "ContextAwareStrategy"

    def test_custom_name_overrides_class_name(self) -> None:
        strategy = ContextAwareStrategy(
            SimpleConfig(symbol=_TEST_SYMBOL),
            name="MyCustomName",
        )
        assert strategy.name == "MyCustomName"

    def test_context_has_strategy_name(self) -> None:
        config = PaperExchangeConfig(
            symbol=_TEST_SYMBOL,
            primary_feed=f"{_TEST_SYMBOL}:{_TEST_TF}",
            warmup_bars=2,
        )
        strategy = ContextAwareStrategy(
            SimpleConfig(symbol=_TEST_SYMBOL),
            name="TestStrategy",
        )

        bars = [make_bar(price=100 + i, ts=f"2024-01-01T{10+i}:00:00Z") for i in range(5)]
        engine = PaperExchange(config, strategy)
        engine.run(bars)

        # Context should have strategy name
        assert strategy.context.strategy_name == "TestStrategy"
