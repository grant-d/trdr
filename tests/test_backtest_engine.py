import pytest

from trdr.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult, Trade
from trdr.data import Bar, Signal, SignalAction


def make_bar(idx: int, open_price: float, close_price: float) -> Bar:
    return Bar(
        timestamp=f"2024-01-01T00:0{idx}:00Z",
        open=open_price,
        high=max(open_price, close_price),
        low=min(open_price, close_price),
        close=close_price,
        volume=100,
    )


def test_next_bar_fill_execution():
    bars = [
        make_bar(0, 100.0, 100.0),
        make_bar(1, 101.0, 101.0),
        make_bar(2, 102.0, 102.0),
        make_bar(3, 103.0, 103.0),
    ]

    def signal_fn(visible_bars, position, _atr, _sl):
        idx = len(visible_bars) - 1
        if idx == 1 and position is None:
            return Signal(
                action=SignalAction.BUY,
                price=visible_bars[-1].close,
                confidence=1.0,
                reason="buy",
                stop_loss=visible_bars[-1].close - 1.0,
                take_profit=visible_bars[-1].close + 1.0,
            )
        if idx == 2 and position is not None:
            return Signal(
                action=SignalAction.CLOSE,
                price=visible_bars[-1].close,
                confidence=1.0,
                reason="close",
            )
        return Signal(
            action=SignalAction.HOLD,
            price=visible_bars[-1].close,
            confidence=0.0,
            reason="hold",
        )

    config = BacktestConfig(symbol="TEST", warmup_bars=1)
    engine = BacktestEngine(config, signal_fn=signal_fn)
    result = engine.run(bars)

    assert result.total_trades == 1
    trade = result.trades[0]
    assert trade.entry_price == bars[2].open
    assert trade.exit_price == bars[3].open
    assert trade.entry_time == bars[2].timestamp
    assert trade.exit_time == bars[3].timestamp


def test_force_close_on_final_bar_close():
    bars = [
        make_bar(0, 100.0, 100.0),
        make_bar(1, 101.0, 101.0),
        make_bar(2, 102.0, 102.0),
        make_bar(3, 103.0, 110.0),
    ]

    def signal_fn(visible_bars, position, _atr, _sl):
        idx = len(visible_bars) - 1
        if idx == 1 and position is None:
            return Signal(
                action=SignalAction.BUY,
                price=visible_bars[-1].close,
                confidence=1.0,
                reason="buy",
                stop_loss=visible_bars[-1].close - 1.0,
                take_profit=visible_bars[-1].close + 1.0,
            )
        return Signal(
            action=SignalAction.HOLD,
            price=visible_bars[-1].close,
            confidence=0.0,
            reason="hold",
        )

    config = BacktestConfig(symbol="TEST", warmup_bars=1)
    engine = BacktestEngine(config, signal_fn=signal_fn)
    result = engine.run(bars)

    assert result.total_trades == 1
    trade = result.trades[0]
    assert trade.entry_price == bars[2].open
    assert trade.exit_price == bars[-1].close
    assert trade.exit_time == bars[-1].timestamp
    assert len(result.equity_curve) == len(bars) - config.warmup_bars


# --- Helper for creating test trades ---


def make_trade(
    net_pnl: float,
    gross_pnl: float | None = None,
    costs: float = 0.0,
    entry_price: float = 100.0,
    size: float = 1.0,
) -> Trade:
    """Create a trade for testing metrics."""
    if gross_pnl is None:
        gross_pnl = net_pnl + costs
    return Trade(
        entry_time="2024-01-01T00:00:00Z",
        exit_time="2024-01-01T01:00:00Z",
        entry_price=entry_price,
        exit_price=entry_price + gross_pnl / size,
        size=size,
        side="long",
        gross_pnl=gross_pnl,
        costs=costs,
        net_pnl=net_pnl,
        entry_reason="test",
        exit_reason="test",
    )


def make_result(trades: list[Trade], equity_curve: list[float] | None = None) -> BacktestResult:
    """Create a BacktestResult for testing."""
    return BacktestResult(
        trades=trades,
        config=BacktestConfig(symbol="TEST"),
        start_time="2024-01-01T00:00:00Z",
        end_time="2024-01-01T12:00:00Z",
        equity_curve=equity_curve or [],
    )


# --- BacktestResult Metrics Tests ---


class TestProfitFactor:
    """Tests for profit_factor calculation.

    CRITICAL: profit_factor must use net_pnl (after costs), not gross_pnl.
    This was a bug that has been fixed - these tests prevent regression.
    """

    def test_profit_factor_uses_net_pnl_not_gross_pnl(self):
        """REGRESSION TEST: profit_factor must use net_pnl, not gross_pnl.

        Scenario: Trade has positive gross_pnl but negative net_pnl due to costs.
        If using gross_pnl (bug): would count as profit
        If using net_pnl (correct): counts as loss
        """
        # Trade with positive gross but negative net (costs exceed gross profit)
        trade = make_trade(
            gross_pnl=50.0,   # Positive before costs
            costs=60.0,       # Costs exceed gross
            net_pnl=-10.0,    # Net is negative
        )
        result = make_result([trade])

        # Bug would return inf (no losses in gross_pnl terms)
        # Correct behavior: profit_factor = 0 (no profits, only losses in net terms)
        assert result.profit_factor == 0.0, (
            "profit_factor should be 0 when only loss exists (net_pnl < 0)"
        )

    def test_profit_factor_mixed_trades_net_pnl(self):
        """profit_factor with mixed wins/losses uses net_pnl correctly."""
        trades = [
            make_trade(net_pnl=100.0),   # Winner
            make_trade(net_pnl=-50.0),   # Loser
            make_trade(net_pnl=30.0),    # Winner
        ]
        result = make_result(trades)

        # net_profits = 100 + 30 = 130
        # net_losses = 50
        # profit_factor = 130 / 50 = 2.6
        assert result.profit_factor == pytest.approx(2.6)

    def test_profit_factor_no_losses_returns_inf(self):
        """profit_factor returns inf when no losing trades."""
        trades = [
            make_trade(net_pnl=100.0),
            make_trade(net_pnl=50.0),
        ]
        result = make_result(trades)

        assert result.profit_factor == float("inf")

    def test_profit_factor_no_profits_returns_zero(self):
        """profit_factor returns 0 when no winning trades."""
        trades = [
            make_trade(net_pnl=-100.0),
            make_trade(net_pnl=-50.0),
        ]
        result = make_result(trades)

        assert result.profit_factor == 0.0

    def test_profit_factor_empty_trades(self):
        """profit_factor returns 0 for empty trade list."""
        result = make_result([])
        assert result.profit_factor == 0.0


class TestWinRate:
    """Tests for win_rate calculation."""

    def test_win_rate_uses_net_pnl(self):
        """win_rate must use net_pnl (via is_winner property)."""
        # Trade with positive gross but negative net
        trade = make_trade(gross_pnl=50.0, costs=60.0, net_pnl=-10.0)
        result = make_result([trade])

        # Should be 0% win rate since net_pnl < 0
        assert result.win_rate == 0.0

    def test_win_rate_calculation(self):
        """win_rate correctly calculates winning percentage."""
        trades = [
            make_trade(net_pnl=100.0),   # Win
            make_trade(net_pnl=-50.0),   # Loss
            make_trade(net_pnl=30.0),    # Win
            make_trade(net_pnl=-20.0),   # Loss
        ]
        result = make_result(trades)

        assert result.win_rate == pytest.approx(0.5)  # 2/4 = 50%

    def test_win_rate_all_winners(self):
        """win_rate is 1.0 when all trades win."""
        trades = [make_trade(net_pnl=100.0) for _ in range(3)]
        result = make_result(trades)

        assert result.win_rate == 1.0

    def test_win_rate_empty_trades(self):
        """win_rate returns 0 for empty trade list."""
        result = make_result([])
        assert result.win_rate == 0.0


class TestPnLMetrics:
    """Tests for P&L-related metrics."""

    def test_total_pnl_sums_net(self):
        """total_pnl sums net_pnl of all trades."""
        trades = [
            make_trade(net_pnl=100.0),
            make_trade(net_pnl=-30.0),
            make_trade(net_pnl=50.0),
        ]
        result = make_result(trades)

        assert result.total_pnl == pytest.approx(120.0)

    def test_gross_pnl_sums_gross(self):
        """gross_pnl sums gross_pnl of all trades."""
        trades = [
            make_trade(gross_pnl=100.0, costs=10.0, net_pnl=90.0),
            make_trade(gross_pnl=50.0, costs=5.0, net_pnl=45.0),
        ]
        result = make_result(trades)

        assert result.gross_pnl == pytest.approx(150.0)

    def test_total_costs_sums_costs(self):
        """total_costs sums costs of all trades."""
        trades = [
            make_trade(gross_pnl=100.0, costs=10.0, net_pnl=90.0),
            make_trade(gross_pnl=50.0, costs=5.0, net_pnl=45.0),
        ]
        result = make_result(trades)

        assert result.total_costs == pytest.approx(15.0)


class TestTradeCountMetrics:
    """Tests for trade counting metrics."""

    def test_winning_losing_trades_count(self):
        """winning_trades and losing_trades count correctly."""
        trades = [
            make_trade(net_pnl=100.0),   # Win
            make_trade(net_pnl=-50.0),   # Loss
            make_trade(net_pnl=30.0),    # Win
        ]
        result = make_result(trades)

        assert result.total_trades == 3
        assert result.winning_trades == 2
        assert result.losing_trades == 1


class TestMaxConsecutiveLosses:
    """Tests for max_consecutive_losses calculation."""

    def test_max_consecutive_losses(self):
        """max_consecutive_losses tracks longest losing streak."""
        trades = [
            make_trade(net_pnl=100.0),   # Win
            make_trade(net_pnl=-50.0),   # Loss 1
            make_trade(net_pnl=-30.0),   # Loss 2
            make_trade(net_pnl=-20.0),   # Loss 3
            make_trade(net_pnl=40.0),    # Win
            make_trade(net_pnl=-10.0),   # Loss 1
        ]
        result = make_result(trades)

        assert result.max_consecutive_losses == 3

    def test_max_consecutive_losses_no_losses(self):
        """max_consecutive_losses is 0 when no losses."""
        trades = [make_trade(net_pnl=100.0) for _ in range(3)]
        result = make_result(trades)

        assert result.max_consecutive_losses == 0

    def test_max_consecutive_losses_empty(self):
        """max_consecutive_losses is 0 for empty trade list."""
        result = make_result([])
        assert result.max_consecutive_losses == 0


class TestMaxDrawdown:
    """Tests for max_drawdown calculation."""

    def test_max_drawdown_calculation(self):
        """max_drawdown correctly calculates peak-to-trough decline."""
        # Equity: 1000 -> 1100 (peak) -> 900 -> 950
        # Max DD = (1100 - 900) / 1100 = 18.18%
        equity_curve = [1000.0, 1100.0, 900.0, 950.0]
        result = make_result([], equity_curve)

        assert result.max_drawdown == pytest.approx(200.0 / 1100.0)

    def test_max_drawdown_no_drawdown(self):
        """max_drawdown is 0 when equity only goes up."""
        equity_curve = [1000.0, 1100.0, 1200.0, 1300.0]
        result = make_result([], equity_curve)

        assert result.max_drawdown == 0.0

    def test_max_drawdown_empty_curve(self):
        """max_drawdown is 0 for empty equity curve."""
        result = make_result([])
        assert result.max_drawdown == 0.0

    def test_max_drawdown_capped_at_100_percent(self):
        """REGRESSION: max_drawdown must cap at 1.0 when equity goes negative.

        Previously, if equity started at 100, went to -2147, the formula
        (100 - (-2147)) / 100 = 22.47 = 2247% which is nonsensical.
        """
        # Equity starts at 10000, goes down to -5000 (below zero)
        equity_curve = [10000.0, 8000.0, 5000.0, -5000.0, 2000.0]
        result = make_result([], equity_curve)

        # Should cap at 1.0 (100%), not exceed it
        assert result.max_drawdown == 1.0

    def test_max_drawdown_abs_calculation(self):
        """max_drawdown_abs returns absolute dollar drawdown."""
        # Equity: 10000 -> 12000 (peak) -> 8000 -> 9000
        # Max DD abs = 12000 - 8000 = 4000
        equity_curve = [10000.0, 12000.0, 8000.0, 9000.0]
        result = make_result([], equity_curve)

        assert result.max_drawdown_abs == 4000.0

    def test_max_drawdown_abs_with_negative_equity(self):
        """max_drawdown_abs works even when equity goes negative."""
        # Equity: 10000 (peak) -> -5000
        # Max DD abs = 10000 - (-5000) = 15000
        equity_curve = [10000.0, 5000.0, -5000.0]
        result = make_result([], equity_curve)

        assert result.max_drawdown_abs == 15000.0


class TestRatioMetrics:
    """Tests for Sharpe and Sortino ratios."""

    def test_sharpe_ratio_calculation(self):
        """sharpe_ratio calculates correctly with sufficient trades."""
        trades = [
            make_trade(net_pnl=10.0, entry_price=100.0, size=1.0),
            make_trade(net_pnl=5.0, entry_price=100.0, size=1.0),
            make_trade(net_pnl=-3.0, entry_price=100.0, size=1.0),
        ]
        result = make_result(trades)

        # Returns: 10%, 5%, -3%
        # Should return a float (exact value depends on annualization)
        assert result.sharpe_ratio is not None
        assert isinstance(result.sharpe_ratio, float)

    def test_sharpe_ratio_insufficient_trades(self):
        """sharpe_ratio returns None with < 2 trades."""
        trades = [make_trade(net_pnl=10.0)]
        result = make_result(trades)

        assert result.sharpe_ratio is None

    def test_sortino_ratio_calculation(self):
        """sortino_ratio calculates correctly with downside deviation."""
        trades = [
            make_trade(net_pnl=10.0, entry_price=100.0, size=1.0),
            make_trade(net_pnl=-5.0, entry_price=100.0, size=1.0),
            make_trade(net_pnl=-3.0, entry_price=100.0, size=1.0),  # Need 2+ downside for std
            make_trade(net_pnl=8.0, entry_price=100.0, size=1.0),
        ]
        result = make_result(trades)

        assert result.sortino_ratio is not None
        assert isinstance(result.sortino_ratio, float)

    def test_sortino_ratio_no_downside(self):
        """sortino_ratio returns inf when no negative returns."""
        trades = [
            make_trade(net_pnl=10.0, entry_price=100.0, size=1.0),
            make_trade(net_pnl=5.0, entry_price=100.0, size=1.0),
        ]
        result = make_result(trades)

        assert result.sortino_ratio == float("inf")


# --- Slippage Tests ---


class TestSlippage:
    """Tests for ATR-based slippage simulation."""

    def test_slippage_increases_entry_price_for_longs(self):
        """Slippage makes long entries more expensive."""
        # Create 50 bars - enough for ATR calculation (needs 15+)
        bars = []
        for i in range(50):
            bars.append(Bar(
                timestamp=f"2024-01-01T{i:02d}:00:00Z",
                open=100.0,
                high=102.0,
                low=98.0,
                close=100.0,
                volume=1000,
            ))

        def buy_signal(visible_bars, position, _atr, _sl):
            # Buy after warmup when we have enough bars for ATR
            if len(visible_bars) == 20 and position is None:
                return Signal(
                    action=SignalAction.BUY,
                    price=100.0,
                    confidence=1.0,
                    reason="buy",
                    stop_loss=90.0,
                )
            if len(visible_bars) == 30 and position is not None:
                return Signal(
                    action=SignalAction.CLOSE,
                    price=100.0,
                    confidence=1.0,
                    reason="close",
                )
            return Signal(action=SignalAction.HOLD, price=100.0, confidence=0.0, reason="hold")

        # Without slippage
        config_no_slip = BacktestConfig(symbol="TEST", warmup_bars=15, slippage_atr=0.0)
        engine_no_slip = BacktestEngine(config_no_slip, signal_fn=buy_signal)
        result_no_slip = engine_no_slip.run(bars)

        # With slippage (1% of ATR per fill)
        config_slip = BacktestConfig(symbol="TEST", warmup_bars=15, slippage_atr=0.01)
        engine_slip = BacktestEngine(config_slip, signal_fn=buy_signal)
        result_slip = engine_slip.run(bars)

        assert result_no_slip.total_trades == 1
        assert result_slip.total_trades == 1

        # Slippage should result in worse P&L (higher entry, lower exit)
        assert result_slip.total_pnl < result_no_slip.total_pnl

    def test_slippage_zero_has_no_effect(self):
        """Zero slippage_atr should not affect prices."""
        bars = [
            Bar(timestamp=f"2024-01-01T{i:02d}:00:00Z", open=100.0, high=101.0, low=99.0, close=100.0, volume=100)
            for i in range(10)
        ]

        def buy_and_close(visible_bars, position, _atr, _sl):
            if len(visible_bars) == 3 and position is None:
                return Signal(action=SignalAction.BUY, price=100.0, confidence=1.0, reason="buy", stop_loss=90.0)
            if len(visible_bars) == 6 and position is not None:
                return Signal(action=SignalAction.CLOSE, price=100.0, confidence=1.0, reason="close")
            return Signal(action=SignalAction.HOLD, price=100.0, confidence=0.0, reason="hold")

        config = BacktestConfig(symbol="TEST", warmup_bars=2, slippage_atr=0.0)
        engine = BacktestEngine(config, signal_fn=buy_and_close)
        result = engine.run(bars)

        assert result.total_trades == 1
        # Entry should be exactly at bar open (100.0)
        assert result.trades[0].entry_price == 100.0

    def test_slippage_scales_with_atr(self):
        """Higher ATR should result in more slippage."""
        # Low volatility bars (50 bars for ATR calculation)
        low_vol_bars = [
            Bar(timestamp=f"2024-01-01T{i:02d}:00:00Z", open=100.0, high=100.5, low=99.5, close=100.0, volume=100)
            for i in range(50)
        ]

        # High volatility bars (wider range)
        high_vol_bars = [
            Bar(timestamp=f"2024-01-01T{i:02d}:00:00Z", open=100.0, high=105.0, low=95.0, close=100.0, volume=100)
            for i in range(50)
        ]

        def buy_signal(visible_bars, position, _atr, _sl):
            # Buy when we have enough bars for ATR
            if len(visible_bars) == 20 and position is None:
                return Signal(action=SignalAction.BUY, price=100.0, confidence=1.0, reason="buy", stop_loss=90.0)
            return Signal(action=SignalAction.HOLD, price=100.0, confidence=0.0, reason="hold")

        config = BacktestConfig(symbol="TEST", warmup_bars=15, slippage_atr=0.01)

        engine_low = BacktestEngine(config, signal_fn=buy_signal)
        result_low = engine_low.run(low_vol_bars)

        engine_high = BacktestEngine(config, signal_fn=buy_signal)
        result_high = engine_high.run(high_vol_bars)

        # High vol should have higher entry price (more slippage)
        assert result_high.trades[0].entry_price > result_low.trades[0].entry_price
