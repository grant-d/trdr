from trdr.backtest.engine import BacktestConfig, BacktestEngine
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
