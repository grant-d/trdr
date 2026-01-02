"""Backtest tests for VolumeAreaBreakout strategy.

Strategy: VAH breakout + VAL bounce with POC target.
Tests VolumeAreaBreakoutStrategy from trdr.strategy.

These tests are for normal development/CI. Feel free to modify thresholds
as needed. SICA uses sica_bench.py instead (which should NOT be modified).

Run with:
  .venv/bin/python -m pytest src/trdr/strategy/volume_area_breakout/test_strategy.py -v
  BACKTEST_SYMBOL=stock:AAPL BACKTEST_TIMEFRAME=1d .venv/bin/python -m pytest ... -v
"""

import asyncio
import math
import os
import re
from pathlib import Path

import pytest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

from trdr.backtest.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult
from trdr.core import load_config
from trdr.data import MarketDataClient
from trdr.strategy import VolumeAreaBreakoutConfig, VolumeAreaBreakoutStrategy

# Load .env for BACKTEST_* vars
load_dotenv()


def get_symbol() -> str:
    """Get symbol from env var. Default: crypto:BTC/USD."""
    return os.environ.get("BACKTEST_SYMBOL", "crypto:BTC/USD")


def get_timeframe_str() -> str:
    """Get timeframe string from env var. Default: 1h."""
    return os.environ.get("BACKTEST_TIMEFRAME", "1h").lower().strip()


def get_timeframe() -> TimeFrame:
    """Get timeframe from env var as Alpaca TimeFrame.

    Supports Alpaca syntax: 1h, 4h, 15m, 1d, etc.
    Also supports simple names: hour, day, minute (defaults to 1x).
    Note: Day only supports amount=1 (Alpaca constraint).
    Default: 1h (hourly).
    """
    tf = get_timeframe_str()

    # Map unit suffix to TimeFrameUnit
    unit_map = {
        "m": TimeFrameUnit.Minute,
        "min": TimeFrameUnit.Minute,
        "minute": TimeFrameUnit.Minute,
        "h": TimeFrameUnit.Hour,
        "hour": TimeFrameUnit.Hour,
        "d": TimeFrameUnit.Day,
        "day": TimeFrameUnit.Day,
    }

    # Try parsing "NNx" format (e.g., "4h", "15m", "1d")
    match = re.match(r"^(\d+)([a-z]+)$", tf)
    if match:
        amount = int(match.group(1))
        unit_str = match.group(2)
        unit = unit_map.get(unit_str)
        if unit:
            # Alpaca constraint: Day only allows amount=1
            if unit == TimeFrameUnit.Day and amount != 1:
                amount = 1
            return TimeFrame(amount, unit)

    # Fallback: simple name (e.g., "hour" -> 1h)
    unit = unit_map.get(tf)
    if unit:
        return TimeFrame(1, unit)

    # Default to 1 hour
    return TimeFrame.Hour


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def bars(event_loop):
    """Fetch bars for backtesting."""
    symbol = get_symbol()
    timeframe = get_timeframe()

    async def fetch():
        config = load_config()
        client = MarketDataClient(config.alpaca, Path("data/cache"))
        bars = await client.get_bars(symbol, lookback=3000, timeframe=timeframe)
        return bars

    return event_loop.run_until_complete(fetch())


@pytest.fixture(scope="module")
def strategy():
    """Create strategy with config from env vars."""
    symbol = get_symbol()
    timeframe = get_timeframe_str()

    config = VolumeAreaBreakoutConfig(
        symbol=symbol,
        timeframe=timeframe,
        atr_threshold=2.0,
        stop_loss_multiplier=1.75,
    )
    return VolumeAreaBreakoutStrategy(config)


@pytest.fixture(scope="module")
def backtest_config():
    """Backtest engine configuration (no strategy params)."""
    symbol = get_symbol()
    return BacktestConfig(
        symbol=symbol,
        warmup_bars=65,
        transaction_cost_pct=0.0025,
        slippage_atr=0.01,
        position_size=0.5,
    )


@pytest.fixture(scope="module")
def backtest_result(bars, backtest_config, strategy) -> BacktestResult:
    """Run single backtest with Volume Profile strategy."""
    engine = BacktestEngine(backtest_config, strategy)
    result = engine.run(bars)

    # Print summary for LLM visibility
    print(f"\n{'='*50}")
    print("BACKTEST SUMMARY")
    print(f"{'='*50}")
    print(f"Strategy: {strategy.name}")
    print(f"Symbol: {strategy.config.symbol}")
    print(f"Timeframe: {strategy.config.timeframe}")
    print(f"Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total P&L: ${result.total_pnl:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.1%} (${result.max_drawdown_abs:.2f})")
    print(f"Sortino: {result.sortino_ratio:.2f}" if result.sortino_ratio else "Sortino: N/A")
    print(f"Sharpe: {result.sharpe_ratio:.2f}" if result.sharpe_ratio else "Sharpe: N/A")
    print(f"Max Consecutive Losses: {result.max_consecutive_losses}")

    # Composite score for SICA ranking (0-1 scale)
    # Uses asymptotic scaling: score = x / (x + k) where k = target value
    # This gives 0.5 at target, approaches 1.0 asymptotically, handles inf

    def asymptotic(x: float, k: float) -> float:
        """Score 0-1 where x=k gives 0.5, x=inf gives 1.0."""
        if x <= 0 or math.isinf(x):
            return 1.0 if x > 0 or math.isinf(x) else 0.0
        return x / (x + k)

    # Weights: WR 20%, PF 20%, DD 15%, Sharpe 15%, Sortino 15%, Calmar 15%
    wr_score = min(result.win_rate / 0.60, 1.0)  # linear cap at 60%
    pf_score = asymptotic(result.profit_factor, 2.0)  # PF=2 → 0.5, PF=inf → 1.0
    dd_score = max(0, 1 - result.max_drawdown / 0.30)  # 0% DD = 1.0, 30% = 0

    sharpe = max(0, result.sharpe_ratio or 0)
    sharpe_score = asymptotic(sharpe, 2.0)  # Sharpe=2 → 0.5

    sortino = max(0, result.sortino_ratio or 0)
    sortino_score = asymptotic(sortino, 2.0)  # Sortino=2 → 0.5

    # Calmar = return / max drawdown (0% DD = perfect score)
    if result.max_drawdown == 0:
        calmar_score = 1.0 if result.total_pnl > 0 else 0.0
    elif result.total_pnl > 0:
        calmar = (result.total_pnl / 10000) / result.max_drawdown
        calmar_score = asymptotic(calmar, 1.0)  # Calmar=1 → 0.5
    else:
        calmar_score = 0.0

    composite = (
        0.20 * wr_score
        + 0.20 * pf_score
        + 0.15 * dd_score
        + 0.15 * sharpe_score
        + 0.15 * sortino_score
        + 0.15 * calmar_score
    )
    print(f"SICA_SCORE: {composite:.3f}")
    print(f"{'='*50}\n")

    return result


class TestAlgoPerformance:
    """Performance tests for Volume Profile strategy."""

    def test_has_trades(self, backtest_result):
        """Strategy must generate enough trades for statistical significance.

        Modify thresholds as needed. SICA uses sica_bench.py.
        """
        total = backtest_result.total_trades
        assert total >= 6, f"Only {total} trades (need >= 6 for significance)"

    def test_win_rate(self, backtest_result):
        """Strategy must have reasonable win rate.

        Modify thresholds as needed. SICA uses sica_bench.py.
        """
        if backtest_result.total_trades == 0:
            pytest.skip("No trades to evaluate")
        wr = backtest_result.win_rate
        assert wr >= 0.40, f"Win rate {wr:.1%} < 40%"

    def test_profit_factor(self, backtest_result):
        """Strategy must have profit factor > 1.0.

        Modify thresholds as needed. SICA uses sica_bench.py.
        """
        if backtest_result.total_trades == 0:
            pytest.skip("No trades to evaluate")
        pf = backtest_result.profit_factor
        assert pf > 1.0, f"Profit factor {pf:.2f} <= 1.0"

    def test_sortino_positive(self, backtest_result):
        """Strategy must have positive Sortino ratio.

        Modify thresholds as needed. SICA uses sica_bench.py.
        """
        sortino = backtest_result.sortino_ratio
        if sortino is not None:
            assert sortino > 0, f"Sortino ratio {sortino:.2f} <= 0"

    def test_max_drawdown(self, backtest_result):
        """Strategy must have controlled drawdown.

        Modify thresholds as needed. SICA uses sica_bench.py.
        """
        max_dd = backtest_result.max_drawdown
        assert max_dd < 0.30, f"Max drawdown {max_dd:.1%} > 30%"


class TestAlgoRobustness:
    """Basic robustness checks.

    Modify thresholds as needed. SICA uses sica_bench.py.
    """

    def test_no_excessive_losing_streak(self, backtest_result):
        """Strategy should not have excessive consecutive losses."""
        max_streak = backtest_result.max_consecutive_losses
        assert max_streak <= 10, f"Max consecutive losses {max_streak} > 10"

    def test_positive_pnl(self, backtest_result):
        """Strategy should be profitable overall."""
        if backtest_result.total_trades == 0:
            pytest.skip("No trades to evaluate")
        pnl = backtest_result.total_pnl
        assert pnl > 0, f"Total P&L ${pnl:.2f} <= 0"


def print_results():
    """Helper to print detailed results."""

    async def run():
        symbol = get_symbol()
        timeframe = get_timeframe()

        config = load_config()
        client = MarketDataClient(config.alpaca, Path("data/cache"))
        bars = await client.get_bars(symbol, lookback=3000, timeframe=timeframe)

        strategy_config = VolumeAreaBreakoutConfig(
            symbol=symbol,
            timeframe=get_timeframe_str(),
            atr_threshold=2.0,
            stop_loss_multiplier=1.75,
        )
        strategy = VolumeAreaBreakoutStrategy(strategy_config)

        bt_config = BacktestConfig(
            symbol=symbol,
            warmup_bars=65,
            transaction_cost_pct=0.0025,
            position_size=1.0,
        )

        engine = BacktestEngine(bt_config, strategy)
        result = engine.run(bars)

        print(f"\n=== Backtest Results ({len(bars)} bars) ===")
        print(f"Strategy: {strategy.name}")
        print(f"Period: {result.start_time} to {result.end_time}")
        print(f"Total trades: {result.total_trades}")
        print(f"Win rate: {result.win_rate:.1%}")
        print(f"Total P&L: ${result.total_pnl:.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        if result.sortino_ratio:
            print(f"Sortino: {result.sortino_ratio:.2f}")
        if result.sharpe_ratio:
            print(f"Sharpe: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.1%} (${result.max_drawdown_abs:.2f})")
        print(f"Max Consecutive Losses: {result.max_consecutive_losses}")

        if result.trades:
            print("\n=== Trades ===")
            for i, t in enumerate(result.trades[:10]):
                print(
                    f"{i+1}. {t.side} @ {t.entry_price:.2f} -> "
                    f"{t.exit_price:.2f}, "
                    f"pnl=${t.net_pnl:.2f} ({t.exit_reason})"
                )
            if len(result.trades) > 10:
                print(f"... and {len(result.trades) - 10} more trades")

    asyncio.run(run())


if __name__ == "__main__":
    print_results()
