"""Backtest tests for MACD strategy - TEMPLATE for strategy tests.

This test file demonstrates the pattern for testing strategies:
1. Define fixtures for data, strategy, and backtest config
2. Run backtest once (module-scoped fixtures)
3. Assert on results

To create tests for a new strategy:
1. Copy this file to your strategy folder
2. Update imports and constants
3. Adjust assertions for your strategy's expected behavior

Run with:
    .venv/bin/python -m pytest src/trdr/strategy/macd/test_strategy.py -v
"""

import asyncio
from pathlib import Path

import pytest

from trdr.backtest.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult
from trdr.core import load_config
from trdr.data import MarketDataClient
from trdr.strategy import MACDConfig, MACDStrategy


# =============================================================================
# STEP 1: Define constants
# =============================================================================
# Set symbol and timeframe for this strategy's tests
# Use different values than other strategies to verify decoupling

SYMBOL = "crypto:ETH/USD"
TIMEFRAME = "4h"


# =============================================================================
# STEP 2: Define fixtures
# =============================================================================
# Fixtures are shared across all tests in this file
# Use scope="module" to run expensive operations (data fetch, backtest) once


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def bars(event_loop):
    """Fetch historical bars for backtesting.

    This fetches data once and reuses across all tests.
    Adjust lookback based on strategy's data requirements.
    """
    from alpaca.data.timeframe import TimeFrame

    async def fetch():
        config = load_config()
        client = MarketDataClient(config.alpaca, Path("data/cache"))
        bars = await client.get_bars(
            SYMBOL,
            lookback=1000,  # Adjust based on strategy needs
            timeframe=TimeFrame(4, TimeFrame.Hour.unit),
        )
        return bars

    return event_loop.run_until_complete(fetch())


@pytest.fixture(scope="module")
def strategy():
    """Create strategy instance with config.

    Set strategy-specific parameters here.
    These should match reasonable defaults or test values.
    """
    config = MACDConfig(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        fast_period=12,
        slow_period=26,
        signal_period=9,
        stop_loss_pct=0.03,  # 3% stop
    )
    return MACDStrategy(config)


@pytest.fixture(scope="module")
def backtest_config():
    """Backtest engine configuration.

    These are engine-level settings (not strategy settings):
    - warmup_bars: Bars before strategy can generate signals
    - transaction_cost_pct: Simulated trading costs
    - slippage_atr: Simulated slippage
    - position_size: Fixed position size for backtesting
    """
    return BacktestConfig(
        symbol=SYMBOL,
        warmup_bars=35,  # slow_period + signal_period = 26 + 9 = 35
        transaction_cost_pct=0.001,
        slippage_atr=0.005,
        position_size=1.0,
    )


@pytest.fixture(scope="module")
def backtest_result(bars, backtest_config, strategy) -> BacktestResult:
    """Run backtest and return results.

    This runs once per test session.
    Print summary for visibility during test runs.
    """
    engine = BacktestEngine(backtest_config, strategy)
    result = engine.run(bars)

    # Print summary for visibility
    print(f"\n{'='*50}")
    print(f"{strategy.name} BACKTEST SUMMARY")
    print(f"{'='*50}")
    print(f"Symbol: {strategy.config.symbol}")
    print(f"Timeframe: {strategy.config.timeframe}")
    print(f"Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total P&L: ${result.total_pnl:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.1%}")
    print(f"{'='*50}\n")

    return result


# =============================================================================
# STEP 3: Write tests
# =============================================================================
# Test categories:
# - Basic sanity checks (strategy name, config)
# - Functional tests (generates trades, uses correct signals)
# - Performance tests (win rate, profit factor) - optional for templates


class TestMACDStrategy:
    """Tests for MACD strategy.

    For a template strategy, focus on:
    - Verifying the abstraction works
    - Basic sanity checks

    For production strategies, add:
    - Performance thresholds (win rate, profit factor)
    - Robustness checks (max drawdown, losing streaks)
    """

    def test_strategy_name(self, strategy):
        """Strategy has correct name."""
        assert strategy.name == "MACD"

    def test_strategy_config(self, strategy):
        """Strategy config has expected values."""
        assert strategy.config.symbol == SYMBOL
        assert strategy.config.timeframe == TIMEFRAME
        assert strategy.config.fast_period == 12
        assert strategy.config.slow_period == 26

    def test_generates_trades(self, backtest_result):
        """Strategy generates trades (basic sanity check).

        For templates, just verify it runs without error.
        For production, set minimum trade thresholds.
        """
        # Template: just verify it works
        assert backtest_result.total_trades >= 0

        # Production example:
        # assert backtest_result.total_trades >= 10, "Need enough trades for significance"

    def test_engine_uses_strategy(self, backtest_result, strategy):
        """Verify engine used this strategy (not a hardcoded default).

        Check that trade reasons contain strategy-specific text.
        """
        if backtest_result.trades:
            entry_reasons = [t.entry_reason for t in backtest_result.trades]
            assert any("MACD" in r for r in entry_reasons), \
                f"Expected MACD in reasons, got: {entry_reasons}"
