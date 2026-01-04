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
from pathlib import Path

import pytest

from trdr.backtest import PaperExchange, PaperExchangeConfig, PaperExchangeResult
from trdr.core import load_config
from trdr.data import MarketDataClient
from trdr.strategy import VolumeAreaBreakoutConfig, VolumeAreaBreakoutStrategy, get_backtest_env
from trdr.strategy.targets import score_result

# Read env vars once at module load
SYMBOL, TIMEFRAME_STR, TIMEFRAME, _ = get_backtest_env(
    default_symbol="crypto:BTC/USD",
    default_timeframe="1h",
)


# IMPORTANT: MUST use scope="function" NOT "module"!
# "module" caches results and WON'T pick up strategy code changes.
# This cost hours of debugging. Do NOT change it back.
@pytest.fixture(scope="function")
def event_loop():
    """Create event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def bars(event_loop):
    """Fetch historical bars for backtesting.

    This fetches data once and reuses across all tests.
    Adjust lookback based on strategy's data requirements.
    """

    async def fetch():
        config = load_config()
        client = MarketDataClient(config.alpaca, Path("data/cache"))
        # Use TIMEFRAME_STR - get_bars handles aggregation for arbitrary timeframes
        bars = await client.get_bars(SYMBOL, lookback=3000, timeframe=TIMEFRAME_STR)
        return bars

    return event_loop.run_until_complete(fetch())


@pytest.fixture(scope="function")
def strategy():
    """Create strategy instance with config.

    Set strategy-specific parameters here.
    These should match reasonable defaults or test values.
    """
    config = VolumeAreaBreakoutConfig(
        symbol=SYMBOL,
        timeframe=TIMEFRAME_STR,
        atr_threshold=2.0,
        stop_loss_multiplier=1.75,
    )
    return VolumeAreaBreakoutStrategy(config)


@pytest.fixture(scope="function")
def backtest_config():
    """Paper exchange configuration.

    Engine-level settings (not strategy settings):
    - warmup_bars: Bars before strategy can generate signals
    - transaction_cost_pct: Simulated trading costs
    - slippage_pct: Simulated slippage as % of price
    - default_position_pct: Position size as % of equity
    """
    return PaperExchangeConfig(
        symbol=SYMBOL,
        warmup_bars=65,
        transaction_cost_pct=0.0025,
        slippage_pct=0.01,
        default_position_pct=0.5,
    )


@pytest.fixture(scope="function")
def backtest_result(bars, backtest_config, strategy) -> PaperExchangeResult:
    """Run single backtest with Volume Profile strategy."""
    engine = PaperExchange(backtest_config, strategy)
    result = engine.run(bars)

    # Print summary for LLM visibility
    print(f"\n{'='*50}")
    print("BACKTEST SUMMARY")
    print(f"{'='*50}")
    print(f"Strategy: {strategy.name}")
    print(f"Symbol: {strategy.config.symbol}")
    print(f"Timeframe: {strategy.config.timeframe}")
    print(f"Trades: {result.total_trades} ({result.trades_per_year:.0f}/yr)")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total P&L: ${result.total_pnl:.2f}")
    print(f"CAGR: {result.cagr:.1%}" if result.cagr else "CAGR: N/A")
    print(f"Max Drawdown: {result.max_drawdown:.1%}")
    print(f"Sharpe: {result.sharpe_ratio:.2f}" if result.sharpe_ratio else "Sharpe: N/A")
    print(f"Sortino: {result.sortino_ratio:.2f}" if result.sortino_ratio else "Sortino: N/A")
    print(f"Calmar: {result.calmar_ratio:.2f}" if result.calmar_ratio else "Calmar: N/A")

    # Use centralized scoring
    score, details = score_result(result)
    print(f"SICA_SCORE: {score:.3f}")
    print(f"{'='*50}\n")

    return result


class TestAlgoPerformance:
    """Sanity checks for Volume Profile strategy.

    These are NOT performance gates - SICA uses sica_bench.py for that.
    These just verify the backtest runs and produces valid metrics.
    """

    def test_has_trades(self, backtest_result):
        """Strategy generates trades."""
        total = backtest_result.total_trades
        assert total >= 1, f"No trades generated"

    def test_win_rate_valid(self, backtest_result):
        """Win rate is valid (0-1 range)."""
        if backtest_result.total_trades == 0:
            pytest.skip("No trades to evaluate")
        wr = backtest_result.win_rate
        assert 0.0 <= wr <= 1.0, f"Win rate {wr:.1%} out of range"

    def test_profit_factor_computed(self, backtest_result):
        """Profit factor is computed (non-negative)."""
        if backtest_result.total_trades == 0:
            pytest.skip("No trades to evaluate")
        pf = backtest_result.profit_factor
        assert pf >= 0.0, f"Profit factor {pf:.2f} is negative"

    def test_sortino_computed(self, backtest_result):
        """Sortino ratio is computed."""
        sortino = backtest_result.sortino_ratio
        # Just verify it's a number (can be negative)
        assert sortino is None or isinstance(sortino, (int, float))

    def test_max_drawdown_valid(self, backtest_result):
        """Max drawdown is valid (0-1 range)."""
        max_dd = backtest_result.max_drawdown
        assert 0.0 <= max_dd <= 1.0, f"Max drawdown {max_dd:.1%} out of range"


class TestAlgoRobustness:
    """Basic robustness checks.

    These are NOT performance gates - SICA uses sica_bench.py for that.
    """

    def test_pnl_computed(self, backtest_result):
        """P&L is computed (any value valid)."""
        if backtest_result.total_trades == 0:
            pytest.skip("No trades to evaluate")
        pnl = backtest_result.total_pnl
        assert isinstance(pnl, (int, float)), f"P&L not computed"


def print_results():
    """Helper to print detailed results."""

    async def run():
        config = load_config()
        client = MarketDataClient(config.alpaca, Path("data/cache"))
        bars = await client.get_bars(SYMBOL, lookback=3000, timeframe=TIMEFRAME_STR)

        strategy_config = VolumeAreaBreakoutConfig(
            symbol=SYMBOL,
            timeframe=TIMEFRAME_STR,
            atr_threshold=2.0,
            stop_loss_multiplier=1.75,
        )
        strategy = VolumeAreaBreakoutStrategy(strategy_config)

        bt_config = PaperExchangeConfig(
            symbol=SYMBOL,
            warmup_bars=65,
            transaction_cost_pct=0.0025,
            default_position_pct=1.0,
        )

        engine = PaperExchange(bt_config, strategy)
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
        print(f"Max Drawdown: {result.max_drawdown:.1%}")

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
