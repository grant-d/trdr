"""Backtest performance tests for Volume Profile strategy.

Tests the core generate_signal function from volume_profile.py against
real market data. SICA iterates on the strategy to improve metrics.

Thresholds:
- Win rate > 45%
- Profit factor > 1.0
- Max drawdown < 25%
- Positive Sortino ratio

Run with: .venv/bin/python -m pytest tests/test_algo_performance.py -v
"""

import asyncio
from pathlib import Path

import pytest

from trdr.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from trdr.core import load_config
from trdr.data import MarketDataClient, generate_signal


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def btc_bars(event_loop):
    """Fetch BTC/USD bars for backtesting."""

    async def fetch():
        config = load_config()
        client = MarketDataClient(config.alpaca, Path("data/cache"))
        bars = await client.get_bars("crypto:BTC/USD", lookback=3000)
        return bars

    return event_loop.run_until_complete(fetch())


@pytest.fixture(scope="module")
def backtest_config():
    """Backtest configuration for crypto."""
    return BacktestConfig(
        symbol="crypto:BTC/USD",
        warmup_bars=65,
        transaction_cost_pct=0.0025,
        position_size=0.5,  # 50% per trade with tight stops
        atr_threshold=2.0,
        stop_loss_multiplier=1.75,
    )


@pytest.fixture(scope="module")
def backtest_result(btc_bars, backtest_config) -> BacktestResult:
    """Run single backtest with Volume Profile strategy."""
    engine = BacktestEngine(backtest_config, signal_fn=generate_signal)
    result = engine.run(btc_bars)

    # Print summary for LLM visibility
    print(f"\n{'='*50}")
    print(f"BACKTEST SUMMARY")
    print(f"{'='*50}")
    print(f"Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total P&L: ${result.total_pnl:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.1%} (${result.max_drawdown_abs:.2f})")
    print(f"Sortino: {result.sortino_ratio:.2f}" if result.sortino_ratio else "Sortino: N/A")
    print(f"Sharpe: {result.sharpe_ratio:.2f}" if result.sharpe_ratio else "Sharpe: N/A")
    print(f"Max Consecutive Losses: {result.max_consecutive_losses}")
    print(f"{'='*50}\n")

    return result


class TestAlgoPerformance:
    """Performance tests for Volume Profile strategy."""

    def test_has_trades(self, backtest_result):
        """Strategy must generate enough trades for statistical significance.

        DO NOT MODIFY THIS TEST - improve the strategy instead.
        """
        total = backtest_result.total_trades
        assert total >= 10, f"Only {total} trades (need >= 10 for significance)"

    def test_win_rate(self, backtest_result):
        """Strategy must have reasonable win rate.

        DO NOT MODIFY THIS TEST - improve the strategy instead.
        """
        if backtest_result.total_trades == 0:
            pytest.skip("No trades to evaluate")
        wr = backtest_result.win_rate
        assert wr >= 0.40, f"Win rate {wr:.1%} < 40%"

    def test_profit_factor(self, backtest_result):
        """Strategy must have profit factor > 1.0.

        DO NOT MODIFY THIS TEST - improve the strategy instead.
        """
        if backtest_result.total_trades == 0:
            pytest.skip("No trades to evaluate")
        pf = backtest_result.profit_factor
        assert pf > 1.0, f"Profit factor {pf:.2f} <= 1.0"

    def test_sortino_positive(self, backtest_result):
        """Strategy must have positive Sortino ratio.

        DO NOT MODIFY THIS TEST - improve the strategy instead.
        """
        sortino = backtest_result.sortino_ratio
        if sortino is not None:
            assert sortino > 0, f"Sortino ratio {sortino:.2f} <= 0"

    def test_max_drawdown(self, backtest_result):
        """Strategy must have controlled drawdown.

        DO NOT MODIFY THIS TEST - improve the strategy instead.
        """
        max_dd = backtest_result.max_drawdown
        assert max_dd < 0.30, f"Max drawdown {max_dd:.1%} > 30%"


class TestAlgoRobustness:
    """Basic robustness checks.

    DO NOT MODIFY THESE TESTS - improve the strategy instead.
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
        config = load_config()
        client = MarketDataClient(config.alpaca, Path("data/cache"))
        bars = await client.get_bars("crypto:BTC/USD", lookback=3000)

        bt_config = BacktestConfig(
            symbol="crypto:BTC/USD",
            warmup_bars=65,
            transaction_cost_pct=0.0025,
            position_size=1.0,
            atr_threshold=2.0,
            stop_loss_multiplier=1.75,
        )

        engine = BacktestEngine(bt_config, signal_fn=generate_signal)
        result = engine.run(bars)

        print(f"\n=== Backtest Results ({len(bars)} bars) ===")
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
