"""Backtest tests for Mean Reversion strategy.

Strategy: Statistical mean reversion with z-score and consecutive day entries.
Tests MeanReversionStrategy from trdr.strategy.

Run with:
  .venv/bin/python -m pytest src/trdr/strategy/mean_reversion/test_strategy.py -v
  BACKTEST_SYMBOL=crypto:BTC/USD BACKTEST_TIMEFRAME=1d .venv/bin/python -m pytest ... -v
"""

import asyncio
from pathlib import Path

import pytest

from trdr.backtest import PaperExchange, PaperExchangeConfig, PaperExchangeResult
from trdr.core import Duration, Feed, Symbol, Timeframe, load_config
from trdr.data import AlpacaDataClient
from trdr.strategy import get_backtest_env
from trdr.strategy.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from trdr.strategy.targets import score_result

# Read env vars once at module load
SYMBOL, TIMEFRAME, LOOKBACK = get_backtest_env(
    default_symbol=Symbol.parse("crypto:BTC/USD"),
    default_timeframe=Timeframe.parse("1d"),
    default_lookback=Duration.parse("300d"),
)


@pytest.fixture(scope="function")
def event_loop():
    """Create event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def bars(event_loop):
    """Fetch historical bars for backtesting."""

    async def fetch():
        config = load_config()
        client = AlpacaDataClient(config.alpaca, Path("data/cache"))
        bars = await client.get_bars(SYMBOL, lookback=3000, timeframe=TIMEFRAME)
        return bars

    return event_loop.run_until_complete(fetch())


@pytest.fixture(scope="function")
def strategy():
    """Create strategy instance with config."""
    config = MeanReversionConfig(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        lookback=Duration.parse("3y"),
        lookback_period=20,
        zscore_entry=1.0,
        zscore_exit=0.0,
        consecutive_down_days=2,
        use_calendar=True,
        stop_loss_atr_mult=2.0,
        max_holding_days=10,
    )
    return MeanReversionStrategy(config)


@pytest.fixture(scope="function")
def backtest_config():
    """Paper exchange configuration."""
    return PaperExchangeConfig(
        primary_feed=Feed(SYMBOL, TIMEFRAME),
        warmup_bars=30,
        transaction_cost_pct=0.0025,
        slippage_pct=0.01,
        default_position_pct=0.5,
    )


@pytest.fixture(scope="function")
def backtest_result(bars, backtest_config, strategy) -> PaperExchangeResult:
    """Run single backtest with Mean Reversion strategy."""
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


@pytest.mark.slow
class TestAlgoPerformance:
    """Sanity checks for Mean Reversion strategy."""

    def test_has_trades(self, backtest_result):
        """Strategy generates trades."""
        total = backtest_result.total_trades
        assert total >= 1, "No trades generated"

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
        assert sortino is None or isinstance(sortino, (int, float))

    def test_max_drawdown_valid(self, backtest_result):
        """Max drawdown is valid (0-1 range)."""
        max_dd = backtest_result.max_drawdown
        assert 0.0 <= max_dd <= 1.0, f"Max drawdown {max_dd:.1%} out of range"


@pytest.mark.slow
class TestAlgoRobustness:
    """Basic robustness checks."""

    def test_pnl_computed(self, backtest_result):
        """P&L is computed (any value valid)."""
        if backtest_result.total_trades == 0:
            pytest.skip("No trades to evaluate")
        pnl = backtest_result.total_pnl
        assert isinstance(pnl, (int, float)), "P&L not computed"


def print_results():
    """Helper to print detailed results."""

    async def run():
        config = load_config()
        client = AlpacaDataClient(config.alpaca, Path("data/cache"))
        bars = await client.get_bars(SYMBOL, lookback=3000, timeframe=TIMEFRAME)

        strategy_config = MeanReversionConfig(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            lookback=Duration.parse("3y"),
            lookback_period=20,
            zscore_entry=1.0,
            consecutive_down_days=2,
        )
        strategy = MeanReversionStrategy(strategy_config)

        bt_config = PaperExchangeConfig(
            primary_feed=Feed(SYMBOL, TIMEFRAME),
            warmup_bars=30,
            transaction_cost_pct=0.0025,
            default_position_pct=0.5,
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
