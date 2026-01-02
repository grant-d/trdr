"""CLI entry point for backtesting.

Usage:
    python -m trdr.backtest --symbol crypto:BTC/USD --lookback 500
    python -m trdr.backtest --symbol AAPL --lookback 1000 --folds 5
    python -m trdr.backtest --symbol crypto:BTC/USD --output results.json
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from ..core import load_config
from ..data import MarketDataClient
from .engine import BacktestConfig, BacktestEngine
from .walk_forward import WalkForwardConfig, run_walk_forward


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backtest trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m trdr.backtest --symbol crypto:BTC/USD --lookback 500
  python -m trdr.backtest --symbol AAPL --lookback 1000 --folds 5
  python -m trdr.backtest --symbol crypto:BTC/USD --output results.json
        """,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Symbol to backtest (e.g., 'crypto:BTC/USD', 'AAPL')",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=500,
        help="Number of bars to fetch (default: 500)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=0,
        help="Number of walk-forward folds. 0 = single backtest (default: 0)",
    )
    parser.add_argument(
        "--train-pct",
        type=float,
        default=0.70,
        help="Training data percentage for walk-forward (default: 0.70)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=65,
        help="Warmup bars before signals (default: 65)",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=1.0,
        help="Position size per trade (default: 1.0)",
    )
    parser.add_argument(
        "--atr-threshold",
        type=float,
        default=2.0,
        help="ATR threshold for signals (default: 2.0)",
    )
    parser.add_argument(
        "--stop-loss-mult",
        type=float,
        default=1.75,
        help="Stop loss ATR multiplier (default: 1.75)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: print to stdout)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


async def fetch_bars(symbol: str, lookback: int, verbose: bool) -> list:
    """Fetch historical bars from Alpaca."""
    try:
        config = load_config()
    except ValueError as e:
        print(f"Config error: {e}", file=sys.stderr)
        print("Ensure .env file exists with ALPACA_API_KEY and ALPACA_SECRET_KEY", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Fetching {lookback} bars for {symbol}...", file=sys.stderr)

    client = MarketDataClient(config.alpaca, Path("data/cache"))
    bars = await client.get_bars(symbol, lookback=lookback)

    if verbose:
        print(f"Fetched {len(bars)} bars", file=sys.stderr)
        if bars:
            print(f"  Period: {bars[0].timestamp} to {bars[-1].timestamp}", file=sys.stderr)

    return bars


def get_transaction_cost(symbol: str) -> float:
    """Get transaction cost for symbol type."""
    if symbol.startswith("crypto:"):
        return 0.0025  # 0.25% for crypto
    return 0.0  # 0% for stocks


def run_single_backtest(bars: list, config: BacktestConfig, verbose: bool) -> dict:
    """Run single backtest and return results dict."""
    if verbose:
        print(f"Running backtest with {len(bars)} bars...", file=sys.stderr)

    engine = BacktestEngine(config)
    result = engine.run(bars)

    if verbose:
        print(f"  Trades: {result.total_trades}", file=sys.stderr)
        print(f"  Win rate: {result.win_rate:.1%}", file=sys.stderr)
        print(f"  P&L: ${result.total_pnl:.2f}", file=sys.stderr)

    return result.to_dict()


def run_walkforward(
    bars: list,
    config: BacktestConfig,
    wf_config: WalkForwardConfig,
    verbose: bool,
) -> dict:
    """Run walk-forward validation and return results dict."""
    if verbose:
        msg = f"Running {wf_config.n_folds}-fold walk-forward with {len(bars)} bars..."
        print(msg, file=sys.stderr)

    result = run_walk_forward(bars, config, wf_config)

    if verbose:
        print(f"  Folds completed: {len(result.folds)}", file=sys.stderr)
        print(f"  Total trades: {result.total_trades}", file=sys.stderr)
        print(f"  Win rate: {result.win_rate:.1%}", file=sys.stderr)
        print(f"  Total P&L: ${result.total_pnl:.2f}", file=sys.stderr)
        if result.avg_sharpe:
            print(f"  Avg Sharpe: {result.avg_sharpe:.2f}", file=sys.stderr)

    return result.to_dict()


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Fetch data
    bars = await fetch_bars(args.symbol, args.lookback, args.verbose)

    if len(bars) < args.warmup + 20:
        min_bars = args.warmup + 20
        print(f"Error: Not enough bars. Need at least {min_bars}, got {len(bars)}", file=sys.stderr)
        sys.exit(1)

    # Create config
    transaction_cost = get_transaction_cost(args.symbol)
    backtest_config = BacktestConfig(
        symbol=args.symbol,
        warmup_bars=args.warmup,
        transaction_cost_pct=transaction_cost,
        position_size=args.position_size,
        atr_threshold=args.atr_threshold,
        stop_loss_multiplier=args.stop_loss_mult,
    )

    # Run backtest
    if args.folds > 0:
        wf_config = WalkForwardConfig(
            n_folds=args.folds,
            train_pct=args.train_pct,
        )
        result = run_walkforward(bars, backtest_config, wf_config, args.verbose)
    else:
        result = run_single_backtest(bars, backtest_config, args.verbose)

    # Add metadata
    result["meta"] = {
        "run_time": datetime.now().isoformat(),
        "bars_fetched": len(bars),
        "mode": "walk_forward" if args.folds > 0 else "single",
    }

    # Output
    output_json = json.dumps(result, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json)
        if args.verbose:
            print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    asyncio.run(main())
