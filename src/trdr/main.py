"""Main entry point for trdr CLI."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")


def _build_strategy_config(config_cls, live_config):
    """Build a strategy config from live config."""
    from .core import Duration, Symbol, Timeframe

    if not live_config.symbol:
        raise ValueError("Live config missing symbol")

    return config_cls(
        symbol=Symbol.parse(live_config.symbol),
        timeframe=Timeframe.parse(live_config.timeframe),
        lookback=Duration.parse("30d"),
    )


def run_live(args: argparse.Namespace) -> None:
    """Run live trading with new harness."""
    from .core import Symbol
    from .live.config import LiveConfig
    from .live.harness import LiveHarness

    # Build live config first to get symbol
    config = LiveConfig.from_env()
    if args.symbol:
        config.symbol = args.symbol

    # Validate symbol format
    try:
        Symbol.parse(config.symbol)
    except ValueError:
        print(f"Invalid symbol format: {config.symbol}")
        print("Expected format: <type>:<SYMBOL> (e.g., crypto:ETH/USD, stock:AAPL)")
        sys.exit(1)

    # Import and instantiate strategy
    strategy_map = {
        "grid": "trdr.strategy.trailing_grid.strategy.TrailingGridStrategy",
        "trailing_grid": "trdr.strategy.trailing_grid.strategy.TrailingGridStrategy",
        "vab": "trdr.strategy.volume_area_breakout.strategy.VolumeAreaBreakoutStrategy",
        "volume_area_breakout": (
            "trdr.strategy.volume_area_breakout.strategy.VolumeAreaBreakoutStrategy"
        ),
    }
    strategy_path = strategy_map.get(args.strategy, args.strategy)
    import importlib

    module_path, class_name = strategy_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    strategy_class = getattr(module, class_name)

    config_class_name = class_name.replace("Strategy", "Config")
    config_cls = getattr(module, config_class_name, None)
    if config_cls is None:
        raise ValueError(f"Strategy config class not found: {module_path}.{config_class_name}")
    try:
        strategy_config = _build_strategy_config(config_cls, config)
        strategy = strategy_class(strategy_config)
    except TypeError as exc:
        print("Strategy config error:", exc)
        print("This strategy requires custom config fields.")
        print("Use a dedicated runner or provide a custom live wrapper.")
        sys.exit(1)

    # Validate
    errors = config.validate()
    if errors:
        print(f"Config errors: {errors}")
        sys.exit(1)

    print(f"Starting live trading: {config.symbol}")
    print(f"Mode: {'PAPER' if config.is_paper else 'LIVE'}")
    print(f"Strategy: {strategy.name}")

    harness = LiveHarness(config, strategy)

    if args.ui:
        harness.run_with_ui()
    else:
        asyncio.run(harness.start())


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="TRDR Trading Bot")
    parser.add_argument("--ui", action="store_true", help="Enable TUI dashboard")
    parser.add_argument("--symbol", type=str, help="Trading symbol (e.g., crypto:ETH/USD)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="trailing_grid",
        help="Strategy name or full path (default: trailing_grid)",
    )

    args = parser.parse_args()
    run_live(args)


if __name__ == "__main__":
    main()
