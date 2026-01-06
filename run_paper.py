#!/usr/bin/env python3
"""Run live trading in paper mode."""

import asyncio
import logging
import signal
import sys

from trdr.core import Duration, Symbol, Timeframe
from trdr.live import LiveConfig, LiveHarness
from trdr.strategy.volume_area_breakout import VolumeAreaBreakoutStrategy
from trdr.strategy.volume_area_breakout.strategy import VolumeAreaBreakoutConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run paper trading."""
    # Load config from env
    config = LiveConfig.from_env(mode="paper")

    errors = config.validate()
    if errors:
        for err in errors:
            logger.error(f"Config error: {err}")
        sys.exit(1)

    # Ensure symbol has asset class prefix
    symbol_str = config.symbol
    if ":" not in symbol_str:
        # Default to crypto if has slash (e.g., ETH/USD), else stock
        if "/" in symbol_str:
            symbol_str = f"crypto:{symbol_str}"
        else:
            symbol_str = f"stock:{symbol_str}"

    logger.info(f"Mode: {config.mode}")
    logger.info(f"Symbol: {symbol_str}")
    logger.info(f"Timeframe: {config.timeframe}")
    logger.info(f"Poll interval: {config.poll_interval_seconds}s")

    # Update config with full symbol
    config.symbol = symbol_str

    # Create strategy
    strategy_config = VolumeAreaBreakoutConfig(
        symbol=Symbol.parse(symbol_str),
        timeframe=Timeframe.parse(config.timeframe),
        lookback=Duration.parse("1M"),
    )
    strategy = VolumeAreaBreakoutStrategy(strategy_config)

    # Create harness
    harness = LiveHarness(
        config=config,
        strategy=strategy,
        on_signal=lambda s: logger.info(f"Signal: {s}"),
        on_error=lambda e: logger.error(f"Error: {e}"),
    )

    # Handle shutdown
    def shutdown(sig: int, frame: object) -> None:
        logger.info("Shutdown requested...")
        asyncio.create_task(harness.stop())

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Run
    logger.info("Starting paper trading...")
    await harness.start()
    logger.info("Stopped.")


if __name__ == "__main__":
    asyncio.run(main())
