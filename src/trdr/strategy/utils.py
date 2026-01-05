"""Strategy test utilities for reading backtest configuration from env vars."""

import os

from dotenv import load_dotenv

from trdr.core import Duration, Symbol, Timeframe

load_dotenv()


def get_backtest_env(
    default_symbol: Symbol,
    default_timeframe: Timeframe,
    default_lookback: Duration,
) -> tuple[Symbol, Timeframe, Duration]:
    """Read backtest config from environment variables.

    Reads BACKTEST_SYMBOL, BACKTEST_TIMEFRAME, BACKTEST_LOOKBACK env vars.
    Supports arbitrary timeframes (60m, 3d, 2w).

    Args:
        default_symbol: Default symbol if env var not set
        default_timeframe: Default timeframe if env var not set
        default_lookback: Default lookback if env var not set

    Returns:
        Tuple of (symbol, timeframe, lookback)
    """
    symbol_str = os.environ.get("BACKTEST_SYMBOL")
    symbol = Symbol.parse(symbol_str) if symbol_str else default_symbol

    timeframe_str = os.environ.get("BACKTEST_TIMEFRAME")
    timeframe = Timeframe.parse(timeframe_str) if timeframe_str else default_timeframe

    lookback_str = os.environ.get("BACKTEST_LOOKBACK")
    lookback = Duration.parse(lookback_str) if lookback_str else default_lookback

    return symbol, timeframe, lookback
