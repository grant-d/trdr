"""Strategy test utilities for reading backtest configuration from env vars."""

import os

from dotenv import load_dotenv

from trdr.backtest import parse_timeframe

load_dotenv()


def get_backtest_env(
    default_symbol: str = "crypto:BTC/USD",
    default_timeframe: str = "1h",
    default_lookback: int = 1000,
) -> tuple[str, str, "TimeFrame", int]:
    """Read backtest config from environment variables.

    Reads BACKTEST_SYMBOL, BACKTEST_TIMEFRAME, BACKTEST_LOOKBACK env vars.

    Args:
        default_symbol: Default symbol if env var not set
        default_timeframe: Default timeframe if env var not set
        default_lookback: Default lookback if env var not set

    Returns:
        Tuple of (symbol, timeframe_str, timeframe, lookback)
    """
    from alpaca.data.timeframe import TimeFrame

    symbol = os.environ.get("BACKTEST_SYMBOL", default_symbol)
    timeframe_str = os.environ.get("BACKTEST_TIMEFRAME", default_timeframe).lower().strip()
    timeframe = parse_timeframe(timeframe_str)
    lookback = int(os.environ.get("BACKTEST_LOOKBACK", str(default_lookback)))

    return symbol, timeframe_str, timeframe, lookback
