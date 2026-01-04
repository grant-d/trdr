"""Trading strategy implementations."""

from .base_strategy import BaseStrategy, StrategyConfig
from ..indicators import (
    VolumeProfile,
    atr,
    bollinger_bands,
    ema,
    ema_series,
    heikin_ashi,
    hma,
    hma_slope,
    hvn_support_strength,
    macd,
    mss,
    multi_timeframe_poc,
    order_flow_imbalance,
    rsi,
    sax_bullish_reversal,
    sax_pattern,
    sma,
    volatility_regime,
    volume_profile,
    volume_trend,
    wma,
)
from .macd_template import MACDConfig, MACDStrategy
from .types import Position, Signal, SignalAction
from .utils import get_backtest_env
from .volume_area_breakout import VolumeAreaBreakoutConfig, VolumeAreaBreakoutStrategy

__all__ = [
    # Base
    "BaseStrategy",
    "StrategyConfig",
    # Indicators - Moving Averages
    "sma",
    "ema",
    "ema_series",
    "wma",
    "hma",
    "hma_slope",
    # Indicators - Volatility
    "atr",
    "bollinger_bands",
    "volatility_regime",
    # Indicators - Momentum
    "rsi",
    "macd",
    "mss",
    # Indicators - Volume
    "volume_profile",
    "volume_trend",
    "order_flow_imbalance",
    "multi_timeframe_poc",
    "hvn_support_strength",
    # Indicators - Pattern Recognition
    "sax_pattern",
    "sax_bullish_reversal",
    "heikin_ashi",
    # Strategies
    "MACDConfig",
    "MACDStrategy",
    "VolumeAreaBreakoutConfig",
    "VolumeAreaBreakoutStrategy",
    # Types
    "Position",
    "Signal",
    "SignalAction",
    "VolumeProfile",
    # Utils
    "get_backtest_env",
]
