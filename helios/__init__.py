"""
Helios Trading Analysis Toolkit

A comprehensive trading analysis system featuring:
- Dollar bars data processing
- Market State Score (MSS) calculation
- Technical indicators (MACD, RSI, etc.)
- Regime-based trading strategies
- Performance evaluation metrics
"""

__version__ = "0.1.0"

from .data_processing import create_dollar_bars, prepare_data, calculate_atr

from .factors import (
    calculate_macd,
    calculate_rsi,
    calculate_stddev,
    calculate_trend_factor,
    calculate_volatility_factor,
    calculate_exhaustion_factor,
    calculate_mss,
    calculate_dynamic_weights,
)

from .strategy_enhanced import EnhancedTradingStrategy, Action, Position, Trade

from .performance import (
    calculate_returns_metrics,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    evaluate_strategy_performance,
    compare_strategies,
    calculate_rolling_metrics,
    generate_performance_report,
)

from .ranges import (
    ParameterRange,
    MinMaxRange,
    LogRange,
    DiscreteRange,
    create_log_range,
    create_discrete_range,
)

from .optimization import (
    WalkForwardOptimizer,
    auto_detect_dollar_thresholds,
    create_enhanced_parameter_ranges,
)

from .generic_algorithm import (
    Individual,
    GeneticAlgorithm,
    estimate_asset_volatility_scale,
)

from .utils import safe_prod

__all__ = [
    # Data processing
    "create_dollar_bars",
    "prepare_data",
    "calculate_atr",
    # Factors
    "calculate_macd",
    "calculate_rsi",
    "calculate_stddev",
    "calculate_trend_factor",
    "calculate_volatility_factor",
    "calculate_exhaustion_factor",
    "calculate_mss",
    "calculate_dynamic_weights",
    # Strategy
    "Action",
    "Position",
    "Trade",
    "EnhancedTradingStrategy",
    # Performance
    "calculate_returns_metrics",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_calmar_ratio",
    "evaluate_strategy_performance",
    "compare_strategies",
    "calculate_rolling_metrics",
    "generate_performance_report",
    # Ranges
    "ParameterRange",
    "MinMaxRange",
    "LogRange",
    "DiscreteRange",
    "create_log_range",
    "create_discrete_range",
    # Optimization
    "WalkForwardOptimizer",
    "auto_detect_dollar_thresholds",
    "create_enhanced_parameter_ranges",
    # Genetic Algorithm
    "Individual",
    "GeneticAlgorithm",
    "estimate_asset_volatility_scale",
    # Utils
    "safe_prod",
]
