---
title: RuntimeContext - Live Portfolio Access in Strategies
date: 2026-01-04
category: feature-implementation
tags:
  - backtest
  - metrics
  - runtime-context
  - strategy
  - adaptive-trading
  - paper-exchange
severity: N/A
component: backtest/paper_exchange
status: resolved
commit: 2d33a0d
---

# RuntimeContext: Live Portfolio Access in Strategies

## Overview

This feature enables trading strategies to access live portfolio state during signal generation, allowing adaptive behavior based on performance metrics like drawdown, win rate, and P&L.

## Problem Statement

Strategies needed to adapt dynamically based on backtest performance:

- Pause trading during high drawdown
- Reduce position size after losing streaks
- Access live metrics (equity, P&L, trade counts)

Previously, strategies had no visibility into portfolio state - they could only see bars and current position.

## Solution

### Architecture

Three core components working together:

1. **TradeMetrics** (`src/trdr/backtest/metrics.py`) - Centralized metrics calculation
2. **RuntimeContext** (`src/trdr/backtest/paper_exchange.py`) - Live state accessor
3. **BaseStrategy.context** (`src/trdr/strategy/base_strategy.py`) - Strategy integration

### TradeMetrics Class

Computes all trading metrics from trades and equity curve. Used by both RuntimeContext (live) and PaperExchangeResult (final):

```python
class TradeMetrics:
    """Computes trading metrics from trades and equity curve."""

    def __init__(
        self,
        trades: list[Trade],
        equity_curve: list[float],
        initial_capital: float,
        asset_type: str,
        start_time: str = "",
        end_time: str = "",
    ):
        ...

    @property
    def win_rate(self) -> float: ...
    @property
    def profit_factor(self) -> float: ...
    @property
    def max_drawdown(self) -> float: ...
    @property
    def sharpe_ratio(self) -> float | None: ...

    def current_drawdown(self, current_equity: float) -> float:
        """Current drawdown from peak as decimal [0.0, 1.0]."""
        ...
```

### RuntimeContext Class

Injected into strategy before each `generate_signal()` call:

```python
class RuntimeContext:
    """Live portfolio state available during signal generation."""

    # Run params
    @property
    def symbol(self) -> str: ...
    @property
    def bar_index(self) -> int: ...
    @property
    def bars_remaining(self) -> int: ...

    # Portfolio state
    @property
    def equity(self) -> float: ...
    @property
    def cash(self) -> float: ...
    @property
    def positions(self) -> dict: ...

    # Live metrics
    @property
    def drawdown(self) -> float: ...
    @property
    def win_rate(self) -> float: ...
    @property
    def profit_factor(self) -> float: ...
```

### Strategy Usage

```python
class MyStrategy(BaseStrategy):
    def generate_signal(self, bars: list[Bar], position: Position | None) -> Signal:
        # Pause during high drawdown
        if self.context.drawdown > 0.15:
            return Signal(action=SignalAction.HOLD, ...)

        # Reduce size after poor performance
        if self.context.win_rate < 0.4 and self.context.total_trades > 10:
            return Signal(..., position_size_pct=0.5)

        # Normal signal generation
        ...
```

## Implementation Details

### Key Bug Fixed: Negative Drawdown

The `current_drawdown()` method was returning negative values when equity exceeded the peak. Fixed with proper clamping:

```python
def current_drawdown(self, current_equity: float) -> float:
    """Current drawdown from peak as decimal."""
    if not self._equity_curve:
        return 0.0
    peak = max(self._equity_curve)
    if peak <= 0:
        return 0.0
    # Clamp to [0, 1] - drawdown is 0 at new highs, max 1.0 (100%)
    dd = (peak - current_equity) / peak
    return max(0.0, min(dd, 1.0))
```

### Strategy Name Feature

Added optional `name` parameter to BaseStrategy:

```python
class BaseStrategy(ABC):
    def __init__(self, config: StrategyConfig, name: str | None = None):
        self.config = config
        self._name = name

    @property
    def name(self) -> str:
        """Returns custom name if set, else class name."""
        return self._name if self._name else self.__class__.__name__
```

Accessible via `self.context.strategy_name` in strategies.

### Circular Reference Prevention

RuntimeContext receives `strategy_name` as string, not the strategy object itself:

```python
# In PaperExchange.run():
self.strategy.context = RuntimeContext(
    ...
    strategy_name=self.strategy.name,  # String, not object
)
```

## Files Changed

| File | Change |
| --- | --- |
| `src/trdr/backtest/metrics.py` | New - TradeMetrics class |
| `src/trdr/backtest/paper_exchange.py` | Add RuntimeContext, refactor Result |
| `src/trdr/backtest/__init__.py` | Export RuntimeContext |
| `src/trdr/strategy/base_strategy.py` | Add context attribute, name param |
| `src/trdr/backtest/STRATEGY_API.md` | Document RuntimeContext |
| `tests/test_metrics.py` | New - 27 tests |

## Testing

27 tests covering:

- TradeMetrics basic calculations
- Empty trades handling
- RuntimeContext run params, portfolio state, live metrics
- Current drawdown edge cases (at peak, new high, zero equity)
- Strategy name feature

## Prevention Strategies

### 1. Clamp Percentage Calculations

Always use `max(0.0, min(value, 1.0))` for percentages.

### 2. Document Side Effects

The Portfolio.open_position() deducts cash - this wasn't obvious. Document all mutations.

### 3. Avoid Circular References

Pass only primitives (strings, numbers) to context, never complex objects.

### 4. Compute On-Demand

Metrics are computed fresh each call - no caching of stale values.

## Cross-References

- Plan: `plans/feat-paper-exchange-backtest-engine.md`
- API Docs: `src/trdr/backtest/STRATEGY_API.md`
- Tests: `tests/test_metrics.py`
- Example Strategy: `src/trdr/strategy/macd_template/strategy.py`
