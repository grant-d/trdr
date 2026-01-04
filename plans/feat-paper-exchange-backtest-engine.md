# Feature: Paper Exchange Backtest Engine

Enhance the backtest/forward engine to act more like a paper exchange with advanced order types, position management, portfolio tracking, and live runtime context for adaptive strategies.

## Design Principles

1. **Simple Strategy Interface** - LLM writes `generate_signal(bars, position) → Signal`. Nothing else.
2. **Data External** - Engine receives bars, doesn't fetch. Same engine for backtest + paper trading.
3. **Complexity Hidden** - Orders, portfolio, calendar handled by engine internals.
4. **Backward Compatible** - Existing strategies work unchanged.
5. **Adaptive Strategies** - Strategy can access live portfolio state via `self.context` for dynamic sizing/behavior.

## Current State (Completed)

`src/trdr/backtest/paper_exchange.py`:

- ✅ PaperExchange replaces BacktestEngine
- ✅ Market orders, Stop Loss, Trailing Stop orders
- ✅ Portfolio with cash + positions MTM
- ✅ Trading calendar (filter by asset type)
- ✅ Slippage (% based) and transaction costs
- ✅ Equity curve tracking

## Requirements

### 1. Order Types (KISS) ✅ DONE

- ✅ Market (existing)
- ✅ Stop Loss (exit on price breach)
- ✅ Trailing Stop Loss (TSL) - follows price, exits on reversal

### 2. Position Management ✅ DONE

- ✅ Multiple entries on same symbol (buy 1 ETH, later buy 2 more)
- ✅ Partial exits (sell 0.5 of 3 ETH position)
- ✅ Track each entry separately for accurate P&L
- ✅ Max position limits (via position_size_pct)

### 3. Portfolio Tracking ✅ DONE

- ✅ Cash balance
- ✅ Equity (cash + positions MTM)
- ✅ Liquidate at end of data (close all positions at market)

### 4. Market Simulation ✅ DONE

- ✅ Slippage (% based)
- ✅ Transaction costs (%)
- ✅ Trading calendar (skip weekends/holidays for stocks)

### 5. RuntimeContext for Adaptive Strategies ✅ DONE

Strategy should access live portfolio state via `self.context` during `generate_signal()`.

**Run Params:**

- `context.symbol` - Trading symbol
- `context.current_bar` - Current Bar object
- `context.bar_index` - Current bar index (0-based)
- `context.total_bars` - Total bars in run
- `context.bars_remaining` - Bars left

**Portfolio State:**

- `context.equity` - Current portfolio value
- `context.cash` - Available cash
- `context.initial_capital` - Starting capital
- `context.positions` - Open positions dict
- `context.pending_orders` - Pending stop/limit orders
- `context.trades` - Completed Trade objects

**Live Metrics (computed on demand):**

- `context.drawdown` - Current drawdown %
- `context.max_drawdown` - Max drawdown %
- `context.total_return` - Total return %
- `context.total_pnl` - Net P&L from closed trades
- `context.win_rate` - Win rate
- `context.profit_factor` - Profits / losses
- `context.expectancy` - Expected value per trade
- `context.sharpe_ratio` - Annualized Sharpe
- `context.sortino_ratio` - Annualized Sortino
- `context.calmar_ratio` - CAGR / max drawdown
- `context.cagr` - Compound annual growth rate

### 6. Centralized Metrics (DRY) ✅ DONE

Extract duplicate metric calculations into shared `TradeMetrics` class:

- Used by both `RuntimeContext` (live) and `PaperExchangeResult` (final)
- Single source of truth for: win_rate, profit_factor, sharpe, sortino, cagr, etc.

## Implementation Plan

### Phase 1-4: Core Engine ✅ DONE

Orders, portfolio, calendar, and market simulation are complete.
See `src/trdr/backtest/` for implementations.

### Phase 5: RuntimeContext & Centralized Metrics ✅ DONE

Enable strategies to access live portfolio state for adaptive behavior.

**Completed in commit `2d33a0d`:**

- TradeMetrics class with all metric calculations
- RuntimeContext with run params, portfolio state, and live metrics
- BaseStrategy with `context` attribute and optional `name` parameter
- PaperExchangeResult delegates to TradeMetrics
- Strategy name accessible via `self.context.strategy_name`

#### 5.1 TradeMetrics Class

```python
# src/trdr/backtest/metrics.py

class TradeMetrics:
    """Computes trading metrics from trades and equity curve.

    Used by both RuntimeContext (live) and PaperExchangeResult (final).
    """

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

    # Properties: total_trades, win_rate, profit_factor, max_drawdown,
    # sharpe_ratio, sortino_ratio, cagr, calmar_ratio, expectancy, etc.

    def current_drawdown(self, current_equity: float) -> float:
        """Current drawdown from peak as decimal."""
        ...

    def cagr_live(self, current_equity: float, current_time: str) -> float | None:
        """CAGR computed from start to current time."""
        ...
```

#### 5.2 RuntimeContext Class

```python
# src/trdr/backtest/paper_exchange.py

class RuntimeContext:
    """Live portfolio state available to strategy during generate_signal().

    Example:
        def generate_signal(self, bars, position):
            if self.context.drawdown > 0.1:
                return Signal(action=SignalAction.HOLD, ...)  # pause during drawdown
            if self.context.win_rate < 0.4 and self.context.total_trades > 10:
                return Signal(..., position_size_pct=0.5)  # reduce size
    """

    def __init__(
        self,
        portfolio: Portfolio,
        order_manager: OrderManager,
        trades: list[Trade],
        equity_curve: list[float],
        config: PaperExchangeConfig,
        current_bar: Bar,
        bar_index: int,
        total_bars: int,
        start_time: str,
    ):
        ...

    # Run params
    @property
    def symbol(self) -> str: ...
    @property
    def current_bar(self) -> Bar: ...
    @property
    def bar_index(self) -> int: ...
    @property
    def total_bars(self) -> int: ...
    @property
    def bars_remaining(self) -> int: ...

    # Portfolio state
    @property
    def equity(self) -> float: ...
    @property
    def cash(self) -> float: ...
    @property
    def initial_capital(self) -> float: ...
    @property
    def positions(self) -> dict: ...
    @property
    def pending_orders(self) -> list[Order]: ...
    @property
    def trades(self) -> list[Trade]: ...

    # Live metrics (delegated to TradeMetrics)
    @property
    def drawdown(self) -> float: ...
    @property
    def max_drawdown(self) -> float: ...
    @property
    def total_return(self) -> float: ...
    @property
    def win_rate(self) -> float: ...
    @property
    def profit_factor(self) -> float: ...
    @property
    def expectancy(self) -> float: ...
    @property
    def sharpe_ratio(self) -> float | None: ...
    @property
    def sortino_ratio(self) -> float | None: ...
    @property
    def cagr(self) -> float | None: ...
    @property
    def calmar_ratio(self) -> float | None: ...
```

#### 5.3 BaseStrategy Update

```python
# src/trdr/strategy/base_strategy.py

class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    Attributes:
        config: Strategy configuration
        context: Live portfolio state (set by engine before generate_signal)

    Example:
        class MyStrategy(BaseStrategy):
            def generate_signal(self, bars, position):
                # Access live portfolio state
                if self.context.drawdown > 0.1:
                    return Signal(action=SignalAction.HOLD, ...)
                if self.context.total_trades > 10 and self.context.win_rate < 0.4:
                    return Signal(..., position_size_pct=0.5)
    """

    context: "RuntimeContext"  # Set by engine before generate_signal()
```

#### 5.4 PaperExchange Integration

```python
# In PaperExchange.run():
for i in range(self.config.warmup_bars, len(filtered_bars)):
    # ... process fills ...

    # Set runtime context for strategy
    self.strategy.context = RuntimeContext(
        portfolio=portfolio,
        order_manager=order_manager,
        trades=trades,
        equity_curve=equity_curve,
        config=self.config,
        current_bar=bar,
        bar_index=i - self.config.warmup_bars,
        total_bars=len(filtered_bars) - self.config.warmup_bars,
        start_time=filtered_bars[self.config.warmup_bars].timestamp,
    )

    signal = self.strategy.generate_signal(visible_bars, strategy_position)
```

#### 5.5 Refactor PaperExchangeResult

Delegate all metrics to TradeMetrics:

```python
@dataclass
class PaperExchangeResult:
    trades: list[Trade]
    config: PaperExchangeConfig
    start_time: str
    end_time: str
    equity_curve: list[float] = field(default_factory=list)
    _metrics: TradeMetrics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_metrics", TradeMetrics(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_capital=self.config.initial_capital,
            asset_type=self.config.asset_type,
            start_time=self.start_time,
            end_time=self.end_time,
        ))

    @property
    def win_rate(self) -> float:
        return self._metrics.win_rate
    # ... delegate all other metrics ...
```

### Phase 6: Testing ✅ DONE

**Completed:** 27 tests in `tests/test_metrics.py` covering all scenarios.

#### 6.1 Unit Tests

| Test | Description |
| --- | --- |
| `test_trade_metrics_basic` | TradeMetrics computes correct values |
| `test_trade_metrics_empty` | Handles empty trades list |
| `test_runtime_context_run_params` | symbol, bar_index, total_bars correct |
| `test_runtime_context_portfolio_state` | equity, cash, positions correct |
| `test_runtime_context_live_metrics` | drawdown, win_rate, etc. computed on demand |
| `test_runtime_context_drawdown` | Current drawdown from peak |
| `test_paper_exchange_result_delegates` | Result delegates to TradeMetrics |
| `test_strategy_accesses_context` | Strategy can use self.context |

#### 6.2 Integration Tests

| Test | Description |
| --- | --- |
| `test_adaptive_strategy` | Strategy reduces size during drawdown |
| `test_context_updates_each_bar` | Context reflects current state |
| `test_context_metrics_accurate` | Metrics match final result |

## File Changes

| File | Change |
| --- | --- |
| `src/trdr/backtest/metrics.py` | New - TradeMetrics class |
| `src/trdr/backtest/paper_exchange.py` | Add RuntimeContext, refactor Result |
| `src/trdr/backtest/__init__.py` | Export RuntimeContext |
| `src/trdr/strategy/base_strategy.py` | Add context attribute |
| `src/trdr/backtest/STRATEGY_API.md` | Document RuntimeContext |
| `tests/test_metrics.py` | New - TradeMetrics tests |
| `tests/test_runtime_context.py` | New - RuntimeContext tests |

## Migration

Backward compatible:

- Existing strategies work unchanged (context is optional to use)
- Signal without new fields works as before
- PaperExchangeResult API unchanged (just refactored internally)
