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
- ✅ All 6 order types: MARKET, LIMIT, STOP, STOP_LIMIT, TRAILING_STOP, TRAILING_STOP_LIMIT
- ✅ Exchange semantics enforced (directional validation)
- ✅ Portfolio with cash + positions MTM
- ✅ Trading calendar (filter by asset type)
- ✅ Slippage (% based) and transaction costs
- ✅ Equity curve tracking
- ✅ RuntimeContext for adaptive strategies
- ✅ 65 tests passing

## Requirements

### 1. Order Types ✅ DONE

- ✅ MARKET - immediate fill at bar open + slippage
- ✅ LIMIT - fill at limit price (no slippage)
- ✅ STOP - triggers when bar range touches stop, fills at stop + slippage
- ✅ STOP_LIMIT - triggers at stop, then acts as limit order (two-phase)
- ✅ TRAILING_STOP - follows price, exits on reversal + slippage
- ✅ TRAILING_STOP_LIMIT - follows price, exits as limit order (no slippage)
- ✅ Exchange semantics enforced (buy stop above, sell stop below, etc.)
- ✅ OCO behavior (stop loss/take profit cancel each other)

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

### Phase 7: Advanced Order Types ✅ DONE

Implement exchange-compliant order types with proper semantics.

#### 7.1 Order Type Consolidation

Consolidated redundant order types into standard exchange semantics:

| Order Type | Trigger | Fill Price | Use Case |
| --- | --- | --- | --- |
| MARKET | Immediately | Bar open + slippage | Immediate entry/exit |
| LIMIT | Price reaches limit | Limit price (no slippage) | Entry at target price |
| STOP | Bar range touches stop | Stop price + slippage | Stop loss |
| STOP_LIMIT | Bar range touches stop | Limit price (no slippage) | Entry on breakout |
| TRAILING_STOP | Trail touched after tracking | Stop price + slippage | Dynamic stop loss |
| TRAILING_STOP_LIMIT | Trail touched after tracking | Limit price (no slippage) | Dynamic with limit |

**Removed:** `STOP_LOSS` and `TAKE_PROFIT` - these are just STOP and LIMIT with proper placement.

#### 7.2 Exchange Semantics Enforcement

Directional validation enforced in `paper_exchange.py`:

```python
def _validate_order_direction(side, order_type, price, stop_price, limit_price) -> str | None:
    """Validate order direction matches exchange semantics."""
    # Buy stop/stop-limit: stop_price must be > current price
    # Sell stop/stop-limit: stop_price must be < current price
    # Buy limit: limit_price must be < current price
    # Sell limit: limit_price must be > current price
```

Signal fields mapped correctly:

- `stop_loss` → STOP order (sell stop below price, triggers on drop)
- `take_profit` → LIMIT order (sell limit above price, fills at limit, no slippage)
- `trailing_stop` → TRAILING_STOP order (tracks highs, exits on reversal)

#### 7.3 STOP_LIMIT Two-Phase Execution

Stop-limit orders have proper two-phase behavior:

1. **Phase 1 (Dormant):** Wait for bar range to touch stop_price, then set `triggered=True`
2. **Phase 2 (Active):** Act as limit order - may fill immediately or hang around

```python
# On same bar: can trigger AND fill if both conditions met
# On later bars: triggered order stays pending until limit fills
if not order.triggered:
    if bar.low <= order.stop_price <= bar.high:
        order.triggered = True
    else:
        return None  # Still waiting for trigger

# Now act as limit order
if order.side == "buy" and bar.low <= order.limit_price:
    fill_price = min(order.limit_price, bar.open)
```

#### 7.4 TRAILING_STOP_LIMIT

New order type combining trailing behavior with limit execution:

- Trails like TRAILING_STOP (ratchets with market)
- When triggered, becomes limit order (no slippage, possible fill risk)

```python
Order(
    side="sell",
    order_type=OrderType.TRAILING_STOP_LIMIT,
    trail_percent=0.02,  # 2% trail
    limit_price=105.0,   # Fills at 105 or better when triggered
)
```

#### 7.5 Files Changed

| File | Change |
| --- | --- |
| `src/trdr/backtest/orders.py` | Consolidated OrderType enum, added TRAILING_STOP_LIMIT, two-phase STOP_LIMIT logic |
| `src/trdr/backtest/paper_exchange.py` | Added `_validate_order_direction()`, take_profit uses LIMIT |
| `src/trdr/backtest/STRATEGY_API.md` | Documented order types and exchange semantics |
| `tests/test_orders.py` | 44 tests for all order types and scenarios |
| `tests/test_paper_exchange.py` | 21 tests including direction validation |

#### 7.6 Test Coverage

**Order Manager Tests (44):**

- STOP triggers when bar range includes stop_price
- STOP_LIMIT triggers then fills (same bar)
- STOP_LIMIT triggers, hangs around, fills later bar
- TRAILING_STOP tracks highs/lows, triggers on reversal
- TRAILING_STOP_LIMIT trails, triggers, fills at limit
- TRAILING_STOP_LIMIT triggers, hangs around, fills later bar
- No slippage on limit orders

**Paper Exchange Tests (21):**

- Stop loss below price works
- Take profit above price works (via LIMIT)
- OCO behavior (one fills, cancels other)
- Direction validation rejects invalid orders

### Phase 8: OTO (One-Triggers-Other) ⏳ FUTURE

True OTO where child orders are only submitted when parent fills.

#### Current Behavior

Exit orders (stop_loss, take_profit, trailing_stop) are submitted immediately with entry order. Works for market entries, but wrong for limit/stop-limit entries that may never fill.

#### Proper OTO Behavior

1. Submit entry order only
2. On entry fill, automatically submit child orders (SL/TP)
3. If entry never fills (limit order), children never get submitted
4. If entry cancelled, children never get submitted

#### Implementation Options

**Option A: Child Orders on Order**

```python
@dataclass
class Order:
    ...
    child_orders: list[Order] = field(default_factory=list)
```

OrderManager submits children only when parent fills.

**Option B: Order Groups**

```python
@dataclass
class OrderGroup:
    parent: Order
    children: list[Order]
    submit_children_on_fill: bool = True
```

**Option C: Conditional Logic in Paper Exchange**

Defer SL/TP submission until entry fill detected. Simplest but less reusable.

#### Use Cases

- Limit entry with bracket: submit limit buy at 98, only create SL/TP if filled
- Stop-limit breakout: submit stop-limit at 105, only create SL/TP if triggered and filled
- Cancel-if-not-filled: if entry expires, no orphaned exit orders

## Migration

Backward compatible:

- Existing strategies work unchanged (context is optional to use)
- Signal without new fields works as before
- PaperExchangeResult API unchanged (just refactored internally)
- `stop_loss` and `take_profit` Signal fields still work (mapped to correct order types internally)
