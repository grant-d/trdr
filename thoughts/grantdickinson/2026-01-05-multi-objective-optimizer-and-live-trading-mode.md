# Multi-Objective Optimizer and Live Trading Mode

## Overview

Two interconnected features for the trdr trading bot:

1. **Multi-Objective Optimizer**: Optimize strategies across competing objectives (Sharpe, drawdown, win rate, profit factor) to generate Pareto-optimal parameter sets
2. **Live Trading Mode**: Poll-based execution harness that runs finalized strategies against real exchanges (Alpaca paper → live)

Both features maintain the existing plugin architecture where strategies implement `BaseStrategy` and remain agnostic to execution context (backtest vs live).

## Problem Statement

**Optimizer**: Current walk-forward system optimizes single objectives. Real trading requires balancing competing goals—high returns vs low drawdown, win rate vs profit factor. Need multi-objective optimization with Pareto frontier selection.

**Live Mode**: Strategies currently run only in backtest via `PaperExchange`. To deploy finalized strategies, need a harness (like `sica_bench` for benchmarking) that:

- Polls strategy at configured frequency
- Submits signals to real exchange
- Handles order lifecycle (fills, partials, retries)
- Provides RuntimeContext from live exchange state

## Proposed Solution

### Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        trdr/                                     │
├─────────────────────────────────────────────────────────────────┤
│  src/trdr/                                                       │
│  ├── strategy/           # Unchanged - plugin strategies         │
│  │   ├── base_strategy.py                                        │
│  │   └── types.py        # Signal, StrategyConfig                │
│  │                                                               │
│  ├── backtest/           # Existing                              │
│  │   ├── paper_exchange.py                                       │
│  │   ├── walk_forward.py                                         │
│  │   └── orders.py                                               │
│  │                                                               │
│  ├── optimize/           # NEW - Multi-objective optimizer       │
│  │   ├── __init__.py                                             │
│  │   ├── multi_objective.py    # NSGA-II/III via pymoo           │
│  │   ├── objectives.py         # Sharpe, drawdown, win_rate, etc │
│  │   ├── pareto.py             # Pareto frontier selection       │
│  │   └── walk_forward_moo.py   # WF + MOO integration            │
│  │                                                               │
│  ├── live/               # NEW - Live trading harness            │
│  │   ├── __init__.py                                             │
│  │   ├── harness.py            # Main orchestrator (like sica_bench) │
│  │   ├── exchange/                                               │
│  │   │   ├── __init__.py                                         │
│  │   │   ├── base.py           # ExchangeInterface ABC           │
│  │   │   ├── alpaca.py         # Alpaca implementation           │
│  │   │   └── adapter.py        # Backtest compat adapter         │
│  │   ├── orders/                                                 │
│  │   │   ├── __init__.py                                         │
│  │   │   ├── manager.py        # Order lifecycle management      │
│  │   │   ├── types.py          # LiveOrder, Fill, etc            │
│  │   │   └── retry.py          # Retry policies                  │
│  │   ├── state/                                                  │
│  │   │   ├── __init__.py                                         │
│  │   │   ├── reconciler.py     # State reconciliation            │
│  │   │   └── context.py        # Live RuntimeContext builder     │
│  │   ├── safety/                                                 │
│  │   │   ├── __init__.py                                         │
│  │   │   ├── circuit_breaker.py                                  │
│  │   │   └── risk_limits.py                                      │
│  │   └── config.py             # Live mode configuration         │
│  │                                                               │
│  └── core/                                                       │
│      └── config.py       # Extended for paper/live keys          │
└─────────────────────────────────────────────────────────────────┘
```

### Exchange Interface (Unified)

```python
# src/trdr/live/exchange/base.py
class ExchangeInterface(ABC):
    """Unified interface for paper (backtest) and live exchanges."""

    @abstractmethod
    async def get_account(self) -> AccountInfo: ...

    @abstractmethod
    async def get_positions(self) -> dict[str, Position]: ...

    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> OrderResponse: ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    async def get_order(self, order_id: str) -> OrderResponse: ...

    @abstractmethod
    async def get_bars(self, symbol: str, timeframe: str, limit: int) -> list[Bar]: ...
```

**Key Decision**: Alpaca supports bracket orders (entry + TP + SL) natively. The existing `PaperExchange` has OCO. Use adapter pattern to map between them.

### Harness Pattern

```python
# src/trdr/live/harness.py
class LiveHarness:
    """Orchestrates strategy execution against live exchange.

    Similar to sica_bench for benchmarking - loads strategy, runs poll loop,
    handles order lifecycle.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        exchange: ExchangeInterface,
        config: LiveConfig,
    ):
        self.strategy = strategy
        self.exchange = exchange
        self.config = config
        self.order_manager = OrderManager(exchange)
        self.circuit_breaker = CircuitBreaker(config.risk_limits)
        self.state_reconciler = StateReconciler(exchange)

    async def run(self) -> None:
        """Main poll loop."""
        await self._startup()

        while not self._should_stop():
            try:
                context = await self._build_context()
                signal = self.strategy.generate_signal(bars, position)

                if self.circuit_breaker.check(context, signal):
                    await self._execute_signal(signal)

                await asyncio.sleep(self.config.poll_interval_seconds)
            except Exception as e:
                await self._handle_error(e)

        await self._shutdown()
```

## Technical Approach

### Phase 1: Multi-Objective Optimizer

#### 1.1 Objective Functions

Define standard trading objectives in `src/trdr/optimize/objectives.py`:

| Objective | Direction | Calculation |
| --- | --- | --- |
| Sharpe Ratio | Maximize | (mean_return - risk_free) / std_return |
| Max Drawdown | Minimize | max(peak - trough) / peak |
| Win Rate | Maximize | winning_trades / total_trades |
| Profit Factor | Maximize | gross_profit / gross_loss |
| Sortino Ratio | Maximize | (mean_return - risk_free) / downside_std |
| Calmar Ratio | Maximize | annual_return / max_drawdown |

```python
# src/trdr/optimize/objectives.py
@dataclass
class ObjectiveResult:
    sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    sortino: float | None = None
    calmar: float | None = None

def calculate_objectives(result: PaperExchangeResult) -> ObjectiveResult:
    """Calculate all objectives from backtest result."""
    ...
```

#### 1.2 pymoo Integration

Use pymoo for NSGA-II (2-3 objectives) or NSGA-III (4+ objectives):

```python
# src/trdr/optimize/multi_objective.py
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem

class StrategyOptimizationProblem(Problem):
    def __init__(
        self,
        strategy_factory: Callable[[dict], BaseStrategy],
        bars: dict[str, list[Bar]],
        param_bounds: dict[str, tuple[float, float]],
        objectives: list[str],  # ["sharpe", "max_drawdown", "win_rate"]
    ):
        self.strategy_factory = strategy_factory
        self.bars = bars
        self.objectives = objectives
        self.param_names = list(param_bounds.keys())

        super().__init__(
            n_var=len(param_bounds),
            n_obj=len(objectives),
            xl=np.array([b[0] for b in param_bounds.values()]),
            xu=np.array([b[1] for b in param_bounds.values()]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        results = []
        for params in x:
            param_dict = dict(zip(self.param_names, params))
            strategy = self.strategy_factory(param_dict)
            bt_result = run_backtest(strategy, self.bars)
            obj_result = calculate_objectives(bt_result)
            results.append(self._extract_objectives(obj_result))

        out["F"] = np.array(results)
```

#### 1.3 Walk-Forward Integration

Run MOO on each training fold, aggregate Pareto frontiers:

```python
# src/trdr/optimize/walk_forward_moo.py
def walk_forward_moo(
    bars: dict[str, list[Bar]],
    strategy_factory: Callable,
    wf_config: WalkForwardConfig,
    moo_config: MOOConfig,
) -> WalkForwardMOOResult:
    """Run multi-objective optimization with walk-forward validation."""

    folds = generate_folds(bars, wf_config)
    fold_results = []

    for fold in folds:
        # Optimize on training data
        problem = StrategyOptimizationProblem(
            strategy_factory=strategy_factory,
            bars=fold.train_bars,
            param_bounds=moo_config.param_bounds,
            objectives=moo_config.objectives,
        )

        algorithm = NSGA2(pop_size=moo_config.population_size)
        res = minimize(problem, algorithm, ('n_gen', moo_config.generations))

        # Validate Pareto front on test data
        validated = validate_pareto_front(res.X, res.F, fold.test_bars)
        fold_results.append(validated)

    return aggregate_fold_results(fold_results)
```

#### 1.4 Pareto Selection Interface

CLI-based selection from Pareto frontier:

```python
# src/trdr/optimize/pareto.py
def select_from_pareto(
    pareto_front: np.ndarray,
    pareto_params: np.ndarray,
    objective_names: list[str],
) -> dict:
    """Interactive CLI selection from Pareto frontier."""

    # Display table
    print("\nPareto Frontier Solutions:")
    print("-" * 80)
    headers = ["#"] + objective_names + ["Params"]
    # ... display table

    # User selection
    choice = int(input("Select solution number: "))
    return dict(zip(param_names, pareto_params[choice]))
```

### Phase 2: Live Trading Mode

#### 2.1 Configuration

```python
# src/trdr/live/config.py
@dataclass
class AlpacaCredentials:
    api_key: str
    api_secret: str

@dataclass
class LiveConfig:
    mode: Literal["paper", "live"]
    paper_credentials: AlpacaCredentials
    live_credentials: AlpacaCredentials
    poll_interval_seconds: float = 60.0
    order_timeout_seconds: float = 30.0
    max_retries: int = 3
    risk_limits: RiskLimits = field(default_factory=RiskLimits)

# Environment variables:
# ALPACA_PAPER_API_KEY, ALPACA_PAPER_API_SECRET
# ALPACA_LIVE_API_KEY, ALPACA_LIVE_API_SECRET
# ALPACA_MODE=paper|live
```

#### 2.2 Alpaca Exchange Implementation

```python
# src/trdr/live/exchange/alpaca.py
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream

class AlpacaExchange(ExchangeInterface):
    def __init__(self, credentials: AlpacaCredentials, paper: bool = True):
        self.client = TradingClient(
            credentials.api_key,
            credentials.api_secret,
            paper=paper,
        )
        self.stream = TradingStream(
            credentials.api_key,
            credentials.api_secret,
            paper=paper,
        )
        self._fill_callbacks: list[Callable] = []

    async def connect(self) -> None:
        self.stream.subscribe_trade_updates(self._handle_trade_update)
        asyncio.create_task(self._run_stream())

    async def submit_order(self, order: OrderRequest) -> OrderResponse:
        # Map to Alpaca order types
        if order.take_profit and order.stop_loss:
            # Use bracket order
            request = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=order.take_profit),
                stop_loss=StopLossRequest(stop_price=order.stop_loss),
            )
        else:
            request = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )

        response = self.client.submit_order(request)
        return self._map_response(response)
```

#### 2.3 Order Manager

```python
# src/trdr/live/orders/manager.py
class OrderManager:
    """Manages order lifecycle: submission, tracking, fills, retries."""

    def __init__(self, exchange: ExchangeInterface, config: LiveConfig):
        self.exchange = exchange
        self.config = config
        self.pending_orders: dict[str, LiveOrder] = {}
        self.positions: dict[str, Position] = {}

    async def submit_signal(self, signal: Signal, context: RuntimeContext) -> LiveOrder:
        """Convert strategy signal to exchange order."""

        # Validate signal
        self._validate_signal(signal, context)

        # Build order request
        order_request = self._signal_to_order(signal, context)

        # Submit with retry
        response = await self._submit_with_retry(order_request)

        # Track order
        live_order = LiveOrder(
            client_id=str(uuid.uuid4()),
            exchange_id=response.order_id,
            signal=signal,
            status=OrderStatus.SUBMITTED,
            submitted_at=datetime.now(UTC),
        )
        self.pending_orders[live_order.client_id] = live_order

        return live_order

    async def _submit_with_retry(self, order: OrderRequest) -> OrderResponse:
        """Submit order with exponential backoff retry."""
        for attempt in range(self.config.max_retries):
            try:
                return await self.exchange.submit_order(order)
            except RetryableError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except NonRetryableError:
                raise
```

#### 2.4 State Reconciliation

```python
# src/trdr/live/state/reconciler.py
class StateReconciler:
    """Reconcile local state with exchange state on startup and periodically."""

    async def reconcile_on_startup(self) -> ReconciliationResult:
        """Called on harness startup to sync state."""

        # Get exchange state
        account = await self.exchange.get_account()
        positions = await self.exchange.get_positions()
        open_orders = await self.exchange.get_open_orders()

        # Load local state (if any)
        local_state = self._load_local_state()

        # Compare and resolve
        discrepancies = self._find_discrepancies(local_state, positions, open_orders)

        if discrepancies:
            # Exchange is source of truth
            self._update_local_state(positions, open_orders)
            return ReconciliationResult(
                success=True,
                discrepancies=discrepancies,
                resolution="synced_to_exchange",
            )

        return ReconciliationResult(success=True, discrepancies=[])
```

#### 2.5 Circuit Breaker

```python
# src/trdr/live/safety/circuit_breaker.py
@dataclass
class RiskLimits:
    max_drawdown_pct: float = 10.0  # Halt if drawdown exceeds
    max_daily_loss_pct: float = 5.0  # Halt if daily loss exceeds
    max_consecutive_losses: int = 5  # Halt after N consecutive losses
    max_position_pct: float = 100.0  # Max position as % of equity
    max_orders_per_hour: int = 100  # Rate limit

class CircuitBreaker:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.halted = False
        self.halt_reason: str | None = None
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.orders_this_hour: list[datetime] = []

    def check(self, context: RuntimeContext, signal: Signal) -> bool:
        """Return False if trading should be halted."""
        if self.halted:
            return False

        # Drawdown check
        if context.drawdown > self.limits.max_drawdown_pct:
            self._halt(f"Drawdown {context.drawdown:.1f}% exceeds limit")
            return False

        # Daily loss check
        if self.daily_pnl < -self.limits.max_daily_loss_pct:
            self._halt(f"Daily loss {-self.daily_pnl:.1f}% exceeds limit")
            return False

        # Consecutive losses check
        if self.consecutive_losses >= self.limits.max_consecutive_losses:
            self._halt(f"{self.consecutive_losses} consecutive losses")
            return False

        # Rate limit check
        self._prune_old_orders()
        if len(self.orders_this_hour) >= self.limits.max_orders_per_hour:
            return False  # Skip signal, don't halt

        return True
```

#### 2.6 RuntimeContext Builder

```python
# src/trdr/live/state/context.py
class LiveContextBuilder:
    """Build RuntimeContext from live exchange state."""

    def __init__(self, exchange: ExchangeInterface):
        self.exchange = exchange
        self.session_start_equity: float | None = None
        self.highest_equity: float = 0.0
        self.trades: list[ClosedTrade] = []

    async def build(self) -> RuntimeContext:
        account = await self.exchange.get_account()
        positions = await self.exchange.get_positions()

        # Track session start
        if self.session_start_equity is None:
            self.session_start_equity = account.equity
            self.highest_equity = account.equity

        # Update highest equity
        self.highest_equity = max(self.highest_equity, account.equity)

        # Calculate drawdown
        drawdown = (self.highest_equity - account.equity) / self.highest_equity * 100

        # Calculate win rate from session trades
        winning = sum(1 for t in self.trades if t.pnl > 0)
        total = len(self.trades)
        win_rate = winning / total if total > 0 else 0.0

        return RuntimeContext(
            current_bar=None,  # Populated separately from market data
            drawdown=drawdown,
            win_rate=win_rate,
            equity=account.equity,
            total_trades=total,
            winning_trades=winning,
            losing_trades=total - winning,
            current_position=self._get_primary_position(positions),
            highest_equity=self.highest_equity,
            profit_factor=self._calculate_profit_factor(),
        )
```

## Acceptance Criteria

### Multi-Objective Optimizer ✅ COMPLETE

- [x] Define 4+ objective functions: Sharpe, max drawdown, win rate, profit factor
  - Implemented in `src/trdr/optimize/objectives.py`: sharpe, max_drawdown, win_rate, profit_factor, sortino, calmar, cagr, alpha, total_trades
- [x] Integrate pymoo NSGA-II for 2-3 objectives
  - Implemented in `src/trdr/optimize/multi_objective.py`
- [x] Integrate pymoo NSGA-III for 4+ objectives
  - Auto-selects based on n_obj in `run_moo()`
- [x] Run MOO on each walk-forward training fold
  - Implemented in `src/trdr/optimize/walk_forward_moo.py`
- [x] Validate Pareto solutions on test folds
  - `validate_oos` parameter in `run_walk_forward_moo()`
- [x] CLI interface for Pareto frontier selection
  - Implemented in `src/trdr/optimize/pareto.py`: `display_pareto_front()`, `select_from_pareto()`
- [x] Persist selected parameters to config file
  - `_dump_params_to_file()` writes to `pareto_params.txt`
- [x] Support constraint handling (e.g., min trades, max drawdown cap)
  - `min_trades` constraint in `MooConfig`, inequality constraint in `StrategyOptimizationProblem`

### Live Trading Mode ✅ COMPLETE

- [x] `ExchangeInterface` ABC matching `PaperExchange` capabilities
  - Implemented in `src/trdr/live/exchange/base.py` with Hydra-prefixed types
- [x] Alpaca exchange implementation with paper/live mode toggle
  - Implemented in `src/trdr/live/exchange/alpaca.py`
- [x] Environment-based credential management (PAPER/LIVE keys)
  - Implemented in `src/trdr/live/config.py` with `LiveConfig.from_env()`
- [x] Poll-based execution harness with configurable interval
  - Implemented in `src/trdr/live/harness.py` as `LiveHarness`
- [x] Order submission with retry and exponential backoff
  - Implemented in `src/trdr/live/orders/retry.py` with `RetryPolicy`
  - OrderManager uses retry logic in `src/trdr/live/orders/manager.py`
- [x] Fill tracking via WebSocket stream
  - Implemented in `AlpacaExchange._handle_trade_update()` and fill callbacks
- [x] State reconciliation on startup
  - Implemented in `src/trdr/live/state/reconciler.py` as `StateReconciler`
- [x] RuntimeContext builder from live exchange state
  - Implemented in `src/trdr/live/state/context.py` as `LiveContextBuilder`
- [x] Circuit breaker with configurable risk limits
  - Implemented in `src/trdr/live/safety/circuit_breaker.py`
  - Supports max drawdown, daily loss, consecutive losses, position size, rate limits
- [x] Graceful shutdown preserving open positions
  - Implemented in `LiveHarness.stop()` with proper cleanup
- [x] Audit logging of all orders and fills
  - Implemented in `OrderManager._audit_log()` with file and logger output

### Integration ✅ COMPLETE

- [x] Same `BaseStrategy` interface works for backtest and live
  - `LiveHarness` accepts any `BaseStrategy` implementation
- [x] Signal contract unchanged
  - Uses existing `Signal` type from strategy module
- [x] OCO/bracket orders mapped correctly to Alpaca
  - Implemented in `AlpacaExchange.submit_order()` with bracket order support
- [x] Existing `PaperExchange` can be adapted to `ExchangeInterface`
  - Types are compatible via `LiveRuntimeContext` adapter

## Dependencies & Prerequisites

### New Dependencies

```text
# requirements.txt additions
pymoo>=0.6.0          # Multi-objective optimization
alpaca-py>=0.13.0     # Alpaca trading API
```

### Existing Dependencies Used

- `numpy` - Array operations for optimization
- `asyncio` - Async execution for live trading
- `dataclasses` - Type definitions

### Prerequisites

1. Alpaca account with paper and/or live API keys
2. Existing strategy implementations
3. Walk-forward optimization system (exists)

## Risk Analysis & Mitigation

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Network failure during order | Unknown order state | Query order status before retry; idempotent order IDs |
| Exchange rate limit | Orders rejected | Exponential backoff; order rate tracking |
| State divergence | Incorrect position sizing | Periodic reconciliation; exchange as source of truth |
| Strategy bug in live | Financial loss | Circuit breakers; paper testing first |
| API key leak | Account compromise | Env vars only; never in code/logs |
| Partial fills | Unexpected position size | Track actual fill qty; update RuntimeContext |

## Implementation Notes

### OCO Compatibility

Alpaca supports bracket orders natively. The existing `PaperExchange` OCO logic should be preserved for backtest but bracket orders used for Alpaca:

```python
# In AlpacaExchange
if signal.stop_loss and signal.take_profit:
    # Use Alpaca bracket order (native OCO equivalent)
    order_class = OrderClass.BRACKET
else:
    order_class = OrderClass.SIMPLE
```

### Retry Policy

| Error Type | Retry? | Max Attempts | Backoff |
| --- | --- | --- | --- |
| Network timeout | Yes | 3 | Exponential (1s, 2s, 4s) |
| Rate limit (429) | Yes | 5 | Exponential + jitter |
| Insufficient funds | No | - | - |
| Invalid params | No | - | - |
| Auth failure | No | - | Halt system |

### Logging Requirements

```python
# Structured logging format
{
    "timestamp": "2026-01-05T10:30:00Z",
    "level": "INFO",
    "event": "order_submitted",
    "order_id": "abc123",
    "symbol": "AAPL",
    "side": "buy",
    "qty": 100,
    "type": "market",
}
```

Log all: signals, orders, fills, errors, state changes, circuit breaker events.

## File Reference

| Component | Path |
| --- | --- |
| Base Strategy | `src/trdr/strategy/base_strategy.py` |
| Strategy Types | `src/trdr/strategy/types.py` |
| Paper Exchange | `src/trdr/backtest/paper_exchange.py` |
| Order Management | `src/trdr/backtest/orders.py` |
| Walk-Forward | `src/trdr/backtest/walk_forward.py` |
| Config | `src/trdr/core/config.py` |
| Old Alpaca Client (reference) | `tmp/martingale-bot/alpaca_client.py` |
| Old Alpaca Env (reference) | `tmp/martingale-bot/alpaca_env.py` |

## References

### Internal

- Strategy API: `src/trdr/backtest/STRATEGY_API.md`
- SICA Runner pattern: `src/trdr/strategy/sica_runner.py`
- feat/helios1 branch: LiveTrader concepts

### External

- [pymoo Multi-Objective](https://pymoo.org/algorithms/moo/nsga2.html)
- [Alpaca Trading API](https://docs.alpaca.markets/docs/trading-api)
- [Alpaca WebSocket Streaming](https://docs.alpaca.markets/docs/websocket-streaming)
- [Alpaca Bracket Orders](https://docs.alpaca.markets/docs/bracket-orders)
- [Optuna NSGAIISampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html)
