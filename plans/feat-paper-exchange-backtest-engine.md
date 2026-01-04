# Feature: Paper Exchange Backtest Engine

Enhance the backtest/forward engine to act more like a paper exchange with advanced order types, position management, and portfolio tracking.

## Design Principles

1. **Simple Strategy Interface** - LLM writes `generate_signal(bars, position) → Signal`. Nothing else.
2. **Data External** - Engine receives bars, doesn't fetch. Same engine for backtest + paper trading.
3. **Complexity Hidden** - Orders, portfolio, calendar handled by engine internals.
4. **Backward Compatible** - Existing strategies work unchanged.

## Current State

`src/trdr/backtest/backtest_engine.py`:

- Market orders only (fill at next bar open)
- Single position at a time (no pyramiding)
- All-in/all-out sizing via `position_size_pct`
- Basic slippage (ATR-based) and transaction costs
- Equity curve tracking

## Requirements

1. **Order Types** (KISS)
   - Market (existing)
   - Stop (trigger at price)
   - Stop Loss (exit on price breach)
   - Trailing Stop Loss (TSL) - follows price, exits on reversal

2. **Position Management**
   - Multiple entries on same symbol (buy 1 ETH, later buy 2 more)
   - Partial exits (sell 0.5 of 3 ETH position)
   - Track each entry separately for accurate P&L
   - Max position limits

3. **Portfolio Tracking**
   - Cash balance
   - Equity (cash + positions MTM)
   - Liquidate method (close all positions at market)

4. **Market Simulation**
   - Slippage (existing ATR-based)
   - Transaction costs (existing %)
   - Trading calendar (skip weekends/holidays for stocks)

## Implementation Plan

### Phase 1: Order Types

Create order abstraction layer.

#### 1.1 Order Dataclass

```python
# src/trdr/backtest/orders.py

@dataclass
class Order:
    id: str
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "stop", "stop_loss", "trailing_stop"]
    quantity: float
    stop_price: float | None = None
    trail_amount: float | None = None  # Absolute $ trail
    trail_percent: float | None = None  # % trail
    status: Literal["pending", "filled", "cancelled"] = "pending"
    created_at: str = ""
    filled_at: str | None = None
    fill_price: float | None = None
```

#### 1.2 Order Manager

```python
class OrderManager:
    def submit(self, order: Order) -> str
    def cancel(self, order_id: str) -> bool
    def process_bar(self, bar: Bar) -> list[Fill]
    def update_trailing_stops(self, bar: Bar) -> None
```

#### 1.3 Trailing Stop Logic

```python
def update_trailing_stops(self, bar: Bar) -> None:
    for order in self.pending_orders:
        if order.order_type != "trailing_stop":
            continue

        if order.side == "sell":  # Long position TSL
            new_stop = bar.high - (order.trail_amount or bar.high * order.trail_percent)
            order.stop_price = max(order.stop_price, new_stop)
        else:  # Short position TSL
            new_stop = bar.low + (order.trail_amount or bar.low * order.trail_percent)
            order.stop_price = min(order.stop_price, new_stop)
```

### Phase 2: Position Management

#### 2.1 Enhanced Position Tracking

```python
@dataclass
class PositionEntry:
    price: float
    quantity: float
    timestamp: str

@dataclass
class Position:
    symbol: str
    side: Literal["long", "short"]
    entries: list[PositionEntry]  # Track each entry separately

    @property
    def total_quantity(self) -> float
    @property
    def avg_price(self) -> float  # Volume-weighted average
    @property
    def unrealized_pnl(self, current_price: float) -> float
```

#### 2.2 Position Config

```python
@dataclass
class PositionConfig:
    max_position_pct: float = 1.0  # Max % of capital in position
```

#### 2.3 Strategy Signal (LLM-facing API)

Strategy just returns a Signal. Engine handles the rest.

```python
@dataclass
class Signal:
    action: SignalAction  # BUY, SELL, CLOSE, HOLD
    reason: str = ""
    # Optional - engine uses defaults if not set:
    stop_loss: float | None = None      # Fixed stop price
    take_profit: float | None = None    # Fixed TP price
    trailing_stop: float | None = None  # Trail % (e.g., 0.02 = 2%)
    quantity: float | None = None       # Units to buy/sell (None = use default sizing)
```

**LLM Strategy Example:**
```python
class SimpleStrategy(BaseStrategy):
    def generate_signal(self, bars: list[Bar], position: Position | None) -> Signal:
        if not position and bars[-1].close > bars[-2].close:
            return Signal(action=SignalAction.BUY, trailing_stop=0.02)
        if position and bars[-1].close < bars[-2].close:
            return Signal(action=SignalAction.CLOSE)
        return Signal(action=SignalAction.HOLD)
```

### Phase 3: Portfolio Tracking

#### 3.1 Portfolio State

```python
@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position]

    @property
    def equity(self, prices: dict[str, float]) -> float:
        mtm = sum(p.unrealized_pnl(prices[p.symbol]) for p in self.positions.values())
        return self.cash + mtm

    @property
    def buying_power(self) -> float:
        return self.cash  # Simple; extend for margin later
```

#### 3.2 Engine Integration

Modify `BacktestEngine.run()`:

```python
def run(self, bars: list[Bar]) -> BacktestResult:
    portfolio = Portfolio(cash=self.config.initial_capital, positions={})
    order_manager = OrderManager()

    for i in range(self.config.warmup_bars, len(bars)):
        bar = bars[i]

        # 1. Update trailing stops
        order_manager.update_trailing_stops(bar)

        # 2. Process fills at bar open
        fills = order_manager.process_bar(bar)
        for fill in fills:
            self._apply_fill(portfolio, fill)

        # 3. Generate signal (point-in-time)
        signal = self.strategy.generate_signal(bars[:i+1], portfolio.positions.get(self.config.symbol))

        # 4. Submit new orders
        if signal.action != SignalAction.HOLD:
            orders = self._signal_to_orders(signal, portfolio, bar)
            for order in orders:
                order_manager.submit(order)

        # 5. Record equity
        equity_curve.append(portfolio.equity({self.config.symbol: bar.close}))
```

### Phase 4: Market Simulation

#### 4.1 Trading Calendar

```python
# src/trdr/backtest/calendar.py

def is_trading_day(timestamp: str, asset_type: str) -> bool:
    """Check if bar falls on a trading day.

    Args:
        timestamp: ISO timestamp
        asset_type: "crypto" (24/7) or "stock" (M-F, exclude holidays)
    """
    if asset_type == "crypto":
        return True

    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    # Skip weekends
    if dt.weekday() >= 5:
        return False
    # Skip US holidays (simplified - major ones)
    # Full impl would use pandas_market_calendars or similar
    return True
```

#### 4.2 Liquidate Method

```python
class Portfolio:
    def liquidate(self, current_prices: dict[str, float]) -> list[Order]:
        """Generate market sell orders for all positions."""
        orders = []
        for symbol, position in self.positions.items():
            orders.append(Order(
                id=str(uuid4()),
                symbol=symbol,
                side="sell",
                order_type="market",
                quantity=position.total_quantity,
            ))
        return orders
```

### Phase 5: Testing

#### 5.1 Unit Tests

| Test | Description |
| --- | --- |
| `test_stop_order` | Stop triggers at price |
| `test_stop_loss` | SL exits position on breach |
| `test_tsl_long` | TSL moves up with price, triggers on drop |
| `test_tsl_short` | TSL moves down with price, triggers on rise |
| `test_multiple_entries` | Buy 1, buy 2 more = 3 total |
| `test_partial_exit` | Sell 0.5 of 3 = 2.5 remaining |
| `test_cash_tracking` | Cash reduces on buy, increases on sell |
| `test_equity_mtm` | Equity = cash + positions MTM |
| `test_liquidate` | Close all positions at market |
| `test_trading_calendar_crypto` | Crypto trades 24/7 |
| `test_trading_calendar_stock` | Stocks skip weekends |

#### 5.2 Integration Tests

- Full backtest with multiple entries
- TSL vs fixed stop comparison
- Stock backtest respects trading calendar

## File Changes

| File | Change |
| --- | --- |
| `src/trdr/backtest/orders.py` | New - Order, Fill, OrderManager |
| `src/trdr/backtest/portfolio.py` | New - Portfolio, PositionEntry, liquidate |
| `src/trdr/backtest/calendar.py` | New - is_trading_day |
| `src/trdr/backtest/backtest_engine.py` | Refactor to use OrderManager, Portfolio |
| `src/trdr/strategy/types.py` | Extend Signal dataclass |
| `tests/test_orders.py` | New - order type tests |
| `tests/test_portfolio.py` | New - portfolio + liquidate tests |
| `tests/test_calendar.py` | New - trading calendar tests |
| `tests/test_backtest_engine.py` | Update for new features |

## Migration

Backward compatible:

- Signal without new fields works as before
- Portfolio initializes from `initial_capital`
- Strategies that don't use multiple entries work unchanged
