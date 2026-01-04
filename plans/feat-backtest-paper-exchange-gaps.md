# Feature: Backtest/Paper Exchange Missing Features

**Status: PHASE 1-2 COMPLETE**

Fix documented-but-missing features. Keep it simple.

## Problem Statement

Two features are documented but not implemented:

1. **Take Profit Orders** - `Signal.take_profit` field exists but engine ignores it
2. **on_trade_complete Callback** - Defined in BaseStrategy but never called

## Reviewer Feedback Summary

**DHH Review:** "Delete features to match reality, not implement features to match documentation."

- Most proposed features have no current use case
- Strategies are long-only momentum plays - no shorts needed
- Volume slippage is premature optimization
- VaR/CVaR adds nothing the current metrics don't cover

**Simplicity Review:** Cut to 3 features: take_profit, on_trade_complete, limit orders.

**Pattern Review:**

- Short selling API (SHORT/COVER) breaks symmetry with industry patterns
- OCO is under-specified - use existing `cancel_all()` instead
- Partial exit fix has logic errors

---

## Revised Implementation Plan

### Phase 1: Fix Documented Features (Critical)

#### 1.1 Take Profit Orders

**Files:** `orders.py`, `paper_exchange.py`

```python
# orders.py - Add to OrderType enum
class OrderType(Enum):
    # ... existing
    TAKE_PROFIT = "take_profit"  # NEW
```

```python
# paper_exchange.py - Update _process_signal()
if signal.take_profit:
    tp_order = Order(
        symbol=self.config.symbol,
        side="sell",
        quantity=quantity,
        order_type=OrderType.TAKE_PROFIT,
        stop_price=signal.take_profit,
    )
    self.order_manager.submit(tp_order)
```

**Fill Logic:** Bar.high >= take_profit triggers fill at max(take_profit, bar.open)

**OCO Behavior:** When any sell order fills (SL or TP), call `order_manager.cancel_all()`. No new oco_group field needed - reuse existing cancel logic.

#### 1.2 on_trade_complete Callback

**Files:** `paper_exchange.py`

```python
# After appending trade to trades list
trades.append(trade)

# Call strategy callback (method exists on BaseStrategy, no hasattr needed)
self.strategy.on_trade_complete(trade.net_pnl, trade.exit_reason)
```

### Phase 2: Limit Orders (High Priority)

#### 2.1 Limit Orders

**Files:** `orders.py`, `paper_exchange.py`, `types.py`

```python
# orders.py - Add to OrderType and Order
class OrderType(Enum):
    # ... existing
    LIMIT = "limit"  # NEW

@dataclass
class Order:
    # ... existing
    limit_price: float | None = None  # NEW
```

```python
# types.py - Add to Signal
@dataclass
class Signal:
    # ... existing
    limit_price: float | None = None  # NEW - for limit entry orders
```

**Fill Logic:**

- Buy limit: fills when bar.low <= limit_price at limit_price
- Sell limit: fills when bar.high >= limit_price at limit_price
- No slippage on limit orders (price guaranteed or better)

### Phase 3: Short Selling (Optional - Config Gated)

Short selling enabled via config parameter, default disabled.

**Files:** `paper_exchange.py`, `types.py`

```python
# paper_exchange.py - Add to config
@dataclass
class PaperExchangeConfig:
    # ... existing
    allow_short: bool = False  # NEW - default disabled

# types.py - Add to SignalAction
class SignalAction(Enum):
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    HOLD = "hold"
    SHORT = "short"   # NEW - sell to open (requires allow_short=True)
    COVER = "cover"   # NEW - buy to close short
```

```python
# paper_exchange.py - Validate in _process_signal()
if signal.action == SignalAction.SHORT:
    if not self.config.allow_short:
        raise ValueError("Short selling disabled. Set allow_short=True in config.")
    # ... short entry logic
```

---

## Deferred (No Current Use Case)

| Feature | Reason | Status |
| --- | --- | --- |
| Partial exit stop adjustment | No strategy uses partial exits | TODO when needed |
| Volume-aware slippage | Premature optimization for small positions | TODO |
| Floating US holidays | Data providers already filter holidays | TODO |
| VaR/CVaR metrics | Current metrics sufficient | TODO |
| Alpha/Beta benchmark | Adds complexity, external analysis can compute | TODO |
| OCO order linking | Use existing cancel_all() on any fill | Not needed |

---

## Acceptance Criteria

### Phase 1 (Critical) ✅ COMPLETE

- [x] `Signal.take_profit` creates TAKE_PROFIT order that fills correctly
- [x] When TP or SL fills, cancel all other pending orders (pseudo-OCO)
- [x] `on_trade_complete(pnl, reason)` called after each trade closes
- [x] All existing tests pass (177+)

### Phase 2 (High) ✅ COMPLETE

- [x] `Signal.limit_price` creates LIMIT order with correct fill logic
- [x] Limit orders have no slippage (guaranteed price)
- [x] Tests for limit order edge cases (gap through, never touched)

### Phase 3 (Optional)

- [ ] `allow_short=True` enables SHORT/COVER actions
- [ ] `allow_short=False` (default) raises error on SHORT signal
- [ ] Short positions track correctly with inverted stop logic

---

## File Changes Summary

| File | Changes |
| --- | --- |
| `src/trdr/backtest/orders.py` | Add TAKE_PROFIT, LIMIT to OrderType; add limit_price to Order |
| `src/trdr/backtest/paper_exchange.py` | Process take_profit, limit orders; call on_trade_complete; add allow_short config |
| `src/trdr/strategy/types.py` | Add Signal.limit_price; add SignalAction.SHORT, COVER |
| `tests/test_orders.py` | Tests for new order types |
| `tests/test_paper_exchange.py` | Tests for take profit, callback, short gating |

---

## Dependencies

None - no new libraries needed.

---

## References

- Current implementation: `src/trdr/backtest/paper_exchange.py`
- API documentation: `src/trdr/backtest/STRATEGY_API.md`
