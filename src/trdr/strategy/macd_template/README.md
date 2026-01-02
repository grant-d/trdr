# MACD Strategy (Template)

Simple MACD crossover strategy.
Use as a template for creating new strategies.

## Creating a New Strategy

### 1. Copy this folder

```bash
cp -r src/trdr/strategy/macd src/trdr/strategy/your_strategy
```

### 2. Rename files and classes

- `strategy.py`: Rename `MACDConfig` → `YourConfig`, `MACDStrategy` → `YourStrategy`
- `test_strategy.py`: Update imports and test class name
- Update `__init__.py` exports

### 3. Implement your logic

**Config** - Define strategy parameters:

```python
@dataclass
class YourConfig(StrategyConfig):
    # StrategyConfig provides: symbol, timeframe
    your_param: float = 1.0
    another_param: int = 10
```

**Strategy** - Implement `generate_signal()`:

```python
class YourStrategy(BaseStrategy):
    def generate_signal(self, bars: list[Bar], position: Position | None) -> Signal:
        # 1. Check minimum data
        if len(bars) < self.min_required:
            return Signal(action=SignalAction.HOLD, ...)

        # 2. Check exits first (if in position)
        if position and position.side == "long":
            if hit_stop_or_target:
                return Signal(action=SignalAction.CLOSE, ...)

        # 3. Check entries (if no position)
        if not position and entry_conditions:
            return Signal(action=SignalAction.BUY, stop_loss=..., take_profit=...)

        return Signal(action=SignalAction.HOLD, ...)
```

### 4. Register in strategy/**init**.py

```python
from .your_strategy import YourConfig, YourStrategy

__all__ = [
    # ... existing exports
    "YourConfig",
    "YourStrategy",
]
```

## Signal Return Values

| Action | When |
| --- | --- |
| `SignalAction.HOLD` | No action needed |
| `SignalAction.BUY` | Enter long position |
| `SignalAction.CLOSE` | Exit current position |

Always include in BUY signals:

- `stop_loss`: Exit price if trade goes against you
- `take_profit`: Target exit price (optional)
- `confidence`: 0.0-1.0, used for position sizing

## Testing

Run strategy tests:

```bash
.venv/bin/python -m pytest src/trdr/strategy/your_strategy/test_strategy.py -v
```

## Files

- `strategy.py` - Config + Strategy class + helper functions
- `test_strategy.py` - Backtest tests
- `README.md` - This file
