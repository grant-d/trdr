# PaperExchange API

Paper exchange with order execution, position management, and portfolio tracking for strategy backtesting.

## Quick Start

```python
from trdr.backtest import PaperExchange, PaperExchangeConfig

config = PaperExchangeConfig(symbol="crypto:ETH/USD")
engine = PaperExchange(config, strategy)
result = engine.run(bars)
```

## Strategy Interface

Implement `generate_signal(bars, position) -> Signal`. Access live portfolio state via `self.context`:

```python
class MyStrategy(BaseStrategy):
    def generate_signal(self, bars: list[Bar], position: Position | None) -> Signal:
        # Access live portfolio metrics
        if self.context.drawdown > 0.15:
            return Signal(action=SignalAction.HOLD, ...)  # pause during drawdown

        if should_buy(bars):
            # Scale position based on performance
            size = 0.5 if self.context.win_rate < 0.4 else 1.0
            return Signal(
                action=SignalAction.BUY,
                price=bars[-1].close,
                confidence=0.8,
                reason="Buy signal",
                stop_loss=bars[-1].close * 0.98,
                take_profit=bars[-1].close * 1.05,
                position_size_pct=size,
            )
        return Signal(action=SignalAction.HOLD, price=bars[-1].close, ...)
```

## Runtime Context

Live portfolio state available via `self.context` in `generate_signal()`:

```python
# Run params
self.context.symbol           # Trading symbol
self.context.current_bar      # Current Bar object
self.context.bar_index        # Current bar index (0-based)
self.context.total_bars       # Total bars in run
self.context.bars_remaining   # Bars left

# Portfolio state
self.context.equity           # Current portfolio value
self.context.cash             # Available cash
self.context.initial_capital  # Starting capital
self.context.positions        # Open positions dict
self.context.pending_orders   # Pending stop/limit orders
self.context.trades           # Completed Trade objects

# Live metrics (computed on demand)
self.context.drawdown         # Current drawdown %
self.context.max_drawdown     # Max drawdown %
self.context.total_return     # Total return %
self.context.total_pnl        # Net P&L from closed trades
self.context.win_rate         # Win rate
self.context.profit_factor    # Profits / losses
self.context.expectancy       # Expected value per trade
self.context.sharpe_ratio     # Annualized Sharpe
self.context.sortino_ratio    # Annualized Sortino
self.context.calmar_ratio     # CAGR / max drawdown
self.context.cagr             # Compound annual growth rate
```

## Signal Examples

```python
# Entry with stop loss and take profit
Signal(action=SignalAction.BUY, price=100.0, confidence=0.75,
       reason="VAH breakout", stop_loss=98.0, take_profit=105.0)

# Scaled position sizing (50% of default)
Signal(action=SignalAction.BUY, price=100.0, confidence=0.6,
       reason="VAL bounce", stop_loss=97.0, take_profit=103.0,
       position_size_pct=0.5)

# Exit on stop loss
Signal(action=SignalAction.CLOSE, price=98.0, confidence=1.0,
       reason="Stop loss hit at 98.00")

# Exit on take profit
Signal(action=SignalAction.CLOSE, price=105.0, confidence=0.9,
       reason="Take profit hit at 105.00")

# Hold existing position
Signal(action=SignalAction.HOLD, price=101.5, confidence=0.5,
       reason="Holding position, awaiting target or stop")

# No entry signal
Signal(action=SignalAction.HOLD, price=100.0, confidence=0.0,
       reason="Insufficient data for analysis")

# Low confidence rejection
Signal(action=SignalAction.HOLD, price=100.0, confidence=0.35,
       reason="Low confidence 0.35 < 0.45 threshold")
```

## Signal Fields

| Field | Type | Description |
| --- | --- | --- |
| `action` | `SignalAction` | BUY, SELL, CLOSE, HOLD |
| `price` | `float` | Current price |
| `confidence` | `float` | 0.0-1.0 signal strength |
| `reason` | `str` | Entry/exit reason |
| `stop_loss` | `float?` | Stop loss price |
| `take_profit` | `float?` | Take profit price |
| `trailing_stop` | `float?` | Trail % (<1) or $ amount |
| `quantity` | `float?` | Explicit size (overrides default) |
| `position_size_pct` | `float` | Size as % of default (1.0 = 100%) |

## Config

```python
PaperExchangeConfig(
    symbol="crypto:ETH/USD",     # Required
    warmup_bars=65,              # Skip before signals
    transaction_cost_pct=0.0025, # 0.25% per trade
    slippage_pct=0.001,          # 0.1% slippage
    default_position_pct=1.0,    # 100% of equity
    initial_capital=10000.0,     # Starting cash
)
```

## Result Properties

| Property | Type | Description |
| --- | --- | --- |
| `total_trades` | `int` | Number of trades |
| `win_rate` | `float` | Winning trades / total |
| `total_pnl` | `float` | Net P&L after costs |
| `profit_factor` | `float` | Profits / losses |
| `max_drawdown` | `float` | Peak-to-trough % |
| `sortino_ratio` | `float?` | Risk-adjusted return |
| `sharpe_ratio` | `float?` | Annualized Sharpe |
| `calmar_ratio` | `float?` | CAGR / max drawdown |
| `cagr` | `float?` | Compound annual growth |
| `total_return` | `float` | Total return decimal |
| `expectancy` | `float` | Expected value per trade |
| `trades` | `list[Trade]` | Trade records |
| `equity_curve` | `list[float]` | Equity at each bar |
