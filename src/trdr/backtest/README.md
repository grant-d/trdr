# PaperExchange API

Simple backtesting for strategy writers.

## Quick Start

```python
from trdr.backtest import PaperExchange, PaperExchangeConfig

config = PaperExchangeConfig(symbol="crypto:ETH/USD")
engine = PaperExchange(config, strategy)
result = engine.run(bars)
```

## Strategy Interface

Implement `generate_signal(bars, position) -> Signal`:

```python
class MyStrategy(BaseStrategy):
    def generate_signal(self, bars: list[Bar], position: Position | None) -> Signal:
        if should_buy(bars):
            return Signal(
                action=SignalAction.BUY,
                price=bars[-1].close,
                confidence=0.8,
                reason="Buy signal",
                stop_loss=bars[-1].close * 0.98,  # 2% stop
                take_profit=bars[-1].close * 1.05,  # 5% target
            )
        if position and should_exit(bars):
            return Signal(action=SignalAction.CLOSE, price=bars[-1].close, ...)
        return Signal(action=SignalAction.HOLD, price=bars[-1].close, ...)
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
| `trades` | `list[Trade]` | Trade records |
| `equity_curve` | `list[float]` | Equity at each bar |
