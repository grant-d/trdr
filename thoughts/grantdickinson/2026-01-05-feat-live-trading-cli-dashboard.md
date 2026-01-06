# Live Trading CLI Dashboard

## Overview

Replace the stream-of-text log output in `LiveHarness` with an interactive Textual TUI showing all critical trading information at a glance.

## Architecture

### Technology Choice: Textual

- Already in use (`src/trdr/ui/`) with established patterns
- Python-native (no Node.js dependency like Ink)
- 120 FPS rendering, CSS styling, reactive attributes
- Existing patterns: key bindings, message handlers, panels

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ TRDR Live │ ETH/USD │ 15m │ ▶ RUNNING │ Circuit: CLOSED        │
├──────────────────────────────────┬──────────────────────────────┤
│ POSITION                         │ CIRCUIT BREAKER              │
│ Side: LONG                       │ State: CLOSED                │
│ Size: 0.5 ETH                    │ Drawdown: 2.3% / 10%         │
│ Entry: $3,245.50                 │ Daily PnL: +$125 / -$500     │
│ Current: $3,289.00               │ Equity: $10,125              │
│ PnL: +$21.75 (+0.67%)            │ HWM: $10,350                 │
├──────────────────────────────────┼──────────────────────────────┤
│ PENDING ORDERS                   │ RECENT SIGNALS               │
│ ○ SELL STOP @ $3,200.00          │ 14:32:15 LONG  score=0.82    │
│ ○ SELL LIMIT @ $3,350.00         │ 14:28:45 EXIT  hit_target    │
│                                  │ 14:15:00 LONG  score=0.75    │
├──────────────────────────────────┴──────────────────────────────┤
│ LOG                                                             │
│ 14:32:15 Bar processed: O=3288 H=3290 L=3285 C=3289 V=1.2k     │
│ 14:32:15 Signal generated: LONG score=0.82                      │
│ 14:32:16 Order submitted: BUY 0.5 ETH @ MARKET                  │
│ 14:32:16 Order filled: 0.5 ETH @ $3,245.50                      │
├─────────────────────────────────────────────────────────────────┤
│ [P]ause [R]esume [S]top [B]reaker Reset [Q]uit                  │
└─────────────────────────────────────────────────────────────────┘
```

### Components

```
src/trdr/live/ui/
├── __init__.py
├── app.py              # LiveDashboard(App)
├── widgets/
│   ├── __init__.py
│   ├── header.py       # StatusHeader - symbol, timeframe, state
│   ├── position.py     # PositionPanel - current position details
│   ├── circuit.py      # CircuitPanel - breaker status, limits
│   ├── orders.py       # OrdersPanel - pending orders list
│   ├── signals.py      # SignalsPanel - recent signal history
│   └── log.py          # LogPanel - scrolling log output
└── styles.tcss         # Textual CSS
```

## Implementation

### Phase 1: Core Dashboard

1. Create `LiveDashboard` app extending `textual.App`
2. Implement `StatusHeader` widget with reactive state
3. Add `LogPanel` capturing harness output
4. Wire key bindings: p/r/s/q

### Phase 2: Trading Panels

1. Implement `PositionPanel` with PnL coloring (green/red)
2. Implement `CircuitPanel` showing breaker status
3. Implement `OrdersPanel` listing pending orders
4. Implement `SignalsPanel` with signal history ring buffer

### Phase 3: Interactivity

1. Add `[B]` breaker reset command
2. Add order cancellation via selection
3. Add position close confirmation dialog
4. Connection status indicator with reconnect

## Key Bindings

| Key | Action |
| --- | --- |
| p | Pause trading (stop new signals) |
| r | Resume trading |
| s | Emergency stop (cancel orders, close position) |
| b | Reset circuit breaker (if OPEN) |
| q | Quit (with confirmation if position open) |

## Data Flow

```
LiveHarness
    │
    ├─► on_signal callback ──► SignalsPanel.add_signal()
    ├─► on_fill callback ────► PositionPanel.update(), OrdersPanel.refresh()
    ├─► on_error callback ───► LogPanel.error(), StatusHeader.set_error()
    │
    └─► get_status() ────────► Periodic refresh (1s)
            │
            ├─► state ───────► StatusHeader
            ├─► position ────► PositionPanel
            ├─► orders ──────► OrdersPanel
            └─► circuit ─────► CircuitPanel
```

## Harness Integration

```python
# In harness.py
class LiveHarness:
    def run_with_ui(self) -> None:
        """Run harness with TUI dashboard."""
        from .ui import LiveDashboard

        app = LiveDashboard(harness=self)
        app.run()

    # Existing callbacks already support this:
    # - on_signal: Callable[[Signal], None]
    # - on_fill: Callable[[Fill], None]
    # - on_error: Callable[[Exception], None]

    # Existing status methods:
    # - get_status() -> dict
    # - state: HarnessState (INITIALIZING, RUNNING, PAUSED, STOPPED)
```

## Color Scheme

| Element | Color |
| --- | --- |
| Positive PnL | green |
| Negative PnL | red |
| RUNNING state | green |
| PAUSED state | yellow |
| STOPPED state | red |
| Circuit CLOSED | green |
| Circuit OPEN | red |
| Circuit HALF_OPEN | yellow |
| Error messages | red bold |
| LONG position | cyan |
| SHORT position | magenta |

## Error States

- **Connection Lost**: Yellow banner, auto-reconnect indicator
- **Circuit Breaker Open**: Red banner, show trigger reason
- **Order Rejected**: Log error, flash orders panel
- **Position Sync Failure**: Red warning, show reconciliation status

## Files to Create

1. `src/trdr/live/ui/__init__.py`
2. `src/trdr/live/ui/app.py`
3. `src/trdr/live/ui/widgets/__init__.py`
4. `src/trdr/live/ui/widgets/header.py`
5. `src/trdr/live/ui/widgets/position.py`
6. `src/trdr/live/ui/widgets/circuit.py`
7. `src/trdr/live/ui/widgets/orders.py`
8. `src/trdr/live/ui/widgets/signals.py`
9. `src/trdr/live/ui/widgets/log.py`
10. `src/trdr/live/ui/styles.tcss`

## Files to Modify

1. `src/trdr/live/harness.py` - Add `run_with_ui()` method
2. `src/trdr/cli.py` - Add `--ui` flag to live command

## Testing

- Unit tests for each widget with mock data
- Integration test: dashboard renders with mock harness
- Manual test: paper trading with real data flow
