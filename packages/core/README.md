# @trdr/core

**Purpose:**
- Implement the core trading engine for TRDR
- Provide event system, market data feeds, consensus, order management, indicators, and utilities
- No UI or database logic allowed
- Used by API and web for all trading logic

**Architecture:**
- Modular TypeScript package
- Event-driven architecture using enhanced event bus
- Pluggable market data feeds (live, backtest, paper)
- Consensus engine for agent voting and decision making
- Order management and execution monitoring
- Technical indicators and signal processing
- Utilities for logging, time, and common logic
- All public classes and interfaces must have JSDoc

**Dataflow:**
- Market data ingested via feeds, processed by indicators and agents
- Events dispatched through event bus to trigger trading actions
- Consensus engine aggregates agent signals for trade decisions
- Orders managed and executed, state tracked and logged
- No direct database or UI interaction

**Main Modules:**

| Module                | Description                                 |
|----------------------|---------------------------------------------|
| `events/`            | Enhanced event bus, event types, filtering  |
| `market-data/`       | Market data feeds, backtest, live, paper    |
| `consensus/`         | Consensus manager, agent voting strategies  |
| `orders/`            | Order state machine, execution, monitoring  |
| `indicators/`        | Technical indicators (EMA, RSI, MACD, etc.) |
| `position-sizing/`   | Position sizing strategies and manager      |
| `utils/`             | Logger, time source, common utilities       |
| `database/`          | Database factory (for dependency injection) |
| `interfaces/`        | Core interfaces for extensibility           |

**Conventions:**
- All public classes and interfaces must have JSDoc
- No UI, API, or database logic in this package
- Only implement trading logic, event flow, and utilities

**Usage:**
- Import core modules in API or web to run trading logic
- Extend or plug in custom feeds, indicators, or strategies

**Example:**
```typescript
import { ConsensusManager, EnhancedEventBus } from '@trdr/core'
``` 