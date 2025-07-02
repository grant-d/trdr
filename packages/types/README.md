# @trdr/types

**Purpose:**
- Provide type contracts for TRDR system
- Re-export shared types for cross-package safety
- Define repository, event, and logger interfaces
- No runtime logic, type-only package

**Architecture:**
- Pure TypeScript module
- Imports types from `@trdr/shared`
- Exports repository, event, logger interfaces
- Used by core and data packages for type contracts

**Dataflow:**
- No runtime dataflow
- Types consumed at build time by all TRDR packages

**Main Exports:**

| Export                | Description                                 |
|----------------------|---------------------------------------------|
| `Candle`             | Market data candle type                     |
| `MarketDataRepository` | Interface for market data storage           |
| `EventBus`           | Event-driven communication interface        |
| `Logger`             | Logging interface for consistent logging    |
| `EventData`          | Base event data type                        |
| `EventHandler`       | Event handler function type                 |
| `EventSubscription`  | Event subscription contract                 |

**Conventions:**
- All interfaces must have JSDoc
- No implementation logic allowed
- Extend only from `@trdr/shared` or local types

**Usage:**
- Import types for repository, event, and logger contracts
- Use in core/data for strong type safety

**Example:**
```typescript
import type { MarketDataRepository, Logger } from '@trdr/types'
``` 