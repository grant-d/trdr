# @trdr/shared

**Purpose:**
- Define all core types for TRDR (market data, agents, orders, config, etc.)
- Provide utility functions (dates, conversions, etc.)
- Ensure type safety and consistency across all packages
- No business logic, only types and stateless utilities

**Architecture:**
- Pure TypeScript module
- Exports interfaces, enums, and utility functions
- No dependencies on other TRDR packages
- Used by all other packages for shared types

**Dataflow:**
- No runtime dataflow except stateless utility functions
- Types and utilities consumed at build time by all TRDR packages

**Main Exports:**

| Export         | Description                                 |
|---------------|---------------------------------------------|
| `Candle`      | OHLCV candle type for market data           |
| `Order`       | Trading order type                          |
| `AgentSignal` | Trading agent signal type                   |
| `AgentConfig` | Agent configuration type                    |
| `MarketData`  | Market data snapshot type                   |
| `SystemConfig`| Full system configuration type              |
| `epochDateNow`| Utility: get current epoch timestamp        |
| `toIsoDate`   | Utility: convert to ISO date string         |
| `toEpochDate` | Utility: convert to epoch timestamp         |

**Conventions:**
- All types and utility functions must have JSDoc
- No business logic or stateful code
- Use only for type definitions and stateless helpers

**Usage:**
- Import types and utilities for use in all TRDR packages
- Use for type safety, conversions, and shared logic

**Example:**
```typescript
import type { Candle, AgentSignal } from '@trdr/shared'
import { epochDateNow, toIsoDate } from '@trdr/shared'
``` 