# @trdr/data

**Purpose:**
- Implement all database logic and data access for TRDR
- Provide repository interfaces for agents, orders, market data, trades
- No business logic, only data access and mapping
- Used by core for persistent storage

**Architecture:**
- TypeScript module
- Implements repository pattern for all entities
- Uses database adapters for storage (Postgres, SQLite, etc.)
- Exposes repository classes for agents, orders, market data, trades
- All repository interfaces must have JSDoc

**Dataflow:**
- Core package calls repository methods for all data persistence
- Data flows from core logic to repositories to database
- No business logic in this package

**Main Modules:**

| Module                | Description                                 |
|----------------------|---------------------------------------------|
| `db/connection-manager` | Manages DB connections and pooling         |
| `db/database`         | Main database interface and setup           |
| `db/schema`           | Database schema definitions                 |
| `repositories/agent-repository` | Agent data access and persistence      |
| `repositories/order-repository` | Order data access and persistence      |
| `repositories/market-data-repository` | Market data access and persistence |
| `repositories/trade-repository` | Trade data access and persistence      |

**Conventions:**
- All repository interfaces and methods must have JSDoc
- No business logic or trading logic allowed
- Only implement data access, mapping, and storage

**Usage:**
- Import repository classes in core to persist and load data
- Use for all agent, order, market data, and trade storage

**Example:**
```typescript
import { AgentRepository, OrderRepository } from '@trdr/data'
``` 