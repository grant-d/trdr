/**
 * Database schema definitions for TRDR
 */

/**
 * SQL statements for creating database tables
 */
export const SCHEMA_STATEMENTS = {
  // Market data tables
  candles: `
    CREATE TABLE IF NOT EXISTS candles (
      id INTEGER PRIMARY KEY,
      symbol TEXT NOT NULL,

      interval TEXT NOT NULL,
      open_time TEXT NOT NULL,
      close_time TEXT NOT NULL,

      open REAL NOT NULL,
      high REAL NOT NULL,
      low REAL NOT NULL,
      close REAL NOT NULL,
      volume REAL NOT NULL,

      quote_volume REAL,
      trades_count INTEGER,

      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
  `,

  candles_indexes: [
    'CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval ON candles(symbol, interval)',
    'CREATE INDEX IF NOT EXISTS idx_candles_open_time ON candles(open_time)',
    'CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON candles(symbol, open_time)',
  ],

  market_ticks: `
    CREATE TABLE IF NOT EXISTS market_ticks (
      id INTEGER PRIMARY KEY,
      symbol TEXT NOT NULL,

      price REAL NOT NULL,
      volume REAL NOT NULL,
      timestamp TEXT NOT NULL,

      bid REAL,
      ask REAL,
      bid_size REAL,
      ask_size REAL,

      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
  `,

  market_ticks_indexes: [
    'CREATE INDEX IF NOT EXISTS idx_ticks_symbol ON market_ticks(symbol)',
    'CREATE INDEX IF NOT EXISTS idx_ticks_timestamp ON market_ticks(timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON market_ticks(symbol, timestamp)',
  ],

  // Order tables
  orders: `
    CREATE TABLE IF NOT EXISTS orders (
      id TEXT PRIMARY KEY,
      symbol TEXT NOT NULL,

      side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
      type TEXT NOT NULL CHECK (type IN ('market', 'limit', 'stop', 'trailing')),
      status TEXT NOT NULL CHECK (status IN ('pending', 'open', 'submitted', 'partial', 'filled', 'cancelled', 'rejected', 'expired')),

      price REAL,
      size REAL NOT NULL,
      filled_size REAL DEFAULT 0,
      average_fill_price REAL,

      stop_price REAL,
      trail_distance REAL,

      agent_id TEXT,

      metadata TEXT,

      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
      submitted_at TEXT,
      filled_at TEXT,
      cancelled_at TEXT
    )
  `,

  orders_indexes: [
    'CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)',
    'CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)',
    'CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at)',
    'CREATE INDEX IF NOT EXISTS idx_orders_agent_id ON orders(agent_id)',
  ],

  // Trade tables
  trades: `
    CREATE TABLE IF NOT EXISTS trades (
      id TEXT PRIMARY KEY,
      order_id TEXT NOT NULL REFERENCES orders(id),
      symbol TEXT NOT NULL,
      side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
      price REAL NOT NULL,
      size REAL NOT NULL,
      fee REAL DEFAULT 0,
      fee_currency TEXT,
      pnl REAL,
      metadata TEXT,
      executed_at TEXT NOT NULL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
  `,

  trades_indexes: [
    'CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades(order_id)',
    'CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)',
    'CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at)',
  ],

  // Agent decision tables
  agent_decisions: `
    CREATE TABLE IF NOT EXISTS agent_decisions (
      id INTEGER PRIMARY KEY,
      agent_id TEXT NOT NULL,
      agent_type TEXT NOT NULL,
      symbol TEXT NOT NULL,
      action TEXT NOT NULL CHECK (action IN ('TRAIL_BUY', 'TRAIL_SELL', 'HOLD')),
      confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
      trail_distance REAL,
      reasoning TEXT NOT NULL,
      market_context TEXT,
      timestamp TEXT NOT NULL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
  `,

  agent_decisions_indexes: [
    'CREATE INDEX IF NOT EXISTS idx_decisions_agent_id ON agent_decisions(agent_id)',
    'CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON agent_decisions(symbol)',
    'CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON agent_decisions(timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_decisions_agent_time ON agent_decisions(agent_id, timestamp)',
  ],

  agent_consensus: `
    CREATE TABLE IF NOT EXISTS agent_consensus (
      id INTEGER PRIMARY KEY,
      symbol TEXT NOT NULL,
      decision TEXT NOT NULL CHECK (decision IN ('buy', 'sell', 'hold')),
      confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
      dissent REAL NOT NULL CHECK (dissent >= 0 AND dissent <= 1),
      votes TEXT NOT NULL,
      timestamp TEXT NOT NULL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
  `,

  agent_consensus_indexes: [
    'CREATE INDEX IF NOT EXISTS idx_consensus_symbol ON agent_consensus(symbol)',
    'CREATE INDEX IF NOT EXISTS idx_consensus_timestamp ON agent_consensus(timestamp)',
  ],

  // Checkpoint tables
  checkpoints: `
    CREATE TABLE IF NOT EXISTS checkpoints (
      id TEXT PRIMARY KEY,
      type TEXT NOT NULL,
      version INTEGER NOT NULL,
      state TEXT NOT NULL,
      metadata TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
  `,

  checkpoints_indexes: [
    'CREATE INDEX IF NOT EXISTS idx_checkpoints_type ON checkpoints(type)',
    'CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at)',
  ],

  // Performance tracking tables
  positions: `
    CREATE TABLE IF NOT EXISTS positions (
      id TEXT PRIMARY KEY,
      symbol TEXT NOT NULL,
      side TEXT NOT NULL CHECK (side IN ('long', 'short')),
      size REAL NOT NULL,
      entry_price REAL NOT NULL,
      current_price REAL,
      unrealized_pnl REAL,
      realized_pnl REAL DEFAULT 0,
      status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
      opened_at TEXT NOT NULL,
      closed_at TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
  `,

  positions_indexes: [
    'CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)',
    'CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)',
    'CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON positions(opened_at)',
  ],

  // Migration tracking
  schema_migrations: `
    CREATE TABLE IF NOT EXISTS schema_migrations (
      version INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      applied_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
  `,
}

/**
 * Get all schema creation statements in order
 */
export function getAllSchemaStatements(): string[] {
  const statements: string[] = []

  // Add table creation statements
  statements.push(SCHEMA_STATEMENTS.candles)
  statements.push(SCHEMA_STATEMENTS.market_ticks)
  statements.push(SCHEMA_STATEMENTS.orders)
  statements.push(SCHEMA_STATEMENTS.trades)
  statements.push(SCHEMA_STATEMENTS.agent_decisions)
  statements.push(SCHEMA_STATEMENTS.agent_consensus)
  statements.push(SCHEMA_STATEMENTS.checkpoints)
  statements.push(SCHEMA_STATEMENTS.positions)
  statements.push(SCHEMA_STATEMENTS.schema_migrations)

  // Add indexes
  statements.push(...SCHEMA_STATEMENTS.candles_indexes)
  statements.push(...SCHEMA_STATEMENTS.market_ticks_indexes)
  statements.push(...SCHEMA_STATEMENTS.orders_indexes)
  statements.push(...SCHEMA_STATEMENTS.trades_indexes)
  statements.push(...SCHEMA_STATEMENTS.agent_decisions_indexes)
  statements.push(...SCHEMA_STATEMENTS.agent_consensus_indexes)
  statements.push(...SCHEMA_STATEMENTS.checkpoints_indexes)
  statements.push(...SCHEMA_STATEMENTS.positions_indexes)

  return statements
}

/**
 * Schema version for tracking migrations
 */
export const CURRENT_SCHEMA_VERSION = 1
