import type { Database as DatabaseType, Statement } from 'better-sqlite3'
import Database from 'better-sqlite3'
import type { OhlcvDto } from '../models'
import { isValidOhlcv } from '../models'
import logger from '../utils/logger'
import type { AttachedDatabase, OhlcvQuery, OhlcvRepository, RepositoryConfig } from './ohlcv-repository.interface'
import { RepositoryConnectionError, RepositoryStorageError, RepositoryValidationError } from './ohlcv-repository.interface'

// Database row type definitions
interface OhlcvRow {
  timestamp: number
  symbol: string
  exchange: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface TimestampRow {
  timestamp: number | null
}

interface CountRow {
  count: number
}

interface StatsRow {
  total_records: number
  unique_symbols: number
  unique_exchanges: number
  earliest: number | null
  latest: number | null
}

interface SymbolRow {
  symbol: string
}

interface ExchangeRow {
  exchange: string
}

/**
 * SQLite-based implementation of the OhlcvRepository interface
 * Optimized for time-series data with proper indexing and transaction support
 */
export class SqliteRepository implements OhlcvRepository {
  private db?: DatabaseType
  private ready = false
  private config?: RepositoryConfig
  private defaultSchema = 'main'
  private readonly attachedSchemas = new Set<string>()

  // Prepared statements for performance
  private statements: {
    insertOhlcv?: Statement
    insertCoefficient?: Statement
    selectOhlcvByRange?: Statement
    selectOhlcvBySymbol?: Statement
    selectLastTimestamp?: Statement
    selectFirstTimestamp?: Statement
    selectCount?: Statement
    selectCoefficient?: Statement
    selectCoefficients?: Statement
    deleteCoefficients?: Statement
    selectSymbols?: Statement
    selectExchanges?: Statement
    deleteBetweenDates?: Statement
    selectStats?: Statement
  } = {}

  /**
   * Initialize the SQLite repository and set up database schema
   */
  async initialize(config: RepositoryConfig): Promise<void> {
    try {
      this.config = config
      this.defaultSchema = config.defaultSchema || 'main'
      
      // Create database connection
      this.db = new Database(config.connectionString, {
        verbose: logger.debug.bind(logger),
        ...config.options
      })

      // Enable WAL mode for better concurrent access
      this.db.exec('PRAGMA journal_mode = WAL')
      this.db.exec('PRAGMA synchronous = NORMAL')
      this.db.exec('PRAGMA cache_size = 1000000')
      this.db.exec('PRAGMA temp_store = MEMORY')
      this.db.exec('PRAGMA mmap_size = 268435456') // 256MB

      // Attach additional databases if specified
      if (config.attachedDatabases) {
        this.attachDatabases(config.attachedDatabases)
      }

      // Create tables and indexes
      this.createTables()
      this.createIndexes()
      
      // Prepare statements
      this.prepareStatements()

      this.ready = true
      logger.info('SQLite repository initialized', {
        connectionString: config.connectionString,
        defaultSchema: this.defaultSchema,
        attachedSchemas: Array.from(this.attachedSchemas),
        options: config.options
      })
      return Promise.resolve()
    } catch (error) {
      logger.error('Failed to initialize SQLite repository', { error })
      throw new RepositoryConnectionError(
        `Failed to initialize SQLite repository: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Attach additional databases for schema support
   */
  private attachDatabases(attachedDatabases: AttachedDatabase[]): void {
    if (!this.db) throw new Error('Database not initialized')

    for (const { schema, path } of attachedDatabases) {
      try {
        // Validate schema name (alphanumeric and underscore only for security)
        if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(schema)) {
          throw new Error(`Invalid schema name: ${schema}`)
        }

        // Attach the database
        this.db.exec(`ATTACH DATABASE '${path}' AS ${schema}`)
        this.attachedSchemas.add(schema)
        
        logger.info('Attached database schema', { schema, path })
      } catch (error) {
        logger.error('Failed to attach database schema', { error, schema, path })
        throw new RepositoryConnectionError(
          `Failed to attach database schema ${schema}: ${String(error)}`,
          error instanceof Error ? error : undefined
        )
      }
    }
  }

  /**
   * Get the full table name with schema prefix
   */
  private getTableName(tableName: string, schema?: string): string {
    const schemaName = schema || this.defaultSchema
    return schemaName === 'main' ? tableName : `${schemaName}.${tableName}`
  }

  /**
   * Create database tables with optimized schema
   */
  private createTables(): void {
    if (!this.db) throw new Error('Database not initialized')

    const ohlcvTable = this.getTableName('ohlcv')
    const coefficientsTable = this.getTableName('coefficients')
    const ohlcvAdditionalTable = this.getTableName('ohlcv_additional')

    // OHLCV data table with optimized column order
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS ${ohlcvTable} (
        timestamp INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        exchange TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
        CONSTRAINT pk_ohlcv PRIMARY KEY (timestamp, symbol, exchange)
      ) WITHOUT ROWID
    `)

    // Coefficients table for storing calculation results
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS ${coefficientsTable} (
        name TEXT NOT NULL,
        symbol TEXT NOT NULL DEFAULT '',
        exchange TEXT NOT NULL DEFAULT '',
        value REAL NOT NULL,
        metadata TEXT,
        timestamp INTEGER NOT NULL,
        created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
        CONSTRAINT pk_coefficients PRIMARY KEY (name, symbol, exchange, timestamp)
      ) WITHOUT ROWID
    `)

    // Additional columns table for dynamic fields added by transforms
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS ${ohlcvAdditionalTable} (
        timestamp INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        exchange TEXT NOT NULL,
        column_name TEXT NOT NULL,
        column_value TEXT NOT NULL,
        CONSTRAINT pk_ohlcv_additional PRIMARY KEY (timestamp, symbol, exchange, column_name),
        FOREIGN KEY (timestamp, symbol, exchange) 
          REFERENCES ${ohlcvTable} (timestamp, symbol, exchange) 
          ON DELETE CASCADE
      ) WITHOUT ROWID
    `)
  }

  /**
   * Create optimized indexes for time-series queries
   */
  private createIndexes(): void {
    if (!this.db) throw new Error('Database not initialized')

    const ohlcvTable = this.getTableName('ohlcv')
    const coefficientsTable = this.getTableName('coefficients')

    // Primary indexes for OHLCV data
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS ix_ohlcv_symbol_timestamp 
      ON ${ohlcvTable} (symbol, timestamp)
    `)
    
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS ix_ohlcv_exchange_timestamp 
      ON ${ohlcvTable} (exchange, timestamp)
    `)
    
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS ix_ohlcv_timestamp 
      ON ${ohlcvTable} (timestamp)
    `)

    // Indexes for coefficients
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS ix_coefficients_name 
      ON ${coefficientsTable} (name)
    `)
    
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS ix_coefficients_symbol 
      ON ${coefficientsTable} (symbol, name)
    `)
    
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS ix_coefficients_timestamp 
      ON ${coefficientsTable} (timestamp)
    `)
  }

  /**
   * Prepare all SQL statements for better performance
   */
  private prepareStatements(): void {
    if (!this.db) throw new Error('Database not initialized')

    const ohlcvTable = this.getTableName('ohlcv')
    const coefficientsTable = this.getTableName('coefficients')

    this.statements.insertOhlcv = this.db.prepare(`
      INSERT OR REPLACE INTO ${ohlcvTable} 
      (timestamp, symbol, exchange, open, high, low, close, volume)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `)

    this.statements.insertCoefficient = this.db.prepare(`
      INSERT OR REPLACE INTO ${coefficientsTable} 
      (name, symbol, exchange, value, metadata, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `)

    this.statements.selectOhlcvByRange = this.db.prepare(`
      SELECT timestamp, symbol, exchange, open, high, low, close, volume
      FROM ${ohlcvTable} 
      WHERE timestamp BETWEEN ? AND ?
        AND (? IS NULL OR symbol = ?)
        AND (? IS NULL OR exchange = ?)
      ORDER BY timestamp ASC
      LIMIT ? OFFSET ?
    `)

    this.statements.selectOhlcvBySymbol = this.db.prepare(`
      SELECT timestamp, symbol, exchange, open, high, low, close, volume
      FROM ${ohlcvTable} 
      WHERE symbol = ?
        AND (? IS NULL OR exchange = ?)
      ORDER BY timestamp DESC
      LIMIT ? OFFSET ?
    `)

    this.statements.selectLastTimestamp = this.db.prepare(`
      SELECT MAX(timestamp) as timestamp
      FROM ${ohlcvTable} 
      WHERE symbol = ?
        AND (? IS NULL OR exchange = ?)
    `)

    this.statements.selectFirstTimestamp = this.db.prepare(`
      SELECT MIN(timestamp) as timestamp
      FROM ${ohlcvTable} 
      WHERE symbol = ?
        AND (? IS NULL OR exchange = ?)
    `)

    this.statements.selectCount = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM ${ohlcvTable} 
      WHERE symbol = ?
        AND (? IS NULL OR exchange = ?)
    `)

    this.statements.selectCoefficient = this.db.prepare(`
      SELECT name, symbol, exchange, value, metadata, timestamp
      FROM ${coefficientsTable} 
      WHERE name = ?
        AND (? = '' OR symbol = ?)
        AND (? = '' OR exchange = ?)
      ORDER BY timestamp DESC
      LIMIT 1
    `)

    this.statements.selectCoefficients = this.db.prepare(`
      SELECT name, symbol, exchange, value, metadata, timestamp
      FROM ${coefficientsTable} 
      WHERE (? = '' OR name GLOB ?)
        AND (? = '' OR symbol = ?)
        AND (? = '' OR exchange = ?)
      ORDER BY name, timestamp DESC
    `)

    this.statements.deleteCoefficients = this.db.prepare(`
      DELETE FROM ${coefficientsTable} 
      WHERE name GLOB ?
        AND (? = '' OR symbol = ?)
        AND (? = '' OR exchange = ?)
    `)

    this.statements.selectSymbols = this.db.prepare(`
      SELECT DISTINCT symbol
      FROM ${ohlcvTable} 
      WHERE (? IS NULL OR exchange = ?)
      ORDER BY symbol
    `)

    this.statements.selectExchanges = this.db.prepare(`
      SELECT DISTINCT exchange
      FROM ${ohlcvTable} 
      ORDER BY exchange
    `)

    this.statements.deleteBetweenDates = this.db.prepare(`
      DELETE FROM ${ohlcvTable} 
      WHERE timestamp BETWEEN ? AND ?
        AND (? IS NULL OR symbol = ?)
        AND (? IS NULL OR exchange = ?)
    `)

    this.statements.selectStats = this.db.prepare(`
      SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT symbol) as unique_symbols,
        COUNT(DISTINCT exchange) as unique_exchanges,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest
      FROM ${ohlcvTable}
    `)
  }

  /**
   * Save a single OHLCV record
   */
  async save(data: OhlcvDto): Promise<void> {
    this.ensureReady()
    
    if (!isValidOhlcv(data)) {
      throw new RepositoryValidationError('Invalid OHLCV data')
    }

    try {
      this.statements.insertOhlcv!.run(
        data.timestamp,
        data.symbol,
        data.exchange,
        data.open,
        data.high,
        data.low,
        data.close,
        data.volume
      )

      // Handle additional columns if present
      this.saveAdditionalColumns(data)
      return Promise.resolve()
    } catch (error) {
      logger.error('Failed to save OHLCV data', { error, data })
      throw new RepositoryStorageError(
        `Failed to save OHLCV data: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Save multiple OHLCV records in a transaction
   */
  async saveMany(data: OhlcvDto[]): Promise<void> {
    this.ensureReady()

    if (data.length === 0) return

    // Validate all data first
    for (const item of data) {
      if (!isValidOhlcv(item)) {
        throw new RepositoryValidationError(`Invalid OHLCV data for ${item.symbol}`)
      }
    }

    try {
      const transaction = this.db!.transaction((items: OhlcvDto[]) => {
        for (const item of items) {
          this.statements.insertOhlcv!.run(
            item.timestamp,
            item.symbol,
            item.exchange,
            item.open,
            item.high,
            item.low,
            item.close,
            item.volume
          )
        }
      })

      transaction(data)
      logger.debug('Saved OHLCV batch', { count: data.length })
      return Promise.resolve()
    } catch (error) {
      logger.error('Failed to save OHLCV batch', { error, count: data.length })
      throw new RepositoryStorageError(
        `Failed to save OHLCV batch: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Append a batch of OHLCV records (alias for saveMany for consistency)
   */
  async appendBatch(data: OhlcvDto[]): Promise<void> {
    return this.saveMany(data)
  }

  /**
   * Save additional columns for OHLCV data
   */
  private saveAdditionalColumns(data: OhlcvDto): void {
    const additionalFields = Object.keys(data).filter(
      key => !['timestamp', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume'].includes(key)
    )

    if (additionalFields.length === 0) return

    const ohlcvAdditionalTable = this.getTableName('ohlcv_additional')
    const insertAdditional = this.db!.prepare(`
      INSERT OR REPLACE INTO ${ohlcvAdditionalTable} 
      (timestamp, symbol, exchange, column_name, column_value)
      VALUES (?, ?, ?, ?, ?)
    `)

    for (const field of additionalFields) {
      const value = data[field]
      insertAdditional.run(
        data.timestamp,
        data.symbol,
        data.exchange,
        field,
        typeof value === 'string' ? value : String(value)
      )
    }
  }

  /**
   * Get OHLCV data within a specific date range
   */
  async getBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<OhlcvDto[]> {
    this.ensureReady()

    try {
      const rows = this.statements.selectOhlcvByRange!.all(
        startTime,
        endTime,
        symbol || null,
        symbol || null,
        exchange || null,
        exchange || null,
        10000, // Default limit
        0      // No offset
      ) as OhlcvRow[]

      return Promise.resolve(this.mapRowsToOhlcv(rows))
    } catch (error) {
      logger.error('Failed to get OHLCV data by date range', { error, startTime, endTime })
      throw new RepositoryStorageError(
        `Failed to get OHLCV data by date range: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get OHLCV data for a specific symbol
   */
  async getBySymbol(
    symbol: string,
    exchange?: string,
    limit?: number,
    offset?: number
  ): Promise<OhlcvDto[]> {
    this.ensureReady()

    try {
      const rows = this.statements.selectOhlcvBySymbol!.all(
        symbol,
        exchange || null,
        exchange || null,
        limit || 1000,
        offset || 0
      ) as OhlcvRow[]

      return Promise.resolve(this.mapRowsToOhlcv(rows))
    } catch (error) {
      logger.error('Failed to get OHLCV data by symbol', { error, symbol })
      throw new RepositoryStorageError(
        `Failed to get OHLCV data by symbol: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get OHLCV data using flexible query parameters
   */
  async query(query: OhlcvQuery): Promise<OhlcvDto[]> {
    if (query.startTime && query.endTime) {
      return this.getBetweenDates(
        query.startTime,
        query.endTime,
        query.symbol,
        query.exchange
      )
    } else if (query.symbol) {
      return this.getBySymbol(
        query.symbol,
        query.exchange,
        query.limit,
        query.offset
      )
    } else {
      // General query - build dynamic SQL
      return this.executeQuery(query)
    }
  }

  /**
   * Execute a dynamic query based on parameters
   */
  private async executeQuery(query: OhlcvQuery): Promise<OhlcvDto[]> {
    this.ensureReady()

    const ohlcvTable = this.getTableName('ohlcv')
    let sql = `SELECT timestamp, symbol, exchange, open, high, low, close, volume FROM ${ohlcvTable} WHERE 1=1`
    const params: (string | number)[] = []

    if (query.symbol) {
      sql += ' AND symbol = ?'
      params.push(query.symbol)
    }

    if (query.exchange) {
      sql += ' AND exchange = ?'
      params.push(query.exchange)
    }

    if (query.startTime) {
      sql += ' AND timestamp >= ?'
      params.push(query.startTime)
    }

    if (query.endTime) {
      sql += ' AND timestamp <= ?'
      params.push(query.endTime)
    }

    sql += ' ORDER BY timestamp ASC'

    if (query.limit) {
      sql += ' LIMIT ?'
      params.push(query.limit)
    }

    if (query.offset) {
      sql += ' OFFSET ?'
      params.push(query.offset)
    }

    try {
      const statement = this.db!.prepare(sql)
      const rows = statement.all(...params) as OhlcvRow[]
      return Promise.resolve(this.mapRowsToOhlcv(rows))
    } catch (error) {
      logger.error('Failed to execute query', { error, query })
      throw new RepositoryStorageError(
        `Failed to execute query: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Map database rows to OhlcvDto objects
   */
  private mapRowsToOhlcv(rows: OhlcvRow[]): OhlcvDto[] {
    return rows.map(row => ({
      timestamp: row.timestamp,
      symbol: row.symbol,
      exchange: row.exchange,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
      volume: row.volume
    }))
  }

  /**
   * Get the most recent timestamp for a specific symbol
   */
  async getLastTimestamp(symbol: string, exchange?: string): Promise<number | null> {
    this.ensureReady()

    try {
      const row = this.statements.selectLastTimestamp!.get(
        symbol,
        exchange || null,
        exchange || null
      ) as TimestampRow | undefined

      return Promise.resolve(row?.timestamp ?? null)
    } catch (error) {
      logger.error('Failed to get last timestamp', { error, symbol })
      throw new RepositoryStorageError(
        `Failed to get last timestamp: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get the earliest timestamp for a specific symbol
   */
  async getFirstTimestamp(symbol: string, exchange?: string): Promise<number | null> {
    this.ensureReady()

    try {
      const row = this.statements.selectFirstTimestamp!.get(
        symbol,
        exchange || null,
        exchange || null
      ) as TimestampRow | undefined

      return Promise.resolve(row?.timestamp ?? null)
    } catch (error) {
      logger.error('Failed to get first timestamp', { error, symbol })
      throw new RepositoryStorageError(
        `Failed to get first timestamp: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get count of records for a symbol
   */
  async getCount(symbol: string, exchange?: string): Promise<number> {
    this.ensureReady()

    try {
      const row = this.statements.selectCount!.get(
        symbol,
        exchange || null,
        exchange || null
      ) as CountRow | undefined

      return Promise.resolve(row?.count ?? 0)
    } catch (error) {
      logger.error('Failed to get count', { error, symbol })
      throw new RepositoryStorageError(
        `Failed to get count: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get all unique symbols in the repository
   */
  async getSymbols(exchange?: string): Promise<string[]> {
    this.ensureReady()

    try {
      const rows = this.statements.selectSymbols!.all(
        exchange || null,
        exchange || null
      ) as SymbolRow[]

      return Promise.resolve(rows.map(row => row.symbol))
    } catch (error) {
      logger.error('Failed to get symbols', { error })
      throw new RepositoryStorageError(
        `Failed to get symbols: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get all unique exchanges in the repository
   */
  async getExchanges(): Promise<string[]> {
    this.ensureReady()

    try {
      const rows = this.statements.selectExchanges!.all() as ExchangeRow[]
      return Promise.resolve(rows.map(row => row.exchange))
    } catch (error) {
      logger.error('Failed to get exchanges', { error })
      throw new RepositoryStorageError(
        `Failed to get exchanges: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Delete OHLCV data within a specific date range
   */
  async deleteBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<number> {
    this.ensureReady()

    try {
      const result = this.statements.deleteBetweenDates!.run(
        startTime,
        endTime,
        symbol || null,
        symbol || null,
        exchange || null,
        exchange || null
      )

      return Promise.resolve(result.changes)
    } catch (error) {
      logger.error('Failed to delete between dates', { error, startTime, endTime })
      throw new RepositoryStorageError(
        `Failed to delete between dates: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Check if the repository is properly initialized and ready to use
   */
  isReady(): boolean {
    return this.ready && !!this.db
  }

  /**
   * Get repository statistics and health information
   */
  async getStats(): Promise<{
    totalRecords: number
    uniqueSymbols: number
    uniqueExchanges: number
    dataDateRange: {
      earliest: number | null
      latest: number | null
    }
    storageSize?: number
  }> {
    this.ensureReady()

    try {
      const row = this.statements.selectStats!.get() as StatsRow | undefined

      // Get database file size if available
      let storageSize: number | undefined
      if (this.config?.connectionString !== ':memory:') {
        try {
          const fs = await import('node:fs/promises')
          const stats = await fs.stat(this.config!.connectionString)
          storageSize = stats.size
        } catch {
          // File may not exist yet or access denied
        }
      }

      return Promise.resolve({
        totalRecords: row?.total_records ?? 0,
        uniqueSymbols: row?.unique_symbols ?? 0,
        uniqueExchanges: row?.unique_exchanges ?? 0,
        dataDateRange: {
          earliest: row?.earliest ?? null,
          latest: row?.latest ?? null
        },
        storageSize
      })
    } catch (error) {
      logger.error('Failed to get stats', { error })
      throw new RepositoryStorageError(
        `Failed to get stats: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Flush any pending writes to ensure data persistence
   */
  async flush(): Promise<void> {
    this.ensureReady()

    try {
      // SQLite automatically flushes with WAL mode, but we can force a checkpoint
      this.db!.exec('PRAGMA wal_checkpoint(PASSIVE)')
      return Promise.resolve()
    } catch (error) {
      logger.error('Failed to flush SQLite database', { error })
      throw new RepositoryStorageError(
        `Failed to flush database: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Close the repository and clean up resources
   */
  async close(): Promise<void> {
    if (this.db) {
      try {
        await this.flush()
        this.db.close()
        this.ready = false
        logger.info('SQLite repository closed')
      } catch (error) {
        logger.error('Error closing SQLite repository', { error })
        throw new RepositoryStorageError(
          `Error closing repository: ${String(error)}`,
          error instanceof Error ? error : undefined
        )
      } finally {
        this.db = undefined
        this.statements = {}
      }
    }
  }

  /**
   * Ensure the repository is ready for operations
   */
  private ensureReady(): void {
    if (!this.ready || !this.db) {
      throw new RepositoryConnectionError('Repository not initialized')
    }
  }
}
