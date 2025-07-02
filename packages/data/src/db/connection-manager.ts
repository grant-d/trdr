import { eventBus } from '@trdr/core'
import { epochDateNow } from '@trdr/shared'
import Database from 'better-sqlite3'
import fs from 'node:fs/promises'
import path from 'node:path'

/**
 * Configuration options for SQLite connection
 */
export interface SQLiteConfig {
  /** Path to the database file. Use ':memory:' for in-memory database */
  readonly databasePath: string
  /** Enable/disable query logging */
  readonly enableLogging?: boolean
  /** Enable WAL mode for better concurrency */
  readonly enableWAL?: boolean
  /** Busy timeout in milliseconds */
  readonly busyTimeout?: number
}

/**
 * Connection manager for SQLite database
 * Handles database initialization, connection lifecycle, and configuration
 */
export class ConnectionManager {
  private static instance: ConnectionManager
  private static readonly testInstances = new Map<string, ConnectionManager>()
  private db: Database.Database | null = null
  private readonly config: SQLiteConfig
  private isInitialized = false

  private constructor(config: SQLiteConfig) {
    this.config = config
  }

  /**
   * Get singleton instance of ConnectionManager
   */
  static getInstance(config?: SQLiteConfig): ConnectionManager {
    // For test environments, create separate instances per database path
    if (config && process.env.NODE_ENV === 'test') {
      const key = config.databasePath
      if (!ConnectionManager.testInstances.has(key)) {
        ConnectionManager.testInstances.set(key, new ConnectionManager(config))
      }
      return ConnectionManager.testInstances.get(key)!
    }

    // For production, use singleton
    if (!ConnectionManager.instance) {
      if (!config) {
        throw new Error('Configuration required for first initialization')
      }
      ConnectionManager.instance = new ConnectionManager(config)
    }
    return ConnectionManager.instance
  }

  /**
   * Reset all instances (for testing)
   */
  static resetInstances(): void {
    ConnectionManager.instance = null as unknown as ConnectionManager
    ConnectionManager.testInstances.clear()
  }

  /**
   * Initialize the database connection
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      return
    }

    try {
      // Ensure directory exists for file-based databases
      if (this.config.databasePath !== ':memory:') {
        const dir = path.dirname(this.config.databasePath)
        await fs.mkdir(dir, { recursive: true })
      }

      // Create SQLite connection
      this.db = new Database(this.config.databasePath, {
        verbose: this.config.enableLogging ? console.warn : undefined,
      })

      // Configure database settings
      if (this.config.enableWAL && this.db) {
        this.db.pragma('journal_mode = WAL')
      }

      if (this.config.busyTimeout && this.db) {
        this.db.pragma(`busy_timeout = ${this.config.busyTimeout}`)
      }

      // Enable foreign keys
      if (this.db) {
        this.db.pragma('foreign_keys = ON')
      }

      this.isInitialized = true

      // Emit system event
      eventBus.emit('system.info', {
        message: 'SQLite connection initialized',
        details: {
          databasePath: this.config.databasePath,
          inMemory: this.config.databasePath === ':memory:',
        },
        timestamp: epochDateNow(),
      })
    } catch (error) {
      this.isInitialized = false
      eventBus.emit('system.error', {
        error,
        context: 'SQLite connection initialization',
        severity: 'critical',
        timestamp: epochDateNow(),
      })
      throw error
    }
  }

  /**
   * Get the database instance
   */
  async getDatabase(): Promise<Database.Database> {
    if (!this.isInitialized || !this.db) {
      await this.initialize()
    }

    if (!this.db) {
      throw new Error('Database not initialized')
    }

    return this.db
  }

  /**
   * Execute a query and return results
   */
  async query<T = unknown>(sql: string, params?: unknown[]): Promise<T[]> {
    const db = await this.getDatabase()

    try {
      const stmt = db.prepare(sql)
      const result = params ? stmt.all(...params) : stmt.all()

      if (this.config.enableLogging) {
        console.warn('Query executed:', sql)
      }

      return result as T[]
    } catch (error) {
      if (this.config.enableLogging) {
        console.error('Query error:', error, '\nSQL:', sql)
      }
      throw error
    }
  }

  /**
   * Execute a statement without returning results
   */
  async execute(sql: string, params?: unknown[]): Promise<void> {
    const db = await this.getDatabase()

    try {
      const stmt = db.prepare(sql)
      if (params) {
        stmt.run(...params)
      } else {
        stmt.run()
      }

      if (this.config.enableLogging) {
        console.warn('Statement executed:', sql)
      }
    } catch (error) {
      if (this.config.enableLogging) {
        console.error('Execute error:', error, '\nSQL:', sql)
      }
      throw error
    }
  }

  /**
   * Execute multiple statements in a transaction
   * Note: SQLite transactions require synchronous functions
   */
  async transaction<T>(fn: (db: Database.Database) => T): Promise<T> {
    const db = await this.getDatabase()

    const transactionFn = db.transaction(() => fn(db))

    return transactionFn()
  }

  /**
   * Close the database connection
   */
  close(): void {
    if (!this.db) {
      return
    }

    this.db.close()
    this.db = null
    this.isInitialized = false

    eventBus.emit('system.info', {
      message: 'SQLite connection closed',
      timestamp: epochDateNow(),
    })
  }

  /**
   * Check if the database is connected
   */
  isConnected(): boolean {
    return this.isInitialized && this.db !== null && this.db?.open === true
  }

  /**
   * Get database statistics
   */
  async getStats(): Promise<{
    readonly sizeBytes?: number | null
    readonly tables: string[]
    readonly rowCounts: Record<string, number>
  }> {
    const stats: {
      sizeBytes?: number | null
      tables: string[]
      rowCounts: Record<string, number>
    } = {
      tables: [],
      rowCounts: {}
    }

    // Get database size
    if (this.config.databasePath !== ':memory:') {
      try {
        const stat = await fs.stat(this.config.databasePath)
        stats.sizeBytes = stat.size
      } catch {
        stats.sizeBytes = null
      }
    }

    // Get table information
    const tables = await this.query<{ name: string }>(`
      SELECT name 
      FROM sqlite_master 
      WHERE type = 'table' 
      AND name NOT LIKE 'sqlite_%'
    `)
    stats.tables = tables.map(t => t.name)

    // Get row counts for each table
    for (const table of tables) {
      const result = await this.query<{ count: number }>(`SELECT COUNT(*) as count FROM ${table.name}`)
      stats.rowCounts[table.name] = result[0]?.count || 0
    }

    return stats
  }
}

/**
 * Helper function to create a connection manager with default config
 */
export function createConnectionManager(config: Partial<SQLiteConfig> = {}): ConnectionManager {
  const defaultConfig: SQLiteConfig = {
    databasePath: process.env.SQLITE_PATH || './data/trdr.db',
    enableLogging: process.env.NODE_ENV === 'development',
    enableWAL: true,
    busyTimeout: 5000,
    ...config,
  }

  return ConnectionManager.getInstance(defaultConfig)
}

// Re-export types for backward compatibility
export type { SQLiteConfig as DuckDBConfig }
