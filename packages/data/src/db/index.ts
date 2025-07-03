import { epochDateNow } from '@trdr/shared'
import { AgentRepository } from '../repositories/agent-repository'
import { MarketDataRepository } from '../repositories/market-data-repository'
import { OrderRepository } from '../repositories/order-repository'
import { TradeRepository } from '../repositories/trade-repository'
import type { EventBus } from '@trdr/types'
import type { ConnectionManager, SQLiteConfig } from './connection-manager'
import { createConnectionManager } from './connection-manager'
import { MigrationRunner } from './migrations'

export { ConnectionManager, createConnectionManager } from './connection-manager'
export type { SQLiteConfig } from './connection-manager'
export { MigrationRunner, MIGRATIONS } from './migrations'
export type { Migration } from './migrations'
export { CURRENT_SCHEMA_VERSION, getAllSchemaStatements } from './schema'

/**
 * Database instance with all repositories
 */
export class Database {
  readonly connectionManager: ConnectionManager
  readonly migrationRunner: MigrationRunner
  readonly marketData: MarketDataRepository
  readonly orders: OrderRepository
  readonly trades: TradeRepository
  readonly agents: AgentRepository
  private readonly eventBus?: EventBus

  constructor(config: Partial<SQLiteConfig> = {}, eventBus?: EventBus) {
    this.connectionManager = createConnectionManager(config)
    this.migrationRunner = new MigrationRunner(this.connectionManager)
    this.eventBus = eventBus
    
    // Set event bus on connection manager if provided
    if (eventBus) {
      this.connectionManager.setEventBus(eventBus)
    }

    // Initialize repositories
    this.marketData = new MarketDataRepository(this.connectionManager)
    this.orders = new OrderRepository(this.connectionManager)
    this.trades = new TradeRepository(this.connectionManager)
    this.agents = new AgentRepository(this.connectionManager)
  }

  /**
   * Initialize the database and run migrations
   */
  async initialize(): Promise<void> {
    try {
      // Initialize connection
      await this.connectionManager.initialize()

      // Run migrations
      await this.migrationRunner.migrate()

      if (this.eventBus) {
        this.eventBus.emit('system.info', {
          message: 'Database initialized successfully',
          timestamp: epochDateNow(),
        })
      }
    } catch (error) {
      if (this.eventBus) {
        this.eventBus.emit('system.error', {
          error: error instanceof Error ? error : new Error(String(error)),
          context: 'Database initialization',
          severity: 'critical',
          timestamp: epochDateNow(),
        })
      }
      throw error
    }
  }

  /**
   * Close the database connection
   */
  close(): void {
    this.connectionManager.close()
  }

  /**
   * Get database statistics
   */
  async getStats(): Promise<{
    readonly connection: {
      readonly sizeBytes?: number | null
      readonly tables: string[]
      readonly rowCounts: Record<string, number>
    }
    readonly migration: {
      readonly currentVersion: number
      readonly targetVersion: number
      readonly needsMigration: boolean
    }
    readonly repositories: {
      readonly candles: number
      readonly ticks: number
      readonly orders: number
      readonly trades: number
      readonly decisions: number
      readonly checkpoints: number
    }
  }> {
    const connectionStats = await this.connectionManager.getStats()
    const migrationStatus = await this.migrationRunner.getStatus()

    return {
      connection: connectionStats,
      migration: {
        currentVersion: migrationStatus.currentVersion,
        targetVersion: migrationStatus.targetVersion,
        needsMigration: migrationStatus.pendingMigrations.length > 0,
      },
      repositories: {
        candles: connectionStats.rowCounts.candles || 0,
        ticks: connectionStats.rowCounts.market_ticks || 0,
        orders: connectionStats.rowCounts.orders || 0,
        trades: connectionStats.rowCounts.trades || 0,
        decisions: connectionStats.rowCounts.agent_decisions || 0,
        checkpoints: connectionStats.rowCounts.checkpoints || 0,
      },
    }
  }

  /**
   * Run database cleanup
   */
  async cleanup(daysToKeep = 90): Promise<{
    marketData: { candlesDeleted: number; ticksDeleted: number }
    ordersDeleted: number
    tradesDeleted: number
    agentData: {
      decisionsDeleted: number
      consensusDeleted: number
      checkpointsDeleted: number
    }
  }> {
    // Delete in order to respect foreign key constraints
    // Trades must be deleted before orders
    const tradesDeleted = await this.trades.cleanup(daysToKeep)
    const ordersDeleted = await this.orders.cleanup(daysToKeep)

    // Market data and agent data can be cleaned up in parallel
    const [marketDataCleanup, agentCleanup] = await Promise.all([
      this.marketData.cleanup(daysToKeep),
      this.agents.cleanup(daysToKeep),
    ])

    return {
      marketData: marketDataCleanup,
      ordersDeleted,
      tradesDeleted,
      agentData: agentCleanup,
    }
  }
}

/**
 * Create and initialize a database instance
 */
export async function createDatabase(config: Partial<SQLiteConfig> = {}, eventBus?: EventBus): Promise<Database> {
  const db = new Database(config, eventBus)
  await db.initialize()
  return db
}
