import { ConnectionManager } from './connection-manager'
import { getAllSchemaStatements } from './schema'
import { eventBus } from '@trdr/core'

/**
 * Initialize the database with all required tables and indexes
 */
export async function initializeDatabase(connectionManager: ConnectionManager): Promise<void> {
  const db = await connectionManager.getDatabase()

  try {
    // Get all schema statements
    const statements = getAllSchemaStatements()

    // Execute all statements in a transaction
    await connectionManager.transaction(() => {
      for (const statement of statements) {
        db.exec(statement)
      }
    })

    eventBus.emit('system.info', {
      message: 'Database schema initialized successfully',
      details: {
        tablesCreated: statements.filter(s => s.includes('CREATE TABLE')).length,
        indexesCreated: statements.filter(s => s.includes('CREATE INDEX')).length,
      },
      timestamp: new Date(),
    })
  } catch (error) {
    eventBus.emit('system.error', {
      error,
      context: 'Database initialization',
      severity: 'critical',
      timestamp: new Date(),
    })
    throw error
  }
}

/**
 * Drop all tables from the database
 */
export async function dropAllTables(connectionManager: ConnectionManager): Promise<void> {
  const tables = [
    'candles',
    'market_ticks',
    'orders',
    'trades',
    'agent_decisions',
    'agent_consensus',
    'checkpoints',
    'positions',
    'schema_migrations',
  ]

  try {
    await connectionManager.transaction((db) => {
      // Disable foreign key constraints temporarily
      db.pragma('foreign_keys = OFF')

      for (const table of tables) {
        db.exec(`DROP TABLE IF EXISTS ${table}`)
      }

      // Re-enable foreign key constraints
      db.pragma('foreign_keys = ON')
    })

    eventBus.emit('system.info', {
      message: 'All database tables dropped successfully',
      timestamp: new Date(),
    })
  } catch (error) {
    eventBus.emit('system.error', {
      error,
      context: 'Database table drop',
      severity: 'warning',
      timestamp: new Date(),
    })
    throw error
  }
}
