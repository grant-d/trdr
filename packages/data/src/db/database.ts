import { epochDateNow } from '@trdr/shared'
import type { EventBus } from '@trdr/types'
import type { ConnectionManager } from './connection-manager'
import { getAllSchemaStatements } from './schema'

/**
 * Initialize the database with all required tables and indexes
 */
export async function initializeDatabase(connectionManager: ConnectionManager, eventBus?: EventBus): Promise<void> {
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

    if (eventBus) {
      eventBus.emit('system.info', {
        message: 'Database schema initialized successfully',
        details: {
          tablesCreated: statements.filter(s => s.includes('CREATE TABLE')).length,
          indexesCreated: statements.filter(s => s.includes('CREATE INDEX')).length,
        },
        timestamp: epochDateNow()
      })
    }
  } catch (error) {
    if (eventBus) {
      eventBus.emit('system.error', {
        error,
        context: 'Database initialization',
        severity: 'critical',
        timestamp: epochDateNow()
      })
    }
    throw error
  }
}

/**
 * Drop all tables from the database
 */
export async function dropAllTables(connectionManager: ConnectionManager, eventBus?: EventBus): Promise<void> {
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

    if (eventBus) {
      eventBus.emit('system.info', {
        message: 'All database tables dropped successfully',
        timestamp: epochDateNow()
      })
    }
  } catch (error) {
    if (eventBus) {
      eventBus.emit('system.error', {
        error,
        context: 'Database table drop',
        severity: 'warning',
        timestamp: epochDateNow()
      })
    }
    throw error
  }
}
