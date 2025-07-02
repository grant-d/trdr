/**
 * Database migrations for SQLite
 */

import type { ConnectionManager } from './connection-manager'
import type Database from 'better-sqlite3'

export interface Migration {
  version: number
  description: string
  up: (db: Database.Database) => void
  down?: (db: Database.Database) => void
}

/**
 * List of all migrations
 */
export const MIGRATIONS: Migration[] = [
  {
    version: 1,
    description: 'Initial schema',
    up: (db) => {
      // Import schema statements and create all tables
      const { getAllSchemaStatements } = require('./schema')
      const statements = getAllSchemaStatements()
      
      for (const statement of statements) {
        db.exec(statement)
      }
    }
  }
]

/**
 * Migration runner for SQLite databases
 */
export class MigrationRunner {
  constructor(private readonly connectionManager: ConnectionManager) {}

  /**
   * Run all pending migrations
   */
  async migrate(): Promise<void> {
    const db = await this.connectionManager.getDatabase()
    
    // Create migrations table if it doesn't exist
    db.exec(`
      CREATE TABLE IF NOT EXISTS migrations (
        version INTEGER PRIMARY KEY,
        description TEXT NOT NULL,
        applied_at INTEGER DEFAULT (strftime('%s', 'now') * 1000)
      )
    `)
    
    // Get current version
    const currentVersion = this.getCurrentVersion(db)
    
    // Run pending migrations
    const pendingMigrations = MIGRATIONS.filter(m => m.version > currentVersion)
    
    for (const migration of pendingMigrations) {
      await this.runMigration(db, migration)
    }
  }

  /**
   * Get migration status
   */
  async getStatus(): Promise<{
    currentVersion: number
    targetVersion: number
    pendingMigrations: Migration[]
  }> {
    const db = await this.connectionManager.getDatabase()
    const currentVersion = this.getCurrentVersion(db)
    const targetVersion = MIGRATIONS[MIGRATIONS.length - 1]?.version || 0
    const pendingMigrations = MIGRATIONS.filter(m => m.version > currentVersion)
    
    return {
      currentVersion,
      targetVersion,
      pendingMigrations
    }
  }

  /**
   * Get current migration version
   */
  private getCurrentVersion(db: Database.Database): number {
    try {
      const result = db.prepare('SELECT MAX(version) as version FROM migrations').get() as { version: number | null }
      return result.version || 0
    } catch {
      return 0
    }
  }

  /**
   * Run a single migration
   */
  private async runMigration(db: Database.Database, migration: Migration): Promise<void> {
    await this.connectionManager.transaction(() => {
      migration.up(db)
      
      // Record migration
      db.prepare('INSERT INTO migrations (version, description) VALUES (?, ?)').run(
        migration.version,
        migration.description
      )
    })
  }
}