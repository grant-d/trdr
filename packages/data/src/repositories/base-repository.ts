import { ConnectionManager } from '../db/connection-manager'
import type Database from 'better-sqlite3'

/**
 * Base repository class with common database operations
 */
export abstract class BaseRepository<T> {
  protected abstract readonly tableName: string

  constructor(protected readonly connectionManager: ConnectionManager) {
  }

  /**
   * Insert a single record
   */
  protected async insert(data: Partial<T>): Promise<void> {
    const fields = Object.keys(data)
    const values = Object.values(data)
    const placeholders = fields.map(() => '?').join(', ')

    const sql = `INSERT INTO ${this.tableName} (${fields.join(', ')}) VALUES (${placeholders})`
    await this.connectionManager.execute(sql, values)
  }

  /**
   * Insert multiple records
   */
  protected async insertBatch(records: Partial<T>[]): Promise<void> {
    if (records.length === 0) return

    const firstRecord = records[0]
    if (!firstRecord) return

    // SQLite handles batch inserts differently - use a transaction for performance
    await this.connectionManager.transaction((db) => {
      const fields = Object.keys(firstRecord)
      const placeholders = fields.map(() => '?').join(', ')
      const sql = `INSERT INTO ${this.tableName} (${fields.join(', ')}) VALUES (${placeholders})`

      const stmt = db.prepare(sql)
      for (const record of records) {
        const values = fields.map(field => (record as any)[field])
        stmt.run(...values)
      }
    })
  }

  /**
   * Update records
   */
  protected async update(
    data: Partial<T>,
    where: string,
    whereParams: any[] = [],
  ): Promise<void> {
    const fields = Object.keys(data)
    const setClause = fields.map(field => `${field} = ?`).join(', ')
    const values = [...Object.values(data), ...whereParams]

    const sql = `UPDATE ${this.tableName} SET ${setClause} WHERE ${where}`
    await this.connectionManager.execute(sql, values)
  }

  /**
   * Delete records
   */
  protected async delete(where: string, params: any[] = []): Promise<void> {
    const sql = `DELETE FROM ${this.tableName} WHERE ${where}`
    await this.connectionManager.execute(sql, params)
  }

  /**
   * Find one record
   */
  protected async findOne(where: string, params: any[] = []): Promise<T | null> {
    const sql = `SELECT * FROM ${this.tableName} WHERE ${where} LIMIT 1`
    const results = await this.connectionManager.query<T>(sql, params)
    return results[0] || null
  }

  /**
   * Find multiple records
   */
  protected async findMany(
    where?: string,
    params: any[] = [],
    orderBy?: string,
    limit?: number,
    offset?: number,
  ): Promise<T[]> {
    let sql = `SELECT * FROM ${this.tableName}`

    if (where) {
      sql += ` WHERE ${where}`
    }

    if (orderBy) {
      sql += ` ORDER BY ${orderBy}`
    }

    if (limit) {
      sql += ` LIMIT ${limit}`
    }

    if (offset) {
      sql += ` OFFSET ${offset}`
    }

    return this.connectionManager.query<T>(sql, params)
  }

  /**
   * Count records
   */
  protected async count(where?: string, params: any[] = []): Promise<number> {
    let sql = `SELECT COUNT(*) as count FROM ${this.tableName}`

    if (where) {
      sql += ` WHERE ${where}`
    }

    const results = await this.connectionManager.query<{ count: number }>(sql, params)
    return results[0]?.count || 0
  }

  /**
   * Execute raw query
   */
  protected async query<R = any>(sql: string, params: any[] = []): Promise<R[]> {
    return this.connectionManager.query<R>(sql, params)
  }

  /**
   * Execute in transaction
   */
  protected async transaction<R>(
    fn: (db: Database.Database) => R,
  ): Promise<R> {
    return this.connectionManager.transaction(fn)
  }
}
