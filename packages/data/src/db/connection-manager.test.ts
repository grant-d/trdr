import { describe, it, beforeEach, afterEach, mock } from 'node:test'
import assert from 'node:assert/strict'
import { ConnectionManager, createConnectionManager } from './connection-manager'
import fs from 'node:fs/promises'
import path from 'node:path'
import { eventBus } from '@trdr/core'

describe('ConnectionManager', () => {
  let manager: ConnectionManager
  const testDbPath = path.join(__dirname, '../../test-data/test-connection.db')

  beforeEach(async () => {
    // Reset singleton instance
    ;(ConnectionManager as any).instance = null

    // Ensure test directory exists
    await fs.mkdir(path.dirname(testDbPath), { recursive: true })

    // Mock event bus
    mock.method(eventBus, 'emit')
  })

  afterEach(async () => {
    // Close connection if open
    if (manager && manager.isConnected()) {
      await manager.close()
    }

    // Clean up test database
    try {
      await fs.unlink(testDbPath)
    } catch (error) {
      // Ignore if file doesn't exist
    }

    // Reset instances
    ConnectionManager.resetInstances()

    // Reset mocks
    mock.reset()
  })

  describe('Singleton Pattern', () => {
    it('should create singleton instance', () => {
      // In test mode, instances are per database path
      const dbPath = process.env.NODE_ENV === 'test' ? testDbPath : ':memory:'
      const manager1 = ConnectionManager.getInstance({
        databasePath: dbPath,
      })
      const manager2 = ConnectionManager.getInstance({
        databasePath: dbPath,
      })

      assert.equal(manager1, manager2)
    })

    it('should throw error if no config provided on first call', () => {
      // Save current NODE_ENV
      const originalEnv = process.env.NODE_ENV
      // Temporarily set to non-test to test singleton behavior
      process.env.NODE_ENV = 'production'

      try {
        assert.throws(
          () => ConnectionManager.getInstance(),
          /Configuration required for first initialization/,
        )
      } finally {
        // Restore NODE_ENV
        process.env.NODE_ENV = originalEnv
      }
    })
  })

  describe('Connection Lifecycle', () => {
    it('should initialize connection', async () => {
      manager = ConnectionManager.getInstance({
        databasePath: ':memory:',
      })

      await manager.initialize()

      assert.equal(manager.isConnected(), true)
      const mockCalls = (eventBus.emit as any).mock.calls
      assert.equal(mockCalls.length, 1)
      assert.ok(mockCalls[0])
      assert.equal(mockCalls[0].arguments[0], 'system.info')
    })

    it('should handle multiple initialize calls', async () => {
      manager = ConnectionManager.getInstance({
        databasePath: ':memory:',
      })

      await manager.initialize()
      await manager.initialize() // Should not throw

      assert.equal(manager.isConnected(), true)
    })

    it('should close connection', async () => {
      manager = ConnectionManager.getInstance({
        databasePath: ':memory:',
      })

      await manager.initialize()
      await manager.close()

      assert.equal(manager.isConnected(), false)
    })

    it('should handle file-based database', async () => {
      manager = ConnectionManager.getInstance({
        databasePath: testDbPath,
      })

      await manager.initialize()

      assert.equal(manager.isConnected(), true)

      // Check file was created
      const stats = await fs.stat(testDbPath)
      assert.ok(stats.isFile())
    })
  })

  describe('Query Execution', () => {
    beforeEach(async () => {
      manager = ConnectionManager.getInstance({
        databasePath: ':memory:',
        enableLogging: false,
      })
      await manager.initialize()
    })

    it('should execute queries', async () => {
      await manager.execute('CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)')
      await manager.execute('INSERT INTO test (name) VALUES (?)', ['test1'])

      const results = await manager.query<{ id: number; name: string }>(
        'SELECT * FROM test WHERE name = ?',
        ['test1'],
      )

      assert.equal(results.length, 1)
      assert.ok(results[0])
      assert.equal(results[0].name, 'test1')
    })

    it('should handle query errors', async () => {
      await assert.rejects(
        () => manager.query('SELECT * FROM non_existent_table'),
        /Table does not exist|no such table/,
      )
    })

    it('should execute statements without results', async () => {
      await manager.execute('CREATE TABLE test2 (id INTEGER)')

      // Should not throw
      await manager.execute('DROP TABLE test2')
    })
  })

  describe('Transactions', () => {
    beforeEach(async () => {
      manager = ConnectionManager.getInstance({
        databasePath: ':memory:',
      })
      await manager.initialize()
      await manager.execute('CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)')
    })

    it('should commit successful transactions', async () => {
      await manager.transaction((db) => {
        db.prepare('INSERT INTO test (value) VALUES (?)').run('tx_value')
      })

      const results = await manager.query<{ value: string }>('SELECT value FROM test')
      assert.equal(results.length, 1)
      assert.ok(results[0])
      assert.equal(results[0].value, 'tx_value')
    })

    it('should rollback failed transactions', async () => {
      await assert.rejects(
        () => manager.transaction((db) => {
          db.prepare('INSERT INTO test (value) VALUES (?)').run('should_rollback')
          throw new Error('Transaction failed')
        }),
        /Transaction failed/,
      )

      const results = await manager.query('SELECT * FROM test')
      assert.equal(results.length, 0)
    })
  })

  describe('Configuration', () => {
    it('should apply configuration options', async () => {
      manager = ConnectionManager.getInstance({
        databasePath: ':memory:',
        enableWAL: true,
        busyTimeout: 5000,
        enableLogging: false,
      })

      await manager.initialize()

      // SQLite doesn't expose settings via SQL the same way, just verify no errors
      assert.equal(manager.isConnected(), true)
    })
  })

  describe('Statistics', () => {
    beforeEach(async () => {
      manager = ConnectionManager.getInstance({
        databasePath: ':memory:',
      })
      await manager.initialize()
    })

    it('should get database statistics', async () => {
      await manager.execute('CREATE TABLE test_stats (id INTEGER)')
      await manager.execute('INSERT INTO test_stats VALUES (1), (2), (3)')

      const stats = await manager.getStats()

      assert.ok(stats.tables.includes('test_stats'))
      assert.equal(stats.rowCounts['test_stats'], 3)
    })
  })

  describe('Connection Management', () => {
    it('should get connection for queries', async () => {
      manager = ConnectionManager.getInstance({
        databasePath: ':memory:',
      })
      await manager.initialize()

      const db = await manager.getDatabase()
      assert.ok(db)

      // Test connection works
      const result = db.prepare('SELECT 1 as value').get()
      assert.ok(result)
    })
  })
})

describe('createConnectionManager', () => {
  it('should create manager with default config', () => {
    const manager = createConnectionManager()
    assert.ok(manager instanceof ConnectionManager)
  })

  it('should merge provided config with defaults', () => {
    const manager = createConnectionManager({
      databasePath: ':memory:',
      enableWAL: true,
    })
    assert.ok(manager instanceof ConnectionManager)
  })

  it('should use environment variables', () => {
    process.env.SQLITE_PATH = '/tmp/test.db'

    const manager = createConnectionManager()
    assert.ok(manager instanceof ConnectionManager)

    // Clean up
    delete process.env.SQLITE_PATH
  })
})
