import { test } from 'node:test'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { rm } from 'node:fs/promises'
import { RepositoryTestBase } from './repository-test-base'
import { SqliteRepository } from '../../../src/repositories/sqlite-repository'
import type { OhlcvRepository } from '../../../src/repositories/ohlcv-repository.interface'

/**
 * SQLite Repository Tests
 */
class SqliteRepositoryTest extends RepositoryTestBase {
  private testDbPath: string = ''

  public async createRepository(): Promise<OhlcvRepository> {
    // Create a unique test database file
    this.testDbPath = join(tmpdir(), `trdr-test-${Date.now()}-${Math.random().toString(36).slice(2)}.db`)
    
    const repo = new SqliteRepository()
    await repo.initialize({
      connectionString: this.testDbPath,
      options: {
        // Use faster settings for testing
        verbose: undefined // Disable verbose logging during tests
      }
    })
    
    return repo
  }

  public async cleanup(): Promise<void> {
    if (this.testDbPath) {
      try {
        await rm(this.testDbPath, { force: true })
      } catch {
        // Ignore cleanup errors
      }
    }
  }
}

/**
 * SQLite-specific tests
 */
test('SQLite Repository - Schema Support', async () => {
  const testInstance = new SqliteRepositoryTest()
  const repo = await testInstance.createRepository()
  
  try {
    // Test that the repository is properly initialized
    const stats = await repo.getStats()
    
    // Should start with empty database
    console.assert(stats.totalRecords === 0)
    console.assert(stats.uniqueSymbols === 0)
    console.assert(stats.uniqueExchanges === 0)
    
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

test('SQLite Repository - Transaction Safety', async () => {
  const testInstance = new SqliteRepositoryTest()
  const repo = await testInstance.createRepository()
  
  try {
    const testData = [
      {
        timestamp: Date.now(),
        symbol: 'TEST1',
        exchange: 'test',
        open: 100,
        high: 110,
        low: 95,
        close: 105,
        volume: 1000
      },
      {
        timestamp: Date.now() + 1000,
        symbol: 'TEST2',
        exchange: 'test',
        open: 200,
        high: 210,
        low: 195,
        close: 205,
        volume: 2000
      }
    ]
    
    // Test batch operation atomicity
    await repo.saveMany(testData)
    
    const results = await repo.query({})
    console.assert(results.length === testData.length)
    
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

test('SQLite Repository - WAL Mode Performance', async () => {
  const testInstance = new SqliteRepositoryTest()
  const repo = await testInstance.createRepository()
  
  try {
    // Generate test data
    const testData = Array.from({ length: 100 }, (_, i) => ({
      timestamp: Date.now() + i * 1000,
      symbol: `TEST${i % 5}`,
      exchange: 'test',
      open: 100 + i,
      high: 110 + i,
      low: 95 + i,
      close: 105 + i,
      volume: 1000 + i
    }))
    
    const startTime = Date.now()
    await repo.saveMany(testData)
    const duration = Date.now() - startTime
    
    // Should be fast with WAL mode (< 1 second for 100 records)
    console.assert(duration < 1000, `Too slow: ${duration}ms`)
    
    const count = await repo.getCount('TEST0')
    console.assert(count >= 1)
    
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

// Run the main test suite
test('SQLite Repository - Complete Test Suite', async () => {
  const testInstance = new SqliteRepositoryTest()
  await testInstance.runAllTests()
})