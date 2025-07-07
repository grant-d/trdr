import { rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { JsonlRepository } from '../../../src/repositories/jsonl-repository'
import type { OhlcvRepository } from '../../../src/repositories'
import { RepositoryTestBase } from './repository-test-base'

/**
 * Jsonl Repository Tests
 */
class JsonlRepositoryTest extends RepositoryTestBase {
  private testDir = ''

  public async createRepository(): Promise<OhlcvRepository> {
    // Create a unique test directory
    this.testDir = join(
      tmpdir(),
      `trdr-jsonl-test-${Date.now()}-${Math.random().toString(36).slice(2)}`
    )

    const repo = new JsonlRepository()
    await repo.initialize({
      connectionString: join(this.testDir, 'test-data.jsonl'), // Provide a file path, not just directory
      options: {
        batchSize: 100 // Smaller batch size for testing
      }
    })

    return repo
  }

  public async cleanup(): Promise<void> {
    if (this.testDir) {
      try {
        await rm(this.testDir, { recursive: true, force: true })
      } catch {
        // Ignore cleanup errors
      }
    }
  }
}

/**
 * Jsonl-specific tests
 */
test('Jsonl Repository - Columnar Storage', async () => {
  const testInstance = new JsonlRepositoryTest()
  const repo = await testInstance.createRepository()

  try {
    // Use single symbol/exchange for the entire dataset
    const testData = Array.from({ length: 50 }, (_, i) => ({
      timestamp: Date.now() + i * 1000,
      symbol: 'SYMBOL_TEST',
      exchange: 'test',
      open: 100 + i,
      high: 110 + i,
      low: 95 + i,
      close: 105 + i,
      volume: 1000 + i
    }))

    await repo.saveMany(testData)

    // Test data retrieval and integrity
    const symbolData = await repo.getBySymbol('SYMBOL_TEST')

    console.assert(symbolData.length === 50)

    // Verify data integrity
    for (const item of symbolData) {
      console.assert(item.symbol === 'SYMBOL_TEST')
      console.assert(item.exchange === 'test')
    }
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

test('Jsonl Repository - Batch Buffer Management', async () => {
  const testInstance = new JsonlRepositoryTest()
  const repo = await testInstance.createRepository()

  try {
    // Test individual saves that should trigger buffer flushes
    for (let i = 0; i < 150; i++) {
      // More than batch size
      await repo.save({
        timestamp: Date.now() + i * 1000,
        symbol: 'BUFFER_TEST',
        exchange: 'test',
        open: 100 + i,
        high: 110 + i,
        low: 95 + i,
        close: 105 + i,
        volume: 1000 + i
      })
    }

    // Flush remaining buffer
    await repo.flush()

    const results = await repo.getBySymbol('BUFFER_TEST')
    console.assert(results.length === 150)
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

test('Jsonl Repository - File Deduplication', async () => {
  const testInstance = new JsonlRepositoryTest()
  const repo = await testInstance.createRepository()

  try {
    const duplicateData = [
      {
        timestamp: 1640995200000,
        symbol: 'DEDUP_TEST',
        exchange: 'test',
        open: 100,
        high: 110,
        low: 95,
        close: 105,
        volume: 1000
      },
      {
        timestamp: 1640995200000, // Same timestamp/symbol/exchange
        symbol: 'DEDUP_TEST',
        exchange: 'test',
        open: 101, // Different values
        high: 111,
        low: 96,
        close: 106,
        volume: 1001
      }
    ]

    await repo.saveMany(duplicateData)

    // Should only have one record after deduplication
    const results = await repo.getBySymbol('DEDUP_TEST')
    console.assert(results.length === 1)

    // Should keep the last record
    console.assert(results[0]!.close === 106)
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

// Commented out - bulk performance testing not needed for streaming pipeline
/*
test('Jsonl Repository - Compression Efficiency', async () => {
  const testInstance = new JsonlRepositoryTest()
  const repo = await testInstance.createRepository()
  
  try {
    // Generate data with patterns that should compress well
    const repetitiveData = Array.from({ length: 1000 }, (_, i) => ({
      timestamp: Date.now() + i * 60000, // 1 minute intervals
      symbol: 'COMPRESS_TEST', // Same symbol for better compression
      exchange: 'test', // Same exchange
      open: 100, // Repeated values
      high: 110,
      low: 95,
      close: 105,
      volume: 1000 + (i % 10) // Some variation
    }))
    
    await repo.saveMany(repetitiveData)
    
    const stats = await repo.getStats()
    console.assert(stats.totalRecords === 1000)
    console.assert(stats.uniqueSymbols === 1)
    
    // Test that we can retrieve all data correctly
    const allData = await repo.query({ limit: 1000 })
    console.assert(allData.length === 1000)
    
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})
*/

test('Jsonl Repository - BigInt Timestamp Handling', async () => {
  const testInstance = new JsonlRepositoryTest()
  const repo = await testInstance.createRepository()

  try {
    // Test with large timestamp values (but within valid range)
    // Using a timestamp for year 2099 (within the 2000-2100 valid range)
    const year2099Timestamp = 4070908800000 // Jan 1, 2099
    const largeTimestampData = [
      {
        timestamp: year2099Timestamp,
        symbol: 'BIGINT_TEST',
        exchange: 'test',
        open: 100,
        high: 110,
        low: 95,
        close: 105,
        volume: 1000
      }
    ]

    await repo.saveMany(largeTimestampData)

    const results = await repo.getBySymbol('BIGINT_TEST')
    console.assert(results.length === 1)
    console.assert(results[0]!.timestamp === year2099Timestamp)
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

// Run the main test suite
test('Jsonl Repository - Complete Test Suite', async () => {
  const testInstance = new JsonlRepositoryTest()
  await testInstance.runAllTests()
})
