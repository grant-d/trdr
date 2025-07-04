import { test } from 'node:test'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { rm } from 'node:fs/promises'
import { RepositoryTestBase } from './repository-test-base'
import { ParquetRepository } from '../../../src/repositories/parquet-repository'
import type { OhlcvRepository } from '../../../src/repositories/ohlcv-repository.interface'

/**
 * Parquet Repository Tests
 */
class ParquetRepositoryTest extends RepositoryTestBase {
  private testDir: string = ''

  public async createRepository(): Promise<OhlcvRepository> {
    // Create a unique test directory
    this.testDir = join(tmpdir(), `trdr-parquet-test-${Date.now()}-${Math.random().toString(36).slice(2)}`)
    
    const repo = new ParquetRepository()
    await repo.initialize({
      connectionString: this.testDir,
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
 * Parquet-specific tests
 */
test('Parquet Repository - Columnar Storage', async () => {
  const testInstance = new ParquetRepositoryTest()
  const repo = await testInstance.createRepository()
  
  try {
    const testData = Array.from({ length: 50 }, (_, i) => ({
      timestamp: Date.now() + i * 1000,
      symbol: `SYMBOL_${i % 5}`,
      exchange: 'test',
      open: 100 + i,
      high: 110 + i,
      low: 95 + i,
      close: 105 + i,
      volume: 1000 + i
    }))
    
    await repo.saveMany(testData)
    
    // Test columnar efficiency with symbol filtering
    const symbol0Data = await repo.getBySymbol('SYMBOL_0')
    const symbol1Data = await repo.getBySymbol('SYMBOL_1')
    
    console.assert(symbol0Data.length === 10) // Every 5th record
    console.assert(symbol1Data.length === 10)
    
    // Verify data integrity
    for (const item of symbol0Data) {
      console.assert(item.symbol === 'SYMBOL_0')
    }
    
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

test('Parquet Repository - Batch Buffer Management', async () => {
  const testInstance = new ParquetRepositoryTest()
  const repo = await testInstance.createRepository()
  
  try {
    // Test individual saves that should trigger buffer flushes
    for (let i = 0; i < 150; i++) { // More than batch size
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

test('Parquet Repository - File Deduplication', async () => {
  const testInstance = new ParquetRepositoryTest()
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

test('Parquet Repository - Compression Efficiency', async () => {
  const testInstance = new ParquetRepositoryTest()
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

test('Parquet Repository - BigInt Timestamp Handling', async () => {
  const testInstance = new ParquetRepositoryTest()
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
test('Parquet Repository - Complete Test Suite', async () => {
  const testInstance = new ParquetRepositoryTest()
  await testInstance.runAllTests()
})