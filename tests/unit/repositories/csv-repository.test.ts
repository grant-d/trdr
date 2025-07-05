import { test } from 'node:test'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { rm } from 'node:fs/promises'
import { RepositoryTestBase } from './repository-test-base'
import { CsvRepository } from '../../../src/repositories/csv-repository'
import type { OhlcvRepository } from '../../../src/repositories/ohlcv-repository.interface'

/**
 * CSV Repository Tests
 */
class CsvRepositoryTest extends RepositoryTestBase {
  private testDir = ''

  public async createRepository(): Promise<OhlcvRepository> {
    // Create a unique test directory
    this.testDir = join(tmpdir(), `trdr-csv-test-${Date.now()}-${Math.random().toString(36).slice(2)}`)
    
    const repo = new CsvRepository()
    await repo.initialize({
      connectionString: this.testDir,
      options: {}
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
 * CSV-specific tests
 */
test('CSV Repository - File Organization', async () => {
  const testInstance = new CsvRepositoryTest()
  const repo = await testInstance.createRepository()
  
  try {
    const testData = [
      {
        timestamp: Date.now(),
        symbol: 'BTCUSD',
        exchange: 'coinbase',
        open: 47000,
        high: 47500,
        low: 46800,
        close: 47200,
        volume: 150
      },
      {
        timestamp: Date.now() + 1000,
        symbol: 'ETHUSD',
        exchange: 'binance',
        open: 3800,
        high: 3850,
        low: 3780,
        close: 3825,
        volume: 500
      }
    ]
    
    await repo.saveMany(testData)
    
    // Verify data can be retrieved correctly
    const btcData = await repo.getBySymbol('BTCUSD', 'coinbase')
    const ethData = await repo.getBySymbol('ETHUSD', 'binance')
    
    console.assert(btcData.length === 1)
    console.assert(ethData.length === 1)
    console.assert(btcData[0]!.symbol === 'BTCUSD')
    console.assert(ethData[0]!.symbol === 'ETHUSD')
    
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

test('CSV Repository - Streaming Writes', async () => {
  const testInstance = new CsvRepositoryTest()
  const repo = await testInstance.createRepository()
  
  try {
    // Test individual saves (should use streaming)
    for (let i = 0; i < 10; i++) {
      await repo.save({
        timestamp: Date.now() + i * 1000,
        symbol: 'STREAM_TEST',
        exchange: 'test',
        open: 100 + i,
        high: 110 + i,
        low: 95 + i,
        close: 105 + i,
        volume: 1000 + i
      })
    }
    
    const results = await repo.getBySymbol('STREAM_TEST')
    console.assert(results.length === 10)
    
    // Verify data ordering
    for (let i = 0; i < results.length - 1; i++) {
      console.assert(results[i]!.timestamp >= results[i + 1]!.timestamp) // Descending order
    }
    
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

test('CSV Repository - CSV Escaping', async () => {
  const testInstance = new CsvRepositoryTest()
  const repo = await testInstance.createRepository()
  
  try {
    // Test symbols and exchanges with special characters
    const testData = [
      {
        timestamp: Date.now(),
        symbol: 'BTC,USD', // Contains comma
        exchange: 'test"exchange', // Contains quote
        open: 47000,
        high: 47500,
        low: 46800,
        close: 47200,
        volume: 150
      },
      {
        timestamp: Date.now() + 1000,
        symbol: 'ETH\nUSD', // Contains newline
        exchange: 'normal',
        open: 3800,
        high: 3850,
        low: 3780,
        close: 3825,
        volume: 500
      }
    ]
    
    await repo.saveMany(testData)
    
    // Verify data can be retrieved correctly despite special characters
    const results = await repo.query({})
    console.assert(results.length === 2)
    
    const specialSymbols = results.map(r => r.symbol)
    console.assert(specialSymbols.includes('BTC,USD'))
    console.assert(specialSymbols.includes('ETH\nUSD'))
    
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

test('CSV Repository - Large File Handling', async () => {
  const testInstance = new CsvRepositoryTest()
  const repo = await testInstance.createRepository()
  
  try {
    // Generate a larger dataset
    const largeDataset = Array.from({ length: 1000 }, (_, i) => ({
      timestamp: Date.now() + i * 1000,
      symbol: 'LARGE_TEST',
      exchange: 'test',
      open: 100 + (i % 100),
      high: 110 + (i % 100),
      low: 95 + (i % 100),
      close: 105 + (i % 100),
      volume: 1000 + i
    }))
    
    const startTime = Date.now()
    await repo.saveMany(largeDataset)
    const duration = Date.now() - startTime
    
    console.log(`CSV write performance: ${duration}ms for ${largeDataset.length} records`)
    
    // Verify all data was written
    const count = await repo.getCount('LARGE_TEST')
    console.assert(count === largeDataset.length)
    
    // Test reading performance
    const readStart = Date.now()
    const results = await repo.getBySymbol('LARGE_TEST')
    const readDuration = Date.now() - readStart
    
    console.log(`CSV read performance: ${readDuration}ms for ${results.length} records`)
    console.assert(results.length === largeDataset.length)
    
  } finally {
    await repo.close()
    await testInstance.cleanup()
  }
})

// Run the main test suite
test('CSV Repository - Complete Test Suite', async () => {
  const testInstance = new CsvRepositoryTest()
  await testInstance.runAllTests()
})