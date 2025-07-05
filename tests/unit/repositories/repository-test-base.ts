import { strict as assert } from 'node:assert'
import { test, afterEach } from 'node:test'
import type { OhlcvDto } from '../../../src/models/ohlcv.dto'
import type { OhlcvRepository } from '../../../src/repositories/ohlcv-repository.interface'
import { forceCleanupAsyncHandles } from '../../helpers/test-cleanup'

/**
 * Base test suite for all OhlcvRepository implementations
 * Ensures consistent behavior across SQLite, CSV, and Jsonl repositories
 */
export abstract class RepositoryTestBase {
  public abstract createRepository(): Promise<OhlcvRepository>
  public abstract cleanup(): Promise<void>

  /**
   * Sample OHLCV data for testing - single symbol/exchange for repository compatibility
   */
  protected readonly sampleData: OhlcvDto[] = [
    {
      timestamp: 1640995200000, // 2022-01-01 00:00:00
      symbol: 'BTCUSD',
      exchange: 'coinbase',
      open: 47000.50,
      high: 47500.25,
      low: 46800.75,
      close: 47200.00,
      volume: 150.25
    },
    {
      timestamp: 1640998800000, // 2022-01-01 01:00:00
      symbol: 'BTCUSD',
      exchange: 'coinbase',
      open: 47200.00,
      high: 47800.50,
      low: 47100.25,
      close: 47650.75,
      volume: 200.50
    },
    {
      timestamp: 1641002400000, // 2022-01-01 02:00:00
      symbol: 'BTCUSD',
      exchange: 'coinbase',
      open: 47500.25,
      high: 47850.75,
      low: 47280.50,
      close: 47625.00,
      volume: 300.75
    },
    {
      timestamp: 1641006000000, // 2022-01-01 03:00:00
      symbol: 'BTCUSD',
      exchange: 'coinbase',
      open: 47625.00,
      high: 48000.00,
      low: 47500.25,
      close: 47900.50,
      volume: 180.25
    }
  ]

  /**
   * Additional sample data for multi-symbol tests (use separate repository instances)
   */
  protected readonly ethSampleData: OhlcvDto[] = [
    {
      timestamp: 1641002400000, // 2022-01-01 02:00:00
      symbol: 'ETHUSD',
      exchange: 'coinbase',
      open: 3800.25,
      high: 3850.75,
      low: 3780.50,
      close: 3825.00,
      volume: 500.75
    }
  ]

  /**
   * Additional sample data for different exchange tests (use separate repository instances)
   */
  protected readonly binanceSampleData: OhlcvDto[] = [
    {
      timestamp: 1641006000000, // 2022-01-01 03:00:00
      symbol: 'BTCUSD',
      exchange: 'binance',
      open: 47650.75,
      high: 48000.00,
      low: 47500.25,
      close: 47900.50,
      volume: 180.25
    }
  ]


  /**
   * Run all repository tests
   */
  async runAllTests(): Promise<void> {
    await test('Repository Interface Tests', async (t) => {
      afterEach(() => {
        forceCleanupAsyncHandles()
      })
      
      await t.test('Basic CRUD Operations', () => this.testBasicCrud())
      await t.test('Batch Operations', () => this.testBatchOperations())
      await t.test('Date Range Queries', () => this.testDateRangeQueries())
      await t.test('Symbol Filtering', () => this.testSymbolFiltering())
      await t.test('Exchange Filtering', () => this.testExchangeFiltering())
      await t.test('Flexible Query Interface', () => this.testFlexibleQuery())
      await t.test('Timestamp Operations', () => this.testTimestampOperations())
      await t.test('Count Operations', () => this.testCountOperations())
      await t.test('Symbol and Exchange Lists', () => this.testSymbolExchangeLists())
      await t.test('Repository Statistics', () => this.testRepositoryStats())
      await t.test('Error Handling', () => this.testErrorHandling())
      await t.test('Edge Cases', () => this.testEdgeCases())
      await t.test('Data Integrity', () => this.testDataIntegrity())
      // Commented out - bulk performance testing not needed for streaming pipeline
      // await t.test('Performance with Large Dataset', () => this.testPerformance())
    })
  }

  /**
   * Test basic CRUD operations
   */
  protected async testBasicCrud(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      // Test save single record
      await repo.save(this.sampleData[0]!)
      
      // Flush any buffers before retrieval
      await repo.flush()
      
      // Test retrieval
      const retrieved = await repo.getBySymbol('BTCUSD', 'coinbase', 1)
      assert.equal(retrieved.length, 1)
      assert.equal(retrieved[0]!.timestamp, this.sampleData[0]!.timestamp)
      assert.equal(retrieved[0]!.symbol, this.sampleData[0]!.symbol)
      assert.equal(retrieved[0]!.close, this.sampleData[0]!.close)
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test batch operations
   */
  protected async testBatchOperations(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      // Test saveMany
      await repo.saveMany(this.sampleData)
      
      // Test appendBatch with additional data for same symbol/exchange
      const additionalData: OhlcvDto[] = [{
        timestamp: 1641009600000,
        symbol: 'BTCUSD',
        exchange: 'coinbase',
        open: 47900.50,
        high: 48100.25,
        low: 47750.50,
        close: 48000.75,
        volume: 250.25
      }]
      
      await repo.appendBatch(additionalData)
      
      // Verify all data was saved
      const allData = await repo.query({})
      assert.equal(allData.length, this.sampleData.length + additionalData.length)
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test date range queries
   */
  protected async testDateRangeQueries(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      // Test date range filtering
      const startTime = 1640995200000
      const endTime = 1641002400000
      
      const filtered = await repo.getBetweenDates(startTime, endTime)
      assert.ok(filtered.length >= 2)
      
      // Verify all results are within range
      for (const item of filtered) {
        assert.ok(item.timestamp >= startTime)
        assert.ok(item.timestamp <= endTime)
      }
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test symbol filtering
   */
  protected async testSymbolFiltering(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      // Test symbol filtering - all our data is BTCUSD
      const btcData = await repo.getBySymbol('BTCUSD')
      const nonExistentData = await repo.getBySymbol('ETHUSD')
      
      assert.ok(btcData.length >= 4)
      assert.equal(nonExistentData.length, 0)
      
      // Verify all results match the symbol
      for (const item of btcData) {
        assert.equal(item.symbol, 'BTCUSD')
      }
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test exchange filtering
   */
  protected async testExchangeFiltering(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      // Test exchange filtering - all our data is coinbase
      const coinbaseData = await repo.getBySymbol('BTCUSD', 'coinbase')
      const nonExistentData = await repo.getBySymbol('BTCUSD', 'binance')
      
      assert.ok(coinbaseData.length >= 4)
      assert.equal(nonExistentData.length, 0)
      
      // Verify all results match the exchange
      for (const item of coinbaseData) {
        assert.equal(item.exchange, 'coinbase')
      }
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test flexible query interface
   */
  protected async testFlexibleQuery(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      // Test complex query with multiple filters
      const results = await repo.query({
        symbol: 'BTCUSD',
        exchange: 'coinbase',
        startTime: 1640995200000,
        endTime: 1641002400000,
        limit: 10,
        offset: 0
      })
      
      assert.ok(results.length >= 1)
      
      for (const item of results) {
        assert.equal(item.symbol, 'BTCUSD')
        assert.equal(item.exchange, 'coinbase')
        assert.ok(item.timestamp >= 1640995200000)
        assert.ok(item.timestamp <= 1641002400000)
      }
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test timestamp operations
   */
  protected async testTimestampOperations(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      // Test getLastTimestamp
      const lastTimestamp = await repo.getLastTimestamp('BTCUSD', 'coinbase')
      assert.ok(lastTimestamp !== null)
      assert.equal(lastTimestamp, 1641006000000)
      
      // Test getFirstTimestamp
      const firstTimestamp = await repo.getFirstTimestamp('BTCUSD', 'coinbase')
      assert.ok(firstTimestamp !== null)
      assert.equal(firstTimestamp, 1640995200000)
      
      // Test with non-existent symbol
      const noTimestamp = await repo.getLastTimestamp('NONEXISTENT')
      assert.equal(noTimestamp, null)
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test count operations
   */
  protected async testCountOperations(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      // Test count by symbol
      const btcCount = await repo.getCount('BTCUSD')
      assert.ok(btcCount >= 2)
      
      // Test count by symbol and exchange
      const coinbaseCount = await repo.getCount('BTCUSD', 'coinbase')
      assert.ok(coinbaseCount >= 1)
      
      // Test count for non-existent symbol
      const noCount = await repo.getCount('NONEXISTENT')
      assert.equal(noCount, 0)
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }




  /**
   * Test symbol and exchange list operations
   */
  protected async testSymbolExchangeLists(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      // Test getSymbols - should only have BTCUSD
      const allSymbols = await repo.getSymbols()
      assert.ok(allSymbols.includes('BTCUSD'))
      assert.equal(allSymbols.length, 1)
      
      // Test getSymbols with exchange filter
      const coinbaseSymbols = await repo.getSymbols('coinbase')
      assert.ok(coinbaseSymbols.includes('BTCUSD'))
      assert.equal(coinbaseSymbols.length, 1)
      
      // Test getExchanges - should only have coinbase
      const exchanges = await repo.getExchanges()
      assert.ok(exchanges.includes('coinbase'))
      assert.equal(exchanges.length, 1)
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test repository statistics
   */
  protected async testRepositoryStats(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      const stats = await repo.getStats()
      
      assert.ok(stats.totalRecords >= this.sampleData.length)
      assert.equal(stats.uniqueSymbols, 1) // Only BTCUSD
      assert.equal(stats.uniqueExchanges, 1) // Only coinbase
      assert.ok(stats.dataDateRange.earliest !== null)
      assert.ok(stats.dataDateRange.latest !== null)
      assert.ok(stats.dataDateRange.earliest <= stats.dataDateRange.latest)
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test error handling
   */
  protected async testErrorHandling(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      // Test invalid OHLCV data
      const invalidData: Partial<OhlcvDto> = {
        timestamp: Date.now(),
        symbol: 'TEST',
        // Missing required fields
      }
      
      let errorThrown = false
      try {
        await repo.save(invalidData as OhlcvDto)
      } catch (error) {
        errorThrown = true
        assert.ok(error instanceof Error)
        assert.ok(error.message.includes('Invalid OHLCV data'))
      }
      
      assert.ok(errorThrown, 'Expected error was not thrown')
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test edge cases
   */
  protected async testEdgeCases(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      // Test empty dataset queries
      const emptyResults = await repo.getBySymbol('NONEXISTENT')
      assert.equal(emptyResults.length, 0)
      
      // Test empty batch operations
      await repo.saveMany([])
      await repo.appendBatch([])
      
      // Test large timestamp values (year 2099)
      const futureData: OhlcvDto = {
        timestamp: 4070908800000, // Jan 1, 2099
        symbol: 'FUTURE',
        exchange: 'test',
        open: 100,
        high: 110,
        low: 95,
        close: 105,
        volume: 1000
      }
      
      await repo.save(futureData)
      await repo.flush() // Flush buffer for single saves
      const retrieved = await repo.getBySymbol('FUTURE')
      assert.equal(retrieved.length, 1)
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test data integrity
   */
  protected async testDataIntegrity(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      // Flush to ensure data is persisted
      await repo.flush()
      
      // Retrieve all data and verify integrity
      const allData = await repo.query({})
      
      // Verify no data corruption
      for (const item of allData) {
        assert.ok(typeof item.timestamp === 'number')
        assert.ok(typeof item.symbol === 'string')
        assert.ok(typeof item.exchange === 'string')
        assert.ok(typeof item.open === 'number')
        assert.ok(typeof item.high === 'number')
        assert.ok(typeof item.low === 'number')
        assert.ok(typeof item.close === 'number')
        assert.ok(typeof item.volume === 'number')
        
        // Verify OHLC relationships
        assert.ok(item.high >= item.open)
        assert.ok(item.high >= item.close)
        assert.ok(item.low <= item.open)
        assert.ok(item.low <= item.close)
        assert.ok(item.volume >= 0)
      }
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }

  /**
   * Test performance with larger dataset
   * Commented out - bulk performance testing not needed for streaming pipeline
   */
  /*
  protected async testPerformance(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      // Generate larger dataset - single symbol/exchange for repository compatibility
      const largeDataset: OhlcvDto[] = []
      const baseTime = 1640995200000
      
      for (let i = 0; i < 1000; i++) {
        const basePrice = 47000
        const open = basePrice + (Math.random() - 0.5) * 1000
        const priceVariation = Math.random() * 500
        const high = Math.max(open, basePrice + priceVariation)
        const low = Math.min(open, basePrice - priceVariation)
        const close = low + Math.random() * (high - low)
        
        largeDataset.push({
          timestamp: baseTime + (i * 60000), // 1 minute intervals
          symbol: 'BTCUSD',
          exchange: 'coinbase',
          open,
          high,
          low,
          close,
          volume: Math.random() * 1000
        })
      }
      
      // Test batch write performance
      const startTime = Date.now()
      await repo.saveMany(largeDataset)
      const writeTime = Date.now() - startTime
      
      // Should complete in reasonable time (< 10 seconds for 1000 records)
      assert.ok(writeTime < 10000, `Write took too long: ${writeTime}ms`)
      
      // Test read performance
      const readStart = Date.now()
      const results = await repo.query({ limit: 1000 })
      const readTime = Date.now() - readStart
      
      assert.ok(readTime < 5000, `Read took too long: ${readTime}ms`)
      assert.ok(results.length >= 1000)
      
    } finally {
      try {
        await repo.close()
      } catch {
        // Force close if normal close fails
        if (typeof (repo as any).forceClose === 'function') {
          (repo as any).forceClose()
        }
      }
      await this.cleanup()
      forceCleanupAsyncHandles()
    }
  }
  */
}