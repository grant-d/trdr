import { strict as assert } from 'node:assert'
import { test } from 'node:test'
import type { OhlcvDto } from '../../../src/models/ohlcv.dto'
import type { CoefficientData, OhlcvRepository } from '../../../src/repositories/ohlcv-repository.interface'

/**
 * Base test suite for all OhlcvRepository implementations
 * Ensures consistent behavior across SQLite, CSV, and Parquet repositories
 */
export abstract class RepositoryTestBase {
  public abstract createRepository(): Promise<OhlcvRepository>
  public abstract cleanup(): Promise<void>

  /**
   * Sample OHLCV data for testing
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
      symbol: 'ETHUSD',
      exchange: 'coinbase',
      open: 3800.25,
      high: 3850.75,
      low: 3780.50,
      close: 3825.00,
      volume: 500.75
    },
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
   * Sample coefficient data for testing
   */
  protected readonly sampleCoefficients: CoefficientData[] = [
    {
      name: 'sma_20',
      symbol: 'BTCUSD',
      exchange: 'coinbase',
      value: 47350.25,
      metadata: { period: 20, type: 'simple' },
      timestamp: 1640995200000
    },
    {
      name: 'rsi_14',
      symbol: 'BTCUSD',
      exchange: 'coinbase',
      value: 65.5,
      metadata: { period: 14, overbought: 70, oversold: 30 },
      timestamp: 1640998800000
    },
    {
      name: 'volume_profile',
      value: 1250.75,
      metadata: { profile_type: 'daily' },
      timestamp: 1641002400000
    }
  ]

  /**
   * Run all repository tests
   */
  async runAllTests(): Promise<void> {
    await test('Repository Interface Tests', async (t) => {
      await t.test('Basic CRUD Operations', () => this.testBasicCrud())
      await t.test('Batch Operations', () => this.testBatchOperations())
      await t.test('Date Range Queries', () => this.testDateRangeQueries())
      await t.test('Symbol Filtering', () => this.testSymbolFiltering())
      await t.test('Exchange Filtering', () => this.testExchangeFiltering())
      await t.test('Flexible Query Interface', () => this.testFlexibleQuery())
      await t.test('Timestamp Operations', () => this.testTimestampOperations())
      await t.test('Count Operations', () => this.testCountOperations())
      await t.test('Coefficient Storage', () => this.testCoefficientStorage())
      await t.test('Coefficient Retrieval', () => this.testCoefficientRetrieval())
      await t.test('Coefficient Deletion', () => this.testCoefficientDeletion())
      await t.test('Symbol and Exchange Lists', () => this.testSymbolExchangeLists())
      await t.test('Repository Statistics', () => this.testRepositoryStats())
      await t.test('Error Handling', () => this.testErrorHandling())
      await t.test('Edge Cases', () => this.testEdgeCases())
      await t.test('Data Integrity', () => this.testDataIntegrity())
      await t.test('Performance with Large Dataset', () => this.testPerformance())
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
      await repo.close()
      await this.cleanup()
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
      
      // Test appendBatch
      const additionalData: OhlcvDto[] = [{
        timestamp: 1641009600000,
        symbol: 'ETHUSD',
        exchange: 'binance',
        open: 3825.00,
        high: 3875.25,
        low: 3810.50,
        close: 3860.75,
        volume: 300.25
      }]
      
      await repo.appendBatch(additionalData)
      
      // Verify all data was saved
      const allData = await repo.query({})
      assert.equal(allData.length, this.sampleData.length + additionalData.length)
      
    } finally {
      await repo.close()
      await this.cleanup()
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
      await repo.close()
      await this.cleanup()
    }
  }

  /**
   * Test symbol filtering
   */
  protected async testSymbolFiltering(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      // Test symbol filtering
      const btcData = await repo.getBySymbol('BTCUSD')
      const ethData = await repo.getBySymbol('ETHUSD')
      
      assert.ok(btcData.length >= 2)
      assert.ok(ethData.length >= 1)
      
      // Verify all results match the symbol
      for (const item of btcData) {
        assert.equal(item.symbol, 'BTCUSD')
      }
      
      for (const item of ethData) {
        assert.equal(item.symbol, 'ETHUSD')
      }
      
    } finally {
      await repo.close()
      await this.cleanup()
    }
  }

  /**
   * Test exchange filtering
   */
  protected async testExchangeFiltering(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      // Test exchange filtering
      const coinbaseData = await repo.getBySymbol('BTCUSD', 'coinbase')
      const binanceData = await repo.getBySymbol('BTCUSD', 'binance')
      
      assert.ok(coinbaseData.length >= 1)
      assert.ok(binanceData.length >= 1)
      
      // Verify all results match the exchange
      for (const item of coinbaseData) {
        assert.equal(item.exchange, 'coinbase')
      }
      
      for (const item of binanceData) {
        assert.equal(item.exchange, 'binance')
      }
      
    } finally {
      await repo.close()
      await this.cleanup()
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
      await repo.close()
      await this.cleanup()
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
      assert.equal(lastTimestamp, 1640998800000)
      
      // Test getFirstTimestamp
      const firstTimestamp = await repo.getFirstTimestamp('BTCUSD', 'coinbase')
      assert.ok(firstTimestamp !== null)
      assert.equal(firstTimestamp, 1640995200000)
      
      // Test with non-existent symbol
      const noTimestamp = await repo.getLastTimestamp('NONEXISTENT')
      assert.equal(noTimestamp, null)
      
    } finally {
      await repo.close()
      await this.cleanup()
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
      await repo.close()
      await this.cleanup()
    }
  }

  /**
   * Test coefficient storage
   */
  protected async testCoefficientStorage(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      // Test single coefficient save
      await repo.saveCoefficient(this.sampleCoefficients[0]!)
      
      // Test batch coefficient save
      await repo.saveCoefficients(this.sampleCoefficients)
      
      // Verify coefficients were saved
      const retrieved = await repo.getCoefficient('sma_20', 'BTCUSD', 'coinbase')
      assert.ok(retrieved !== null)
      assert.equal(retrieved.name, 'sma_20')
      assert.equal(retrieved.value, 47350.25)
      assert.ok(retrieved.metadata)
      assert.equal((retrieved.metadata as { period: number }).period, 20)
      
    } finally {
      await repo.close()
      await this.cleanup()
    }
  }

  /**
   * Test coefficient retrieval
   */
  protected async testCoefficientRetrieval(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveCoefficients(this.sampleCoefficients)
      
      // Test pattern matching
      const smaCoefficients = await repo.getCoefficients('sma_*')
      assert.ok(smaCoefficients.length >= 1)
      
      // Test symbol filtering
      const btcCoefficients = await repo.getCoefficients(undefined, 'BTCUSD')
      assert.ok(btcCoefficients.length >= 2)
      
      // Test exchange filtering
      const coinbaseCoefficients = await repo.getCoefficients(undefined, 'BTCUSD', 'coinbase')
      assert.ok(coinbaseCoefficients.length >= 2)
      
      // Test global coefficients (no symbol/exchange)
      const globalCoefficients = await repo.getCoefficients('volume_*')
      assert.ok(globalCoefficients.length >= 1)
      
    } finally {
      await repo.close()
      await this.cleanup()
    }
  }

  /**
   * Test coefficient deletion
   */
  protected async testCoefficientDeletion(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveCoefficients(this.sampleCoefficients)
      
      // Test deletion by pattern
      const deletedCount = await repo.deleteCoefficients('rsi_*')
      assert.ok(deletedCount >= 1)
      
      // Verify deletion
      const remaining = await repo.getCoefficient('rsi_14', 'BTCUSD', 'coinbase')
      assert.equal(remaining, null)
      
      // Verify other coefficients remain
      const smaCoeff = await repo.getCoefficient('sma_20', 'BTCUSD', 'coinbase')
      assert.ok(smaCoeff !== null)
      
    } finally {
      await repo.close()
      await this.cleanup()
    }
  }

  /**
   * Test symbol and exchange list operations
   */
  protected async testSymbolExchangeLists(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      await repo.saveMany(this.sampleData)
      
      // Test getSymbols
      const allSymbols = await repo.getSymbols()
      assert.ok(allSymbols.includes('BTCUSD'))
      assert.ok(allSymbols.includes('ETHUSD'))
      
      // Test getSymbols with exchange filter
      const coinbaseSymbols = await repo.getSymbols('coinbase')
      assert.ok(coinbaseSymbols.includes('BTCUSD'))
      assert.ok(coinbaseSymbols.includes('ETHUSD'))
      
      // Test getExchanges
      const exchanges = await repo.getExchanges()
      assert.ok(exchanges.includes('coinbase'))
      assert.ok(exchanges.includes('binance'))
      
    } finally {
      await repo.close()
      await this.cleanup()
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
      assert.ok(stats.uniqueSymbols >= 2)
      assert.ok(stats.uniqueExchanges >= 2)
      assert.ok(stats.dataDateRange.earliest !== null)
      assert.ok(stats.dataDateRange.latest !== null)
      assert.ok(stats.dataDateRange.earliest! <= stats.dataDateRange.latest!)
      
    } finally {
      await repo.close()
      await this.cleanup()
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
      await repo.close()
      await this.cleanup()
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
      await repo.close()
      await this.cleanup()
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
      await repo.close()
      await this.cleanup()
    }
  }

  /**
   * Test performance with larger dataset
   */
  protected async testPerformance(): Promise<void> {
    const repo = await this.createRepository()
    
    try {
      // Generate larger dataset
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
          symbol: i % 2 === 0 ? 'BTCUSD' : 'ETHUSD',
          exchange: i % 3 === 0 ? 'coinbase' : 'binance',
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
      await repo.close()
      await this.cleanup()
    }
  }
}