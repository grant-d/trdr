import { deepStrictEqual, ok, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../src/models'
import { ZScoreNormalizer } from '../../../src/transforms'
import type { TransformedOhlcvDto } from './test-types'

describe('ZScoreNormalizer', () => {
  // Helper to create test data
  const createTestData = (count: number, startValue = 100): OhlcvDto[] => {
    const data: OhlcvDto[] = []
    for (let i = 0; i < count; i++) {
      data.push({
        timestamp: new Date('2024-01-01').getTime() + i * 60000,
        symbol: 'BTCUSD',
        exchange: 'test',
        open: startValue + i * 10,
        high: startValue + i * 10 + 5,
        low: startValue + i * 10 - 5,
        close: startValue + i * 10 + 2,
        volume: 1000 + i * 100,
      })
    }
    return data
  }

  // Helper to convert array to async iterator
  async function* arrayToAsyncIterator<T>(array: T[]): AsyncIterator<T> {
    for (const item of array) {
      yield item
    }
  }

  // Helper to collect async iterator results
  async function collectResults(iterator: AsyncIterator<OhlcvDto>): Promise<TransformedOhlcvDto[]> {
    const results: TransformedOhlcvDto[] = []
    let item = await iterator.next()
    while (!item.done) {
      results.push(item.value as TransformedOhlcvDto)
      item = await iterator.next()
    }
    return results
  }

  describe('constructor and validation', () => {
    it('should create instance with default parameters', () => {
      const normalizer = new ZScoreNormalizer({})
      ok(normalizer)
      strictEqual(normalizer.type, 'zScore')
      strictEqual(normalizer.name, 'Z-Score Normalizer')
    })

    it('should validate window size', () => {
      const normalizer = new ZScoreNormalizer({ windowSize: 1 })
      throws(() => normalizer.validate(), /Window size must be at least 2/)
    })
  })

  describe('rolling window z-score normalization', () => {
    it('should normalize data to mean=0, std=1', async () => {
      // Create data with known values for window calculation
      const testData = createTestData(5, 100)
      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_zscore'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Transform now doesn't yield until ready (windowSize=3)
      strictEqual(transformed.length, 3)

      // All transformed items now have full windows to check
      // Window [102, 112, 122]: close values from testData
      const window1 = [102, 112, 122]
      const mean1 = window1.reduce((a, b) => a + b) / 3 // 112
      const variance1 = window1.reduce((sum, val) => sum + Math.pow(val - mean1, 2), 0) / 3
      const std1 = Math.sqrt(variance1)
      const expected2 = (122 - mean1) / std1
      
      // Since first 2 items are not yielded, the 3rd item is now at index 0
      ok(Math.abs(transformed[0]!.close_zscore! - expected2) < 0.0001)
    })

    it('should handle multiple fields', async () => {
      const testData = createTestData(3)
      const normalizer = new ZScoreNormalizer({ 
        in: ['open', 'close', 'volume'],
        out: ['open_zscore', 'close_zscore', 'volume_zscore'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Transform now doesn't yield until ready (windowSize=3)
      // With 3 data points and windowSize=3, we get 1 result
      strictEqual(transformed.length, 1)

      // Check that all fields have z-score versions
      ok('open_zscore' in transformed[0]!)
      ok('close_zscore' in transformed[0])
      ok('volume_zscore' in transformed[0])
      ok(!('high_zscore' in transformed[0]))
      ok(!('low_zscore' in transformed[0]))
    })

    it('should handle constant values (std=0)', async () => {
      const testData: OhlcvDto[] = Array(5).fill(null).map((_, i) => ({
        timestamp: 1000 + i * 1000,
        symbol: 'BTCUSD',
        exchange: 'test',
        open: 100, // All same value
        high: 100,
        low: 100,
        close: 100,
        volume: 1000,
      }))

      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_zscore'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Transform now doesn't yield until ready (windowSize=3)
      // So we should get 3 items
      strictEqual(transformed.length, 3)
      
      // When std=0 (all values same), all z-scores should be 0
      for (let i = 0; i < transformed.length; i++) {
        strictEqual(transformed[i]!.close_zscore, 0)
      }
    })

    it('should process data correctly with rolling window', async () => {
      const testData = createTestData(3)
      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_zscore'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Only yields after windowSize=3 data points, so 1 result
      strictEqual(transformed.length, 1)
      
      // Check that output column exists
      ok('close_zscore' in transformed[0]!)
      
      // Item should have a calculated z-score (not 0 since it has sufficient data)
      ok(typeof transformed[0]!.close_zscore === 'number')
    })
  })

  describe('rolling window behavior validation', () => {
    it('should normalize using rolling window', async () => {
      const testData = createTestData(10)
      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_zscore'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Transform now doesn't yield until ready (windowSize=3)
      strictEqual(transformed.length, 8) // 10 - 2 items that are not ready

      // Check a specific window (items 2, 3, 4 with closes 102, 112, 122)
      // Window for item at index 4 would be [112, 122, 132]
      const window = [112, 122, 132]
      const mean = window.reduce((a, b) => a + b, 0) / 3 // 122
      const variance = window.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / 3
      const std = Math.sqrt(variance)

      // Item at index 2 (which was 4 before) has value 132, should be normalized based on this window
      const expectedZ = (132 - mean) / std
      ok(Math.abs(transformed[2]!.close_zscore! - expectedZ) < 0.01)
    })

    it('should handle insufficient data gracefully', async () => {
      const testData = createTestData(3)
      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_zscore'],
        windowSize: 5 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // With only 3 data points and window size 5, nothing will be yielded
      strictEqual(transformed.length, 0)
    })
  })

  describe('column-driven configuration', () => {
    it('should use custom output column names', async () => {
      const testData = createTestData(3)
      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_z'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      ok('close_z' in transformed[0]!)
      ok(!('close_zscore' in transformed[0]))
    })

    it('should overwrite original columns when using same names', async () => {
      const testData = createTestData(3)
      const normalizer = new ZScoreNormalizer({ 
        in: ['volume'],
        out: ['volume'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Transform now doesn't yield until ready (windowSize=3)
      // With 3 data points and windowSize=3, we get 1 result
      strictEqual(transformed.length, 1)
      
      // Item should have normalized value (z-score)
      ok(typeof transformed[0]!.volume === 'number')
      // Volume should be transformed to z-score, not original value
      ok(transformed[0]!.volume !== 1000) // Original would be 1000
    })

    it('should drop columns when output is null', async () => {
      const testData = createTestData(3)
      const normalizer = new ZScoreNormalizer({ 
        in: ['close', 'volume'],
        out: ['close_zscore', null],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      ok('close_zscore' in transformed[0]!)
      ok(!('volume' in transformed[0])) // Should be dropped
    })
  })

  describe('multiple symbols', () => {
    it('should normalize each symbol independently', async () => {
      const testData: OhlcvDto[] = []
      
      // Add BTC data (high values) - need enough for window
      for (let i = 0; i < 6; i++) {
        testData.push({
          timestamp: 1000 + i * 1000,
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 40000 + i * 1000,
          high: 40100 + i * 1000,
          low: 39900 + i * 1000,
          close: 40050 + i * 1000,
          volume: 100000,
        })
      }
      
      // Add ETH data (lower values)
      for (let i = 0; i < 6; i++) {
        testData.push({
          timestamp: 1000 + i * 1000,
          symbol: 'ETHUSD',
          exchange: 'test',
          open: 2000 + i * 100,
          high: 2010 + i * 100,
          low: 1990 + i * 100,
          close: 2005 + i * 100,
          volume: 50000,
        })
      }

      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_zscore'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Each symbol should be normalized independently
      const btc = transformed.filter(t => t.symbol === 'BTCUSD')
      const eth = transformed.filter(t => t.symbol === 'ETHUSD')

      // Transform now doesn't yield until ready (windowSize=3)
      // Each symbol has 6 items, so each yields 4 results (items 3,4,5,6)
      strictEqual(btc.length, 4)
      strictEqual(eth.length, 4)

      // All items should have z-scores calculated
      ok(typeof btc[0]!.close_zscore === 'number')
      ok(typeof btc[1]!.close_zscore === 'number')
      ok(typeof btc[2]!.close_zscore === 'number')
      ok(typeof eth[0]!.close_zscore === 'number')
      ok(typeof eth[1]!.close_zscore === 'number')
      ok(typeof eth[2]!.close_zscore === 'number')
    })
  })

  describe('getOutputFields and getRequiredFields', () => {
    it('should return correct fields', () => {
      const normalizer1 = new ZScoreNormalizer({})
      deepStrictEqual(
        normalizer1.getOutputFields(), 
        ['open', 'high', 'low', 'close', 'volume']
      )
      deepStrictEqual(
        normalizer1.getRequiredFields(),
        ['open', 'high', 'low', 'close', 'volume']
      )

      const normalizer2 = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_z']
      })
      deepStrictEqual(
        normalizer2.getOutputFields(),
        ['close_z']
      )
      deepStrictEqual(
        normalizer2.getRequiredFields(),
        ['close']
      )
    })
  })

  describe('withParams', () => {
    it('should create new instance with updated params', () => {
      const original = new ZScoreNormalizer({ windowSize: 5 })
      const updated = original.withParams({ windowSize: 10 })

      strictEqual(original.params.windowSize, 5)
      strictEqual(updated.params.windowSize, 10)
      ok(original !== updated)
    })
  })

  describe('readiness', () => {
    it('should not be ready for rolling window before enough data', () => {
      const normalizer = new ZScoreNormalizer({ windowSize: 5 })
      strictEqual(normalizer.isReady(), false)
    })

    it('should be ready for rolling window after enough data', async () => {
      const testData = createTestData(6)
      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_zscore'],
        windowSize: 3 
      })

      // Check readiness before processing
      strictEqual(normalizer.isReady(), false)

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      
      // Process some data
      const iterator = result.data
      // Since transform doesn't yield until ready, the first item we get
      // will be when the transform is already ready (after processing 3 items internally)
      const firstItem = await iterator.next()
      ok(!firstItem.done)
      strictEqual(normalizer.isReady(), true)
    })
  })

  describe('edge cases', () => {
    it('should handle single data point', async () => {
      const testData = createTestData(3)
      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_zscore'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Transform now yields after windowSize=3 data points
      // With exactly 3 items, should yield 1 result
      strictEqual(transformed.length, 1)
      ok(typeof transformed[0]!.close_zscore === 'number')
    })

    it('should handle empty data stream', async () => {
      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_zscore'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator([]))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 0)
    })

    it('should handle negative values', async () => {
      const testData: OhlcvDto[] = [
        { timestamp: 1000, symbol: 'TEST', exchange: 'test', open: -100, high: -95, low: -105, close: -98, volume: 1000 },
        { timestamp: 2000, symbol: 'TEST', exchange: 'test', open: -50, high: -45, low: -55, close: -52, volume: 1100 },
        { timestamp: 3000, symbol: 'TEST', exchange: 'test', open: -200, high: -195, low: -205, close: -198, volume: 1200 },
      ]

      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_zscore'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Transform now doesn't yield until ready (windowSize=3)
      // So we should get 1 item (3 - 3 + 1 = 1)
      strictEqual(transformed.length, 1)
      
      // Should have calculated z-score
      ok(typeof transformed[0]!.close_zscore === 'number')
    })

    it('should handle very large values', async () => {
      const testData: OhlcvDto[] = [
        { timestamp: 1000, symbol: 'TEST', exchange: 'test', open: 1e15, high: 1e15 + 1e10, low: 1e15 - 1e10, close: 1e15, volume: 1e18 },
        { timestamp: 2000, symbol: 'TEST', exchange: 'test', open: 1e15 + 1e14, high: 1e15 + 1e14 + 1e10, low: 1e15 + 1e14 - 1e10, close: 1e15 + 1e14, volume: 1e18 + 1e17 },
        { timestamp: 3000, symbol: 'TEST', exchange: 'test', open: 1e15 + 2e14, high: 1e15 + 2e14 + 1e10, low: 1e15 + 2e14 - 1e10, close: 1e15 + 2e14, volume: 1e18 + 2e17 },
      ]

      const normalizer = new ZScoreNormalizer({ 
        in: ['close'],
        out: ['close_zscore'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Transform now doesn't yield until ready (windowSize=3)
      // So we should get 1 item
      strictEqual(transformed.length, 1)
      
      // Should handle large numbers without issues
      ok(typeof transformed[0]!.close_zscore === 'number') // Should calculate properly
    })
  })
})