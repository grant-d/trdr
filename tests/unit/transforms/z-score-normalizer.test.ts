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
    for await (const item of { [Symbol.asyncIterator]: () => iterator }) {
      results.push(item as TransformedOhlcvDto)
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

  describe('global z-score normalization', () => {
    it('should normalize data to mean=0, std=1', async () => {
      // Create data with known mean and std
      // Values: 100, 110, 120, 130, 140
      // Mean: 120, Std: ~15.81
      const testData = createTestData(5, 100)
      const normalizer = new ZScoreNormalizer({ 
        fields: ['close'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 5)

      // Calculate expected z-scores
      const closes = [102, 112, 122, 132, 142]
      const mean = closes.reduce((a, b) => a + b, 0) / closes.length // 122
      const variance = closes.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / closes.length
      const std = Math.sqrt(variance) // ~15.81

      // Check z-scores
      for (let i = 0; i < 5; i++) {
        const expected = (closes[i]! - mean) / std
        ok(Math.abs(transformed[i]!.close_zscore! - expected) < 0.0001)
      }

      // Verify mean of z-scores is ~0 and std is ~1
      const zScores = transformed.map(t => t.close_zscore!)
      const zMean = zScores.reduce((a, b) => a + b, 0) / zScores.length
      ok(Math.abs(zMean) < 0.0001, `Z-score mean should be ~0, got ${zMean}`)
    })

    it('should handle multiple fields', async () => {
      const testData = createTestData(3)
      const normalizer = new ZScoreNormalizer({ 
        fields: ['open', 'close', 'volume'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Check that all fields have z-score versions
      ok('open_zscore' in transformed[0]!)
      ok('close_zscore' in transformed[0]!)
      ok('volume_zscore' in transformed[0]!)
      ok(!('high_zscore' in transformed[0]!))
      ok(!('low_zscore' in transformed[0]!))
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
        fields: ['close'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // When std=0, all z-scores should be 0
      transformed.forEach(item => {
        strictEqual(item.close_zscore, 0)
      })
    })

    it('should store coefficients', async () => {
      const testData = createTestData(3)
      const normalizer = new ZScoreNormalizer({ 
        fields: ['close'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      await collectResults(result.data)

      // For global normalization, check coefficients on the normalizer instance
      const coefficients = (normalizer as any).getCoefficients() // TODO: Remove any cast
      ok(coefficients)
      strictEqual(coefficients.type, 'zScore')
      ok('close_mean' in coefficients.values)
      ok('close_std' in coefficients.values)
    })
  })

  describe('rolling window z-score normalization', () => {
    it('should normalize using rolling window', async () => {
      const testData = createTestData(10)
      const normalizer = new ZScoreNormalizer({ 
        fields: ['close'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 10)

      // First two items should have z-score based on available data
      // From third item onwards, should use window of 3

      // Check a specific window (items 2, 3, 4 with closes 122, 132, 142)
      const window = [122, 132, 142]
      const mean = window.reduce((a, b) => a + b, 0) / 3 // 132
      const variance = window.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / 3
      const std = Math.sqrt(variance) // ~8.165

      // Item at index 4 should be normalized based on this window
      const expectedZ = (142 - mean) / std
      ok(Math.abs(transformed[4]!.close_zscore! - expectedZ) < 0.01)
    })

    it('should handle insufficient data gracefully', async () => {
      const testData = createTestData(1)
      const normalizer = new ZScoreNormalizer({ 
        fields: ['close'],
        windowSize: 5 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // With only 1 data point and window size 5, should output 0
      strictEqual(transformed[0]!.close_zscore, 0)
    })
  })

  describe('custom suffix and field selection', () => {
    it('should use custom suffix', async () => {
      const testData = createTestData(3)
      const normalizer = new ZScoreNormalizer({ 
        fields: ['close'],
        suffix: '_z',
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      ok('close_z' in transformed[0]!)
      ok(!('close_zscore' in transformed[0]!))
    })

    it('should not add suffix when disabled', async () => {
      const testData = createTestData(3)
      const normalizer = new ZScoreNormalizer({ 
        fields: ['volume'],
        addSuffix: false,
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Original volume should be replaced
      ok(typeof transformed[0]!.volume === 'number')
      // And it should be a z-score (not the original value)
      ok(Math.abs(transformed[0]!.volume) < 5) // Z-scores are typically < 3
    })
  })

  describe('multiple symbols', () => {
    it('should normalize each symbol independently', async () => {
      const testData: OhlcvDto[] = []
      
      // Add BTC data (high values)
      for (let i = 0; i < 3; i++) {
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
      for (let i = 0; i < 3; i++) {
        testData.push({
          timestamp: 1000 + i * 1000,
          symbol: 'ETHUSD',
          exchange: 'test',
          open: 2000 + i * 50,
          high: 2010 + i * 50,
          low: 1990 + i * 50,
          close: 2005 + i * 50,
          volume: 50000,
        })
      }

      const normalizer = new ZScoreNormalizer({ 
        fields: ['close'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Each symbol should have its own normalization
      const btcZScores = transformed.filter(t => t.symbol === 'BTCUSD').map(t => t.close_zscore!)
      const ethZScores = transformed.filter(t => t.symbol === 'ETHUSD').map(t => t.close_zscore!)

      // Both should have mean ~0
      const btcMean = btcZScores.reduce((a, b) => a + b, 0) / btcZScores.length
      const ethMean = ethZScores.reduce((a, b) => a + b, 0) / ethZScores.length
      
      ok(Math.abs(btcMean) < 0.0001)
      ok(Math.abs(ethMean) < 0.0001)
    })
  })

  describe('reverse transform', () => {
    it('should correctly reverse z-score normalization', async () => {
      const testData = createTestData(5)
      const normalizer = new ZScoreNormalizer({ 
        fields: ['close', 'volume'],
        windowSize: null 
      })

      // First, apply normalization
      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const normalized = await collectResults(result.data)

      // Get coefficients after processing
      const coefficients = (normalizer as any).getCoefficients() // TODO: Remove any cast

      // Then reverse it
      const reversed = await collectResults(
        normalizer.reverse!(arrayToAsyncIterator(normalized), coefficients)
      )

      // Check that original values are restored
      for (let i = 0; i < testData.length; i++) {
        ok(Math.abs(reversed[i]!.close - testData[i]!.close) < 0.0001)
        ok(Math.abs(reversed[i]!.volume - testData[i]!.volume) < 0.0001)
        
        // Z-score fields should be removed
        ok(!('close_zscore' in reversed[i]!))
        ok(!('volume_zscore' in reversed[i]!))
      }
    })
  })

  describe('getOutputFields and getRequiredFields', () => {
    it('should return correct fields', () => {
      const normalizer1 = new ZScoreNormalizer({})
      deepStrictEqual(
        normalizer1.getOutputFields(), 
        ['open_zscore', 'high_zscore', 'low_zscore', 'close_zscore', 'volume_zscore']
      )
      deepStrictEqual(
        normalizer1.getRequiredFields(),
        ['open', 'high', 'low', 'close', 'volume']
      )

      const normalizer2 = new ZScoreNormalizer({ 
        fields: ['close', 'volume'],
        suffix: '_normalized'
      })
      deepStrictEqual(
        normalizer2.getOutputFields(),
        ['close_normalized', 'volume_normalized']
      )
      deepStrictEqual(
        normalizer2.getRequiredFields(),
        ['close', 'volume']
      )
    })
  })

  describe('withParams', () => {
    it('should create new instance with updated params', () => {
      const original = new ZScoreNormalizer({ windowSize: 10 })
      const updated = original.withParams({ windowSize: 20 })

      strictEqual(original.params.windowSize, 10)
      strictEqual(updated.params.windowSize, 20)
      ok(original !== updated)
    })
  })

  describe('edge cases', () => {
    it('should handle single data point', async () => {
      const testData = createTestData(1)
      const normalizer = new ZScoreNormalizer({ 
        fields: ['close'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      // With only one data point, z-score should be 0 (undefined std)
      strictEqual(transformed[0]!.close_zscore, 0)
    })

    it('should handle empty data stream', async () => {
      const normalizer = new ZScoreNormalizer({ 
        fields: ['close'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator([]))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 0)
    })

    it('should handle all zeros', async () => {
      const testData: OhlcvDto[] = Array(5).fill(null).map((_, i) => ({
        timestamp: 1000 + i * 1000,
        symbol: 'TEST',
        exchange: 'test',
        open: 0,
        high: 0,
        low: 0,
        close: 0,
        volume: 0,
      }))

      const normalizer = new ZScoreNormalizer({ 
        fields: ['close', 'volume'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // All z-scores should be 0 when all values are 0
      transformed.forEach(item => {
        strictEqual(item.close_zscore, 0)
        strictEqual(item.volume_zscore, 0)
      })
    })

    it('should handle very large values', async () => {
      const testData: OhlcvDto[] = [
        {
          timestamp: 1000,
          symbol: 'TEST',
          exchange: 'test',
          open: 1e15,
          high: 1e15 + 1,
          low: 1e15 - 1,
          close: 1e15,
          volume: 1e18,
        },
        {
          timestamp: 2000,
          symbol: 'TEST',
          exchange: 'test',
          open: 1e15 + 10,
          high: 1e15 + 11,
          low: 1e15 + 9,
          close: 1e15 + 10,
          volume: 1e18 + 1e12,
        },
      ]

      const normalizer = new ZScoreNormalizer({ 
        fields: ['close'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Should handle large numbers without overflow
      ok(Number.isFinite(transformed[0]!.close_zscore!))
      ok(Number.isFinite(transformed[1]!.close_zscore!))
    })
  })
})