import { deepStrictEqual, ok, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../src/models'
import { MinMaxNormalizer } from '../../../src/transforms'
import type { TransformedOhlcvDto } from './test-types'

describe('MinMaxNormalizer', () => {
  // Helper to create test data
  const createTestData = (values: number[]): OhlcvDto[] => {
    return values.map((value, i) => ({
      timestamp: new Date('2024-01-01').getTime() + i * 60000,
      symbol: 'BTCUSD',
      exchange: 'test',
      open: value,
      high: value + 10,
      low: value - 10,
      close: value + 5,
      volume: value * 10,
    }))
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
      const normalizer = new MinMaxNormalizer({})
      ok(normalizer)
      strictEqual(normalizer.type, 'minMax')
      strictEqual(normalizer.name, 'Min-Max Normalizer')
      strictEqual(normalizer.description, 'Normalizes data to range [0, 1]')
    })

    it('should create instance with custom range', () => {
      const normalizer = new MinMaxNormalizer({ targetMin: -1, targetMax: 1 })
      strictEqual(normalizer.description, 'Normalizes data to range [-1, 1]')
    })

    it('should validate window size', () => {
      const normalizer = new MinMaxNormalizer({ windowSize: 1 })
      throws(() => normalizer.validate(), /Window size must be at least 2/)
    })

    it('should validate target range', () => {
      const normalizer = new MinMaxNormalizer({ targetMin: 1, targetMax: 0 })
      throws(() => normalizer.validate(), /targetMax must be greater than targetMin/)
    })
  })

  describe('global min-max normalization', () => {
    it('should normalize data to [0, 1] range by default', async () => {
      // Values: 0, 50, 100
      const testData = createTestData([0, 50, 100])
      const normalizer = new MinMaxNormalizer({ 
        fields: ['open'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 3)

      // Check normalized values
      strictEqual(transformed[0]!.open_norm, 0)    // (0-0)/(100-0) = 0
      strictEqual(transformed[1]!.open_norm, 0.5)  // (50-0)/(100-0) = 0.5
      strictEqual(transformed[2]!.open_norm, 1)    // (100-0)/(100-0) = 1
    })

    it('should normalize to custom range', async () => {
      const testData = createTestData([10, 20, 30])
      const normalizer = new MinMaxNormalizer({ 
        fields: ['open'],
        targetMin: -1,
        targetMax: 1,
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Min=10, Max=30, Range=20
      // Formula: (x-min)/range * targetRange + targetMin
      strictEqual(transformed[0]!.open_norm, -1)   // (10-10)/20 * 2 + (-1) = -1
      strictEqual(transformed[1]!.open_norm, 0)    // (20-10)/20 * 2 + (-1) = 0
      strictEqual(transformed[2]!.open_norm, 1)    // (30-10)/20 * 2 + (-1) = 1
    })

    it('should handle multiple fields', async () => {
      const testData = createTestData([100])
      const normalizer = new MinMaxNormalizer({ 
        fields: ['open', 'close', 'volume'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Check that all fields have normalized versions
      ok('open_norm' in transformed[0]!)
      ok('close_norm' in transformed[0]!)
      ok('volume_norm' in transformed[0]!)
      ok(!('high_norm' in transformed[0]!))
      ok(!('low_norm' in transformed[0]!))
    })

    it('should handle constant values (range=0)', async () => {
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

      const normalizer = new MinMaxNormalizer({ 
        fields: ['close'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // When all values are the same, should use middle of target range
      transformed.forEach(item => {
        strictEqual(item.close_norm, 0.5) // Middle of [0, 1]
      })
    })

    it('should store coefficients', async () => {
      const testData = createTestData([10, 20, 30])
      const normalizer = new MinMaxNormalizer({ 
        fields: ['close'],
        targetMin: -1,
        targetMax: 1,
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      
      // Consume at least one item to trigger coefficient setting
      const firstItem = await result.data.next()
      ok(!firstItem.done)
      
      // Now check coefficients - they're set on the normalizer instance
      const coefficients = normalizer['coefficients']
      ok(coefficients)
      strictEqual(coefficients.type, 'minMax')
      strictEqual(coefficients.values.close_min, 15) // close = open + 5
      strictEqual(coefficients.values.close_max, 35)
      strictEqual(coefficients.values.target_min, -1)
      strictEqual(coefficients.values.target_max, 1)
      
      // Consume remaining data
      await collectResults(result.data)
    })
  })

  describe('rolling window min-max normalization', () => {
    it('should normalize using rolling window', async () => {
      const testData = createTestData([10, 20, 30, 25, 15])
      const normalizer = new MinMaxNormalizer({ 
        fields: ['open'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 5)

      // Check specific windows
      // Window [20, 30, 25]: min=20, max=30
      const window3Value = 25
      const expected3 = (window3Value - 20) / (30 - 20) // 0.5
      strictEqual(transformed[3]!.open_norm, expected3)

      // Window [30, 25, 15]: min=15, max=30
      const window4Value = 15
      const expected4 = (window4Value - 15) / (30 - 15) // 0
      strictEqual(transformed[4]!.open_norm, expected4)
    })

    it('should handle insufficient data gracefully', async () => {
      const testData = createTestData([100])
      const normalizer = new MinMaxNormalizer({ 
        fields: ['open'],
        windowSize: 5 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // With only 1 data point, should output targetMin
      strictEqual(transformed[0]!.open_norm, 0)
    })
  })

  describe('custom suffix and field selection', () => {
    it('should use custom suffix', async () => {
      const testData = createTestData([10, 20])
      const normalizer = new MinMaxNormalizer({ 
        fields: ['close'],
        suffix: '_scaled',
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      ok('close_scaled' in transformed[0]!)
      ok(!('close_norm' in transformed[0]!))
    })

    it('should not add suffix when disabled', async () => {
      const testData = createTestData([10, 20])
      const normalizer = new MinMaxNormalizer({ 
        fields: ['volume'],
        addSuffix: false,
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Original volume should be replaced with normalized value
      strictEqual(transformed[0]!.volume, 0) // Min value
      strictEqual(transformed[1]!.volume, 1) // Max value
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
          open: 40000 + i * 10000, // 40k, 50k, 60k
          high: 40100,
          low: 39900,
          close: 40050,
          volume: 100000,
        })
      }
      
      // Add ETH data (lower values)
      for (let i = 0; i < 3; i++) {
        testData.push({
          timestamp: 1000 + i * 1000,
          symbol: 'ETHUSD',
          exchange: 'test',
          open: 2000 + i * 100, // 2000, 2100, 2200
          high: 2010,
          low: 1990,
          close: 2005,
          volume: 50000,
        })
      }

      const normalizer = new MinMaxNormalizer({ 
        fields: ['open'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // BTC: min=40k, max=60k
      const btc = transformed.filter(t => t.symbol === 'BTCUSD')
      strictEqual(btc[0]!.open_norm, 0)   // 40k
      strictEqual(btc[1]!.open_norm, 0.5) // 50k
      strictEqual(btc[2]!.open_norm, 1)   // 60k

      // ETH: min=2000, max=2200
      const eth = transformed.filter(t => t.symbol === 'ETHUSD')
      strictEqual(eth[0]!.open_norm, 0)   // 2000
      strictEqual(eth[1]!.open_norm, 0.5) // 2100
      strictEqual(eth[2]!.open_norm, 1)   // 2200
    })
  })

  describe('reverse transform', () => {
    it('should correctly reverse min-max normalization', async () => {
      const testData = createTestData([10, 30, 20])
      const normalizer = new MinMaxNormalizer({ 
        fields: ['open', 'volume'],
        targetMin: -1,
        targetMax: 1,
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
        ok(Math.abs(reversed[i]!.open - testData[i]!.open) < 0.0001)
        ok(Math.abs(reversed[i]!.volume - testData[i]!.volume) < 0.0001)
        
        // Normalized fields should be removed
        ok(!('open_norm' in reversed[i]!))
        ok(!('volume_norm' in reversed[i]!))
      }
    })

    it('should handle edge case where targetRange is 0', async () => {
      // This is an edge case that shouldn't happen in practice
      const normalizer = new MinMaxNormalizer({ 
        fields: ['open'],
        windowSize: null 
      })

      const coefficients = {
        type: 'minMax' as any, // TODO: Remove any cast
        symbol: 'BTCUSD',
        timestamp: Date.now(),
        values: {
          open_min: 10,
          open_max: 20,
          target_min: 1,
          target_max: 1, // Same as min!
        }
      }

      const normalized: OhlcvDto[] = [{
        timestamp: 1000,
        symbol: 'BTCUSD',
        exchange: 'test',
        open: 15,
        high: 15,
        low: 15,
        close: 15,
        volume: 150,
        open_norm: 1, // Would be stuck at target_min
      }]

      const reversed = await collectResults(
        normalizer.reverse!(arrayToAsyncIterator(normalized), coefficients)
      )

      // Should return min value when targetRange is 0
      strictEqual(reversed[0]!.open, 10)
    })
  })

  describe('getOutputFields and getRequiredFields', () => {
    it('should return correct fields', () => {
      const normalizer1 = new MinMaxNormalizer({})
      deepStrictEqual(
        normalizer1.getOutputFields(), 
        ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm']
      )
      deepStrictEqual(
        normalizer1.getRequiredFields(),
        ['open', 'high', 'low', 'close', 'volume']
      )

      const normalizer2 = new MinMaxNormalizer({ 
        fields: ['close'],
        suffix: '_01'
      })
      deepStrictEqual(
        normalizer2.getOutputFields(),
        ['close_01']
      )
      deepStrictEqual(
        normalizer2.getRequiredFields(),
        ['close']
      )
    })
  })

  describe('withParams', () => {
    it('should create new instance with updated params', () => {
      const original = new MinMaxNormalizer({ targetMin: 0, targetMax: 1 })
      const updated = original.withParams({ targetMin: -1, targetMax: 1 })

      strictEqual(original.params.targetMin, 0)
      strictEqual(updated.params.targetMin, -1)
      ok(original !== updated)
    })
  })

  describe('edge cases', () => {
    it('should handle single data point', async () => {
      const testData = createTestData([100])
      const normalizer = new MinMaxNormalizer({ 
        fields: ['close'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      // With only one value, should use middle of range
      strictEqual(transformed[0]!.close_norm, 0.5)
    })

    it('should handle empty data stream', async () => {
      const normalizer = new MinMaxNormalizer({ 
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

      const normalizer = new MinMaxNormalizer({ 
        fields: ['close', 'volume'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // All values same, should use middle of range
      transformed.forEach(item => {
        strictEqual(item.close_norm, 0.5)
        strictEqual(item.volume_norm, 0.5)
      })
    })

    it('should handle negative values', async () => {
      const testData = createTestData([-100, -50, -200])
      const normalizer = new MinMaxNormalizer({ 
        fields: ['open'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Min=-200, Max=-50, range=150
      // (-100-(-200))/150 = 100/150 = 0.667
      strictEqual(transformed[0]!.open_norm.toFixed(3), '0.667')
      // (-50-(-200))/150 = 150/150 = 1
      strictEqual(transformed[1]!.open_norm, 1)
      // (-200-(-200))/150 = 0/150 = 0
      strictEqual(transformed[2]!.open_norm, 0)
    })

    it('should handle very large values', async () => {
      const testData: OhlcvDto[] = [
        {
          timestamp: 1000,
          symbol: 'TEST',
          exchange: 'test',
          open: 1e15,
          high: 1e15 + 1e10,
          low: 1e15 - 1e10,
          close: 1e15,
          volume: 1e18,
        },
        {
          timestamp: 2000,
          symbol: 'TEST',
          exchange: 'test',
          open: 1e15 + 1e14,
          high: 1e15 + 1e14 + 1e10,
          low: 1e15 + 1e14 - 1e10,
          close: 1e15 + 1e14,
          volume: 1e18 + 1e17,
        },
      ]

      const normalizer = new MinMaxNormalizer({ 
        fields: ['open'],
        windowSize: null 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Should handle large numbers without precision loss
      strictEqual(transformed[0]!.open_norm, 0)
      strictEqual(transformed[1]!.open_norm, 1)
    })
  })
})