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
    let item = await iterator.next()
    while (!item.done) {
      results.push(item.value as TransformedOhlcvDto)
      item = await iterator.next()
    }
    return results
  }

  describe('constructor and validation', () => {
    it('should create instance with default parameters', () => {
      const normalizer = new MinMaxNormalizer({})
      ok(normalizer)
      strictEqual(normalizer.type, 'minMax')
      strictEqual(normalizer.name, 'Min-Max Normalizer')
      strictEqual(normalizer.description, 'Normalizes data to range [0, 1] using 20 period rolling window')
    })

    it('should create instance with custom range', () => {
      const normalizer = new MinMaxNormalizer({ min: -1, max: 1 })
      strictEqual(normalizer.description, 'Normalizes data to range [-1, 1] using 20 period rolling window')
    })

    it('should validate window size', () => {
      const normalizer = new MinMaxNormalizer({ windowSize: 1 })
      throws(() => normalizer.validate(), /Window size must be at least 2/)
    })

    it('should validate target range', () => {
      const normalizer = new MinMaxNormalizer({ min: 1, max: 0 })
      throws(() => normalizer.validate(), /targetMax must be greater than targetMin/)
    })
  })

  describe('rolling window min-max normalization', () => {
    it('should normalize data to [0, 1] range by default', async () => {
      // Values: 0, 50, 100
      const testData = createTestData([0, 50, 100])
      const normalizer = new MinMaxNormalizer({ 
        in: ['open'],
        out: ['open_norm'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 3)

      // First two items have insufficient data (window < 3), so output 0
      strictEqual(transformed[0]!.open_norm, 0)
      strictEqual(transformed[1]!.open_norm, 0)
      
      // Third item has full window [0, 50, 100]: min=0, max=100, value=100
      strictEqual(transformed[2]!.open_norm, 1)    // (100-0)/(100-0) = 1
    })

    it('should normalize to custom range', async () => {
      const testData = createTestData([10, 20, 30])
      const normalizer = new MinMaxNormalizer({ 
        in: ['open'],
        out: ['open_norm'],
        min: -1,
        max: 1,
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // First two items have insufficient data, so output 0
      strictEqual(transformed[0]!.open_norm, 0)
      strictEqual(transformed[1]!.open_norm, 0)
      
      // Third item has window [10, 20, 30]: min=10, max=30, value=30
      // Formula: (x-min)/range * targetRange + targetMin = (30-10)/20 * 2 + (-1) = 1
      strictEqual(transformed[2]!.open_norm, 1)
    })

    it('should handle multiple fields', async () => {
      const testData = createTestData([100, 200])
      const normalizer = new MinMaxNormalizer({  
        in: ['open', 'close', 'volume'],
        out: ['open_norm', 'close_norm', 'volume_norm'],
        windowSize: 2 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // First item has insufficient data (window size 2 needs 2 items)
      strictEqual(transformed[0]!.open_norm, 0)
      strictEqual(transformed[0]!.close_norm, 0)
      strictEqual(transformed[0]!.volume_norm, 0)
      
      // Second item should have normalized values  
      ok('open_norm' in transformed[1]!)
      ok('close_norm' in transformed[1])
      ok('volume_norm' in transformed[1])
      ok(!('high_norm' in transformed[1]))
      ok(!('low_norm' in transformed[1]))
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
        in: ['close'],
        out: ['close_norm'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // First two items have insufficient data
      strictEqual(transformed[0]!.close_norm, 0)
      strictEqual(transformed[1]!.close_norm, 0)
      
      // Remaining items: when all values in window are the same, should use middle of target range
      for (let i = 2; i < transformed.length; i++) {
        strictEqual(transformed[i]!.close_norm, 0.5) // Middle of [0, 1]
      }
    })
  })

  describe('rolling window behavior validation', () => {
    it('should normalize using rolling window', async () => {
      const testData = createTestData([10, 20, 30, 25, 15])
      const normalizer = new MinMaxNormalizer({  
        in: ['open'],
        out: ['open_norm'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 5)

      // First two items have insufficient data
      strictEqual(transformed[0]!.open_norm, 0)
      strictEqual(transformed[1]!.open_norm, 0)
      
      // Window [10, 20, 30]: min=10, max=30, value=30
      strictEqual(transformed[2]!.open_norm, 1) // (30-10)/(30-10) = 1
      
      // Window [20, 30, 25]: min=20, max=30, value=25
      const expected3 = (25 - 20) / (30 - 20) // 0.5
      strictEqual(transformed[3]!.open_norm, expected3)

      // Window [30, 25, 15]: min=15, max=30, value=15
      const expected4 = (15 - 15) / (30 - 15) // 0
      strictEqual(transformed[4]!.open_norm, expected4)
    })

    it('should handle insufficient data gracefully', async () => {
      const testData = createTestData([100])
      const normalizer = new MinMaxNormalizer({  
        in: ['open'],
        out: ['open_norm'],
        windowSize: 5 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // With only 1 data point (less than window size), should output 0
      strictEqual(transformed[0]!.open_norm, 0)
    })
  })

  describe('column-driven configuration', () => {
    it('should use custom output column names', async () => {
      const testData = createTestData([10, 20, 30])
      const normalizer = new MinMaxNormalizer({  
        in: ['close'],
        out: ['close_scaled'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      ok('close_scaled' in transformed[0]!)
      ok(!('close_norm' in transformed[0]))
    })

    it('should overwrite original columns when using same names', async () => {
      const testData = createTestData([10, 20, 30])
      const normalizer = new MinMaxNormalizer({  
        in: ['volume'],
        out: ['volume'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // First two items will have zero values (not enough data)
      strictEqual(transformed[0]!.volume, 0)
      strictEqual(transformed[1]!.volume, 0)
      
      // Third item should have normalized value based on window [10, 20, 30]
      // Volume: 300, window volumes: [100, 200, 300], min=100, max=300, range=200
      // (300-100)/200 = 1
      strictEqual(transformed[2]!.volume, 1)
    })

    it('should drop columns when output is null', async () => {
      const testData = createTestData([10, 20])
      const normalizer = new MinMaxNormalizer({  
        in: ['close', 'volume'],
        out: ['close_norm', null],
        windowSize: 2 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      ok('close_norm' in transformed[0]!)
      ok(!('volume' in transformed[0])) // Should be dropped
    })
  })

  describe('multiple symbols', () => {
    it('should normalize each symbol independently', async () => {
      const testData: OhlcvDto[] = []
      
      // Add BTC data (high values) - need enough for window size
      for (let i = 0; i < 6; i++) {
        testData.push({
          timestamp: 1000 + i * 1000,
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 40000 + i * 10000, // 40k, 50k, 60k, 70k, 80k, 90k
          high: 40100,
          low: 39900,
          close: 40050,
          volume: 100000,
        })
      }
      
      // Add ETH data (lower values)
      for (let i = 0; i < 6; i++) {
        testData.push({
          timestamp: 1000 + i * 1000,
          symbol: 'ETHUSD',
          exchange: 'test',
          open: 2000 + i * 100, // 2000, 2100, 2200, 2300, 2400, 2500
          high: 2010,
          low: 1990,
          close: 2005,
          volume: 50000,
        })
      }

      const normalizer = new MinMaxNormalizer({  
        in: ['open'],
        out: ['open_norm'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // BTC data - first 2 items insufficient data, then windows available
      const btc = transformed.filter(t => t.symbol === 'BTCUSD')
      strictEqual(btc[0]!.open_norm, 0)   // Insufficient data
      strictEqual(btc[1]!.open_norm, 0)   // Insufficient data
      strictEqual(btc[2]!.open_norm, 1)   // Window [40k, 50k, 60k], value=60k

      // ETH data - same pattern
      const eth = transformed.filter(t => t.symbol === 'ETHUSD')
      strictEqual(eth[0]!.open_norm, 0)   // Insufficient data
      strictEqual(eth[1]!.open_norm, 0)   // Insufficient data
      strictEqual(eth[2]!.open_norm, 1)   // Window [2000, 2100, 2200], value=2200
    })
  })

  describe('getOutputFields and getRequiredFields', () => {
    it('should return correct fields', () => {
      const normalizer1 = new MinMaxNormalizer({ })
      deepStrictEqual(
        normalizer1.getOutputFields(), 
        ['open', 'high', 'low', 'close', 'volume']
      )
      deepStrictEqual(
        normalizer1.getRequiredFields(),
        ['open', 'high', 'low', 'close', 'volume']
      )

      const normalizer2 = new MinMaxNormalizer({  
        in: ['close'],
        out: ['close_01']
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
      const original = new MinMaxNormalizer({ min: 0, max: 1 })
      const updated = original.withParams({ min: -1, max: 1 })

      strictEqual(original.params.min, 0)
      strictEqual(updated.params.min, -1)
      ok(original !== updated)
    })
  })

  describe('readiness', () => {
    it('should not be ready for rolling window before enough data', () => {
      const normalizer = new MinMaxNormalizer({ windowSize: 5 })
      strictEqual(normalizer.isReady(), false)
    })

    it('should be ready for rolling window after enough data', async () => {
      const testData = createTestData([1, 2, 3, 4, 5, 6])
      const normalizer = new MinMaxNormalizer({ 
        in: ['open'],
        out: ['open_norm'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      
      // Process some data
      const iterator = result.data
      await iterator.next() // 1st
      await iterator.next() // 2nd
      strictEqual(normalizer.isReady(), false)
      
      await iterator.next() // 3rd - should be ready now
      strictEqual(normalizer.isReady(), true)
    })
  })

  describe('edge cases', () => {
    it('should handle single data point', async () => {
      const testData = createTestData([100])
      const normalizer = new MinMaxNormalizer({  
        in: ['close'],
        out: ['close_norm'],
        windowSize: 2 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      // With insufficient data (1 item but window size 2), should output 0
      strictEqual(transformed[0]!.close_norm, 0)
    })

    it('should handle empty data stream', async () => {
      const normalizer = new MinMaxNormalizer({  
        in: ['close'],
        out: ['close_norm'],
        windowSize: 5 
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
        in: ['close', 'volume'],
        out: ['close_norm', 'volume_norm'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // First two items have insufficient data
      for (let i = 0; i < 2; i++) {
        strictEqual(transformed[i]!.close_norm, 0)
        strictEqual(transformed[i]!.volume_norm, 0)
      }
      
      // Remaining items: all values same, should use middle of range
      for (let i = 2; i < transformed.length; i++) {
        strictEqual(transformed[i]!.close_norm, 0.5)
        strictEqual(transformed[i]!.volume_norm, 0.5)
      }
    })

    it('should handle negative values', async () => {
      const testData = createTestData([-100, -50, -200])
      const normalizer = new MinMaxNormalizer({  
        in: ['open'],
        out: ['open_norm'],
        windowSize: 3 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // First two items have insufficient data
      strictEqual(transformed[0]!.open_norm, 0)
      strictEqual(transformed[1]!.open_norm, 0)
      
      // Window [-100, -50, -200]: min=-200, max=-50, value=-200
      // (-200-(-200))/(-50-(-200)) = 0/150 = 0
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
        in: ['open'],
        out: ['open_norm'],
        windowSize: 2 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // First item has insufficient data
      strictEqual(transformed[0]!.open_norm, 0)
      // Second item has window of 2, should normalize properly
      strictEqual(transformed[1]!.open_norm, 1)
    })
  })
})