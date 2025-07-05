import { deepStrictEqual, ok, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../src/models'
import { LogReturnsNormalizer } from '../../../src/transforms'
import type { TransformedOhlcvDto } from './test-types'

describe('LogReturnsNormalizer', () => {
  // Helper to create test data
  const createTestData = (prices: number[]): OhlcvDto[] => {
    return prices.map((price, i) => ({
      timestamp: new Date('2024-01-01').getTime() + i * 60000,
      symbol: 'BTCUSD',
      exchange: 'test',
      open: price - 1,
      high: price + 1,
      low: price - 2,
      close: price,
      volume: 1000,
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
      const normalizer = new LogReturnsNormalizer({})
      ok(normalizer)
      strictEqual(normalizer.type, 'logReturns')
      strictEqual(normalizer.name, 'Log Returns Normalizer')
    })

    it('should validate price field', () => {
      const normalizer = new LogReturnsNormalizer({ priceField: 'invalid' as any }) // Intentionally invalid
      throws(() => normalizer.validate(), /Invalid price field/)
    })
  })

  describe('log returns calculation', () => {
    it('should calculate natural log returns for close prices by default', async () => {
      const prices = [100, 110, 121, 115.95]
      const testData = createTestData(prices)
      const normalizer = new LogReturnsNormalizer({})

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 4)

      // First bar should have 0 return (no previous price)
      strictEqual(transformed[0]!.close_log_return, 0)

      // Second bar: ln(110/100) = ln(1.1) ≈ 0.0953
      ok(Math.abs(transformed[1]!.close_log_return! - Math.log(1.1)) < 0.0001)

      // Third bar: ln(121/110) = ln(1.1) ≈ 0.0953
      ok(Math.abs(transformed[2]!.close_log_return! - Math.log(1.1)) < 0.0001)

      // Fourth bar: ln(115.95/121) = ln(0.958264...) ≈ -0.0426
      ok(Math.abs(transformed[3]!.close_log_return! - Math.log(115.95/121)) < 0.0001)
    })

    it('should calculate log10 returns when specified', async () => {
      const prices = [100, 1000, 10000]
      const testData = createTestData(prices)
      const normalizer = new LogReturnsNormalizer({ base: 'log10' })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // log10(1000/100) = log10(10) = 1
      ok(Math.abs(transformed[1]!.close_log_return! - 1) < 0.0001)

      // log10(10000/1000) = log10(10) = 1
      ok(Math.abs(transformed[2]!.close_log_return! - 1) < 0.0001)
    })

    it('should use specified price field', async () => {
      const prices = [100, 110, 105]
      const testData = createTestData(prices)
      const normalizer = new LogReturnsNormalizer({ 
        priceField: 'high',
        outputField: 'high_return'
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Check that high prices are used (price + 1)
      strictEqual(transformed[0]!.high_return, 0)
      
      // ln(111/101) ≈ 0.0943
      ok(Math.abs(transformed[1]!.high_return! - Math.log(111/101)) < 0.0001)
    })

    it('should handle multiple symbols', async () => {
      const testData: OhlcvDto[] = [
        {
          timestamp: 1000,
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 100,
          high: 102,
          low: 98,
          close: 101,
          volume: 1000,
        },
        {
          timestamp: 2000,
          symbol: 'ETHUSD',
          exchange: 'test',
          open: 50,
          high: 52,
          low: 48,
          close: 51,
          volume: 500,
        },
        {
          timestamp: 3000,
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 102,
          high: 104,
          low: 100,
          close: 103,
          volume: 1100,
        },
        {
          timestamp: 4000,
          symbol: 'ETHUSD',
          exchange: 'test',
          open: 51,
          high: 53,
          low: 49,
          close: 52,
          volume: 600,
        },
      ]

      const normalizer = new LogReturnsNormalizer({})
      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // First occurrence of each symbol should have 0 return
      strictEqual(transformed[0]!.close_log_return, 0) // BTC first
      strictEqual(transformed[1]!.close_log_return, 0) // ETH first

      // Second occurrence should have proper returns
      ok(Math.abs(transformed[2]!.close_log_return! - Math.log(103/101)) < 0.0001) // BTC
      ok(Math.abs(transformed[3]!.close_log_return! - Math.log(52/51)) < 0.0001) // ETH
    })

    it('should handle zero and negative prices', async () => {
      const testData: OhlcvDto[] = [
        {
          timestamp: 1000,
          symbol: 'TEST',
          exchange: 'test',
          open: 100,
          high: 100,
          low: 100,
          close: 100,
          volume: 1000,
        },
        {
          timestamp: 2000,
          symbol: 'TEST',
          exchange: 'test',
          open: 0,
          high: 0,
          low: 0,
          close: 0,
          volume: 0,
        },
        {
          timestamp: 3000,
          symbol: 'TEST',
          exchange: 'test',
          open: 50,
          high: 50,
          low: 50,
          close: 50,
          volume: 500,
        },
      ]

      const normalizer = new LogReturnsNormalizer({})
      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // When current or previous price is 0 or negative, return should be 0
      strictEqual(transformed[1]!.close_log_return, 0)
      strictEqual(transformed[2]!.close_log_return, 0)
    })
  })

  describe('coefficients', () => {
    it('should store coefficients on first transform', async () => {
      const testData = createTestData([100, 110])
      const normalizer = new LogReturnsNormalizer({ 
        priceField: 'open',
        base: 'log10' 
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      
      // Consume at least one item to trigger coefficient setting
      const firstItem = await result.data.next()
      ok(!firstItem.done)
      
      // Now check coefficients - they're set on the normalizer instance
      const coefficients = normalizer['coefficients']
      ok(coefficients)
      strictEqual(coefficients.type, 'logReturns')
      strictEqual(coefficients.symbol, 'BTCUSD')
      strictEqual(coefficients.values.priceField, 1) // 'open' = 1
      strictEqual(coefficients.values.base, 10)
      
      // Consume remaining data
      await collectResults(result.data)
    })
  })

  describe('getOutputFields and getRequiredFields', () => {
    it('should return correct output fields', () => {
      const normalizer1 = new LogReturnsNormalizer({})
      deepStrictEqual(normalizer1.getOutputFields(), ['close_log_return'])

      const normalizer2 = new LogReturnsNormalizer({ 
        priceField: 'high',
        outputField: 'high_lr' 
      })
      deepStrictEqual(normalizer2.getOutputFields(), ['high_lr'])
    })

    it('should return correct required fields', () => {
      const normalizer1 = new LogReturnsNormalizer({})
      deepStrictEqual(normalizer1.getRequiredFields(), ['close'])

      const normalizer2 = new LogReturnsNormalizer({ priceField: 'open' })
      deepStrictEqual(normalizer2.getRequiredFields(), ['open'])
    })
  })

  describe('withParams', () => {
    it('should create new instance with updated params', () => {
      const original = new LogReturnsNormalizer({ priceField: 'close' })
      const updated = original.withParams({ priceField: 'open' })

      strictEqual(original.params.priceField, 'close')
      strictEqual(updated.params.priceField, 'open')
      ok(original !== updated)
    })
  })

  describe('edge cases', () => {
    it('should handle single data point', async () => {
      const testData = createTestData([100])
      const normalizer = new LogReturnsNormalizer({})

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      strictEqual(transformed[0]!.close_log_return, 0) // No previous price
    })

    it('should handle all zero prices', async () => {
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

      const normalizer = new LogReturnsNormalizer({})
      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // All returns should be 0 when prices are 0
      transformed.forEach(item => {
        strictEqual(item.close_log_return, 0)
      })
    })

    it('should handle constant non-zero prices', async () => {
      const testData: OhlcvDto[] = Array(5).fill(null).map((_, i) => ({
        timestamp: 1000 + i * 1000,
        symbol: 'TEST',
        exchange: 'test',
        open: 100,
        high: 100,
        low: 100,
        close: 100,
        volume: 1000,
      }))

      const normalizer = new LogReturnsNormalizer({})
      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // First should be 0, rest should be log(1) = 0
      strictEqual(transformed[0]!.close_log_return, 0)
      for (let i = 1; i < transformed.length; i++) {
        strictEqual(transformed[i]!.close_log_return, 0) // log(100/100) = log(1) = 0
      }
    })

    it('should handle empty data stream', async () => {
      const normalizer = new LogReturnsNormalizer({})
      const result = await normalizer.apply(arrayToAsyncIterator([]))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 0)
    })
  })

  describe('reverse transform', () => {
    it('should throw error indicating limitation', async () => {
      const normalizer = new LogReturnsNormalizer({})
      const data = arrayToAsyncIterator([])
      const coefficients = { type: 'logReturns', values: {} }

      try {
        await normalizer.reverse(data, coefficients).next()
        ok(false, 'Should have thrown')
      } catch (error) {
        ok((error as Error).message.includes('initial price'))
      }
    })
  })
})