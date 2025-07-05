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
    let item = await iterator.next()
    while (!item.done) {
      results.push(item.value as TransformedOhlcvDto)
      item = await iterator.next()
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

    it('should create instance with custom base', () => {
      const normalizer = new LogReturnsNormalizer({ base: 'log10' })
      strictEqual(normalizer.description, 'Calculates logarithmic returns from price data')
    })
  })

  describe('log returns calculation', () => {
    it('should calculate natural log returns for close prices by default', async () => {
      const testData = createTestData([100, 105, 110])
      const normalizer = new LogReturnsNormalizer({
        in: ['close'],
        out: ['close_returns']
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 3)
      
      // First item has no previous data, so returns should be 0
      strictEqual(transformed[0]!.close_returns, 0)
      
      // Calculate expected log returns: ln(105/100) and ln(110/105)
      const expectedReturn1 = Math.log(105 / 100)
      const expectedReturn2 = Math.log(110 / 105)
      
      ok(Math.abs(transformed[1]!.close_returns! - expectedReturn1) < 0.0001)
      ok(Math.abs(transformed[2]!.close_returns! - expectedReturn2) < 0.0001)
    })

    it('should calculate log10 returns when specified', async () => {
      const testData = createTestData([100, 110])
      const normalizer = new LogReturnsNormalizer({
        in: ['close'],
        out: ['close_log10'],
        base: 'log10'
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed[0]!.close_log10, 0) // First item
      
      const expectedReturn = Math.log10(110 / 100)
      ok(Math.abs(transformed[1]!.close_log10! - expectedReturn) < 0.0001)
    })

    it('should use specified price field', async () => {
      const testData = createTestData([100, 105])
      const normalizer = new LogReturnsNormalizer({
        in: ['open'],
        out: ['open_returns']
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed[0]!.open_returns, 0) // First item
      
      // Open values are price - 1, so 99 -> 104
      const expectedReturn = Math.log(104 / 99)
      ok(Math.abs(transformed[1]!.open_returns! - expectedReturn) < 0.0001)
    })

    it('should handle multiple symbols', async () => {
      const testData: OhlcvDto[] = [
        { timestamp: 1000, symbol: 'BTCUSD', exchange: 'test', open: 99, high: 101, low: 98, close: 100, volume: 1000 },
        { timestamp: 2000, symbol: 'BTCUSD', exchange: 'test', open: 104, high: 106, low: 103, close: 105, volume: 1000 },
        { timestamp: 3000, symbol: 'ETHUSD', exchange: 'test', open: 199, high: 201, low: 198, close: 200, volume: 2000 },
        { timestamp: 4000, symbol: 'ETHUSD', exchange: 'test', open: 209, high: 211, low: 208, close: 210, volume: 2000 },
      ]

      const normalizer = new LogReturnsNormalizer({
        in: ['close'],
        out: ['close_returns']
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // First item of each symbol should have 0 returns
      strictEqual(transformed[0]!.close_returns, 0) // BTC first
      strictEqual(transformed[2]!.close_returns, 0) // ETH first

      // Check BTC return: ln(105/100)
      const btcReturn = Math.log(105 / 100)
      ok(Math.abs(transformed[1]!.close_returns! - btcReturn) < 0.0001)

      // Check ETH return: ln(210/200)  
      const ethReturn = Math.log(210 / 200)
      ok(Math.abs(transformed[3]!.close_returns! - ethReturn) < 0.0001)
    })

    it('should handle zero and negative prices', async () => {
      const testData: OhlcvDto[] = [
        { timestamp: 1000, symbol: 'TEST', exchange: 'test', open: 100, high: 101, low: 99, close: 100, volume: 1000 },
        { timestamp: 2000, symbol: 'TEST', exchange: 'test', open: 0, high: 1, low: -1, close: 0, volume: 1000 },
        { timestamp: 3000, symbol: 'TEST', exchange: 'test', open: -10, high: -9, low: -11, close: -10, volume: 1000 },
      ]

      const normalizer = new LogReturnsNormalizer({
        in: ['close'],
        out: ['close_returns']
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed[0]!.close_returns, 0) // First item
      strictEqual(transformed[1]!.close_returns, 0) // Zero/negative price -> 0 return
      strictEqual(transformed[2]!.close_returns, 0) // Negative price -> 0 return
    })
  })

  describe('getOutputFields and getRequiredFields', () => {
    it('should return correct output fields', () => {
      const normalizer1 = new LogReturnsNormalizer({})
      deepStrictEqual(
        normalizer1.getOutputFields(),
        ['open', 'high', 'low', 'close', 'volume']
      )

      const normalizer2 = new LogReturnsNormalizer({
        in: ['close'],
        out: ['close_returns']
      })
      deepStrictEqual(
        normalizer2.getOutputFields(),
        ['close_returns']
      )
    })

    it('should return correct required fields', () => {
      const normalizer1 = new LogReturnsNormalizer({})
      deepStrictEqual(
        normalizer1.getRequiredFields(),
        ['open', 'high', 'low', 'close', 'volume']
      )

      const normalizer2 = new LogReturnsNormalizer({
        in: ['close'],
        out: ['close_returns']
      })
      deepStrictEqual(
        normalizer2.getRequiredFields(),
        ['close']
      )
    })
  })

  describe('withParams', () => {
    it('should create new instance with updated params', () => {
      const original = new LogReturnsNormalizer({ base: 'natural' })
      const updated = original.withParams({ base: 'log10' })

      strictEqual(original.params.base, 'natural')
      strictEqual(updated.params.base, 'log10')
      ok(original !== updated)
    })
  })

  describe('readiness', () => {
    it('should be ready after processing 2 data points', async () => {
      const testData = createTestData([100, 105, 110])
      const normalizer = new LogReturnsNormalizer({
        in: ['close'],
        out: ['close_returns']
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      
      // Process data points
      const iterator = result.data
      await iterator.next() // 1st point
      strictEqual(normalizer.isReady(), false)
      
      await iterator.next() // 2nd point - should be ready now
      strictEqual(normalizer.isReady(), true)
    })
  })

  describe('column-driven configuration', () => {
    it('should overwrite original columns when using same names', async () => {
      const testData = createTestData([100, 105])
      const normalizer = new LogReturnsNormalizer({
        in: ['close'],
        out: ['close'] // Overwrite close column
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // Close should be overwritten with log returns
      strictEqual(transformed[0]!.close, 0) // First item
      const expectedReturn = Math.log(105 / 100)
      ok(Math.abs(transformed[1]!.close - expectedReturn) < 0.0001)
    })

    it('should drop columns when output is null', async () => {
      const testData = createTestData([100, 105])
      const normalizer = new LogReturnsNormalizer({
        in: ['close', 'volume'],
        out: ['close_returns', null] // Drop volume
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      ok('close_returns' in transformed[0]!)
      ok(!('volume' in transformed[0])) // Should be dropped
    })
  })

  describe('edge cases', () => {
    it('should handle single data point', async () => {
      const testData = createTestData([100])
      const normalizer = new LogReturnsNormalizer({
        in: ['close'],
        out: ['close_returns']
      })

      const result = await normalizer.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      strictEqual(transformed[0]!.close_returns, 0) // No previous data
    })

    it('should handle empty data stream', async () => {
      const normalizer = new LogReturnsNormalizer({
        in: ['close'],
        out: ['close_returns']
      })

      const result = await normalizer.apply(arrayToAsyncIterator([]))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 0)
    })
  })
})