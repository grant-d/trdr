import type { Candle } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import { SMAIndicator } from './sma'

describe('SMAIndicator', () => {
  let candles: Candle[]

  beforeEach(() => {
    // Create test candles with known values
    candles = [
      { timestamp: 1000, open: 10, high: 12, low: 9, close: 11, volume: 100 },
      { timestamp: 2000, open: 11, high: 13, low: 10, close: 12, volume: 110 },
      { timestamp: 3000, open: 12, high: 14, low: 11, close: 13, volume: 120 },
      { timestamp: 4000, open: 13, high: 15, low: 12, close: 14, volume: 130 },
      { timestamp: 5000, open: 14, high: 16, low: 13, close: 15, volume: 140 },
      { timestamp: 6000, open: 15, high: 17, low: 14, close: 16, volume: 150 },
      { timestamp: 7000, open: 16, high: 18, low: 15, close: 17, volume: 160 },
      { timestamp: 8000, open: 17, high: 19, low: 16, close: 18, volume: 170 },
      { timestamp: 9000, open: 18, high: 20, low: 17, close: 19, volume: 180 },
      { timestamp: 10000, open: 19, high: 21, low: 18, close: 20, volume: 190 },
    ]
  })

  describe('calculate', () => {
    it('should calculate SMA correctly for valid period', () => {
      const sma = new SMAIndicator({ period: 3 })
      const result = sma.calculate(candles.slice(0, 3))

      assert.ok(result)
      // SMA(3) = (11 + 12 + 13) / 3 = 36 / 3 = 12
      assert.equal(result.value, 12)
      assert.equal(result.timestamp, 3000)
    })

    it('should return null when not enough candles', () => {
      const sma = new SMAIndicator({ period: 5 })
      const result = sma.calculate(candles.slice(0, 3))

      assert.equal(result, null)
    })

    it('should calculate SMA for longer period', () => {
      const sma = new SMAIndicator({ period: 5 })
      const result = sma.calculate(candles.slice(0, 5))

      assert.ok(result)
      // SMA(5) = (11 + 12 + 13 + 14 + 15) / 5 = 65 / 5 = 13
      assert.equal(result.value, 13)
      assert.equal(result.timestamp, 5000)
    })

    it('should use cache for repeated calculations', () => {
      const sma = new SMAIndicator({ period: 3, cacheEnabled: true })

      // First calculation
      const result1 = sma.calculate(candles.slice(0, 5))
      assert.ok(result1)

      // Second calculation with same data should use cache
      const result2 = sma.calculate(candles.slice(0, 5))
      assert.ok(result2)
      assert.equal(result1.value, result2.value)
      assert.equal(result1.timestamp, result2.timestamp)
    })

    it('should throw error for invalid period', () => {
      assert.throws(() => new SMAIndicator({ period: 0 }), /SMA period must be at least 1/)
    })
  })

  describe('calculateAll', () => {
    it('should calculate SMA for all valid positions', () => {
      const sma = new SMAIndicator({ period: 3 })
      const results = sma.calculateAll(candles)

      // With 10 candles and period 3, we should get 8 results
      assert.equal(results.length, 8)

      // First SMA: (11 + 12 + 13) / 3 = 12
      assert.equal(results[0]?.value, 12)
      assert.equal(results[0]?.timestamp, 3000)

      // Last SMA: (18 + 19 + 20) / 3 = 19
      assert.equal(results[7]?.value, 19)
      assert.equal(results[7]?.timestamp, 10000)
    })

    it('should return empty array when not enough candles', () => {
      const sma = new SMAIndicator({ period: 20 })
      const results = sma.calculateAll(candles)

      assert.equal(results.length, 0)
    })
  })

  describe('static methods', () => {
    it('should calculate SMA using static method', () => {
      const result = SMAIndicator.calculate(candles.slice(0, 3), 3)

      assert.equal(result, 12)
    })

    it('should return null for invalid inputs', () => {
      const result = SMAIndicator.calculate(candles.slice(0, 2), 3)
      assert.equal(result, null)
    })

    it('should calculate optimized SMA using typed arrays', () => {
      const closes = new Float64Array(candles.map((c) => c.close))
      const results = SMAIndicator.calculateOptimized(closes, 3)

      assert.ok(results)
      assert.equal(results.length, 8)
      assert.equal(results[0], 12)
      assert.equal(results[7], 19)
    })
  })

  describe('getMinimumCandles', () => {
    it('should return the period as minimum candles', () => {
      const sma = new SMAIndicator({ period: 10 })
      assert.equal(sma.getMinimumCandles(), 10)
    })
  })

  describe('reset', () => {
    it('should clear the cache', () => {
      const sma = new SMAIndicator({ period: 3, cacheEnabled: true })

      // Calculate to populate cache
      const result1 = sma.calculate(candles.slice(0, 5))
      assert.ok(result1)

      // Reset
      sma.reset()

      // Next calculation should not be from cache (we can't directly test this
      // but we can verify it still works)
      const result2 = sma.calculate(candles.slice(0, 5))
      assert.ok(result2)
      assert.equal(result1.value, result2.value)
    })
  })
})
