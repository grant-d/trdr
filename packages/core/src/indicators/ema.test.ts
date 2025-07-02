import { toEpochDate, type Candle } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import { EMAIndicator } from './ema'

describe('EMAIndicator', () => {
  let candles: Candle[]

  beforeEach(() => {
    // Create test candles with known values
    candles = [
      { timestamp: toEpochDate(1000), open: 10, high: 12, low: 9, close: 10, volume: 100 },
      { timestamp: toEpochDate(2000), open: 11, high: 13, low: 10, close: 12, volume: 110 },
      { timestamp: toEpochDate(3000), open: 12, high: 14, low: 11, close: 14, volume: 120 },
      { timestamp: toEpochDate(4000), open: 13, high: 15, low: 12, close: 16, volume: 130 },
      { timestamp: toEpochDate(5000), open: 14, high: 16, low: 13, close: 18, volume: 140 },
      { timestamp: toEpochDate(6000), open: 15, high: 17, low: 14, close: 16, volume: 150 },
      { timestamp: toEpochDate(7000), open: 16, high: 18, low: 15, close: 14, volume: 160 },
      { timestamp: toEpochDate(8000), open: 17, high: 19, low: 16, close: 12, volume: 170 },
      { timestamp: toEpochDate(9000), open: 18, high: 20, low: 17, close: 10, volume: 180 },
      { timestamp: toEpochDate(10000), open: 19, high: 21, low: 18, close: 8, volume: 190 },
    ]
  })

  describe('calculate', () => {
    it('should calculate EMA correctly', () => {
      const ema = new EMAIndicator({ period: 3 })
      const result = ema.calculate(candles.slice(0, 5))

      assert.ok(result)
      // First EMA uses SMA: (10 + 12 + 14) / 3 = 12
      // Multiplier = 2 / (3 + 1) = 0.5
      // EMA4 = (16 - 12) * 0.5 + 12 = 14
      // EMA5 = (18 - 14) * 0.5 + 14 = 16
      assert.equal(result.value, 16)
      assert.equal(result.timestamp, 5000)
    })

    it('should return null when not enough candles', () => {
      const ema = new EMAIndicator({ period: 5 })
      const result = ema.calculate(candles.slice(0, 3))

      assert.equal(result, null)
    })

    it('should handle period of 1', () => {
      const ema = new EMAIndicator({ period: 1 })
      const result = ema.calculate(candles.slice(0, 1))

      assert.ok(result)
      // With period 1, EMA equals the close price
      assert.equal(result.value, 10)
    })

    it('should use cache for repeated calculations', () => {
      const ema = new EMAIndicator({ period: 3, cacheEnabled: true })

      // First calculation
      const result1 = ema.calculate(candles.slice(0, 5))
      assert.ok(result1)

      // Second calculation with same data should use cache
      const result2 = ema.calculate(candles.slice(0, 5))
      assert.ok(result2)
      assert.equal(result1.value, result2.value)
      assert.equal(result1.timestamp, result2.timestamp)
    })

    it('should throw error for invalid period', () => {
      assert.throws(() => new EMAIndicator({ period: 0 }), /EMA period must be at least 1/)
    })
  })

  describe('calculateAll', () => {
    it('should calculate EMA for all valid positions', () => {
      const ema = new EMAIndicator({ period: 3 })
      const results = ema.calculateAll(candles.slice(0, 6))

      // With 6 candles and period 3, we should get 4 results
      assert.equal(results.length, 4)

      // First EMA uses SMA: (10 + 12 + 14) / 3 = 12
      assert.equal(results[0]?.value, 12)
      assert.equal(results[0]?.timestamp, 3000)

      // Multiplier = 2 / (3 + 1) = 0.5
      // EMA4 = (16 - 12) * 0.5 + 12 = 14
      assert.equal(results[1]?.value, 14)
      assert.equal(results[1]?.timestamp, 4000)

      // EMA5 = (18 - 14) * 0.5 + 14 = 16
      assert.equal(results[2]?.value, 16)
      assert.equal(results[2]?.timestamp, 5000)

      // EMA6 = (16 - 16) * 0.5 + 16 = 16
      assert.equal(results[3]?.value, 16)
      assert.equal(results[3]?.timestamp, 6000)
    })

    it('should return empty array when not enough candles', () => {
      const ema = new EMAIndicator({ period: 20 })
      const results = ema.calculateAll(candles)

      assert.equal(results.length, 0)
    })
  })

  describe('static methods', () => {
    it('should calculate EMA using static method', () => {
      const result = EMAIndicator.calculate(candles.slice(0, 5), 3)

      assert.equal(result, 16)
    })

    it('should return null for invalid inputs', () => {
      const result = EMAIndicator.calculate(candles.slice(0, 2), 3)
      assert.equal(result, null)
    })

    it('should calculate optimized EMA using typed arrays', () => {
      const closes = new Float64Array(candles.slice(0, 6).map((c) => c.close))
      const results = EMAIndicator.calculateOptimized(closes, 3)

      assert.ok(results)
      assert.equal(results.length, 4)
      assert.equal(results[0], 12) // SMA
      assert.equal(results[1], 14)
      assert.equal(results[2], 16)
      assert.equal(results[3], 16)
    })
  })

  describe('multiplier calculation', () => {
    it('should calculate correct multiplier', () => {
      // Period 3: 2 / (3 + 1) = 0.5
      const ema3 = new EMAIndicator({ period: 3 })
      assert.equal(ema3['multiplier'], 0.5)

      // Period 9: 2 / (9 + 1) = 0.2
      const ema9 = new EMAIndicator({ period: 9 })
      assert.equal(ema9['multiplier'], 0.2)

      // Period 19: 2 / (19 + 1) = 0.1
      const ema19 = new EMAIndicator({ period: 19 })
      assert.equal(ema19['multiplier'], 0.1)
    })
  })

  describe('getMinimumCandles', () => {
    it('should return the period as minimum candles', () => {
      const ema = new EMAIndicator({ period: 10 })
      assert.equal(ema.getMinimumCandles(), 10)
    })
  })

  describe('reset', () => {
    it('should clear the cache', () => {
      const ema = new EMAIndicator({ period: 3, cacheEnabled: true })

      // Calculate to populate cache
      const result1 = ema.calculate(candles.slice(0, 5))
      assert.ok(result1)

      // Reset
      ema.reset()

      // Next calculation should work correctly
      const result2 = ema.calculate(candles.slice(0, 5))
      assert.ok(result2)
      assert.equal(result1.value, result2.value)
    })
  })
})
