import { toEpochDate, type Candle } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import { ATRIndicator } from './atr'

describe('ATRIndicator', () => {
  let candles: Candle[]

  beforeEach(() => {
    // Create test candles with known values
    candles = [
      { timestamp: toEpochDate(1000), open: 10, high: 12, low: 9, close: 11, volume: 100 },
      { timestamp: toEpochDate(2000), open: 11, high: 14, low: 10, close: 13, volume: 110 },
      { timestamp: toEpochDate(3000), open: 13, high: 15, low: 11, close: 12, volume: 120 },
      { timestamp: toEpochDate(4000), open: 12, high: 16, low: 11, close: 15, volume: 130 },
      { timestamp: toEpochDate(5000), open: 15, high: 18, low: 14, close: 17, volume: 140 },
      { timestamp: toEpochDate(6000), open: 17, high: 19, low: 15, close: 16, volume: 150 },
      { timestamp: toEpochDate(7000), open: 16, high: 17, low: 14, close: 14, volume: 160 },
      { timestamp: toEpochDate(8000), open: 14, high: 15, low: 12, close: 13, volume: 170 },
      { timestamp: toEpochDate(9000), open: 13, high: 14, low: 11, close: 12, volume: 180 },
      { timestamp: toEpochDate(10000), open: 12, high: 13, low: 10, close: 11, volume: 190 },
      { timestamp: toEpochDate(11000), open: 11, high: 12, low: 9, close: 10, volume: 200 },
      { timestamp: toEpochDate(12000), open: 10, high: 11, low: 8, close: 9, volume: 210 },
      { timestamp: toEpochDate(13000), open: 9, high: 10, low: 7, close: 8, volume: 220 },
      { timestamp: toEpochDate(14000), open: 8, high: 9, low: 6, close: 7, volume: 230 },
      { timestamp: toEpochDate(15000), open: 7, high: 8, low: 5, close: 6, volume: 240 },
      { timestamp: toEpochDate(16000), open: 6, high: 7, low: 4, close: 5, volume: 250 },
    ]
  })

  describe('constructor', () => {
    it('should create with default parameters', () => {
      const atr = new ATRIndicator()

      assert.equal(atr.config.period, 14)
      assert.equal(atr.config.smoothing, 'wilder')
      assert.equal(atr.config.cacheEnabled, true)
    })

    it('should accept custom parameters', () => {
      const atr = new ATRIndicator({
        period: 10,
        smoothing: 'sma',
        cacheEnabled: false,
      })

      assert.equal(atr.config.period, 10)
      assert.equal(atr.config.smoothing, 'sma')
      assert.equal(atr.config.cacheEnabled, false)
    })

    it('should throw error for invalid period', () => {
      assert.throws(() => new ATRIndicator({ period: 0 }), /ATR period must be at least 1/)
    })
  })

  describe('calculate', () => {
    it('should calculate ATR correctly with Wilder smoothing', () => {
      const atr = new ATRIndicator({ period: 14 })
      const result = atr.calculate(candles)

      assert.ok(result)
      assert.equal(typeof result.value, 'number')
      assert.ok(result.value > 0)
      assert.equal(result.timestamp, candles[candles.length - 1]?.timestamp)
    })

    it('should calculate ATR correctly with SMA smoothing', () => {
      const atr = new ATRIndicator({ period: 14, smoothing: 'sma' })
      const result = atr.calculate(candles)

      assert.ok(result)
      assert.equal(typeof result.value, 'number')
      assert.ok(result.value > 0)
    })

    it('should return null when not enough candles', () => {
      const atr = new ATRIndicator({ period: 14 })
      const result = atr.calculate(candles.slice(0, 10))

      assert.equal(result, null)
    })

    it('should calculate true range correctly', () => {
      // Test with simple candles where we can verify TR calculation
      const testCandles: Candle[] = [
        { timestamp: toEpochDate(1000), open: 10, high: 15, low: 8, close: 12, volume: 100 },
        { timestamp: toEpochDate(2000), open: 12, high: 16, low: 10, close: 14, volume: 100 },
      ]

      const atr = new ATRIndicator({ period: 1 })
      const result = atr.calculate(testCandles)

      assert.ok(result)
      // TR = max(H-L, |H-PC|, |L-PC|) = max(6, 4, 2) = 6
      assert.equal(result.value, 6)
    })

    it('should use cache for repeated calculations', () => {
      const atr = new ATRIndicator({ period: 14, cacheEnabled: true })

      // First calculation
      const result1 = atr.calculate(candles)
      assert.ok(result1)

      // Second calculation should use cache
      const result2 = atr.calculate(candles)
      assert.ok(result2)

      assert.equal(result1.value, result2.value)
    })
  })

  describe('calculateAll', () => {
    it('should calculate ATR for all valid positions', () => {
      const atr = new ATRIndicator({ period: 5 })
      const results = atr.calculateAll(candles.slice(0, 10))

      // With 10 candles and period 5, we need 6 candles minimum
      // So we should get 10 - 5 = 5 results
      assert.equal(results.length, 5)

      // Each result should have valid ATR value
      for (const result of results) {
        assert.equal(typeof result.value, 'number')
        assert.ok(result.value > 0)
      }

      // Results should be in chronological order
      for (let i = 1; i < results.length; i++) {
        assert.ok(results[i]!.timestamp > results[i - 1]!.timestamp)
      }
    })

    it('should return empty array when not enough candles', () => {
      const atr = new ATRIndicator({ period: 20 })
      const results = atr.calculateAll(candles.slice(0, 10))

      assert.equal(results.length, 0)
    })
  })

  describe('getMinimumCandles', () => {
    it('should return period + 1', () => {
      const atr = new ATRIndicator({ period: 14 })
      assert.equal(atr.getMinimumCandles(), 15)
    })
  })

  describe('reset', () => {
    it('should clear the cache', () => {
      const atr = new ATRIndicator({ period: 14, cacheEnabled: true })

      // Calculate to populate cache
      const result1 = atr.calculate(candles)
      assert.ok(result1)

      // Reset
      atr.reset()

      // Should still calculate correctly
      const result2 = atr.calculate(candles)
      assert.ok(result2)
      assert.equal(result1.value, result2.value)
    })
  })

  describe('static methods', () => {
    it('should calculate ATR using static method', () => {
      const result = ATRIndicator.calculate(candles, 14, 'wilder')

      assert.ok(result !== null)
      assert.equal(typeof result, 'number')
      assert.ok(result > 0)
    })

    it('should return null for invalid inputs', () => {
      const result = ATRIndicator.calculate(candles.slice(0, 5), 14)
      assert.equal(result, null)
    })

    it('should calculate optimized ATR using typed arrays', () => {
      const highs = new Float64Array(candles.map((c) => c.high))
      const lows = new Float64Array(candles.map((c) => c.low))
      const closes = new Float64Array(candles.map((c) => c.close))

      const results = ATRIndicator.calculateOptimized(highs, lows, closes, 14, 'wilder')

      assert.ok(results)
      assert.equal(results.length, 2) // 16 candles - 14 period = 2

      for (let i = 0; i < results.length; i++) {
        assert.equal(typeof results[i], 'number')
        assert.ok(results[i]! > 0)
      }
    })
  })

  describe('smoothing methods', () => {
    it('should produce different results for different smoothing methods', () => {
      const wilderATR = new ATRIndicator({ period: 14, smoothing: 'wilder' })
      const smaATR = new ATRIndicator({ period: 14, smoothing: 'sma' })

      const wilderResult = wilderATR.calculate(candles)
      const smaResult = smaATR.calculate(candles)

      assert.ok(wilderResult)
      assert.ok(smaResult)

      // Results should be different (though potentially close)
      assert.notEqual(wilderResult.value, smaResult.value)
    })
  })
})
