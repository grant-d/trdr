import { toEpochDate, type Candle } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import { BollingerBandsIndicator } from './bollinger-bands'

describe('BollingerBandsIndicator', () => {
  let candles: Candle[]

  beforeEach(() => {
    // Create test candles with predictable values
    candles = []
    for (let i = 0; i < 30; i++) {
      // Create some volatility with a sine wave pattern
      const basePrice = 100
      const amplitude = 10
      const price = basePrice + amplitude * Math.sin(i * 0.5)

      candles.push({
        timestamp: toEpochDate((i + 1) * 1000),
        open: price - 0.5,
        high: price + 1,
        low: price - 1,
        close: price,
        volume: 1000 + i * 10,
      })
    }
  })

  describe('constructor', () => {
    it('should create with default parameters', () => {
      const bb = new BollingerBandsIndicator()

      assert.equal(bb.config.period, 20)
      assert.equal(bb.config.stdDevMultiplier, 2)
      assert.equal(bb.config.cacheEnabled, true)
    })

    it('should accept custom parameters', () => {
      const bb = new BollingerBandsIndicator({
        period: 10,
        stdDevMultiplier: 2.5,
        cacheEnabled: false,
      })

      assert.equal(bb.config.period, 10)
      assert.equal(bb.config.stdDevMultiplier, 2.5)
      assert.equal(bb.config.cacheEnabled, false)
    })

    it('should throw error for invalid period', () => {
      assert.throws(
        () => new BollingerBandsIndicator({ period: 1 }),
        /Bollinger Bands period must be at least 2/
      )
    })

    it('should throw error for invalid multiplier', () => {
      assert.throws(
        () => new BollingerBandsIndicator({ stdDevMultiplier: 0 }),
        /Standard deviation multiplier must be positive/
      )

      assert.throws(
        () => new BollingerBandsIndicator({ stdDevMultiplier: -1 }),
        /Standard deviation multiplier must be positive/
      )
    })
  })

  describe('calculate', () => {
    it('should calculate Bollinger Bands correctly', () => {
      const bb = new BollingerBandsIndicator({ period: 20 })
      const result = bb.calculate(candles)

      assert.ok(result)
      assert.equal(typeof result.value, 'number')
      assert.equal(typeof result.upper, 'number')
      assert.equal(typeof result.middle, 'number')
      assert.equal(typeof result.lower, 'number')
      assert.equal(typeof result.bandwidth, 'number')
      assert.equal(typeof result.percentB, 'number')
      assert.equal(result.timestamp, candles[candles.length - 1]?.timestamp)

      // Middle band should equal the value
      assert.equal(result.middle, result.value)

      // Upper band should be above middle, lower below
      assert.ok(result.upper > result.middle)
      assert.ok(result.lower < result.middle)

      // Bandwidth should be positive
      assert.ok(result.bandwidth > 0)

      // %B should be between -1 and 2 typically (can go outside in extreme cases)
      assert.ok(result.percentB >= -2 && result.percentB <= 3)
    })

    it('should return null when not enough candles', () => {
      const bb = new BollingerBandsIndicator({ period: 20 })
      const result = bb.calculate(candles.slice(0, 10))

      assert.equal(result, null)
    })

    it('should calculate with constant prices', () => {
      // Create candles with constant price
      const constantCandles: Candle[] = []
      for (let i = 0; i < 20; i++) {
        constantCandles.push({
          timestamp: toEpochDate((i + 1) * 1000),
          open: 100,
          high: 100,
          low: 100,
          close: 100,
          volume: 1000,
        })
      }

      const bb = new BollingerBandsIndicator({ period: 20, stdDevMultiplier: 2 })
      const result = bb.calculate(constantCandles)

      assert.ok(result)
      // With constant prices, std dev is 0, so all bands should be equal
      assert.equal(result.upper, 100)
      assert.equal(result.middle, 100)
      assert.equal(result.lower, 100)
      assert.equal(result.bandwidth, 0)
      // %B is undefined when upper = lower, but should handle gracefully
      assert.ok(!isFinite(result.percentB) || result.percentB === 0)
    })

    it('should use cache for repeated calculations', () => {
      const bb = new BollingerBandsIndicator({ period: 20, cacheEnabled: true })

      // First calculation
      const result1 = bb.calculate(candles)
      assert.ok(result1)

      // Second calculation should use cache
      const result2 = bb.calculate(candles)
      assert.ok(result2)

      assert.equal(result1.upper, result2.upper)
      assert.equal(result1.middle, result2.middle)
      assert.equal(result1.lower, result2.lower)
    })
  })

  describe('calculateAll', () => {
    it('should calculate Bollinger Bands for all valid positions', () => {
      const bb = new BollingerBandsIndicator({ period: 10 })
      const results = bb.calculateAll(candles)

      // With 30 candles and period 10, we should get 21 results
      assert.equal(results.length, 21)

      // Each result should have all BB components
      for (const result of results) {
        assert.equal(typeof result.upper, 'number')
        assert.equal(typeof result.middle, 'number')
        assert.equal(typeof result.lower, 'number')
        assert.equal(typeof result.bandwidth, 'number')
        assert.equal(typeof result.percentB, 'number')
        assert.ok(result.upper >= result.middle)
        assert.ok(result.lower <= result.middle)
      }

      // Results should be in chronological order
      for (let i = 1; i < results.length; i++) {
        assert.ok(results[i]!.timestamp > results[i - 1]!.timestamp)
      }
    })

    it('should return empty array when not enough candles', () => {
      const bb = new BollingerBandsIndicator({ period: 50 })
      const results = bb.calculateAll(candles)

      assert.equal(results.length, 0)
    })
  })

  describe('getMinimumCandles', () => {
    it('should return the period', () => {
      const bb = new BollingerBandsIndicator({ period: 25 })
      assert.equal(bb.getMinimumCandles(), 25)
    })
  })

  describe('reset', () => {
    it('should clear the cache', () => {
      const bb = new BollingerBandsIndicator({ period: 20, cacheEnabled: true })

      // Calculate to populate cache
      const result1 = bb.calculate(candles)
      assert.ok(result1)

      // Reset
      bb.reset()

      // Should still calculate correctly
      const result2 = bb.calculate(candles)
      assert.ok(result2)
      assert.equal(result1.upper, result2.upper)
    })
  })

  describe('static methods', () => {
    it('should calculate Bollinger Bands using static method', () => {
      const result = BollingerBandsIndicator.calculate(candles, 20, 2)

      assert.ok(result)
      assert.equal(typeof result.upper, 'number')
      assert.equal(typeof result.middle, 'number')
      assert.equal(typeof result.lower, 'number')
    })

    it('should return null for invalid inputs', () => {
      let result = BollingerBandsIndicator.calculate(candles.slice(0, 5), 20)
      assert.equal(result, null)

      result = BollingerBandsIndicator.calculate(candles, 1, 2)
      assert.equal(result, null)

      result = BollingerBandsIndicator.calculate(candles, 20, 0)
      assert.equal(result, null)
    })

    it('should calculate optimized BB using typed arrays', () => {
      const closes = new Float64Array(candles.map((c) => c.close))
      const results = BollingerBandsIndicator.calculateOptimized(closes, 20, 2)

      assert.ok(results)
      assert.equal(results.upper.length, 11) // 30 - 20 + 1
      assert.equal(results.middle.length, 11)
      assert.equal(results.lower.length, 11)
      assert.equal(results.bandwidth.length, 11)
      assert.equal(results.percentB.length, 11)

      // Verify relationships
      for (let i = 0; i < results.upper.length; i++) {
        assert.ok(results.upper[i]! >= results.middle[i]!)
        assert.ok(results.lower[i]! <= results.middle[i]!)
        assert.ok(results.bandwidth[i]! >= 0)
      }
    })
  })

  describe('edge cases', () => {
    it('should handle volatile price movements', () => {
      // Create candles with high volatility
      const volatileCandles: Candle[] = []
      for (let i = 0; i < 25; i++) {
        const price = i % 2 === 0 ? 100 : 150 // Oscillate between 100 and 150
        volatileCandles.push({
          timestamp: toEpochDate((i + 1) * 1000),
          open: price,
          high: price + 5,
          low: price - 5,
          close: price,
          volume: 1000,
        })
      }

      const bb = new BollingerBandsIndicator({ period: 20 })
      const result = bb.calculate(volatileCandles)

      assert.ok(result)
      // With high volatility, bands should be wide
      const bandWidth = result.upper - result.lower
      assert.ok(bandWidth > 20) // Expect wide bands with this volatility
    })

    it('should handle different multipliers', () => {
      const bb1 = new BollingerBandsIndicator({ period: 20, stdDevMultiplier: 1 })
      const bb2 = new BollingerBandsIndicator({ period: 20, stdDevMultiplier: 2 })
      const bb3 = new BollingerBandsIndicator({ period: 20, stdDevMultiplier: 3 })

      const result1 = bb1.calculate(candles)
      const result2 = bb2.calculate(candles)
      const result3 = bb3.calculate(candles)

      assert.ok(result1)
      assert.ok(result2)
      assert.ok(result3)

      // Larger multipliers should produce wider bands
      const width1 = result1.upper - result1.lower
      const width2 = result2.upper - result2.lower
      const width3 = result3.upper - result3.lower

      assert.ok(width1 < width2)
      assert.ok(width2 < width3)
    })
  })
})
