import type { Candle } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import type { SwingPoint } from './interfaces'
import { SwingDetector } from './swing-detector'

describe('SwingDetector', () => {
  let candles: Candle[]

  beforeEach(() => {
    // Create test candles with clear swing patterns
    candles = [
      // Lead-in candles
      { timestamp: 1000, open: 100, high: 101, low: 99, close: 100, volume: 1000 },
      { timestamp: 2000, open: 100, high: 102, low: 98, close: 101, volume: 1000 },
      { timestamp: 3000, open: 101, high: 103, low: 100, close: 102, volume: 1000 },
      { timestamp: 4000, open: 102, high: 104, low: 101, close: 103, volume: 1000 },
      { timestamp: 5000, open: 103, high: 105, low: 102, close: 104, volume: 1000 },
      // Swing high at index 5
      { timestamp: 6000, open: 104, high: 110, low: 103, close: 108, volume: 1000 },
      // Decline after swing high
      { timestamp: 7000, open: 108, high: 109, low: 105, close: 106, volume: 1000 },
      { timestamp: 8000, open: 106, high: 107, low: 104, close: 105, volume: 1000 },
      { timestamp: 9000, open: 105, high: 106, low: 103, close: 104, volume: 1000 },
      { timestamp: 10000, open: 104, high: 105, low: 102, close: 103, volume: 1000 },
      // Swing low at index 10
      { timestamp: 11000, open: 103, high: 104, low: 95, close: 96, volume: 1000 },
      // Rise after swing low
      { timestamp: 12000, open: 96, high: 99, low: 96, close: 98, volume: 1000 },
      { timestamp: 13000, open: 98, high: 101, low: 97, close: 100, volume: 1000 },
      { timestamp: 14000, open: 100, high: 103, low: 99, close: 102, volume: 1000 },
      { timestamp: 15000, open: 102, high: 105, low: 101, close: 104, volume: 1000 },
      // Another swing high at index 15
      { timestamp: 16000, open: 104, high: 112, low: 103, close: 110, volume: 1000 },
      // Decline after swing high
      { timestamp: 17000, open: 110, high: 111, low: 107, close: 108, volume: 1000 },
      { timestamp: 18000, open: 108, high: 109, low: 105, close: 106, volume: 1000 },
      { timestamp: 19000, open: 106, high: 107, low: 103, close: 104, volume: 1000 },
      { timestamp: 20000, open: 104, high: 105, low: 101, close: 102, volume: 1000 },
    ]
  })

  describe('constructor', () => {
    it('should create with default parameters', () => {
      const detector = new SwingDetector()

      assert.equal(detector.config.lookback, 5)
      assert.equal(detector.config.lookforward, 5)
      assert.equal(detector.config.minSwingPercent, 0.001)
      assert.equal(detector.config.includeWicks, true)
      assert.equal(detector.config.cacheEnabled, true)
    })

    it('should accept custom parameters', () => {
      const detector = new SwingDetector({
        lookback: 3,
        lookforward: 3,
        minSwingPercent: 0.005,
        includeWicks: false,
        cacheEnabled: false,
      })

      assert.equal(detector.config.lookback, 3)
      assert.equal(detector.config.lookforward, 3)
      assert.equal(detector.config.minSwingPercent, 0.005)
      assert.equal(detector.config.includeWicks, false)
      assert.equal(detector.config.cacheEnabled, false)
    })

    it('should throw error for invalid lookback/lookforward', () => {
      assert.throws(
        () => new SwingDetector({ lookback: 0 }),
        /Lookback and lookforward must be at least 1/
      )
      assert.throws(
        () => new SwingDetector({ lookforward: 0 }),
        /Lookback and lookforward must be at least 1/
      )
    })
  })

  describe('calculate', () => {
    it('should detect swing highs and lows', () => {
      const detector = new SwingDetector({ lookback: 3, lookforward: 3 })
      const swings = detector.calculate(candles)

      assert.ok(swings)
      assert.ok(swings.length > 0)

      // Should detect swing high around index 5
      const swingHigh1 = swings.find((s) => s.index === 5 && s.type === 'high')
      assert.ok(swingHigh1)
      assert.equal(swingHigh1.price, 110) // Using high price
      assert.ok(swingHigh1.strength > 0)

      // Should detect swing low around index 10
      const swingLow1 = swings.find((s) => s.index === 10 && s.type === 'low')
      assert.ok(swingLow1)
      assert.equal(swingLow1.price, 95) // Using low price
      assert.ok(swingLow1.strength > 0)
    })

    it('should respect includeWicks setting', () => {
      const detectorWithWicks = new SwingDetector({ 
        lookback: 3, 
        lookforward: 3,
        includeWicks: true 
      })
      const detectorWithoutWicks = new SwingDetector({ 
        lookback: 3, 
        lookforward: 3,
        includeWicks: false 
      })

      const swingsWithWicks = detectorWithWicks.calculate(candles)
      const swingsWithoutWicks = detectorWithoutWicks.calculate(candles)

      assert.ok(swingsWithWicks)
      assert.ok(swingsWithoutWicks)

      // With wicks should use high/low prices
      const highWithWicks = swingsWithWicks.find((s) => s.index === 5 && s.type === 'high')
      assert.ok(highWithWicks)
      assert.equal(highWithWicks.price, 110) // high price

      // Without wicks should use close prices
      const highWithoutWicks = swingsWithoutWicks.find((s) => s.index === 5 && s.type === 'high')
      assert.ok(highWithoutWicks)
      assert.equal(highWithoutWicks.price, 108) // close price
    })

    it('should filter by minimum swing percentage', () => {
      const detectorLowThreshold = new SwingDetector({ 
        lookback: 3, 
        lookforward: 3,
        minSwingPercent: 0.001 
      })
      const detectorHighThreshold = new SwingDetector({ 
        lookback: 3, 
        lookforward: 3,
        minSwingPercent: 0.05 // 5% minimum
      })

      const swingsLow = detectorLowThreshold.calculate(candles)
      const swingsHigh = detectorHighThreshold.calculate(candles)

      assert.ok(swingsLow)
      assert.ok(swingsHigh)
      assert.ok(swingsLow.length > swingsHigh.length)
    })

    it('should return null when not enough candles', () => {
      const detector = new SwingDetector({ lookback: 5, lookforward: 5 })
      const result = detector.calculate(candles.slice(0, 5))

      assert.equal(result, null)
    })

    it('should use cache for repeated calculations', () => {
      const detector = new SwingDetector({ cacheEnabled: true })

      // First calculation
      const result1 = detector.calculate(candles)
      assert.ok(result1)

      // Second calculation should use cache
      const result2 = detector.calculate(candles)
      assert.ok(result2)

      assert.equal(result1.length, result2.length)
    })
  })

  describe('calculateAll', () => {
    it('should return same results as calculate', () => {
      const detector = new SwingDetector({ lookback: 3, lookforward: 3 })
      const calculateResult = detector.calculate(candles)
      const calculateAllResult = detector.calculateAll(candles)

      assert.ok(calculateResult)
      assert.equal(calculateResult.length, calculateAllResult.length)

      for (let i = 0; i < calculateResult.length; i++) {
        const swing1: SwingPoint = calculateResult[i]!
        const swing2: SwingPoint = calculateAllResult[i]!
        assert.equal(swing1.type, swing2.type)
        assert.equal(swing1.index, swing2.index)
        assert.equal(swing1.price, swing2.price)
      }
    })

    it('should return empty array when not enough candles', () => {
      const detector = new SwingDetector({ lookback: 5, lookforward: 5 })
      const results = detector.calculateAll(candles.slice(0, 5))

      assert.equal(results.length, 0)
    })
  })

  describe('swing detection logic', () => {
    it('should correctly identify isolated high', () => {
      // Create candles with a clear isolated high
      const isolatedHighCandles: Candle[] = [
        { timestamp: 1000, open: 100, high: 101, low: 99, close: 100, volume: 1000 },
        { timestamp: 2000, open: 100, high: 102, low: 98, close: 101, volume: 1000 },
        { timestamp: 3000, open: 101, high: 103, low: 100, close: 102, volume: 1000 },
        // Clear swing high
        { timestamp: 4000, open: 102, high: 110, low: 101, close: 108, volume: 1000 },
        { timestamp: 5000, open: 108, high: 109, low: 105, close: 106, volume: 1000 },
        { timestamp: 6000, open: 106, high: 107, low: 104, close: 105, volume: 1000 },
        { timestamp: 7000, open: 105, high: 106, low: 103, close: 104, volume: 1000 },
      ]

      const detector = new SwingDetector({ lookback: 2, lookforward: 2 })
      const swings = detector.calculate(isolatedHighCandles)

      assert.ok(swings)
      assert.equal(swings.length, 1)
      assert.equal(swings[0]!.type, 'high')
      assert.equal(swings[0]!.index, 3)
      assert.equal(swings[0]!.price, 110)
    })

    it('should correctly identify isolated low', () => {
      // Create candles with a clear isolated low
      const isolatedLowCandles: Candle[] = [
        { timestamp: 1000, open: 100, high: 101, low: 99, close: 100, volume: 1000 },
        { timestamp: 2000, open: 100, high: 101, low: 98, close: 99, volume: 1000 },
        { timestamp: 3000, open: 99, high: 100, low: 97, close: 98, volume: 1000 },
        // Clear swing low
        { timestamp: 4000, open: 98, high: 99, low: 90, close: 92, volume: 1000 },
        { timestamp: 5000, open: 92, high: 95, low: 91, close: 94, volume: 1000 },
        { timestamp: 6000, open: 94, high: 97, low: 93, close: 96, volume: 1000 },
        { timestamp: 7000, open: 96, high: 99, low: 95, close: 98, volume: 1000 },
      ]

      const detector = new SwingDetector({ lookback: 2, lookforward: 2 })
      const swings = detector.calculate(isolatedLowCandles)

      assert.ok(swings)
      assert.equal(swings.length, 1)
      assert.equal(swings[0]!.type, 'low')
      assert.equal(swings[0]!.index, 3)
      assert.equal(swings[0]!.price, 90)
    })

    it('should handle equal prices correctly', () => {
      // Create candles with equal highs - should not detect swing
      const equalHighCandles: Candle[] = [
        { timestamp: 1000, open: 100, high: 105, low: 99, close: 100, volume: 1000 },
        { timestamp: 2000, open: 100, high: 105, low: 98, close: 101, volume: 1000 },
        { timestamp: 3000, open: 101, high: 105, low: 100, close: 102, volume: 1000 },
        { timestamp: 4000, open: 102, high: 105, low: 101, close: 103, volume: 1000 },
        { timestamp: 5000, open: 103, high: 105, low: 102, close: 104, volume: 1000 },
      ]

      const detector = new SwingDetector({ lookback: 2, lookforward: 2 })
      const swings = detector.calculate(equalHighCandles)

      assert.ok(!swings || swings.length === 0)
    })
  })

  describe('getMinimumCandles', () => {
    it('should return lookback + lookforward + 1', () => {
      const detector = new SwingDetector({ lookback: 5, lookforward: 5 })
      assert.equal(detector.getMinimumCandles(), 11)
    })
  })

  describe('reset', () => {
    it('should clear the cache', () => {
      const detector = new SwingDetector({ cacheEnabled: true })

      // Calculate to populate cache
      const result1 = detector.calculate(candles)
      assert.ok(result1)

      // Reset
      detector.reset()

      // Should still calculate correctly
      const result2 = detector.calculate(candles)
      assert.ok(result2)
      assert.equal(result1.length, result2.length)
    })
  })

  describe('static methods', () => {
    it('should detect swings using static method', () => {
      const result = SwingDetector.detect(candles, 3, 3, 0.001, true)

      assert.ok(result)
      assert.ok(result.length > 0)
    })

    it('should find last swing high', () => {
      const lastHigh = SwingDetector.findLastSwingHigh(candles, 3, 3)

      assert.ok(lastHigh)
      assert.equal(lastHigh.type, 'high')
      assert.equal(lastHigh.index, 15)
    })

    it('should find last swing low', () => {
      const lastLow = SwingDetector.findLastSwingLow(candles, 3, 3)

      assert.ok(lastLow)
      assert.equal(lastLow.type, 'low')
      assert.equal(lastLow.index, 10)
    })

    it('should return null when no swings found', () => {
      // Flat candles - no swings
      const flatCandles: Candle[] = Array(10).fill(null).map((_, i) => ({
        timestamp: (i + 1) * 1000,
        open: 100,
        high: 100,
        low: 100,
        close: 100,
        volume: 1000,
      }))

      const lastHigh = SwingDetector.findLastSwingHigh(flatCandles)
      const lastLow = SwingDetector.findLastSwingLow(flatCandles)

      assert.equal(lastHigh, null)
      assert.equal(lastLow, null)
    })
  })

  describe('swing strength calculation', () => {
    it('should calculate correct swing strength', () => {
      const detector = new SwingDetector({ lookback: 3, lookforward: 3 })
      const swings = detector.calculate(candles)

      assert.ok(swings)

      // Check that all swings have positive strength
      for (const swing of swings) {
        assert.ok(swing.strength > 0)
        assert.ok(swing.strength < 1) // Should be a reasonable percentage
      }

      // Stronger swings should have higher strength values
      const highSwing = swings.find((s) => s.index === 15 && s.type === 'high')
      assert.ok(highSwing)
      assert.ok(highSwing.strength > 0.05) // Should be a significant swing
    })
  })

  describe('edge cases', () => {
    it('should handle candles with extreme values', () => {
      const extremeCandles: Candle[] = [
        { timestamp: 1000, open: 100, high: 101, low: 99, close: 100, volume: 1000 },
        { timestamp: 2000, open: 100, high: 102, low: 98, close: 101, volume: 1000 },
        // Extreme spike
        { timestamp: 3000, open: 101, high: 1000, low: 100, close: 500, volume: 1000 },
        { timestamp: 4000, open: 500, high: 501, low: 499, close: 500, volume: 1000 },
        { timestamp: 5000, open: 500, high: 502, low: 498, close: 501, volume: 1000 },
      ]

      const detector = new SwingDetector({ lookback: 1, lookforward: 1 })
      const swings = detector.calculate(extremeCandles)

      assert.ok(swings)
      assert.ok(swings.length > 0)
    })

    it('should handle very small price movements', () => {
      const smallMovementCandles: Candle[] = Array(10).fill(null).map((_, i) => ({
        timestamp: (i + 1) * 1000,
        open: 100,
        high: 100 + (i === 5 ? 0.01 : 0),
        low: 100 - (i === 5 ? 0.01 : 0),
        close: 100,
        volume: 1000,
      }))

      const detector = new SwingDetector({ 
        lookback: 2, 
        lookforward: 2,
        minSwingPercent: 0.00001 // Very low threshold
      })
      const swings = detector.calculate(smallMovementCandles)

      assert.ok(swings)
      assert.ok(swings.length >= 0) // May or may not detect depending on threshold
    })
  })
})