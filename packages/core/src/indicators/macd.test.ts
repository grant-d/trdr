import type { Candle } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import { MACDIndicator } from './macd'

describe('MACDIndicator', () => {
  let candles: Candle[]

  beforeEach(() => {
    // Create test candles - need at least 35 for default MACD (26 + 9)
    const prices = [
      50, 51, 52, 51, 50, 49, 48, 49, 50, 51, 52, 53, 54, 55, 54, 53, 52, 51, 50, 49, 48, 47, 48,
      49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 59, 58, 57, 56, 55,
    ]

    candles = prices.map((close, i) => ({
      timestamp: (i + 1) * 1000,
      open: close - 0.5,
      high: close + 1,
      low: close - 1,
      close,
      volume: 1000 + i * 10,
    }))
  })

  describe('constructor', () => {
    it('should create with default parameters', () => {
      const macd = new MACDIndicator()

      assert.equal(macd.config.fastPeriod, 12)
      assert.equal(macd.config.slowPeriod, 26)
      assert.equal(macd.config.signalPeriod, 9)
    })

    it('should accept custom parameters', () => {
      const macd = new MACDIndicator({
        fastPeriod: 10,
        slowPeriod: 20,
        signalPeriod: 5,
      })

      assert.equal(macd.config.fastPeriod, 10)
      assert.equal(macd.config.slowPeriod, 20)
      assert.equal(macd.config.signalPeriod, 5)
    })

    it('should throw error if fast period >= slow period', () => {
      assert.throws(
        () => new MACDIndicator({ fastPeriod: 26, slowPeriod: 12 }),
        /Fast period must be less than slow period/
      )
    })
  })

  describe('calculate', () => {
    it('should calculate MACD values correctly', () => {
      const macd = new MACDIndicator()
      const result = macd.calculate(candles)

      assert.ok(result)
      assert.equal(typeof result.macd, 'number')
      assert.equal(typeof result.signal, 'number')
      assert.equal(typeof result.histogram, 'number')
      assert.equal(result.value, result.macd) // value should equal macd
      assert.equal(result.timestamp, candles[candles.length - 1]?.timestamp)

      // Histogram should be MACD - Signal
      assert.ok(Math.abs(result.histogram - (result.macd - result.signal)) < 0.0001)
    })

    it('should return null when not enough candles', () => {
      const macd = new MACDIndicator()
      const result = macd.calculate(candles.slice(0, 30))

      assert.equal(result, null)
    })

    it('should use cache for repeated calculations', () => {
      const macd = new MACDIndicator({ cacheEnabled: true })

      // First calculation
      const result1 = macd.calculate(candles)
      assert.ok(result1)

      // Second calculation should use cache
      const result2 = macd.calculate(candles)
      assert.ok(result2)

      assert.equal(result1.macd, result2.macd)
      assert.equal(result1.signal, result2.signal)
      assert.equal(result1.histogram, result2.histogram)
    })
  })

  describe('calculateAll', () => {
    it('should calculate MACD for all valid positions', () => {
      const macd = new MACDIndicator()
      const results = macd.calculateAll(candles)

      assert.ok(results.length > 0)

      // Each result should have all MACD components
      for (const result of results) {
        assert.equal(typeof result.macd, 'number')
        assert.equal(typeof result.signal, 'number')
        assert.equal(typeof result.histogram, 'number')
        assert.equal(result.value, result.macd)
        assert.ok(Math.abs(result.histogram - (result.macd - result.signal)) < 0.0001)
      }

      // Results should be in chronological order
      for (let i = 1; i < results.length; i++) {
        assert.ok(results[i]!.timestamp > results[i - 1]!.timestamp)
      }
    })

    it('should return empty array when not enough candles', () => {
      const macd = new MACDIndicator()
      const results = macd.calculateAll(candles.slice(0, 30))

      assert.equal(results.length, 0)
    })
  })

  describe('getMinimumCandles', () => {
    it('should return correct minimum for default parameters', () => {
      const macd = new MACDIndicator()
      // 26 (slow) + 9 (signal) - 1 = 34
      assert.equal(macd.getMinimumCandles(), 34)
    })

    it('should return correct minimum for custom parameters', () => {
      const macd = new MACDIndicator({
        fastPeriod: 10,
        slowPeriod: 20,
        signalPeriod: 5,
      })
      // 20 (slow) + 5 (signal) - 1 = 24
      assert.equal(macd.getMinimumCandles(), 24)
    })
  })

  describe('reset', () => {
    it('should clear internal caches', () => {
      const macd = new MACDIndicator({ cacheEnabled: true })

      // Calculate to populate caches
      const result1 = macd.calculate(candles)
      assert.ok(result1)

      // Reset
      macd.reset()

      // Should still calculate correctly
      const result2 = macd.calculate(candles)
      assert.ok(result2)
      assert.equal(result1.macd, result2.macd)
    })
  })

  describe('static calculate', () => {
    it('should calculate MACD using static method', () => {
      const result = MACDIndicator.calculate(candles)

      assert.ok(result)
      assert.equal(typeof result.macd, 'number')
      assert.equal(typeof result.signal, 'number')
      assert.equal(typeof result.histogram, 'number')
    })

    it('should accept custom config', () => {
      const result = MACDIndicator.calculate(candles, {
        fastPeriod: 10,
        slowPeriod: 20,
        signalPeriod: 5,
      })

      assert.ok(result)
    })
  })

  describe('trending market behavior', () => {
    it('should show positive MACD in uptrend', () => {
      // Create uptrending candles
      const uptrendCandles: Candle[] = []
      for (let i = 0; i < 50; i++) {
        uptrendCandles.push({
          timestamp: i * 1000,
          open: 100 + i * 0.8,
          high: 100 + i * 0.8 + 1,
          low: 100 + i * 0.8 - 0.5,
          close: 100 + i * 0.8,
          volume: 1000,
        })
      }

      const macd = new MACDIndicator()
      const result = macd.calculate(uptrendCandles)

      assert.ok(result)
      // In a strong uptrend, MACD should be positive
      assert.ok(result.macd > 0)
    })

    it('should show negative MACD in downtrend', () => {
      // Create downtrending candles
      const downtrendCandles: Candle[] = []
      for (let i = 0; i < 50; i++) {
        downtrendCandles.push({
          timestamp: i * 1000,
          open: 100 - i * 0.8,
          high: 100 - i * 0.8 + 0.5,
          low: 100 - i * 0.8 - 1,
          close: 100 - i * 0.8,
          volume: 1000,
        })
      }

      const macd = new MACDIndicator()
      const result = macd.calculate(downtrendCandles)

      assert.ok(result)
      // In a strong downtrend, MACD should be negative
      assert.ok(result.macd < 0)
    })
  })
})
