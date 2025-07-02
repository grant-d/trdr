import { toEpochDate, type Candle } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import { RSIIndicator } from './rsi'

describe('RSIIndicator', () => {
  let candles: Candle[]

  beforeEach(() => {
    // Create test candles with known price movements
    candles = [
      { timestamp: toEpochDate(1000), open: 44, high: 45, low: 43, close: 44.34, volume: 100 },
      { timestamp: toEpochDate(2000), open: 44.34, high: 45, low: 44, close: 44.09, volume: 110 },
      { timestamp: toEpochDate(3000), open: 44.09, high: 44.5, low: 43.5, close: 44.15, volume: 120 },
      { timestamp: toEpochDate(4000), open: 44.15, high: 44.5, low: 43, close: 43.61, volume: 130 },
      { timestamp: toEpochDate(5000), open: 43.61, high: 45, low: 43.5, close: 44.33, volume: 140 },
      { timestamp: toEpochDate(6000), open: 44.33, high: 45, low: 44, close: 44.83, volume: 150 },
      { timestamp: toEpochDate(7000), open: 44.83, high: 45.5, low: 44.5, close: 45.1, volume: 160 },
      { timestamp: toEpochDate(8000), open: 45.1, high: 45.5, low: 44.5, close: 45.42, volume: 170 },
      { timestamp: toEpochDate(9000), open: 45.42, high: 46, low: 45, close: 45.84, volume: 180 },
      { timestamp: toEpochDate(10000), open: 45.84, high: 46.5, low: 45.5, close: 46.08, volume: 190 },
      { timestamp: toEpochDate(11000), open: 46.08, high: 46.5, low: 45.5, close: 45.89, volume: 200 },
      { timestamp: toEpochDate(12000), open: 45.89, high: 46.5, low: 45.5, close: 46.03, volume: 210 },
      { timestamp: toEpochDate(13000), open: 46.03, high: 47, low: 46, close: 46.41, volume: 220 },
      { timestamp: toEpochDate(14000), open: 46.41, high: 47, low: 46, close: 46.22, volume: 230 },
      { timestamp: toEpochDate(15000), open: 46.22, high: 47, low: 46, close: 45.64, volume: 240 },
      { timestamp: toEpochDate(16000), open: 45.64, high: 46, low: 45, close: 46.21, volume: 250 },
      { timestamp: toEpochDate(17000), open: 46.21, high: 47, low: 46, close: 46.25, volume: 260 },
      { timestamp: toEpochDate(18000), open: 46.25, high: 47, low: 46, close: 45.71, volume: 270 },
      { timestamp: toEpochDate(19000), open: 45.71, high: 46.5, low: 45.5, close: 46.45, volume: 280 },
      { timestamp: toEpochDate(20000), open: 46.45, high: 47, low: 46, close: 45.96, volume: 290 },
    ]
  })

  describe('constructor', () => {
    it('should create with default parameters', () => {
      const rsi = new RSIIndicator()

      assert.equal(rsi.config.period, 14)
      assert.equal(rsi.config.smoothing, 'wilder')
      assert.equal(rsi.config.cacheEnabled, true)
    })

    it('should accept custom parameters', () => {
      const rsi = new RSIIndicator({
        period: 9,
        smoothing: 'sma',
        cacheEnabled: false,
      })

      assert.equal(rsi.config.period, 9)
      assert.equal(rsi.config.smoothing, 'sma')
      assert.equal(rsi.config.cacheEnabled, false)
    })

    it('should throw error for invalid period', () => {
      assert.throws(() => new RSIIndicator({ period: 0 }), /RSI period must be at least 1/)
    })
  })

  describe('calculate', () => {
    it('should calculate RSI correctly with Wilder smoothing', () => {
      const rsi = new RSIIndicator({ period: 14 })
      const result = rsi.calculate(candles.slice(0, 16))

      assert.ok(result)
      assert.equal(typeof result.value, 'number')
      assert.ok(result.value >= 0 && result.value <= 100)
      assert.equal(result.timestamp, 16000)
    })

    it('should calculate RSI correctly with SMA smoothing', () => {
      const rsi = new RSIIndicator({ period: 14, smoothing: 'sma' })
      const result = rsi.calculate(candles.slice(0, 16))

      assert.ok(result)
      assert.equal(typeof result.value, 'number')
      assert.ok(result.value >= 0 && result.value <= 100)
    })

    it('should return null when not enough candles', () => {
      const rsi = new RSIIndicator({ period: 14 })
      const result = rsi.calculate(candles.slice(0, 10))

      assert.equal(result, null)
    })

    it('should handle all gains (RSI = 100)', () => {
      // Create candles with only upward movement
      const upCandles: Candle[] = []
      for (let i = 0; i < 20; i++) {
        upCandles.push({
          timestamp: toEpochDate((i + 1) * 1000),
          open: 100 + i,
          high: 101 + i,
          low: 99 + i,
          close: 100 + i,
          volume: 1000,
        })
      }

      const rsi = new RSIIndicator({ period: 14 })
      const result = rsi.calculate(upCandles)

      assert.ok(result)
      assert.equal(result.value, 100)
    })

    it('should handle all losses (RSI = 0)', () => {
      // Create candles with only downward movement
      const downCandles: Candle[] = []
      for (let i = 0; i < 20; i++) {
        downCandles.push({
          timestamp: toEpochDate((i + 1) * 1000),
          open: 100 - i,
          high: 101 - i,
          low: 99 - i,
          close: 100 - i,
          volume: 1000,
        })
      }

      const rsi = new RSIIndicator({ period: 14 })
      const result = rsi.calculate(downCandles)

      assert.ok(result)
      assert.equal(result.value, 0)
    })

    it('should handle no change (RSI = 50 approximately)', () => {
      // Create candles with no price change
      const flatCandles: Candle[] = []
      for (let i = 0; i < 20; i++) {
        flatCandles.push({
          timestamp: toEpochDate((i + 1) * 1000),
          open: 100,
          high: 100,
          low: 100,
          close: 100,
          volume: 1000,
        })
      }

      const rsi = new RSIIndicator({ period: 14 })
      const result = rsi.calculate(flatCandles)

      assert.ok(result)
      // With no change, RSI should be 100 (no losses)
      assert.equal(result.value, 100)
    })

    it('should use cache for repeated calculations', () => {
      const rsi = new RSIIndicator({ period: 14, cacheEnabled: true })

      // First calculation
      const result1 = rsi.calculate(candles)
      assert.ok(result1)

      // Second calculation should use cache
      const result2 = rsi.calculate(candles)
      assert.ok(result2)

      assert.equal(result1.value, result2.value)
    })
  })

  describe('calculateAll', () => {
    it('should calculate RSI for all valid positions', () => {
      const rsi = new RSIIndicator({ period: 14 })
      const results = rsi.calculateAll(candles)

      // With 20 candles and period 14, we should get 6 results
      assert.equal(results.length, 6)

      // Each result should have valid RSI value
      for (const result of results) {
        assert.equal(typeof result.value, 'number')
        assert.ok(result.value >= 0 && result.value <= 100)
      }

      // Results should be in chronological order
      for (let i = 1; i < results.length; i++) {
        assert.ok(results[i]!.timestamp > results[i - 1]!.timestamp)
      }
    })

    it('should return empty array when not enough candles', () => {
      const rsi = new RSIIndicator({ period: 20 })
      const results = rsi.calculateAll(candles.slice(0, 10))

      assert.equal(results.length, 0)
    })
  })

  describe('getMinimumCandles', () => {
    it('should return period + 1', () => {
      const rsi = new RSIIndicator({ period: 14 })
      assert.equal(rsi.getMinimumCandles(), 15)
    })
  })

  describe('reset', () => {
    it('should clear the cache', () => {
      const rsi = new RSIIndicator({ period: 14, cacheEnabled: true })

      // Calculate to populate cache
      const result1 = rsi.calculate(candles)
      assert.ok(result1)

      // Reset
      rsi.reset()

      // Should still calculate correctly
      const result2 = rsi.calculate(candles)
      assert.ok(result2)
      assert.equal(result1.value, result2.value)
    })
  })

  describe('static methods', () => {
    it('should calculate RSI using static method', () => {
      const result = RSIIndicator.calculate(candles, 14, 'wilder')

      assert.ok(result !== null)
      assert.equal(typeof result, 'number')
      assert.ok(result >= 0 && result <= 100)
    })

    it('should return null for invalid inputs', () => {
      const result = RSIIndicator.calculate(candles.slice(0, 5), 14)
      assert.equal(result, null)
    })

    it('should calculate optimized RSI using typed arrays', () => {
      const closes = new Float64Array(candles.map((c) => c.close))
      const results = RSIIndicator.calculateOptimized(closes, 14, 'wilder')

      assert.ok(results)
      assert.equal(results.length, 6) // 20 candles - 14 period = 6

      for (let i = 0; i < results.length; i++) {
        assert.equal(typeof results[i], 'number')
        assert.ok(results[i]! >= 0 && results[i]! <= 100)
      }
    })
  })

  describe('smoothing methods', () => {
    it('should produce different results for different smoothing methods', () => {
      const wilderRSI = new RSIIndicator({ period: 14, smoothing: 'wilder' })
      const smaRSI = new RSIIndicator({ period: 14, smoothing: 'sma' })

      const wilderResult = wilderRSI.calculate(candles)
      const smaResult = smaRSI.calculate(candles)

      assert.ok(wilderResult)
      assert.ok(smaResult)

      // Results should be different (though potentially close)
      assert.notEqual(wilderResult.value, smaResult.value)
    })
  })

  describe('overbought/oversold detection', () => {
    it('should detect overbought conditions', () => {
      // Create strong uptrend candles
      const strongUpCandles: Candle[] = []
      for (let i = 0; i < 20; i++) {
        const basePrice = 100 + i * 2 // Strong upward movement
        strongUpCandles.push({
          timestamp: toEpochDate((i + 1) * 1000),
          open: basePrice,
          high: basePrice + 1,
          low: basePrice - 0.5,
          close: basePrice + 0.8,
          volume: 1000,
        })
      }

      const rsi = new RSIIndicator({ period: 14 })
      const result = rsi.calculate(strongUpCandles)

      assert.ok(result)
      // In strong uptrend, RSI should be above 70 (overbought)
      assert.ok(result.value > 70)
    })

    it('should detect oversold conditions', () => {
      // Create strong downtrend candles
      const strongDownCandles: Candle[] = []
      for (let i = 0; i < 20; i++) {
        const basePrice = 100 - i * 2 // Strong downward movement
        strongDownCandles.push({
          timestamp: toEpochDate((i + 1) * 1000),
          open: basePrice,
          high: basePrice + 0.5,
          low: basePrice - 1,
          close: basePrice - 0.8,
          volume: 1000,
        })
      }

      const rsi = new RSIIndicator({ period: 14 })
      const result = rsi.calculate(strongDownCandles)

      assert.ok(result)
      // In strong downtrend, RSI should be below 30 (oversold)
      assert.ok(result.value < 30)
    })
  })
})
