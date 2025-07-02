import { toEpochDate, type Candle } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import { VWAPIndicator } from './vwap'

describe('VWAPIndicator', () => {
  let candles: Candle[]

  beforeEach(() => {
    // Create test candles with known values over multiple days
    candles = []

    // Day 1 - timestamps 0-23 hours
    for (let i = 0; i < 8; i++) {
      candles.push({
        timestamp: toEpochDate(Date.UTC(2024, 0, 1, i + 9, 0, 0)), // 9 AM - 4 PM
        open: 100 + i,
        high: 102 + i,
        low: 99 + i,
        close: 101 + i,
        volume: 1000 + i * 100,
      })
    }

    // Day 2
    for (let i = 0; i < 8; i++) {
      candles.push({
        timestamp: toEpochDate(Date.UTC(2024, 0, 2, i + 9, 0, 0)),
        open: 105 + i,
        high: 107 + i,
        low: 104 + i,
        close: 106 + i,
        volume: 1200 + i * 100,
      })
    }

    // Day 3
    for (let i = 0; i < 8; i++) {
      candles.push({
        timestamp: toEpochDate(Date.UTC(2024, 0, 3, i + 9, 0, 0)),
        open: 110 + i,
        high: 112 + i,
        low: 109 + i,
        close: 111 + i,
        volume: 1400 + i * 100,
      })
    }
  })

  describe('constructor', () => {
    it('should create with default parameters', () => {
      const vwap = new VWAPIndicator()

      assert.equal(vwap.config.anchorTime, 'session')
      assert.equal(vwap.config.cacheEnabled, true)
    })

    it('should accept custom parameters', () => {
      const vwap = new VWAPIndicator({
        anchorTime: 'week',
        cacheEnabled: false,
      })

      assert.equal(vwap.config.anchorTime, 'week')
      assert.equal(vwap.config.cacheEnabled, false)
    })
  })

  describe('calculate', () => {
    it('should calculate VWAP correctly for session anchor', () => {
      const vwap = new VWAPIndicator({ anchorTime: 'session' })
      // Use only day 1 candles
      const result = vwap.calculate(candles.slice(0, 8))

      assert.ok(result)
      assert.equal(typeof result.value, 'number')
      assert.ok(result.value > 0)
      assert.equal(result.timestamp, candles[7]!.timestamp)

      // VWAP should be weighted average of typical prices
      // Manual calculation for verification
      let pv = 0
      let vol = 0
      for (let i = 0; i < 8; i++) {
        const candle = candles[i]!
        const typicalPrice = (candle.high + candle.low + candle.close) / 3
        pv += typicalPrice * candle.volume
        vol += candle.volume
      }
      const expectedVWAP = pv / vol

      assert.ok(Math.abs(result.value - expectedVWAP) < 0.0001)
    })

    it('should reset VWAP at session boundaries', () => {
      const vwap = new VWAPIndicator({ anchorTime: 'session' })
      
      // Calculate VWAP for all three days
      const allResults = vwap.calculateAll(candles)
      
      // Check that VWAP resets at day boundaries
      // Day 1 last VWAP
      const day1LastVWAP = allResults[7]!.value
      // Day 2 first VWAP
      const day2FirstVWAP = allResults[8]!.value

      // Day 2 first VWAP should be equal to its typical price (no accumulation from day 1)
      const day2FirstCandle = candles[8]!
      const day2FirstTypicalPrice = (day2FirstCandle.high + day2FirstCandle.low + day2FirstCandle.close) / 3
      
      assert.ok(Math.abs(day2FirstVWAP - day2FirstTypicalPrice) < 0.0001)
      assert.notEqual(day2FirstVWAP, day1LastVWAP)
    })

    it('should return null for empty candles', () => {
      const vwap = new VWAPIndicator()
      const result = vwap.calculate([])

      assert.equal(result, null)
    })

    it('should handle single candle', () => {
      const vwap = new VWAPIndicator()
      const singleCandle = candles.slice(0, 1)
      const result = vwap.calculate(singleCandle)

      assert.ok(result)
      // VWAP should equal typical price for single candle
      const typicalPrice = (singleCandle[0]!.high + singleCandle[0]!.low + singleCandle[0]!.close) / 3
      assert.ok(Math.abs(result.value - typicalPrice) < 0.0001)
    })

    it('should handle zero volume gracefully', () => {
      const zeroVolumeCandles: Candle[] = candles.slice(0, 3).map((c) => ({
        ...c,
        volume: 0,
      }))

      const vwap = new VWAPIndicator()
      const result = vwap.calculate(zeroVolumeCandles)

      assert.equal(result, null)
    })

    it('should use cache for repeated calculations', () => {
      const vwap = new VWAPIndicator({ cacheEnabled: true })

      // First calculation
      const result1 = vwap.calculate(candles.slice(0, 8))
      assert.ok(result1)

      // Second calculation should use cache
      const result2 = vwap.calculate(candles.slice(0, 8))
      assert.ok(result2)

      assert.equal(result1.value, result2.value)
    })
  })

  describe('calculateAll', () => {
    it('should calculate VWAP for all candles with session resets', () => {
      const vwap = new VWAPIndicator({ anchorTime: 'session' })
      const results = vwap.calculateAll(candles)

      // Should have one result per candle
      assert.equal(results.length, candles.length)

      // Check that values reset at day boundaries
      // VWAP should accumulate within each day
      for (let day = 0; day < 3; day++) {
        const dayStart = day * 8
        const dayEnd = dayStart + 8

        // Within a day, VWAP should generally change as more data accumulates
        for (let i = dayStart + 1; i < dayEnd; i++) {
          assert.ok(results[i]!.value !== results[i - 1]!.value || i === dayStart)
        }
      }

      // Results should be in chronological order
      for (let i = 1; i < results.length; i++) {
        assert.ok(results[i]!.timestamp > results[i - 1]!.timestamp)
      }
    })

    it('should return empty array for empty candles', () => {
      const vwap = new VWAPIndicator()
      const results = vwap.calculateAll([])

      assert.equal(results.length, 0)
    })
  })

  describe('anchor time variations', () => {
    it('should handle week anchor time', () => {
      // Create candles spanning two weeks
      const weekCandles: Candle[] = []
      
      // Week 1
      for (let day = 0; day < 7; day++) {
        for (let hour = 0; hour < 8; hour++) {
          weekCandles.push({
            timestamp: toEpochDate(Date.UTC(2024, 0, day + 1, hour + 9, 0, 0)),
            open: 100,
            high: 102,
            low: 99,
            close: 101,
            volume: 1000,
          })
        }
      }

      // Week 2
      for (let day = 0; day < 7; day++) {
        for (let hour = 0; hour < 8; hour++) {
          weekCandles.push({
            timestamp: toEpochDate(Date.UTC(2024, 0, day + 8, hour + 9, 0, 0)),
            open: 110,
            high: 112,
            low: 109,
            close: 111,
            volume: 1000,
          })
        }
      }

      const vwap = new VWAPIndicator({ anchorTime: 'week' })
      const results = vwap.calculateAll(weekCandles)

      assert.ok(results.length > 0)
      
      // VWAP should be different between weeks
      const week2FirstIdx = 7 * 8
      
      // Week 2 should start fresh
      const week2FirstCandle = weekCandles[week2FirstIdx]!
      const week2FirstTypicalPrice = (week2FirstCandle.high + week2FirstCandle.low + week2FirstCandle.close) / 3
      
      assert.ok(Math.abs(results[week2FirstIdx]!.value - week2FirstTypicalPrice) < 0.0001)
    })

    it('should handle month anchor time', () => {
      // Create candles spanning two months
      const monthCandles: Candle[] = []
      
      // Month 1 (January)
      for (let day = 0; day < 20; day++) {
        monthCandles.push({
          timestamp: toEpochDate(Date.UTC(2024, 0, day + 1, 12, 0, 0)),
          open: 100,
          high: 102,
          low: 99,
          close: 101,
          volume: 1000,
        })
      }

      // Month 2 (February)
      for (let day = 0; day < 20; day++) {
        monthCandles.push({
          timestamp: toEpochDate(Date.UTC(2024, 1, day + 1, 12, 0, 0)),
          open: 110,
          high: 112,
          low: 109,
          close: 111,
          volume: 1000,
        })
      }

      const vwap = new VWAPIndicator({ anchorTime: 'month' })
      const results = vwap.calculateAll(monthCandles)

      assert.ok(results.length > 0)
      
      // VWAP should reset at month boundary
      const month1Last = results[19]!.value
      const month2First = results[20]!.value
      
      assert.notEqual(month1Last, month2First)
    })
  })

  describe('getMinimumCandles', () => {
    it('should return 1 as minimum', () => {
      const vwap = new VWAPIndicator()
      assert.equal(vwap.getMinimumCandles(), 1)
    })
  })

  describe('reset', () => {
    it('should clear the cache', () => {
      const vwap = new VWAPIndicator({ cacheEnabled: true })

      // Calculate to populate cache
      const result1 = vwap.calculate(candles.slice(0, 8))
      assert.ok(result1)

      // Reset
      vwap.reset()

      // Should still calculate correctly
      const result2 = vwap.calculate(candles.slice(0, 8))
      assert.ok(result2)
      assert.equal(result1.value, result2.value)
    })
  })

  describe('static methods', () => {
    it('should calculate VWAP using static method', () => {
      const result = VWAPIndicator.calculate(candles.slice(0, 8), 'session')

      assert.ok(result !== null)
      assert.equal(typeof result, 'number')
      assert.ok(result > 0)
    })

    it('should return null for empty candles', () => {
      const result = VWAPIndicator.calculate([], 'session')
      assert.equal(result, null)
    })

    it('should calculate optimized VWAP using typed arrays', () => {
      const slice = candles.slice(0, 8)
      const highs = new Float64Array(slice.map((c) => c.high))
      const lows = new Float64Array(slice.map((c) => c.low))
      const closes = new Float64Array(slice.map((c) => c.close))
      const volumes = new Float64Array(slice.map((c) => c.volume))
      const timestamps = new Float64Array(slice.map((c) => c.timestamp))

      const results = VWAPIndicator.calculateOptimized(
        highs,
        lows,
        closes,
        volumes,
        timestamps,
        'session'
      )

      assert.ok(results)
      assert.equal(results.length, 8)

      for (let i = 0; i < results.length; i++) {
        assert.equal(typeof results[i], 'number')
        assert.ok(results[i]! > 0)
      }
    })
  })

  describe('edge cases', () => {
    it('should handle irregular time gaps', () => {
      // Create candles with gaps
      const gappedCandles: Candle[] = [
        {
          timestamp: toEpochDate(Date.UTC(2024, 0, 1, 9, 0, 0)),
          open: 100,
          high: 102,
          low: 99,
          close: 101,
          volume: 1000,
        },
        // Skip several hours
        {
          timestamp: toEpochDate(Date.UTC(2024, 0, 1, 14, 0, 0)),
          open: 105,
          high: 107,
          low: 104,
          close: 106,
          volume: 1500,
        },
        // Next day
        {
          timestamp: toEpochDate(Date.UTC(2024, 0, 2, 10, 0, 0)),
          open: 110,
          high: 112,
          low: 109,
          close: 111,
          volume: 2000,
        },
      ]

      const vwap = new VWAPIndicator({ anchorTime: 'session' })
      const results = vwap.calculateAll(gappedCandles)

      assert.equal(results.length, 3)
      
      // Check that day boundary is still detected correctly
      const day2Candle = gappedCandles[2]!
      const day2TypicalPrice = (day2Candle.high + day2Candle.low + day2Candle.close) / 3
      assert.ok(Math.abs(results[2]!.value - day2TypicalPrice) < 0.0001)
    })

    it('should handle very high volume correctly', () => {
      const highVolumeCandles: Candle[] = candles.slice(0, 3).map((c, i) => ({
        ...c,
        volume: 1e9 * (i + 1), // Billions
      }))

      const vwap = new VWAPIndicator()
      const result = vwap.calculate(highVolumeCandles)

      assert.ok(result)
      assert.ok(result.value > 0)
      assert.ok(isFinite(result.value))
    })
  })
})