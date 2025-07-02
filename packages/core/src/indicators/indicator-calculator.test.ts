import { toEpochDate, type Candle } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import { IndicatorCalculator } from './indicator-calculator'

describe('IndicatorCalculator', () => {
  let calculator: IndicatorCalculator
  let candles: Candle[]

  beforeEach(() => {
    calculator = new IndicatorCalculator()

    // Create test candles with realistic price movements
    candles = []
    let basePrice = 100
    for (let i = 0; i < 50; i++) {
      // Add some price movement
      const change = Math.sin(i * 0.2) * 5 + Math.random() * 2 - 1
      basePrice += change

      const high = basePrice + Math.random() * 2
      const low = basePrice - Math.random() * 2
      const close = low + Math.random() * (high - low)
      const open = i > 0 ? candles[i - 1]!.close : basePrice

      candles.push({
        timestamp: toEpochDate((i + 1) * 60000), // 1 minute intervals
        open,
        high,
        low,
        close,
        volume: 1000 + Math.random() * 1000,
      })
    }
  })

  describe('constructor', () => {
    it('should create calculator with default cache size', () => {
      const calc = new IndicatorCalculator()
      assert.ok(calc)
    })

    it('should create calculator with custom cache size', () => {
      const calc = new IndicatorCalculator(500)
      assert.ok(calc)
    })
  })

  describe('Trend Indicators', () => {
    describe('sma', () => {
      it('should calculate simple moving average', () => {
        const result = calculator.sma(candles, 10)
        assert.ok(result)
        assert.equal(typeof result.value, 'number')
        assert.ok(result.value > 0)
        assert.equal(result.timestamp, candles[candles.length - 1]!.timestamp)
      })

      it('should return null when not enough data', () => {
        const result = calculator.sma(candles.slice(0, 5), 10)
        assert.equal(result, null)
      })
    })

    describe('ema', () => {
      it('should calculate exponential moving average', () => {
        const result = calculator.ema(candles, 10)
        assert.ok(result)
        assert.equal(typeof result.value, 'number')
        assert.ok(result.value > 0)
      })

      it('should return null when not enough data', () => {
        const result = calculator.ema(candles.slice(0, 5), 10)
        assert.equal(result, null)
      })
    })

    describe('macd', () => {
      it('should calculate MACD with default config', () => {
        const result = calculator.macd(candles)
        assert.ok(result)
        assert.equal(typeof result.macd, 'number')
        assert.equal(typeof result.signal, 'number')
        assert.equal(typeof result.histogram, 'number')
      })

      it('should calculate MACD with custom config', () => {
        const result = calculator.macd(candles, {
          fastPeriod: 10,
          slowPeriod: 20,
          signalPeriod: 5,
        })
        assert.ok(result)
        assert.equal(typeof result.macd, 'number')
      })

      it('should return null when not enough data', () => {
        const result = calculator.macd(candles.slice(0, 20))
        assert.equal(result, null)
      })
    })
  })

  describe('Volatility Indicators', () => {
    describe('atr', () => {
      it('should calculate ATR with default period', () => {
        const result = calculator.atr(candles)
        assert.ok(result)
        assert.equal(typeof result.value, 'number')
        assert.ok(result.value > 0)
      })

      it('should calculate ATR with custom period', () => {
        const result = calculator.atr(candles, 10)
        assert.ok(result)
        assert.equal(typeof result.value, 'number')
      })

      it('should calculate ATR with custom config', () => {
        const result = calculator.atr(candles, 14, { smoothing: 'sma' })
        assert.ok(result)
        assert.equal(typeof result.value, 'number')
      })

      it('should return null when not enough data', () => {
        const result = calculator.atr(candles.slice(0, 10), 14)
        assert.equal(result, null)
      })
    })

    describe('bollingerBands', () => {
      it('should calculate Bollinger Bands with default config', () => {
        const result = calculator.bollingerBands(candles)
        assert.ok(result)
        assert.equal(typeof result.upper, 'number')
        assert.equal(typeof result.middle, 'number')
        assert.equal(typeof result.lower, 'number')
        assert.equal(typeof result.bandwidth, 'number')
        assert.equal(typeof result.percentB, 'number')

        // Verify bands are in correct order
        assert.ok(result.upper > result.middle)
        assert.ok(result.middle > result.lower)
      })

      it('should calculate Bollinger Bands with custom config', () => {
        const result = calculator.bollingerBands(candles, {
          period: 10,
          stdDevMultiplier: 1.5,
        })
        assert.ok(result)
        assert.ok(result.upper > result.middle)
      })

      it('should return null when not enough data', () => {
        const result = calculator.bollingerBands(candles.slice(0, 10), { period: 20 })
        assert.equal(result, null)
      })
    })
  })

  describe('Momentum Indicators', () => {
    describe('rsi', () => {
      it('should calculate RSI with default period', () => {
        const result = calculator.rsi(candles)
        assert.ok(result)
        assert.equal(typeof result.value, 'number')
        assert.ok(result.value >= 0 && result.value <= 100)
      })

      it('should calculate RSI with custom period', () => {
        const result = calculator.rsi(candles, 10)
        assert.ok(result)
        assert.ok(result.value >= 0 && result.value <= 100)
      })

      it('should calculate RSI with custom config', () => {
        const result = calculator.rsi(candles, 14, { smoothing: 'sma' })
        assert.ok(result)
        assert.ok(result.value >= 0 && result.value <= 100)
      })

      it('should return null when not enough data', () => {
        const result = calculator.rsi(candles.slice(0, 10), 14)
        assert.equal(result, null)
      })
    })
  })

  describe('Volume Indicators', () => {
    describe('vwap', () => {
      it('should calculate VWAP with default config', () => {
        const result = calculator.vwap(candles)
        assert.ok(result)
        assert.equal(typeof result.value, 'number')
        assert.ok(result.value > 0)
      })

      it('should calculate VWAP with custom config', () => {
        const result = calculator.vwap(candles, { anchorTime: 'week' })
        assert.ok(result)
        assert.equal(typeof result.value, 'number')
      })

      it('should return null for empty candles', () => {
        const result = calculator.vwap([])
        assert.equal(result, null)
      })
    })
  })

  describe('Swing Detection', () => {
    describe('swingHighLow', () => {
      it('should detect swing highs and lows with default config', () => {
        const swings = calculator.swingHighLow(candles)
        assert.ok(Array.isArray(swings))

        // Should find at least some swings in 50 candles
        assert.ok(swings.length > 0)

        // Check swing properties
        for (const swing of swings) {
          assert.ok(['high', 'low'].includes(swing.type))
          assert.equal(typeof swing.index, 'number')
          assert.equal(typeof swing.price, 'number')
          assert.equal(typeof swing.timestamp, 'number')
          assert.equal(typeof swing.strength, 'number')
        }
      })

      it('should detect swings with custom config', () => {
        const swings = calculator.swingHighLow(candles, {
          lookback: 3,
          lookforward: 3,
          minSwingPercent: 0.01,
        })
        assert.ok(Array.isArray(swings))
      })

      it('should return empty array when not enough data', () => {
        const swings = calculator.swingHighLow(candles.slice(0, 5), {
          lookback: 5,
          lookforward: 5,
        })
        assert.equal(swings.length, 0)
      })
    })
  })

  describe('calculateAll', () => {
    it('should calculate all standard indicators', () => {
      const results = calculator.calculateAll(candles)

      // Check trend indicators
      assert.ok(results.sma20)
      assert.equal(typeof results.sma20.value, 'number')

      assert.ok(results.sma50) // Should have exactly enough data for 50 period
      assert.equal(typeof results.sma50.value, 'number')
      
      assert.ok(results.ema20)
      assert.equal(typeof results.ema20.value, 'number')

      // Check momentum indicators
      assert.ok(results.rsi)
      assert.ok(results.rsi.value >= 0 && results.rsi.value <= 100)

      assert.ok(results.macd)
      assert.equal(typeof results.macd.macd, 'number')

      // Check volatility indicators
      assert.ok(results.atr)
      assert.ok(results.atr.value > 0)

      assert.ok(results.bollingerBands)
      assert.ok(results.bollingerBands.upper > results.bollingerBands.lower)

      // Check volume indicators
      assert.ok(results.vwap)
      assert.ok(results.vwap.value > 0)
    })

    it('should handle partial results gracefully', () => {
      const results = calculator.calculateAll(candles.slice(0, 15))

      // Some indicators should work
      assert.ok(results.sma20 === undefined) // Not enough data
      assert.ok(results.ema20 === undefined) // Not enough data
      assert.ok(results.rsi)
      assert.ok(results.atr)
      
      // MACD needs more data
      assert.ok(results.macd === undefined)
    })

    it('should handle empty candles', () => {
      const results = calculator.calculateAll([])

      assert.equal(results.sma20, undefined)
      assert.equal(results.sma50, undefined)
      assert.equal(results.ema20, undefined)
      assert.equal(results.rsi, undefined)
      assert.equal(results.macd, undefined)
      assert.equal(results.atr, undefined)
      assert.equal(results.bollingerBands, undefined)
      assert.equal(results.vwap, undefined)
    })
  })

  describe('Cache Management', () => {
    it('should clear cache', () => {
      // Calculate some indicators to populate cache
      calculator.sma(candles, 10)
      calculator.ema(candles, 10)
      calculator.rsi(candles)

      // Clear cache
      calculator.clearCache()

      // Should still work after clearing
      const result = calculator.sma(candles, 10)
      assert.ok(result)
    })

    it('should get cache statistics', () => {
      // Calculate some indicators
      calculator.sma(candles, 10)
      calculator.ema(candles, 10)
      calculator.rsi(candles)

      const stats = calculator.getCacheStats()
      assert.ok(stats)
      assert.equal(typeof stats.size, 'number')
      assert.equal(typeof stats.hits, 'number')
      assert.equal(typeof stats.misses, 'number')
      assert.equal(typeof stats.evictions, 'number')
    })

    it('should use cache for repeated calculations', () => {
      // First calculation
      const result1 = calculator.sma(candles, 10)
      assert.ok(result1)

      // Second calculation (should return same value)
      const result2 = calculator.sma(candles, 10)
      assert.ok(result2)
      assert.equal(result1.value, result2.value)

      // Third calculation with different period should be different
      const result3 = calculator.sma(candles, 20)
      assert.ok(result3)
      assert.notEqual(result1.value, result3.value)
    })
  })

  describe('Edge Cases', () => {
    it('should handle single candle', () => {
      const singleCandle = [candles[0]!]

      const sma = calculator.sma(singleCandle, 1)
      assert.ok(sma)
      assert.equal(sma.value, singleCandle[0]!.close)

      const vwap = calculator.vwap(singleCandle)
      assert.ok(vwap)

      const atr = calculator.atr(singleCandle, 1)
      assert.equal(atr, null) // Needs at least 2 candles
    })

    it('should handle candles with zero volume', () => {
      const zeroVolumeCandles = candles.slice(0, 20).map(c => ({
        ...c,
        volume: 0,
      }))

      const vwap = calculator.vwap(zeroVolumeCandles)
      assert.equal(vwap, null) // VWAP needs volume
    })

    it('should handle extreme price values', () => {
      const extremeCandles: Candle[] = [
        { timestamp: toEpochDate(1000), open: 1, high: 10000, low: 0.001, close: 5000, volume: 1000 },
        { timestamp: toEpochDate(2000), open: 5000, high: 9999, low: 1, close: 100, volume: 1000 },
        { timestamp: toEpochDate(3000), open: 100, high: 1000, low: 10, close: 500, volume: 1000 },
      ]

      const atr = calculator.atr(extremeCandles, 2)
      assert.ok(atr)
      assert.ok(atr.value > 0)
    })
  })
})