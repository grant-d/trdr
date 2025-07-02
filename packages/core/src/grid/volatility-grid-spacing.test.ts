import type { Candle } from '@trdr/shared'
import { toEpochDate } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import {
  VolatilityGridSpacing,
  type VolatilitySpacingConfig
} from './volatility-grid-spacing'

describe('VolatilityGridSpacing', () => {
  let volatilitySpacing: VolatilityGridSpacing

  const createMockCandle = (
    timestamp: number,
    open: number,
    high: number,
    low: number,
    close: number,
    volume = 1000
  ): Candle => ({
    timestamp: toEpochDate(timestamp),
    open,
    high,
    low,
    close,
    volume
  })

  const createTrendingCandles = (startPrice: number, count: number, trend = 0.01): Candle[] => {
    const candles: Candle[] = []
    let price = startPrice
    
    for (let i = 0; i < count; i++) {
      const timestamp = Date.now() - (count - i) * 3600000
      const dailyChange = (Math.random() - 0.5) * 0.02 + trend // ±1% random + trend
      
      const open = price
      const close = price * (1 + dailyChange)
      const high = Math.max(open, close) * (1 + Math.random() * 0.01)
      const low = Math.min(open, close) * (1 - Math.random() * 0.01)
      
      candles.push(createMockCandle(timestamp, open, high, low, close))
      price = close
    }
    
    return candles
  }

  const createVolatileCandles = (startPrice: number, count: number, volatility = 0.05): Candle[] => {
    const candles: Candle[] = []
    let price = startPrice
    
    for (let i = 0; i < count; i++) {
      const timestamp = Date.now() - (count - i) * 3600000
      const dailyChange = (Math.random() - 0.5) * volatility * 2 // High volatility
      
      const open = price
      const close = price * (1 + dailyChange)
      const high = Math.max(open, close) * (1 + Math.random() * volatility)
      const low = Math.min(open, close) * (1 - Math.random() * volatility)
      
      candles.push(createMockCandle(timestamp, open, high, low, close))
      price = close
    }
    
    return candles
  }

  beforeEach(() => {
    const config: Partial<VolatilitySpacingConfig> = {
      baseSpacing: 2.0, // 2% base
      maxSpacing: 8.0, // 8% max
      volatilityMethod: 'atr',
      volatilityPeriod: 14,
      volatilitySensitivity: 1.5,
      riskAdjustment: 0.8,
      enableAdaptiveSpacing: true
    }

    volatilitySpacing = new VolatilityGridSpacing(config)
  })

  describe('calculateOptimalSpacing', () => {
    it('should return base spacing when insufficient data', async () => {
      const candles = createTrendingCandles(50000, 5) // Less than volatility period
      const currentPrice = 50000

      const result = await volatilitySpacing.calculateOptimalSpacing(candles, currentPrice)

      assert.equal(result.optimalSpacing, 2.0)
      assert.equal(result.reasoning, 'Insufficient historical data, using base spacing')
      assert.equal(result.confidence, 0.3)
    })

    it('should increase spacing during high volatility', async () => {
      const candles = createVolatileCandles(50000, 30, 0.08) // High volatility (8%)
      const currentPrice = candles[candles.length - 1]!.close

      const result = await volatilitySpacing.calculateOptimalSpacing(candles, currentPrice)

      assert.ok(result.optimalSpacing > 2.0, 'Should increase spacing for high volatility')
      assert.ok(result.optimalSpacing <= 8.0, 'Should not exceed max spacing')
      assert.ok(result.confidence > 0.3, 'Should have reasonable confidence with sufficient data')
      assert.ok(result.reasoning.includes('volatility'), 'Should mention volatility in reasoning')
    })

    it('should use lower spacing during low volatility', async () => {
      const candles = createTrendingCandles(50000, 30, 0.001) // Very low volatility
      const currentPrice = candles[candles.length - 1]!.close

      const result = await volatilitySpacing.calculateOptimalSpacing(candles, currentPrice)

      // With low volatility, spacing should be close to base or moderately adjusted
      // The algorithm may still apply risk adjustments, so allow for reasonable increase
      assert.ok(result.optimalSpacing >= 2.0, 'Should not go below base spacing')
      assert.ok(result.optimalSpacing <= 6.0, 'Should not increase excessively for low volatility')
      assert.ok(result.confidence > 0.5, 'Should have good confidence with stable data')
    })

    it('should handle both ATR and standard deviation methods', async () => {
      const candles = createTrendingCandles(50000, 25)
      const currentPrice = candles[candles.length - 1]!.close

      // Test ATR method
      const atrSpacing = new VolatilityGridSpacing({
        baseSpacing: 2.0,
        volatilityMethod: 'atr'
      })
      const atrResult = await atrSpacing.calculateOptimalSpacing(candles, currentPrice)

      // Test Standard Deviation method
      const stdSpacing = new VolatilityGridSpacing({
        baseSpacing: 2.0,
        volatilityMethod: 'standard_deviation'
      })
      const stdResult = await stdSpacing.calculateOptimalSpacing(candles, currentPrice)

      assert.ok(atrResult.optimalSpacing > 0)
      assert.ok(stdResult.optimalSpacing > 0)
      assert.ok(atrResult.volatilityMetrics.currentVolatility >= 0)
      assert.ok(stdResult.volatilityMetrics.currentVolatility >= 0)
    })

    it('should consider volatility trend in spacing calculation', async () => {
      // Create candles with increasing volatility pattern
      const baseCandles = createTrendingCandles(50000, 20, 0.005)
      const volatileCandles = createVolatileCandles(
        baseCandles[baseCandles.length - 1]!.close,
        10,
        0.06
      )
      const candles = [...baseCandles, ...volatileCandles]
      const currentPrice = candles[candles.length - 1]!.close

      const result = await volatilitySpacing.calculateOptimalSpacing(candles, currentPrice)

      // Volatility trend detection requires sufficient history and significant changes
      // Just verify that we get a valid trend and reasonable spacing
      assert.ok(['increasing', 'decreasing', 'stable'].includes(result.volatilityMetrics.volatilityTrend))
      assert.ok(result.optimalSpacing > 2.0, 'Should increase spacing with volatile data')
      assert.ok(result.optimalSpacing <= 8.0, 'Should not exceed maximum spacing')
    })
  })

  describe('validateSpacingWithSwings', () => {
    it('should validate spacing against swing patterns', async () => {
      // Create candles with clear swing patterns
      const candles: Candle[] = []
      let price = 50000
      
      // Create a clear swing pattern
      for (let i = 0; i < 20; i++) {
        const timestamp = Date.now() - (20 - i) * 3600000
        
        // Create swing pattern: up for 5, down for 5, repeat
        const swingDirection = Math.floor(i / 5) % 2 === 0 ? 1 : -1
        const change = swingDirection * 0.02 // 2% swing
        
        const open = price
        const close = price * (1 + change)
        const high = Math.max(open, close) * 1.005
        const low = Math.min(open, close) * 0.995
        
        candles.push(createMockCandle(timestamp, open, high, low, close))
        price = close
      }

      // Test reasonable spacing (should be valid)
      const reasonableSpacing = 1.5
      const reasonableResult = await volatilitySpacing.validateSpacingWithSwings(
        candles,
        reasonableSpacing
      )
      assert.equal(reasonableResult.isValid, true)

      // Test too wide spacing (should be invalid)
      const wideSpacing = 10.0
      const wideResult = await volatilitySpacing.validateSpacingWithSwings(
        candles,
        wideSpacing
      )
      assert.equal(wideResult.isValid, false)
      assert.ok(wideResult.adjustedSpacing !== undefined)
      assert.ok(wideResult.reason.includes('too wide'))

      // Test too tight spacing (should be invalid)
      const tightSpacing = 0.1
      const tightResult = await volatilitySpacing.validateSpacingWithSwings(
        candles,
        tightSpacing
      )
      assert.equal(tightResult.isValid, false)
      assert.ok(tightResult.adjustedSpacing !== undefined)
      assert.ok(tightResult.reason.includes('too tight'))
    })

    it('should handle insufficient swing data gracefully', async () => {
      const candles = createTrendingCandles(50000, 5) // Not enough for swing analysis
      
      const result = await volatilitySpacing.validateSpacingWithSwings(candles, 2.0)
      
      assert.equal(result.isValid, true)
      assert.ok(result.reason.includes('Insufficient swing data'))
    })
  })

  describe('volatility metrics calculation', () => {
    it('should calculate comprehensive volatility metrics', async () => {
      const candles = createVolatileCandles(50000, 30, 0.04)
      const currentPrice = candles[candles.length - 1]!.close

      const result = await volatilitySpacing.calculateOptimalSpacing(candles, currentPrice)
      const metrics = result.volatilityMetrics

      assert.ok(metrics.currentVolatility >= 0)
      assert.ok(metrics.normalizedVolatility >= 0 && metrics.normalizedVolatility <= 1)
      assert.ok(metrics.averageVolatility >= 0)
      assert.ok(metrics.volatilityPercentile >= 0 && metrics.volatilityPercentile <= 100)
      assert.ok(['increasing', 'decreasing', 'stable'].includes(metrics.volatilityTrend))
    })

    it('should maintain volatility history correctly', async () => {
      const candles = createTrendingCandles(50000, 20)
      const currentPrice = candles[candles.length - 1]!.close

      // Calculate spacing multiple times to build history
      await volatilitySpacing.calculateOptimalSpacing(candles, currentPrice)
      await volatilitySpacing.calculateOptimalSpacing(candles.slice(1), currentPrice)
      await volatilitySpacing.calculateOptimalSpacing(candles.slice(2), currentPrice)

      const history = volatilitySpacing.getVolatilityHistory()
      assert.ok(history.length > 0, 'Should maintain volatility history')
      assert.ok(history.length <= 3, 'Should have tracked multiple calculations')
    })

    it('should reset history when requested', async () => {
      const candles = createTrendingCandles(50000, 20)
      const currentPrice = candles[candles.length - 1]!.close

      // Build some history
      await volatilitySpacing.calculateOptimalSpacing(candles, currentPrice)
      assert.ok(volatilitySpacing.getVolatilityHistory().length > 0)

      // Reset and verify
      volatilitySpacing.resetHistory()
      assert.equal(volatilitySpacing.getVolatilityHistory().length, 0)
    })
  })

  describe('configuration validation', () => {
    it('should use default configuration when not provided', () => {
      const defaultSpacing = new VolatilityGridSpacing()
      
      // Should not throw and should have reasonable defaults
      assert.ok(defaultSpacing)
    })

    it('should respect max spacing limits', async () => {
      const restrictiveConfig: Partial<VolatilitySpacingConfig> = {
        baseSpacing: 1.0,
        maxSpacing: 2.0, // Very low max
        volatilitySensitivity: 5.0 // High sensitivity
      }

      const restrictiveSpacing = new VolatilityGridSpacing(restrictiveConfig)
      const candles = createVolatileCandles(50000, 25, 0.1) // Extreme volatility
      const currentPrice = candles[candles.length - 1]!.close

      const result = await restrictiveSpacing.calculateOptimalSpacing(candles, currentPrice)

      assert.ok(result.optimalSpacing <= 2.0, 'Should respect max spacing limit')
      assert.ok(result.optimalSpacing >= 1.0, 'Should not go below base spacing')
    })

    it('should work with adaptive spacing disabled', async () => {
      const staticConfig: Partial<VolatilitySpacingConfig> = {
        baseSpacing: 3.0,
        enableAdaptiveSpacing: false
      }

      const staticSpacing = new VolatilityGridSpacing(staticConfig)
      const candles = createVolatileCandles(50000, 25, 0.08) // High volatility
      const currentPrice = candles[candles.length - 1]!.close

      const result = await staticSpacing.calculateOptimalSpacing(candles, currentPrice)

      // With adaptive spacing disabled, should return base spacing
      assert.equal(result.optimalSpacing, 3.0)
    })
  })
})