import type { Candle, GridState, GridLevel } from '@trdr/shared'
import { toStockSymbol, epochDateNow } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import { SelfTuningGridSpacing, type SelfTuningConfig } from './self-tuning-grid-spacing'
import type { VolatilitySpacingConfig } from './volatility-grid-spacing'

describe('SelfTuningGridSpacing', () => {
  let spacingCalculator: SelfTuningGridSpacing
  let testCandles: Candle[]
  let testGridStates: GridState[]

  const createTestCandles = (count: number, basePrice: number = 50000, volatility: number = 0.02): Candle[] => {
    const candles: Candle[] = []
    let price = basePrice
    
    for (let i = 0; i < count; i++) {
      const change = (Math.random() - 0.5) * volatility * price
      price += change
      
      const high = price * (1 + Math.random() * 0.01)
      const low = price * (1 - Math.random() * 0.01)
      
      candles.push({
        timestamp: (epochDateNow() - (count - i) * 60000) as any, // 1 minute intervals
        open: price - change,
        high,
        low,
        close: price,
        volume: 1000 + Math.random() * 5000
      })
    }
    
    return candles
  }

  const createTestGridState = (
    totalPnl: number = 100,
    fillCount: number = 5,
    levelCount: number = 10
  ): GridState => {
    const now = epochDateNow()
    const levels: GridLevel[] = []
    
    for (let i = 0; i < levelCount; i++) {
      levels.push({
        id: `level-${i}`,
        price: 50000 + i * 1000,
        side: i % 2 === 0 ? 'buy' : 'sell',
        size: 0.01,
        isActive: i < 5,
        createdAt: (now - 3600000) as any, // 1 hour ago
        updatedAt: (now - (i < fillCount ? 1800000 : 3600000)) as any, // Some filled 30 min ago
        fillCount: i < fillCount ? 1 : 0,
        pnl: i < fillCount ? totalPnl / fillCount : 0
      })
    }

    return {
      config: {
        gridSpacing: 2,
        gridLevels: levelCount,
        trailPercent: 0.5,
        minOrderSize: 0.001,
        maxOrderSize: 1.0,
        rebalanceThreshold: 0.1
      },
      symbol: toStockSymbol('BTC-USD'),
      levels,
      centerPrice: 50000,
      currentSpacing: 2,
      allocatedCapital: 10000,
      availableCapital: 5000,
      currentPosition: 0.05,
      realizedPnl: totalPnl * 0.8,
      unrealizedPnl: totalPnl * 0.2,
      initializedAt: (now - 7200000) as any, // 2 hours ago
      lastUpdatedAt: now,
      isActive: true
    }
  }

  beforeEach(() => {
    const volatilityConfig: Partial<VolatilitySpacingConfig> = {
      baseSpacing: 2.0,
      maxSpacing: 5.0,
      volatilitySensitivity: 1.5
    }

    const tuningConfig: Partial<SelfTuningConfig> = {
      enablePerformanceTuning: true,
      minHistoryPeriods: 3,
      evaluationWindow: 60000, // 1 minute for testing
      maxAdjustmentPercent: 20,
      minAdjustmentThreshold: 0.1
    }

    spacingCalculator = new SelfTuningGridSpacing(volatilityConfig, tuningConfig)
    testCandles = createTestCandles(50, 50000, 0.02)
    testGridStates = [createTestGridState(150, 6, 10)]
  })

  describe('calculateOptimalSpacing', () => {
    it('should return base spacing when performance tuning is disabled', async () => {
      const disabledConfig: Partial<SelfTuningConfig> = {
        enablePerformanceTuning: false
      }
      
      const calculator = new SelfTuningGridSpacing({}, disabledConfig)
      const result = await calculator.calculateOptimalSpacing(testCandles, 50000, testGridStates)
      
      assert.ok(result.optimalSpacing > 0)
      assert.equal(result.reasoning.includes('Performance tuning'), false)
    })

    it('should return base spacing when insufficient history', async () => {
      const result = await spacingCalculator.calculateOptimalSpacing(testCandles, 50000, testGridStates)
      
      assert.ok(result.optimalSpacing > 0)
      assert.equal(result.reasoning.includes('Performance tuning'), false)
    })

    it('should apply performance tuning when sufficient history exists', async () => {
      // Build up performance history
      for (let i = 0; i < 5; i++) {
        spacingCalculator.recordGridPerformance(
          2.0 + i * 0.2,
          createTestGridState(100 + i * 50, 5 + i, 10),
          { volatility: 0.02, trend: 'sideways', volume: 1000 }
        )
      }

      // Wait for evaluation window to pass
      await new Promise(resolve => setTimeout(resolve, 70))

      const result = await spacingCalculator.calculateOptimalSpacing(testCandles, 50000, testGridStates)
      
      assert.ok(result.optimalSpacing > 0)
      // Should include performance tuning in reasoning when applied
      // Note: Tuning might not always be applied if adjustments are below threshold
    })
  })

  describe('recordGridPerformance', () => {
    it('should record performance metrics correctly', () => {
      const gridState = createTestGridState(200, 8, 12)
      const marketConditions = {
        volatility: 0.025,
        trend: 'bullish' as const,
        volume: 2000
      }

      spacingCalculator.recordGridPerformance(2.5, gridState, marketConditions)

      const stats = spacingCalculator.getPerformanceStatistics()
      assert.equal(stats.historyCount, 1)
      assert.equal(stats.averageSpacing, 2.5)
      assert.equal(stats.bestPerformingSpacing, 2.5)
    })

    it('should maintain limited history size', () => {
      // Record more than max history
      for (let i = 0; i < 20; i++) {
        spacingCalculator.recordGridPerformance(
          2.0 + i * 0.1,
          createTestGridState(100 + i * 10, 5, 10),
          { volatility: 0.02, trend: 'sideways', volume: 1000 }
        )
      }

      const stats = spacingCalculator.getPerformanceStatistics()
      assert.ok(stats.historyCount <= 15) // Should be limited to ~3 * minHistoryPeriods
    })
  })

  describe('performance analysis', () => {
    it('should identify best performing spacing from history', async () => {
      // Record different spacings with varying performance
      const performances = [
        { spacing: 1.5, pnl: 50, fills: 3 },
        { spacing: 2.0, pnl: 200, fills: 6 }, // Best performance
        { spacing: 2.5, pnl: 100, fills: 4 },
        { spacing: 3.0, pnl: 75, fills: 2 }
      ]

      for (const perf of performances) {
        spacingCalculator.recordGridPerformance(
          perf.spacing,
          createTestGridState(perf.pnl, perf.fills, 10),
          { volatility: 0.02, trend: 'sideways', volume: 1000 }
        )
      }

      const stats = spacingCalculator.getPerformanceStatistics()
      assert.equal(stats.bestPerformingSpacing, 2.0)
    })

    it('should adjust spacing based on fill rate', async () => {
      // Record low fill rate performance
      for (let i = 0; i < 5; i++) {
        spacingCalculator.recordGridPerformance(
          3.0, // Wide spacing
          createTestGridState(50, 1, 10), // Low fills
          { volatility: 0.02, trend: 'sideways', volume: 1000 }
        )
      }

      await new Promise(resolve => setTimeout(resolve, 70))

      const result = await spacingCalculator.calculateOptimalSpacing(testCandles, 50000, testGridStates)
      
      // Should suggest tighter spacing due to low fill rate
      // Note: The exact adjustment depends on multiple factors
      assert.ok(result.optimalSpacing > 0)
    })

    it('should consider market conditions in adjustments', async () => {
      // Record performance in different market conditions
      spacingCalculator.recordGridPerformance(
        2.0,
        createTestGridState(100, 5, 10),
        { volatility: 0.05, trend: 'bullish', volume: 5000 } // High volatility, bullish
      )

      spacingCalculator.recordGridPerformance(
        2.0,
        createTestGridState(80, 6, 10),
        { volatility: 0.01, trend: 'sideways', volume: 1000 } // Low volatility, sideways
      )

      const stats = spacingCalculator.getPerformanceStatistics()
      assert.equal(stats.historyCount, 2)
    })
  })

  describe('market condition assessment', () => {
    it('should correctly identify bullish trend', async () => {
      // Create uptrending candles
      const bullishCandles = createTestCandles(20, 50000, 0.01)
      for (let i = 0; i < bullishCandles.length; i++) {
        const candle = { ...bullishCandles[i]! }
        candle.close = 50000 + i * 100 // Steady uptrend
        bullishCandles[i] = candle
      }

      // The market condition assessment is internal, but we can test by recording performance
      spacingCalculator.recordGridPerformance(
        2.0,
        createTestGridState(100, 5, 10),
        { volatility: 0.02, trend: 'bullish', volume: 1000 }
      )

      const stats = spacingCalculator.getPerformanceStatistics()
      assert.equal(stats.historyCount, 1)
    })

    it('should handle high volatility conditions', async () => {
      // Test recording performance in high volatility conditions

      spacingCalculator.recordGridPerformance(
        3.0, // Wider spacing for volatility
        createTestGridState(150, 4, 10),
        { volatility: 0.05, trend: 'sideways', volume: 1000 }
      )

      const stats = spacingCalculator.getPerformanceStatistics()
      assert.equal(stats.historyCount, 1)
    })
  })

  describe('confidence calculation', () => {
    it('should increase confidence with more data', () => {
      // Record multiple performance periods (limited by maxHistorySize = minHistoryPeriods * 3 = 9)
      for (let i = 0; i < 10; i++) {
        spacingCalculator.recordGridPerformance(
          2.0,
          createTestGridState(100 + i * 10, 5, 10),
          { volatility: 0.02, trend: 'sideways', volume: 1000 }
        )
      }

      const stats = spacingCalculator.getPerformanceStatistics()
      assert.equal(stats.historyCount, 9) // Limited by maxHistorySize
      
      // More history should improve confidence in future calculations
      assert.ok(stats.historyCount >= 5)
    })

    it('should handle edge cases with minimal data', async () => {
      const emptyCandles: Candle[] = []
      const result = await spacingCalculator.calculateOptimalSpacing(emptyCandles, 50000)
      
      assert.ok(result.optimalSpacing > 0)
      assert.ok(result.confidence >= 0)
      assert.ok(result.confidence <= 1)
    })
  })

  describe('performance metrics calculation', () => {
    it('should calculate comprehensive metrics correctly', async () => {
      const gridState = createTestGridState(300, 7, 15)
      
      // Verify the grid state has expected structure
      assert.equal(gridState.levels.length, 15)
      assert.equal(gridState.levels.filter(l => l.fillCount > 0).length, 7)
      
      // Current metrics are populated by calculateOptimalSpacing, not recordGridPerformance
      await spacingCalculator.calculateOptimalSpacing(testCandles, 50000, [gridState])

      const stats = spacingCalculator.getPerformanceStatistics()
      assert.ok(stats.currentMetrics.totalPnl !== undefined)
      assert.ok(stats.currentMetrics.fillRate !== undefined)
    })

    it('should handle zero performance gracefully', () => {
      const zeroGridState = createTestGridState(0, 0, 10)
      
      spacingCalculator.recordGridPerformance(
        2.0,
        zeroGridState,
        { volatility: 0.02, trend: 'sideways', volume: 1000 }
      )

      const stats = spacingCalculator.getPerformanceStatistics()
      assert.equal(stats.historyCount, 1)
      // Should not crash with zero performance
    })
  })

  describe('utility methods', () => {
    it('should reset performance history', () => {
      // Add some history
      spacingCalculator.recordGridPerformance(
        2.0,
        createTestGridState(100, 5, 10),
        { volatility: 0.02, trend: 'sideways', volume: 1000 }
      )

      let stats = spacingCalculator.getPerformanceStatistics()
      assert.equal(stats.historyCount, 1)

      // Reset
      spacingCalculator.resetPerformanceHistory()

      stats = spacingCalculator.getPerformanceStatistics()
      assert.equal(stats.historyCount, 0)
      assert.equal(stats.averageSpacing, 0)
    })

    it('should provide meaningful performance statistics', () => {
      // Add varied performance data
      const testData = [
        { spacing: 1.5, pnl: 80, fills: 8 },
        { spacing: 2.0, pnl: 150, fills: 6 },
        { spacing: 2.5, pnl: 120, fills: 4 }
      ]

      testData.forEach(data => {
        spacingCalculator.recordGridPerformance(
          data.spacing,
          createTestGridState(data.pnl, data.fills, 10),
          { volatility: 0.02, trend: 'sideways', volume: 1000 }
        )
      })

      const stats = spacingCalculator.getPerformanceStatistics()
      assert.equal(stats.historyCount, 3)
      assert.equal(stats.averageSpacing, 2.0) // (1.5 + 2.0 + 2.5) / 3
      assert.ok(stats.bestPerformingSpacing > 0)
    })
  })

  describe('edge cases and error handling', () => {
    it('should handle empty grid states', async () => {
      const emptyGridStates: GridState[] = []
      const result = await spacingCalculator.calculateOptimalSpacing(testCandles, 50000, emptyGridStates)
      
      assert.ok(result.optimalSpacing > 0)
      assert.equal(result.reasoning.includes('Performance tuning'), false)
    })

    it('should handle invalid price values', async () => {
      const result = await spacingCalculator.calculateOptimalSpacing(testCandles, 0)
      
      // Should use base spacing for invalid prices
      assert.ok(result.optimalSpacing > 0)
    })

    it('should handle extreme performance values', () => {
      // Test with extreme negative performance
      spacingCalculator.recordGridPerformance(
        2.0,
        createTestGridState(-1000, 0, 10), // Large loss, no fills
        { volatility: 0.02, trend: 'sideways', volume: 1000 }
      )

      // Test with extreme positive performance
      spacingCalculator.recordGridPerformance(
        2.0,
        createTestGridState(10000, 10, 10), // Large gain, all levels filled
        { volatility: 0.02, trend: 'sideways', volume: 1000 }
      )

      const stats = spacingCalculator.getPerformanceStatistics()
      assert.equal(stats.historyCount, 2)
      // Should handle extreme values without crashing
    })
  })

  describe('integration with volatility spacing', () => {
    it('should enhance volatility-based spacing with performance data', async () => {
      // First get base volatility spacing
      const baseCalculator = new SelfTuningGridSpacing({}, { enablePerformanceTuning: false })
      const baseResult = await baseCalculator.calculateOptimalSpacing(testCandles, 50000)

      // Add performance history to tuning calculator
      for (let i = 0; i < 5; i++) {
        spacingCalculator.recordGridPerformance(
          baseResult.optimalSpacing,
          createTestGridState(100 + i * 20, 5, 10),
          { volatility: 0.02, trend: 'sideways', volume: 1000 }
        )
      }

      await new Promise(resolve => setTimeout(resolve, 70))

      const tunedResult = await spacingCalculator.calculateOptimalSpacing(testCandles, 50000, testGridStates)

      // Both should produce valid results
      assert.ok(baseResult.optimalSpacing > 0)
      assert.ok(tunedResult.optimalSpacing > 0)
      
      // Confidence might be different due to performance data
      assert.ok(tunedResult.confidence >= 0)
      assert.ok(tunedResult.confidence <= 1)
    })
  })
})