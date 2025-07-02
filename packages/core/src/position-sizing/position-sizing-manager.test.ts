import { describe, it, beforeEach } from 'node:test'
import assert from 'node:assert/strict'
import { PositionSizingManager } from './position-sizing-manager'
import { FixedFractionalStrategy } from './strategies/fixed-fractional'
import { KellyCriterionStrategy } from './strategies/kelly-criterion'
import { VolatilityAdjustedStrategy } from './strategies/volatility-adjusted'
import type { PositionSizingInput, PositionSizingConfig } from './interfaces'
import { EventBus } from '../events/event-bus'

describe('PositionSizingManager', () => {
  let manager: PositionSizingManager
  let config: PositionSizingConfig
  let baseInput: PositionSizingInput

  beforeEach(() => {
    // Reset event bus to avoid interference between tests
    EventBus.getInstance().reset()
    
    config = {
      defaultStrategy: 'fixed',
      strategies: ['kelly', 'fixed', 'volatility'],
      enableAdaptive: true,
      enableMarketAdjustments: true,
      minPositionSize: 0.001, // 0.001 BTC
      maxPositionSize: 1.0, // 1 BTC
      enableBacktesting: false
    }

    manager = new PositionSizingManager(config)

    baseInput = {
      side: 'buy',
      entryPrice: 50000,
      stopLoss: 49000,
      winRate: 0.6,
      riskRewardRatio: 2.0,
      confidence: 0.8,
      riskParams: {
        accountBalance: 10000,
        maxRiskPerTrade: 0.02,
        maxPortfolioRisk: 0.06,
        currentExposure: 0,
        openPositions: 0,
        maxPositions: 5,
        riskFreeRate: 0.02
      },
      marketConditions: {
        volatility: 0.5,
        spread: 0.0005,
        trendStrength: 0.2,
        relativeVolume: 1.0,
        regime: 'trending',
        timeOfDayFactor: 1.0
      }
    }
  })

  describe('Strategy Registration', () => {
    it('should have default strategies registered', () => {
      const strategies = manager.getAvailableStrategies()
      assert.ok(strategies.includes('kelly'))
      assert.ok(strategies.includes('fixed'))
      assert.ok(strategies.includes('volatility'))
    })

    it('should register custom strategy', () => {
      const customStrategy = new FixedFractionalStrategy(0.015)
      manager.registerStrategy('custom', customStrategy)
      
      const strategies = manager.getAvailableStrategies()
      assert.ok(strategies.includes('custom'))
    })

    it('should set active strategy', () => {
      manager.setActiveStrategy('kelly')
      assert.equal(manager.getActiveStrategy(), 'kelly')
    })

    it('should throw on invalid strategy', () => {
      assert.throws(
        () => manager.setActiveStrategy('invalid'),
        /Strategy 'invalid' not found/
      )
    })

    it('should allow custom Kelly parameters', () => {
      const customKelly = new KellyCriterionStrategy(0.5, 0.5) // Half Kelly, 50% max
      manager.registerStrategy('halfKelly', customKelly)
      manager.setActiveStrategy('halfKelly')
      
      const output = manager.calculatePositionSize(baseInput)
      
      // Kelly calculates position allocation, not risk percentage
      // With 0.5 safety factor (vs default 0.25), we should see roughly double the risk
      // Default Kelly would give ~0.001-0.002 risk percentage
      assert.ok(output.riskPercentage > 0.002) // Should be higher than default
      assert.ok(output.riskPercentage < 0.01) // But still reasonable
      assert.equal(output.method, 'Kelly Criterion')
      
      // Check the reasoning shows the higher Kelly fraction
      assert.ok(output.reasoning.includes('20.00%')) // 0.4 * 0.5 = 0.2 = 20%
    })

    it('should allow custom volatility parameters', () => {
      const customVol = new VolatilityAdjustedStrategy(0.02, 20, 5) // 2% base, higher multiplier
      manager.registerStrategy('customVol', customVol)
      manager.setActiveStrategy('customVol')
      
      const output = manager.calculatePositionSize(baseInput)
      
      // Should use 2% base risk instead of 1%
      assert.ok(output.riskPercentage > 0.01)
      assert.equal(output.method, 'Volatility Adjusted')
    })
  })

  describe('Position Size Calculation', () => {
    it('should calculate position size with fixed fractional', () => {
      manager.setActiveStrategy('fixed')
      const output = manager.calculatePositionSize(baseInput)

      assert.ok(output.positionSize > 0)
      assert.equal(output.method, 'Fixed Fractional')
      // Fixed fractional uses 1% base risk, multiplied by confidence (0.8)
      assert.equal(output.riskPercentage, 0.01 * 0.8) // 1% * confidence
      assert.ok(output.confidence > 0)
    })

    it('should calculate position size with Kelly criterion', () => {
      manager.setActiveStrategy('kelly')
      const output = manager.calculatePositionSize(baseInput)

      assert.ok(output.positionSize > 0)
      assert.equal(output.method, 'Kelly Criterion')
      assert.ok(output.reasoning.includes('Kelly'))
      
      // Kelly formula: f = (bp - q) / b
      // where f = fraction, b = risk/reward ratio, p = win rate, q = 1 - p
      // With safety factor of 0.25 (1/4 Kelly)
      const p = baseInput.winRate // 0.6
      const q = 1 - p // 0.4
      const b = baseInput.riskRewardRatio // 2.0
      const kellyFraction = (b * p - q) / b // (2*0.6 - 0.4)/2 = 0.8/2 = 0.4
      const safetyFactor = 0.25 // Default safety factor
      const expectedFraction = kellyFraction * safetyFactor * baseInput.confidence // 0.4 * 0.25 * 0.8 = 0.08
      
      // After volatility adjustment (50% volatility reduces by 25%)
      const volatilityAdjustment = 1 - (baseInput.marketConditions.volatility * 0.5) // 1 - 0.25 = 0.75
      const expectedPositionFraction = expectedFraction * volatilityAdjustment // 0.08 * 0.75 = 0.06
      
      // Risk percentage = position fraction * risk per trade
      const riskPerTrade = Math.abs(baseInput.entryPrice - baseInput.stopLoss) / baseInput.entryPrice // 0.02
      const expectedRiskPercentage = expectedPositionFraction * riskPerTrade // 0.06 * 0.02 = 0.0012
      
      // Risk percentage should be around 0.12% (0.0012)
      assert.ok(Math.abs(output.riskPercentage - expectedRiskPercentage) < 0.001, 
        `Expected risk ~${expectedRiskPercentage.toFixed(4)}, got ${output.riskPercentage.toFixed(4)}`)
    })

    it('should handle negative Kelly edge correctly', () => {
      manager.setActiveStrategy('kelly')
      const losingInput = {
        ...baseInput,
        winRate: 0.3, // Low win rate
        riskRewardRatio: 1.0 // Poor risk/reward
      }
      
      const output = manager.calculatePositionSize(losingInput)
      
      // Kelly should return 0 position size for negative edge
      assert.equal(output.positionSize, 0)
      assert.equal(output.confidence, 0)
      assert.ok(output.warnings.some(w => w.includes('Negative Kelly fraction')))
    })

    it('should calculate position size with volatility adjusted', () => {
      manager.setActiveStrategy('volatility')
      const output = manager.calculatePositionSize(baseInput)

      assert.ok(output.positionSize > 0)
      assert.equal(output.method, 'Volatility Adjusted')
      assert.ok(output.adjustments.some(a => a.type === 'volatility'))
      
      // Volatility adjusted uses 1% base risk with volatility scaling
      // With 0.5 volatility (normal), multiplier should be around 1.0
      const baseRisk = 0.01
      const volatilityMultiplier = 1 / (0.5 + baseInput.marketConditions.volatility) // 1/1 = 1.0
      const expectedRisk = baseRisk * volatilityMultiplier * baseInput.confidence
      
      assert.ok(Math.abs(output.riskPercentage - expectedRisk) < 0.005,
        `Expected risk ~${expectedRisk.toFixed(3)}, got ${output.riskPercentage.toFixed(3)}`)
    })

    it('should reduce position size in high volatility', () => {
      manager.setActiveStrategy('volatility')
      const highVolInput = {
        ...baseInput,
        marketConditions: {
          ...baseInput.marketConditions,
          volatility: 0.9 // High volatility
        }
      }
      
      const normalOutput = manager.calculatePositionSize(baseInput)
      const highVolOutput = manager.calculatePositionSize(highVolInput)
      
      // High volatility should reduce position size
      assert.ok(highVolOutput.positionSize < normalOutput.positionSize)
      assert.ok(highVolOutput.riskPercentage < normalOutput.riskPercentage)
      assert.ok(highVolOutput.adjustments.some(a => 
        a.type === 'volatility' && a.factor < 1.0
      ))
    })

    it('should increase position size in low volatility', () => {
      manager.setActiveStrategy('volatility')
      const lowVolInput = {
        ...baseInput,
        marketConditions: {
          ...baseInput.marketConditions,
          volatility: 0.2 // Low volatility
        }
      }
      
      const normalOutput = manager.calculatePositionSize(baseInput)
      const lowVolOutput = manager.calculatePositionSize(lowVolInput)
      
      // Low volatility should increase position size (up to max)
      assert.ok(lowVolOutput.positionSize > normalOutput.positionSize)
      assert.ok(lowVolOutput.adjustments.some(a => 
        a.type === 'volatility' && a.factor > 1.0
      ))
    })

    it('should calculate position size within reasonable risk bounds', () => {
      const highRiskInput = {
        ...baseInput,
        stopLoss: 45000, // 10% stop loss
        riskParams: {
          ...baseInput.riskParams,
          maxRiskPerTrade: 0.02 // 2% max risk
        }
      }

      const output = manager.calculatePositionSize(highRiskInput)
      // Fixed fractional with 1% base risk * 0.8 confidence = 0.8% risk
      assert.ok(output.riskPercentage <= baseInput.riskParams.maxRiskPerTrade)
      assert.ok(output.riskPercentage > 0)
      assert.equal(output.method, 'Fixed Fractional')
    })

    it('should apply minimum position size constraint', () => {
      const smallAccountInput = {
        ...baseInput,
        riskParams: {
          ...baseInput.riskParams,
          accountBalance: 100 // Very small account
        }
      }

      const output = manager.calculatePositionSize(smallAccountInput)
      
      if (output.positionSize > 0) {
        assert.ok(output.positionSize >= config.minPositionSize)
      } else {
        assert.ok(output.warnings.some(w => w.includes('minimum')))
      }
    })

    it('should apply maximum position size constraint', () => {
      const largeAccountInput = {
        ...baseInput,
        stopLoss: 49900, // Very tight stop
        riskParams: {
          ...baseInput.riskParams,
          accountBalance: 1000000, // Large account
          maxRiskPerTrade: 0.1 // High risk tolerance
        }
      }

      const output = manager.calculatePositionSize(largeAccountInput)
      assert.ok(output.positionSize <= config.maxPositionSize)
      
      if (output.positionSize === config.maxPositionSize) {
        assert.ok(output.adjustments.some(a => a.type === 'risk_limit'))
      }
    })
  })

  describe('Market Condition Adjustments', () => {
    it('should reduce size in high volatility', () => {
      const highVolInput = {
        ...baseInput,
        marketConditions: {
          ...baseInput.marketConditions,
          volatility: 0.9
        }
      }

      const normalOutput = manager.calculatePositionSize(baseInput)
      const volOutput = manager.calculatePositionSize(highVolInput)

      assert.ok(volOutput.positionSize < normalOutput.positionSize)
      assert.ok(volOutput.adjustments.some(a => a.type === 'volatility'))
    })

    it('should reduce size in low volume', () => {
      const lowVolumeInput = {
        ...baseInput,
        marketConditions: {
          ...baseInput.marketConditions,
          relativeVolume: 0.3
        }
      }

      const normalOutput = manager.calculatePositionSize(baseInput)
      const lowVolOutput = manager.calculatePositionSize(lowVolumeInput)

      assert.ok(lowVolOutput.positionSize < normalOutput.positionSize)
      assert.ok(lowVolOutput.adjustments.some(a => 
        a.type === 'market_conditions' && a.reason.includes('volume')
      ))
    })

    it('should reduce size with high spread', () => {
      const highSpreadInput = {
        ...baseInput,
        marketConditions: {
          ...baseInput.marketConditions,
          spread: 0.005 // 0.5%
        }
      }

      const normalOutput = manager.calculatePositionSize(baseInput)
      const spreadOutput = manager.calculatePositionSize(highSpreadInput)

      assert.ok(spreadOutput.positionSize < normalOutput.positionSize)
      assert.ok(spreadOutput.adjustments.some(a => 
        a.type === 'market_conditions' && a.reason.includes('spread')
      ))
    })
  })

  describe('Adaptive Sizing', () => {
    it('should reduce size during drawdown', () => {
      const drawdownInput = {
        ...baseInput,
        historicalMetrics: {
          avgWinRate: 0.5,
          avgRiskReward: 1.5,
          maxConsecutiveLosses: 5,
          currentConsecutiveLosses: 4,
          sharpeRatio: 0.8,
          maxDrawdown: 0.2,
          profitFactor: 1.2
        }
      }

      const normalOutput = manager.calculatePositionSize(baseInput)
      const drawdownOutput = manager.calculatePositionSize(drawdownInput)

      assert.ok(drawdownOutput.positionSize < normalOutput.positionSize)
    })

    it('should increase confidence during winning streak', () => {
      const winningInput = {
        ...baseInput,
        confidence: 0.7,
        historicalMetrics: {
          avgWinRate: 0.7,
          avgRiskReward: 2.5,
          maxConsecutiveLosses: 2,
          currentConsecutiveLosses: 0,
          sharpeRatio: 2.0,
          maxDrawdown: 0.05,
          profitFactor: 3.0
        }
      }

      const output = manager.calculatePositionSize(winningInput)
      assert.ok(output.confidence > winningInput.confidence)
    })
  })

  describe('Strategy Recommendation', () => {
    it('should recommend volatility strategy in high volatility', () => {
      const recommendation = manager.getRecommendedStrategy({
        volatility: 0.8,
        spread: 0.001,
        trendStrength: 0.1,
        relativeVolume: 1.0,
        regime: 'volatile',
        timeOfDayFactor: 1.0
      })

      assert.equal(recommendation.strategy, 'volatility')
      assert.ok(recommendation.reasoning.includes('volatility'))
    })

    it('should recommend Kelly in trending market', () => {
      const recommendation = manager.getRecommendedStrategy({
        volatility: 0.3,
        spread: 0.0005,
        trendStrength: 0.6,
        relativeVolume: 1.2,
        regime: 'trending',
        timeOfDayFactor: 1.0
      })

      assert.equal(recommendation.strategy, 'kelly')
      assert.ok(recommendation.reasoning.includes('trend'))
    })

    it('should recommend fixed fractional as default', () => {
      const recommendation = manager.getRecommendedStrategy({
        volatility: 0.5,
        spread: 0.001,
        trendStrength: 0.1,
        relativeVolume: 0.9,
        regime: 'ranging',
        timeOfDayFactor: 0.8
      })

      assert.equal(recommendation.strategy, 'fixed')
      assert.ok(recommendation.reasoning.includes('Standard'))
    })
  })

  describe('Kelly Strategy Edge Cases', () => {
    it('should cap Kelly fraction at maximum', () => {
      manager.setActiveStrategy('kelly')
      const highEdgeInput = {
        ...baseInput,
        winRate: 0.9, // Very high win rate
        riskRewardRatio: 5.0 // Excellent risk/reward
      }
      
      const output = manager.calculatePositionSize(highEdgeInput)
      
      // Kelly formula: f = (bp - q) / b
      // p = 0.9, q = 0.1, b = 5.0
      // kellyFraction = (5*0.9 - 0.1)/5 = 4.4/5 = 0.88
      // With safety factor of 0.25, we get 0.88 * 0.25 = 0.22
      // This is less than max Kelly fraction (0.25), so it won't be capped
      // The test expectation was incorrect
      
      // Should apply Kelly formula with safety factor
      assert.ok(output.riskPercentage <= 0.25 * highEdgeInput.confidence)
      assert.ok(output.positionSize > 0)
      
      // Check that Kelly calculation was used
      assert.ok(output.reasoning.includes('Kelly'))
    })

    it('should handle edge case of 100% win rate', () => {
      manager.setActiveStrategy('kelly')
      const perfectInput = {
        ...baseInput,
        winRate: 1.0, // 100% win rate (unrealistic but should handle)
        riskRewardRatio: 2.0
      }
      
      const output = manager.calculatePositionSize(perfectInput)
      
      // Kelly strategy validates win rate must be between 0 and 1 (exclusive)
      // So 1.0 is considered invalid
      assert.equal(output.positionSize, 0)
      assert.ok(output.warnings.some(w => w.includes('Win rate must be between 0 and 1')))
      assert.equal(output.reasoning, 'Validation failed')
    })

    it('should handle near-100% win rate correctly', () => {
      manager.setActiveStrategy('kelly')
      const nearPerfectInput = {
        ...baseInput,
        winRate: 0.99, // 99% win rate (near perfect)
        riskRewardRatio: 2.0
      }
      
      const output = manager.calculatePositionSize(nearPerfectInput)
      
      // With 99% win rate, Kelly should give very high position size
      // f = (bp - q) / b = (2*0.99 - 0.01) / 2 = 1.97/2 = 0.985
      // With safety factor 0.25, we get 0.985 * 0.25 = 0.24625
      // This is just under the 0.25 cap
      
      assert.ok(output.positionSize > 0)
      assert.ok(output.riskPercentage <= 0.25) // Should be close to max
      assert.ok(output.reasoning.includes('Kelly'))
    })

    it('should actually cap Kelly fraction when it exceeds maximum', () => {
      manager.setActiveStrategy('kelly')
      // Create scenario where Kelly fraction will exceed cap
      // Need very high win rate and risk/reward to get raw Kelly > 1
      const extremeEdgeInput = {
        ...baseInput,
        winRate: 0.95, // 95% win rate
        riskRewardRatio: 10.0 // 10:1 risk/reward
      }
      
      // Kelly formula: f = (bp - q) / b
      // p = 0.95, q = 0.05, b = 10.0
      // rawKellyFraction = (10*0.95 - 0.05)/10 = 9.45/10 = 0.945
      // With safety factor of 0.25, we get 0.945 * 0.25 = 0.23625
      // This is still less than 0.25 cap, so let's use custom Kelly with higher safety factor
      
      // Create custom Kelly with higher safety factor to test capping
      const customKelly = new KellyCriterionStrategy(0.5, 0.2) // 50% safety, 20% max
      manager.registerStrategy('testKelly', customKelly)
      manager.setActiveStrategy('testKelly')
      
      const cappedOutput = manager.calculatePositionSize(extremeEdgeInput)
      
      // Now with 0.5 safety factor: 0.945 * 0.5 = 0.4725, which exceeds 0.2 cap
      assert.ok(cappedOutput.warnings.some(w => w.includes('capped at 20%')))
      assert.ok(cappedOutput.positionSize > 0)
    })
  })

  describe('Volatility Strategy Edge Cases', () => {
    it('should cap volatility multiplier at maximum', () => {
      manager.setActiveStrategy('volatility')
      const veryLowVolInput = {
        ...baseInput,
        marketConditions: {
          ...baseInput.marketConditions,
          volatility: 0.05 // Very low volatility
        }
      }
      
      const output = manager.calculatePositionSize(veryLowVolInput)
      
      // Multiplier should be capped (default max is 3)
      const maxMultiplier = 3
      const expectedMaxRisk = 0.01 * maxMultiplier * baseInput.confidence
      assert.ok(output.riskPercentage <= expectedMaxRisk * 1.1) // Allow small margin
    })

    it('should handle extreme volatility gracefully', () => {
      manager.setActiveStrategy('volatility')
      const extremeVolInput = {
        ...baseInput,
        marketConditions: {
          ...baseInput.marketConditions,
          volatility: 1.0 // Maximum volatility
        }
      }
      
      const output = manager.calculatePositionSize(extremeVolInput)
      
      // Should still produce valid position size, just very small
      assert.ok(output.positionSize > 0)
      assert.ok(output.riskPercentage < 0.01) // Should be well below base risk
    })
  })

  describe('Validation', () => {
    it('should handle invalid input', () => {
      const invalidInput = {
        ...baseInput,
        entryPrice: -100 // Invalid negative price
      }

      const output = manager.calculatePositionSize(invalidInput)
      assert.equal(output.positionSize, 0)
      assert.equal(output.confidence, 0)
      assert.ok(output.warnings.length > 0)
    })

    it('should handle sell orders correctly', () => {
      const sellInput = {
        ...baseInput,
        side: 'sell' as const,
        stopLoss: 51000 // Stop above entry for sell
      }

      const output = manager.calculatePositionSize(sellInput)
      assert.ok(output.positionSize > 0)
    })
  })

  describe('Static Methods', () => {
    it('should create risk parameters', () => {
      const riskParams = PositionSizingManager.createRiskParameters(50000, 10000, 2)
      
      assert.equal(riskParams.accountBalance, 50000)
      assert.equal(riskParams.currentExposure, 10000)
      assert.equal(riskParams.openPositions, 2)
      assert.equal(riskParams.maxRiskPerTrade, 0.02)
      assert.equal(riskParams.maxPortfolioRisk, 0.06)
    })

    it('should assess market conditions', () => {
      const recentPrices = Array.from({ length: 100 }, (_, i) => 50000 + Math.sin(i/10) * 1000)
      const conditions = PositionSizingManager.assessMarketConditions(
        50500,
        recentPrices,
        1500000,
        1000000
      )

      assert.ok(conditions.volatility >= 0 && conditions.volatility <= 1)
      assert.ok(conditions.relativeVolume > 0)
      assert.ok(['trending', 'ranging', 'volatile'].includes(conditions.regime))
      assert.ok(conditions.timeOfDayFactor >= 0 && conditions.timeOfDayFactor <= 1)
    })
  })
})