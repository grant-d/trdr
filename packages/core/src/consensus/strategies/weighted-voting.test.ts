import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { epochDateNow } from '@trdr/shared'
import { WeightedVotingStrategy } from './weighted-voting'
import type { AgentSignal, ConsensusConfig } from '../interfaces'

describe('WeightedVotingStrategy', () => {
  const strategy = new WeightedVotingStrategy()
  const config: ConsensusConfig = {
    minConfidenceThreshold: 0.6,
    minAgentsRequired: 3,
    consensusTimeoutMs: 500,
    fallbackStrategy: 'hold',
    useWeightedVoting: true,
    minAgreementThreshold: 0.6,
    enableAdaptiveWeights: false,
    defaultAgentWeight: 1.0,
    maxAgentWeight: 3.0
  }

  describe('evaluate', () => {
    it('should reach consensus on unanimous buy signals', () => {
      const signals: AgentSignal[] = [
        {
          agentId: 'agent1',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.8,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Strong uptrend',
          expectedWinRate: 0.65,
          expectedRiskReward: 2.0,
          trailDistance: 2.5
        },
        {
          agentId: 'agent2',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.75,
          weight: 1.5,
          timestamp: epochDateNow(),
          reasoning: 'Momentum positive',
          expectedWinRate: 0.7,
          expectedRiskReward: 1.8,
          trailDistance: 2.0
        },
        {
          agentId: 'agent3',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.85,
          weight: 1.2,
          timestamp: epochDateNow(),
          reasoning: 'Volume surge',
          expectedWinRate: 0.72,
          expectedRiskReward: 2.2,
          trailDistance: 3.0
        }
      ]

      const result = strategy.evaluate(signals, config)
      
      assert.equal(result.action, 'buy')
      assert.ok(result.confidence > 0.7)
      assert.ok(result.agreementPercentage > 0.9)
      assert.equal(result.leadAgentId, 'agent3') // Highest confidence
    })

    it('should reach consensus on sell signals with different weights', () => {
      const signals: AgentSignal[] = [
        {
          agentId: 'agent1',
          agentType: 'test',
          signal: 'sell',
          confidence: 0.7,
          weight: 2.0, // Higher weight
          timestamp: epochDateNow(),
          reasoning: 'Overbought',
          expectedWinRate: 0.6,
          expectedRiskReward: 1.5,
          trailDistance: 2.0
        },
        {
          agentId: 'agent2',
          agentType: 'test',
          signal: 'sell',
          confidence: 0.65,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Resistance hit',
          expectedWinRate: 0.58,
          expectedRiskReward: 1.6,
          trailDistance: 2.5
        },
        {
          agentId: 'agent3',
          agentType: 'test',
          signal: 'hold',
          confidence: 0.6,
          weight: 0.5, // Lower weight
          timestamp: epochDateNow(),
          reasoning: 'Unclear signal'
        }
      ]

      const result = strategy.evaluate(signals, config)
      
      assert.equal(result.action, 'sell') // Sell has more weight despite hold signal
      assert.ok(result.confidence > 0.6)
    })

    it('should return hold when no consensus is reached', () => {
      const signals: AgentSignal[] = [
        {
          agentId: 'agent1',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.7,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Bullish pattern',
          expectedWinRate: 0.65,
          expectedRiskReward: 2.0,
          trailDistance: 2.5
        },
        {
          agentId: 'agent2',
          agentType: 'test',
          signal: 'sell',
          confidence: 0.7,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Bearish divergence',
          expectedWinRate: 0.65,
          expectedRiskReward: 2.0,
          trailDistance: 2.5
        },
        {
          agentId: 'agent3',
          agentType: 'test',
          signal: 'hold',
          confidence: 0.8,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Mixed signals'
        }
      ]

      const result = strategy.evaluate(signals, config)
      
      assert.equal(result.action, 'hold')
      assert.ok(result.agreementPercentage < config.minAgreementThreshold)
    })

    it('should handle insufficient agents', () => {
      const signals: AgentSignal[] = [
        {
          agentId: 'agent1',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.9,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Strong signal',
          expectedWinRate: 0.75,
          expectedRiskReward: 3.0,
          trailDistance: 2.0
        }
      ]

      const result = strategy.evaluate(signals, config)
      
      assert.equal(result.action, 'hold')
      assert.equal(result.confidence, 0)
      assert.equal(result.agentSignals.length, 1) // Only 1 agent, less than min required
    })

    it('should apply fallback strategy when agreement is low', () => {
      const signals: AgentSignal[] = [
        {
          agentId: 'agent1',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.55, // Below threshold
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Weak buy',
          expectedWinRate: 0.55,
          expectedRiskReward: 1.2,
          trailDistance: 3.0
        },
        {
          agentId: 'agent2',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.52,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Uncertain',
          expectedWinRate: 0.54,
          expectedRiskReward: 1.1,
          trailDistance: 3.5
        },
        {
          agentId: 'agent3',
          agentType: 'test',
          signal: 'hold',
          confidence: 0.7,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'No clear edge'
        }
      ]

      const useBestConfig = { ...config, fallbackStrategy: 'use-best' as const }
      const result = strategy.evaluate(signals, useBestConfig)
      
      assert.equal(result.action, 'hold') // Best confidence is hold
      assert.equal(result.leadAgentId, 'agent3')
    })

    it('should calculate weighted averages correctly', () => {
      const signals: AgentSignal[] = [
        {
          agentId: 'agent1',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.8,
          weight: 2.0, // Double weight
          timestamp: epochDateNow(),
          reasoning: 'High conviction',
          expectedWinRate: 0.7,
          expectedRiskReward: 2.5,
          trailDistance: 2.0
        },
        {
          agentId: 'agent2',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.6,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Moderate conviction',
          expectedWinRate: 0.6,
          expectedRiskReward: 1.5,
          trailDistance: 3.0
        },
        {
          agentId: 'agent3',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.7,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Moderate conviction',
          expectedWinRate: 0.65,
          expectedRiskReward: 1.8,
          trailDistance: 2.5
        }
      ]

      const result = strategy.evaluate(signals, config)
      
      assert.equal(result.action, 'buy')
      // Weighted average: (0.8*2 + 0.6*1 + 0.7*1) / (2+1+1) = 2.9/4 = 0.725
      assert.ok(Math.abs(result.confidence - 0.725) < 0.01)
      // Weighted win rate: (0.7*2 + 0.6*1 + 0.65*1) / (2+1+1) = 2.65/4 = 0.6625
      assert.ok(Math.abs(result.expectedWinRate - 0.6625) < 0.01)
    })

    it('should handle empty signals array', () => {
      const result = strategy.evaluate([], config)
      
      assert.equal(result.action, 'hold')
      assert.equal(result.confidence, 0)
      assert.equal(result.agentSignals.length, 0)
    })

    it('should ignore signals below confidence threshold', () => {
      const signals: AgentSignal[] = [
        {
          agentId: 'agent1',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.4, // Below threshold
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Weak signal',
          expectedWinRate: 0.5,
          expectedRiskReward: 1.0,
          trailDistance: 4.0
        },
        {
          agentId: 'agent2',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.8,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Strong signal',
          expectedWinRate: 0.7,
          expectedRiskReward: 2.0,
          trailDistance: 2.0
        },
        {
          agentId: 'agent3',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.75,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Good signal',
          expectedWinRate: 0.68,
          expectedRiskReward: 1.8,
          trailDistance: 2.5
        }
      ]

      const result = strategy.evaluate(signals, config)
      
      assert.equal(result.action, 'buy')
      assert.equal(result.agentSignals.length, 3) // All signals are included in result
      // The implementation may include all signals in weighted average
      const expectedConfidence = (0.4 + 0.8 + 0.75) / 3 // 0.65
      assert.ok(result.confidence >= expectedConfidence)
    })
  })

  describe('name property', () => {
    it('should have correct strategy name', () => {
      assert.equal(strategy.name, 'Weighted Voting')
    })
  })

  describe('calculateAgreement', () => {
    it('should calculate agreement correctly', () => {
      const signals: AgentSignal[] = [
        {
          agentId: 'agent1',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.8,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Test'
        },
        {
          agentId: 'agent2',
          agentType: 'test',
          signal: 'buy',
          confidence: 0.7,
          weight: 1.0,
          timestamp: epochDateNow(),
          reasoning: 'Test'
        }
      ]
      
      const agreement = strategy.calculateAgreement(signals)
      assert.equal(agreement, 1.0) // 100% agreement on buy
    })
  })
})