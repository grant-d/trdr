import { describe, it, beforeEach } from 'node:test'
import assert from 'node:assert/strict'
import { PerformanceTracker } from './performance-tracker'
import type { AgentSignal, Trade } from './types'
import type { TradeOutcome } from './performance-tracker'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import { epochDateNow, toEpochDate } from '@trdr/shared'

describe('PerformanceTracker', () => {
  let tracker: PerformanceTracker
  let eventBus: EventBus
  
  beforeEach(() => {
    eventBus = EventBus.getInstance()
    Object.values(EventTypes).forEach(type => {
      eventBus.registerEvent(type)
    })
    
    tracker = new PerformanceTracker(eventBus, {
      minSignalsForAdjustment: 2, // Lower for testing
      adjustmentSensitivity: 0.5
    })
  })
  
  describe('signal recording', () => {
    it('should record agent signals', () => {
      const signal: AgentSignal = {
        action: 'buy',
        confidence: 0.8,
        reason: 'Test signal',
        timestamp: epochDateNow()
      }
      
      tracker.recordSignal('agent1', signal, 100)
      
      const history = tracker.getPerformanceHistory('agent1')
      assert.ok(history)
      assert.equal(history.metrics.length, 1)
      assert.equal(history.metrics[0]!.signal.action, 'buy')
      assert.equal(history.metrics[0]!.executionTime, 100)
    })
    
    it('should maintain max metrics limit', () => {
      const customTracker = new PerformanceTracker(eventBus, {
        maxMetricsPerAgent: 3
      })
      
      // Record 5 signals
      for (let i = 0; i < 5; i++) {
        const signal: AgentSignal = {
          action: 'buy',
          confidence: 0.8,
          reason: `Signal ${i}`,
          timestamp: epochDateNow()
        }
        customTracker.recordSignal('agent1', signal, 100)
      }
      
      const history = customTracker.getPerformanceHistory('agent1')
      assert.equal(history?.metrics.length, 3) // Should only keep last 3
      assert.equal(history?.metrics[0]?.signal.reason, 'Signal 2')
    })
  })
  
  describe('trade outcome recording', () => {
    it('should record trade outcomes', () => {
      const signalTime = epochDateNow()
      const signal: AgentSignal = {
        action: 'buy',
        confidence: 0.8,
        reason: 'Test signal',
        timestamp: signalTime
      }
      
      tracker.recordSignal('agent1', signal, 100)
      
      const outcome: TradeOutcome = {
        trade: {
          id: 'trade1',
          timestamp: epochDateNow(),
          side: 'sell',
          price: 105,
          size: 1,
          pnl: 5,
          fee: 0.1
        },
        pnl: 5,
        pnlPercent: 0.05,
        holdTime: 60000, // 1 minute
        maxDrawdown: 0.02,
        success: true
      }
      
      tracker.recordTradeOutcome('agent1', signalTime, outcome)
      
      const history = tracker.getPerformanceHistory('agent1')
      assert.ok(history?.metrics[0]?.outcome)
      assert.equal(history?.metrics[0]?.outcome?.pnl, 5)
      assert.equal(history?.metrics[0]?.outcome?.success, true)
    })
    
    it('should update summary after recording outcome', () => {
      const signalTime = epochDateNow()
      const signal: AgentSignal = {
        action: 'buy',
        confidence: 0.8,
        reason: 'Test signal',
        timestamp: signalTime
      }
      
      tracker.recordSignal('agent1', signal, 100)
      
      const outcome: TradeOutcome = {
        trade: {} as Trade,
        pnl: 10,
        pnlPercent: 0.1,
        holdTime: 60000,
        maxDrawdown: 0.02,
        success: true
      }
      
      tracker.recordTradeOutcome('agent1', signalTime, outcome)
      
      const summary = tracker.getPerformanceSummary('agent1')
      assert.ok(summary)
      assert.equal(summary.totalSignals, 1)
      assert.equal(summary.successfulSignals, 1)
      assert.equal(summary.winRate, 1)
      assert.equal(summary.averageReturn, 0.1)
    })
  })
  
  describe('weight adjustment calculations', () => {
    it('should calculate weight adjustments based on performance', () => {
      // Create two agents with different performance
      setupAgentWithPerformance('agent1', [
        { pnl: 10, pnlPercent: 0.1, success: true },
        { pnl: 5, pnlPercent: 0.05, success: true },
        { pnl: -2, pnlPercent: -0.02, success: false }
      ])
      
      setupAgentWithPerformance('agent2', [
        { pnl: -5, pnlPercent: -0.05, success: false },
        { pnl: -3, pnlPercent: -0.03, success: false },
        { pnl: 1, pnlPercent: 0.01, success: true }
      ])
      
      const currentWeights = new Map([
        ['agent1', 1.0],
        ['agent2', 1.0]
      ])
      
      const adjustments = tracker.calculateWeightAdjustments(currentWeights)
      
      assert.equal(adjustments.size, 2)
      
      const agent1Adj = adjustments.get('agent1')
      const agent2Adj = adjustments.get('agent2')
      
      assert.ok(agent1Adj)
      assert.ok(agent2Adj)
      
      
      // Agent 1 should have higher weight due to better performance
      assert.ok(agent1Adj.newWeight > agent1Adj.oldWeight)
      assert.ok(agent2Adj.newWeight < agent2Adj.oldWeight)
    })
    
    it('should respect min/max weight multipliers', () => {
      const customTracker = new PerformanceTracker(eventBus, {
        minSignalsForAdjustment: 1,
        maxWeightMultiplier: 1.5,
        minWeightMultiplier: 0.5,
        adjustmentSensitivity: 1.0 // Max sensitivity
      })
      
      // Setup agent with excellent performance
      const signal: AgentSignal = {
        action: 'buy',
        confidence: 0.9,
        reason: 'Test',
        timestamp: epochDateNow()
      }
      
      customTracker.recordSignal('agent1', signal, 100)
      customTracker.recordTradeOutcome('agent1', signal.timestamp, {
        trade: {} as Trade,
        pnl: 100,
        pnlPercent: 1.0, // 100% return
        holdTime: 60000,
        maxDrawdown: 0,
        success: true
      })
      
      const adjustments = customTracker.calculateWeightAdjustments(
        new Map([['agent1', 1.0]])
      )
      
      const adj = adjustments.get('agent1')
      assert.ok(adj)
      assert.ok(adj.newWeight <= 1.5) // Should not exceed max multiplier
    })
  })
  
  describe('performance metrics calculation', () => {
    it('should calculate Sharpe ratio correctly', () => {
      setupAgentWithPerformance('agent1', [
        { pnl: 10, pnlPercent: 0.1, success: true },
        { pnl: 5, pnlPercent: 0.05, success: true },
        { pnl: -2, pnlPercent: -0.02, success: false },
        { pnl: 8, pnlPercent: 0.08, success: true },
        { pnl: -1, pnlPercent: -0.01, success: false }
      ])
      
      const summary = tracker.getPerformanceSummary('agent1')
      assert.ok(summary)
      assert.ok(summary.sharpeRatio !== 0)
      // Sharpe should be positive for profitable strategy
      assert.ok(summary.sharpeRatio > 0)
    })
    
    it('should calculate max drawdown correctly', () => {
      setupAgentWithPerformance('agent1', [
        { pnl: 10, pnlPercent: 0.1, success: true },   // 10%
        { pnl: 5, pnlPercent: 0.05, success: true },   // 15%
        { pnl: -8, pnlPercent: -0.08, success: false }, // 7%
        { pnl: -5, pnlPercent: -0.05, success: false }, // 2%
        { pnl: 3, pnlPercent: 0.03, success: true }     // 5%
      ])
      
      const summary = tracker.getPerformanceSummary('agent1')
      assert.ok(summary)
      // Max drawdown from 15% to 2% = 13/115 ≈ 0.113
      assert.ok(summary.maxDrawdown > 0.1)
      assert.ok(summary.maxDrawdown < 0.15)
    })
    
    it('should calculate profit factor correctly', () => {
      setupAgentWithPerformance('agent1', [
        { pnl: 10, pnlPercent: 0.1, success: true },
        { pnl: 5, pnlPercent: 0.05, success: true },
        { pnl: -3, pnlPercent: -0.03, success: false },
        { pnl: -2, pnlPercent: -0.02, success: false }
      ])
      
      const summary = tracker.getPerformanceSummary('agent1')
      assert.ok(summary)
      // Profit factor = gross profit / gross loss = 15 / 5 = 3
      assert.equal(summary.profitFactor, 3)
    })
    
    it('should calculate consistency score', () => {
      // Consistent returns
      setupAgentWithPerformance('consistent', [
        { pnl: 5, pnlPercent: 0.05, success: true },
        { pnl: 4, pnlPercent: 0.04, success: true },
        { pnl: 6, pnlPercent: 0.06, success: true },
        { pnl: 5, pnlPercent: 0.05, success: true }
      ])
      
      // Inconsistent returns
      const tracker2 = new PerformanceTracker(eventBus, { minSignalsForAdjustment: 1 })
      setupAgentWithPerformance('inconsistent', [
        { pnl: 50, pnlPercent: 0.5, success: true },
        { pnl: -40, pnlPercent: -0.4, success: false },
        { pnl: 30, pnlPercent: 0.3, success: true },
        { pnl: -20, pnlPercent: -0.2, success: false }
      ], tracker2)
      
      const consistentSummary = tracker.getPerformanceSummary('consistent')
      const inconsistentSummary = tracker2.getPerformanceSummary('inconsistent')
      
      assert.ok(consistentSummary)
      assert.ok(inconsistentSummary)
      
      // Consistent agent should have higher consistency score
      assert.ok(consistentSummary.consistency > inconsistentSummary.consistency)
      assert.ok(consistentSummary.consistency > 0.8)
      assert.ok(inconsistentSummary.consistency < 0.5)
    })
  })
  
  describe('recent performance tracking', () => {
    it('should track recent performance windows', () => {
      const now = epochDateNow()
      
      // Add signals at different times
      const signals = [
        { timestamp: now - 2 * 60 * 60 * 1000, pnl: 10 }, // 2 hours ago
        { timestamp: now - 25 * 60 * 60 * 1000, pnl: -5 }, // 25 hours ago
        { timestamp: now - 5 * 24 * 60 * 60 * 1000, pnl: 15 }, // 5 days ago
        { timestamp: now - 10 * 24 * 60 * 60 * 1000, pnl: -3 }, // 10 days ago
        { timestamp: now - 40 * 24 * 60 * 60 * 1000, pnl: 20 } // 40 days ago
      ]
      
      for (const sig of signals) {
        const signal: AgentSignal = {
          action: 'buy',
          confidence: 0.8,
          reason: 'Test',
          timestamp: toEpochDate(sig.timestamp)
        }
        
        tracker.recordSignal('agent1', signal, 100)
        tracker.recordTradeOutcome('agent1', signal.timestamp, {
          trade: {} as Trade,
          pnl: sig.pnl,
          pnlPercent: sig.pnl / 100,
          holdTime: 60000,
          maxDrawdown: 0.02,
          success: sig.pnl > 0
        })
      }
      
      const summary = tracker.getPerformanceSummary('agent1')
      assert.ok(summary)
      
      // Check recent performance windows
      assert.equal(summary.recentPerformance.last24h.signals, 1) // Only 2 hours ago
      assert.equal(summary.recentPerformance.last7d.signals, 3) // 2 hours, 25 hours, and 5 days
      assert.equal(summary.recentPerformance.last30d.signals, 4) // Not 40 days
    })
  })
  
  describe('data management', () => {
    it('should clear agent data', () => {
      setupAgentWithPerformance('agent1', [
        { pnl: 10, pnlPercent: 0.1, success: true }
      ])
      
      assert.ok(tracker.getPerformanceHistory('agent1'))
      
      tracker.clearAgentData('agent1')
      
      assert.equal(tracker.getPerformanceHistory('agent1'), undefined)
    })
    
    it('should export performance data', () => {
      setupAgentWithPerformance('agent1', [
        { pnl: 10, pnlPercent: 0.1, success: true }
      ])
      
      setupAgentWithPerformance('agent2', [
        { pnl: 5, pnlPercent: 0.05, success: true }
      ])
      
      const exported = tracker.exportPerformanceData()
      
      assert.ok(exported.agent1)
      assert.ok(exported.agent2)
      assert.equal(exported.agent1.metrics.length, 1)
      assert.equal(exported.agent2.metrics.length, 1)
    })
  })
  
  // Helper function to setup agent with performance
  function setupAgentWithPerformance(
    agentId: string,
    outcomes: Array<Partial<TradeOutcome>>,
    customTracker?: PerformanceTracker
  ): void {
    const t = customTracker || tracker
    const now = epochDateNow()
    
    for (let i = 0; i < outcomes.length; i++) {
      const outcome = outcomes[i]!
      // Spread signals over time to avoid timestamp collision
      const signal: AgentSignal = {
        action: 'buy',
        confidence: 0.8,
        reason: 'Test signal',
        timestamp: toEpochDate(now - (outcomes.length - i) * 1000) // Each signal 1 second apart
      }
      
      t.recordSignal(agentId, signal, 100)
      
      t.recordTradeOutcome(agentId, signal.timestamp, {
        trade: {} as Trade,
        pnl: outcome.pnl || 0,
        pnlPercent: outcome.pnlPercent || 0,
        holdTime: outcome.holdTime || 60000,
        maxDrawdown: outcome.maxDrawdown || 0,
        success: outcome.success || false
      })
    }
  }
})