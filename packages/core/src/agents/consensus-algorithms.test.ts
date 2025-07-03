import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import {
  calculateConfidenceInterval,
  normalizeSignals,
  performanceWeightedConsensus,
  exponentialWeightedConsensus,
  confidenceWeightedConsensus,
  bayesianConsensus,
  vetoConsensus
} from './consensus-algorithms'
import type { AgentSignal, ITradeAgent } from './types'
import { BaseAgent } from './base-agent'
import { epochDateNow, toEpochDate } from '@trdr/shared'

// Mock agent for testing
class MockAgent extends BaseAgent {
  protected async onInitialize(): Promise<void> {}
  
  protected async performAnalysis(): Promise<AgentSignal> {
    return this.createSignal('hold', 0.5, 'Test signal')
  }
  
  // Override performance for testing
  setPerformance(winRate: number, sharpeRatio: number): void {
    this.performanceData.winRate = winRate
    this.performanceData.sharpeRatio = sharpeRatio
  }
}

describe('Consensus Algorithms', () => {
  describe('calculateConfidenceInterval', () => {
    it('should calculate confidence interval correctly', () => {
      const signals = new Map<string, AgentSignal>([
        ['agent1', { action: 'buy', confidence: 0.8, reason: 'Bullish', timestamp: epochDateNow() }],
        ['agent2', { action: 'buy', confidence: 0.7, reason: 'Also bullish', timestamp: epochDateNow() }],
        ['agent3', { action: 'buy', confidence: 0.9, reason: 'Very bullish', timestamp: epochDateNow() }]
      ])
      
      const weights = new Map([
        ['agent1', 1.0],
        ['agent2', 1.0],
        ['agent3', 1.0]
      ])
      
      const consensus = 0.8 // Average confidence
      const interval = calculateConfidenceInterval(signals, weights, consensus)
      
      assert.ok(interval.lower < consensus)
      assert.ok(interval.upper > consensus)
      assert.ok(interval.lower >= 0)
      assert.ok(interval.upper <= 1)
    })
  })
  
  describe('normalizeSignals', () => {
    it('should normalize signals within action groups', () => {
      const signals = new Map<string, AgentSignal>([
        ['agent1', { action: 'buy', confidence: 0.6, reason: 'Bullish', timestamp: epochDateNow() }],
        ['agent2', { action: 'buy', confidence: 0.9, reason: 'Very bullish', timestamp: epochDateNow() }],
        ['agent3', { action: 'sell', confidence: 0.5, reason: 'Bearish', timestamp: epochDateNow() }]
      ])
      
      const normalized = normalizeSignals(signals)
      
      // Buy signals should be normalized to 0 and 1
      const buySignals = Array.from(normalized.values()).filter(s => s.action === 'buy')
      const buyConfidences = buySignals.map(s => s.confidence)
      assert.ok(buyConfidences.includes(0))
      assert.ok(buyConfidences.includes(1))
      
      // Sell signal should remain unchanged (only one in group)
      const sellSignal = Array.from(normalized.values()).find(s => s.action === 'sell')
      assert.equal(sellSignal?.confidence, 0.5)
    })
  })
  
  describe('performanceWeightedConsensus', () => {
    it('should weight agents by performance', async () => {
      const agents = new Map<string, ITradeAgent>()
      
      // Create agents with different performance
      const goodAgent = new MockAgent({
        id: 'good-agent',
        name: 'Good Agent',
        version: '1.0.0',
        description: 'High performing',
        type: 'custom'
      })
      goodAgent.setPerformance(0.8, 1.5) // 80% win rate, 1.5 Sharpe
      
      const poorAgent = new MockAgent({
        id: 'poor-agent',
        name: 'Poor Agent',
        version: '1.0.0',
        description: 'Low performing',
        type: 'custom'
      })
      poorAgent.setPerformance(0.3, -0.5) // 30% win rate, negative Sharpe
      
      agents.set('good-agent', goodAgent)
      agents.set('poor-agent', poorAgent)
      
      const signals = new Map<string, AgentSignal>([
        ['good-agent', { action: 'buy', confidence: 0.8, reason: 'Strong buy', timestamp: epochDateNow() }],
        ['poor-agent', { action: 'sell', confidence: 0.7, reason: 'Sell signal', timestamp: epochDateNow() }]
      ])
      
      const weights = new Map([
        ['good-agent', 1.0],
        ['poor-agent', 1.0]
      ])
      
      const consensus = performanceWeightedConsensus.calculateConsensus(signals, weights, agents)
      
      // Good agent should dominate due to better performance
      assert.equal(consensus.action, 'buy')
      assert.ok(consensus.reason.includes('w='))
      assert.ok(consensus.confidenceInterval)
    })
  })
  
  describe('exponentialWeightedConsensus', () => {
    it('should give more weight to recent signals', () => {
      const now = epochDateNow()
      const oldTimestamp = toEpochDate(Date.now() - 10 * 60 * 1000) // 10 minutes ago
      
      const signals = new Map<string, AgentSignal>([
        ['recent-agent', { action: 'buy', confidence: 0.7, reason: 'Recent signal', timestamp: now }],
        ['old-agent', { action: 'sell', confidence: 0.9, reason: 'Old signal', timestamp: oldTimestamp }]
      ])
      
      const weights = new Map([
        ['recent-agent', 1.0],
        ['old-agent', 1.0]
      ])
      
      const consensus = exponentialWeightedConsensus.calculateConsensus(signals, weights)
      
      // Recent signal should have more influence despite lower confidence
      assert.equal(consensus.action, 'buy')
    })
  })
  
  describe('confidenceWeightedConsensus', () => {
    it('should give more weight to high confidence signals', () => {
      const signals = new Map<string, AgentSignal>([
        ['confident-agent', { action: 'buy', confidence: 0.95, reason: 'Very confident', timestamp: epochDateNow() }],
        ['unsure-agent1', { action: 'sell', confidence: 0.55, reason: 'Not sure', timestamp: epochDateNow() }],
        ['unsure-agent2', { action: 'sell', confidence: 0.60, reason: 'Also unsure', timestamp: epochDateNow() }]
      ])
      
      const weights = new Map([
        ['confident-agent', 1.0],
        ['unsure-agent1', 1.0],
        ['unsure-agent2', 1.0]
      ])
      
      const consensus = confidenceWeightedConsensus.calculateConsensus(signals, weights)
      
      // High confidence signal should win despite being outnumbered
      assert.equal(consensus.action, 'buy')
      assert.ok(consensus.reason.includes('95%'))
    })
  })
  
  describe('bayesianConsensus', () => {
    it('should incorporate prior beliefs', () => {
      const signals = new Map<string, AgentSignal>([
        ['agent1', { action: 'buy', confidence: 0.6, reason: 'Weak buy', timestamp: epochDateNow() }],
        ['agent2', { action: 'hold', confidence: 0.7, reason: 'Hold', timestamp: epochDateNow() }]
      ])
      
      const weights = new Map([
        ['agent1', 1.0],
        ['agent2', 1.0]
      ])
      
      // Strong prior for selling
      const priors = { buy: 0.2, sell: 0.5, hold: 0.3 }
      
      const consensus = bayesianConsensus.calculateConsensus(signals, weights, undefined, priors)
      
      assert.ok(consensus.posteriorProbabilities)
      assert.ok(consensus.reason.includes('Bayesian posterior'))
      
      // Should have posterior probabilities for all actions
      assert.ok('buy' in consensus.posteriorProbabilities)
      assert.ok('sell' in consensus.posteriorProbabilities)
      assert.ok('hold' in consensus.posteriorProbabilities)
    })
  })
  
  describe('vetoConsensus', () => {
    it('should apply veto for high confidence dissenting signals', () => {
      const signals = new Map<string, AgentSignal>([
        ['agent1', { action: 'buy', confidence: 0.7, reason: 'Buy signal', timestamp: epochDateNow() }],
        ['agent2', { action: 'buy', confidence: 0.6, reason: 'Also buy', timestamp: epochDateNow() }],
        ['veto-agent', { action: 'sell', confidence: 0.95, reason: 'DANGER!', timestamp: epochDateNow() }]
      ])
      
      const weights = new Map([
        ['agent1', 1.0],
        ['agent2', 1.0],
        ['veto-agent', 1.0]
      ])
      
      const consensus = vetoConsensus.calculateConsensus(signals, weights)
      
      // Veto should override majority
      assert.equal(consensus.action, 'sell')
      assert.ok(consensus.reason.includes('VETO'))
      assert.equal(consensus.vetoApplied, true)
    })
    
    it('should use weighted voting when no veto applies', () => {
      const signals = new Map<string, AgentSignal>([
        ['agent1', { action: 'buy', confidence: 0.7, reason: 'Buy signal', timestamp: epochDateNow() }],
        ['agent2', { action: 'buy', confidence: 0.6, reason: 'Also buy', timestamp: epochDateNow() }],
        ['agent3', { action: 'sell', confidence: 0.8, reason: 'Sell signal', timestamp: epochDateNow() }]
      ])
      
      const weights = new Map([
        ['agent1', 1.0],
        ['agent2', 1.0],
        ['agent3', 1.0]
      ])
      
      const consensus = vetoConsensus.calculateConsensus(signals, weights)
      
      // No veto, majority wins
      assert.equal(consensus.action, 'buy')
      assert.equal(consensus.vetoApplied, false)
    })
  })
})