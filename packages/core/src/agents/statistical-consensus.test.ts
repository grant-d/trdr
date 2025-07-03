import { describe, it, beforeEach } from 'node:test'
import assert from 'node:assert/strict'
import { 
  calculateStatisticalStopLoss, 
  calculateStatisticalLimitPrice,
  calculateConsensusPositionSize,
  enhanceConsensusWithPriceLevels
} from './statistical-consensus'
import type { AgentSignal, ConsensusResult } from './types'
import { epochDateNow } from '@trdr/shared'

describe('Statistical Consensus', () => {
  let mockSignals: Map<string, AgentSignal>
  let mockWeights: Map<string, number>
  
  beforeEach(() => {
    mockSignals = new Map([
      ['rsi', {
        action: 'buy',
        confidence: 0.8,
        reason: 'RSI oversold',
        stopLoss: 95,
        limitPrice: 101,
        positionSize: 0.05,
        timestamp: epochDateNow()
      }],
      ['macd', {
        action: 'buy',
        confidence: 0.7,
        reason: 'MACD bullish crossover',
        stopLoss: 93,
        limitPrice: 100.5,
        positionSize: 0.03,
        timestamp: epochDateNow()
      }],
      ['bollinger', {
        action: 'buy',
        confidence: 0.9,
        reason: 'Price at lower band',
        stopLoss: 94,
        limitPrice: 100.8,
        positionSize: 0.04,
        timestamp: epochDateNow()
      }]
    ])
    
    mockWeights = new Map([
      ['rsi', 1.0],
      ['macd', 0.8],
      ['bollinger', 1.2]
    ])
  })
  
  describe('calculateStatisticalStopLoss', () => {
    it('should calculate weighted stop loss for buy orders', () => {
      const result = calculateStatisticalStopLoss(mockSignals, mockWeights, 100, 'buy')
      
      assert.ok(result, 'Result should be defined')
      assert.ok(result.stopLoss < 100, 'Stop loss should be below current price for buy orders')
      assert.ok(result.confidence > 0, 'Confidence should be positive')
      assert.ok(result.stdDev >= 0, 'Standard deviation should be non-negative')
    })
    
    it('should calculate weighted stop loss for sell orders', () => {
      // Update signals for sell action
      for (const signal of mockSignals.values()) {
        ;(signal as any).action = 'sell'
        ;(signal as any).stopLoss = signal.stopLoss! + 10 // Stop loss above current price
      }
      
      const result = calculateStatisticalStopLoss(mockSignals, mockWeights, 100, 'sell')
      
      assert.ok(result, 'Result should be defined')
      assert.ok(result.stopLoss > 100, 'Stop loss should be above current price for sell orders')
    })
    
    it('should provide default stop loss when no agent signals include stop loss', () => {
      // Remove stop loss from all signals
      for (const signal of mockSignals.values()) {
        delete (signal as any).stopLoss
      }
      
      const result = calculateStatisticalStopLoss(mockSignals, mockWeights, 100, 'buy')
      
      assert.ok(result, 'Result should be defined')
      assert.ok(result.stopLoss < 100, 'Default stop loss should be below current price for buy')
      assert.ok(result.confidence < 0.8, 'Default should have lower confidence')
    })
  })
  
  describe('calculateStatisticalLimitPrice', () => {
    it('should calculate weighted limit price for buy orders', () => {
      const result = calculateStatisticalLimitPrice(mockSignals, mockWeights, 100, 'buy')
      
      assert.ok(result, 'Result should be defined')
      assert.ok(result.limitPrice <= 101.5, 'Limit price should be reasonable for buy orders')
      assert.ok(result.confidence > 0, 'Confidence should be positive')
    })
    
    it('should ensure buy limit price is not above current price', () => {
      // Set all limit prices above current price
      for (const signal of mockSignals.values()) {
        ;(signal as any).limitPrice = 105
      }
      
      const result = calculateStatisticalLimitPrice(mockSignals, mockWeights, 100, 'buy')
      
      assert.ok(result, 'Result should be defined')
      assert.ok(result.limitPrice <= 100, 'Buy limit should not exceed current price')
    })
    
    it('should ensure sell limit price is not below current price', () => {
      // Update for sell orders
      for (const signal of mockSignals.values()) {
        ;(signal as any).action = 'sell'
        ;(signal as any).limitPrice = 95 // Below current price
      }
      
      const result = calculateStatisticalLimitPrice(mockSignals, mockWeights, 100, 'sell')
      
      assert.ok(result, 'Result should be defined')
      assert.ok(result.limitPrice >= 100, 'Sell limit should not be below current price')
    })
  })
  
  describe('calculateConsensusPositionSize', () => {
    it('should calculate weighted position size', () => {
      const result = calculateConsensusPositionSize(mockSignals, mockWeights)
      
      assert.ok(result >= 0.01, 'Position size should be at least 1%')
      assert.ok(result <= 0.25, 'Position size should not exceed 25%')
    })
    
    it('should use default when no position sizes provided', () => {
      // Remove position sizes
      for (const signal of mockSignals.values()) {
        delete (signal as any).positionSize
      }
      
      const result = calculateConsensusPositionSize(mockSignals, mockWeights, 0.1)
      
      assert.ok(result > 0, 'Should return positive position size')
      assert.ok(result <= 0.1, 'Should be influenced by average confidence')
    })
  })
  
  describe('enhanceConsensusWithPriceLevels', () => {
    it('should enhance consensus with statistical price levels', () => {
      const baseConsensus: ConsensusResult = {
        action: 'buy',
        confidence: 0.8,
        reason: 'Test consensus',
        agentSignals: Object.fromEntries(mockSignals),
        agreement: 0.8,
        participatingAgents: 3,
        timestamp: epochDateNow()
      }
      
      const enhanced = enhanceConsensusWithPriceLevels(
        baseConsensus,
        mockSignals,
        mockWeights,
        100
      )
      
      assert.equal(enhanced.action, 'buy', 'Action should be preserved')
      assert.ok(enhanced.stopLoss, 'Stop loss should be calculated')
      assert.ok(enhanced.limitPrice, 'Limit price should be calculated')
      assert.ok(enhanced.positionSize, 'Position size should be calculated')
      assert.ok(enhanced.priceConfidence, 'Price confidence should be included')
      assert.ok(enhanced.priceConfidence.stopLoss.confidence > 0, 'Stop loss confidence should be positive')
      assert.ok(enhanced.priceConfidence.limitPrice.confidence > 0, 'Limit price confidence should be positive')
    })
    
    it('should not enhance hold signals', () => {
      const baseConsensus: ConsensusResult = {
        action: 'hold',
        confidence: 0.5,
        reason: 'No clear signal',
        agentSignals: {},
        agreement: 0.5,
        participatingAgents: 0,
        timestamp: epochDateNow()
      }
      
      const enhanced = enhanceConsensusWithPriceLevels(
        baseConsensus,
        new Map(),
        new Map(),
        100
      )
      
      assert.equal(enhanced.action, 'hold', 'Hold action should be preserved')
      assert.equal(enhanced.stopLoss, undefined, 'No stop loss for hold')
      assert.equal(enhanced.limitPrice, undefined, 'No limit price for hold')
    })
  })
})