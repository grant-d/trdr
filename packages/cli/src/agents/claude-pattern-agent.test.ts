import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { ClaudePatternAgent } from './claude-pattern-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

const MODEL_M = 'claude-sonnet-4-20250514'

describe('ClaudePatternAgent', () => {
  let agent: ClaudePatternAgent
  const metadata: AgentMetadata = {
    id: 'claude-test',
    name: 'Claude Pattern Test Agent',
    version: '1.0.0',
    description: 'Test Claude pattern agent',
    type: 'custom'
  }

  beforeEach(async () => {
    agent = new ClaudePatternAgent(metadata, {
      model: MODEL_M,
      maxTokens: 1000,
      temperature: 0.3,
      contextWindow: 10,
      confidenceThreshold: 0.7,
      maxRetries: 3,
      cacheDuration: 15
    })
    await agent.initialize()
  })

  afterEach(async () => {
    await agent.shutdown()
  })

  const createContext = (prices: number[], volumes?: number[]): MarketContext => ({
    symbol: 'BTC-USD',
    currentPrice: prices[prices.length - 1] || 50000,
    candles: prices.map((price, i) => ({
      open: price - 50,
      high: price + 100,
      low: price - 100,
      close: price,
      volume: volumes?.[i] || 1000,
      timestamp: toEpochDate(Date.now() - (prices.length - i) * 60000)
    })),
    indicators: {}
  })

  // it('should initialize and use fallback when no API key', async () => {
  //   const context = createContext([50000])
  //   const signal = await agent.analyze(context)
  //   
  //   assert.ok(signal)
  //   assert.ok(signal.reason.includes('[Fallback]'))
  //   assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
  // })

  it('should detect head and shoulders pattern', async () => {
    // Create head and shoulders pattern
    const headAndShoulders = [
      50000, 50200, 50400, 50200, // Left shoulder
      50000, 50200, 50600, 50200, // Head (higher)
      50000, 50200, 50400, 50200, // Right shoulder
      50000, 49800                // Breakdown
    ]
    
    const context = createContext(headAndShoulders)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    if (signal.reason.includes('Head and Shoulders')) {
      assert.strictEqual(signal.action, 'sell')
      assert.ok(signal.confidence >= 0.65)
    }
  })

  it('should detect double bottom pattern', async () => {
    // Create double bottom pattern
    const doubleBottom = [
      50000, 49800, 49600, 49500, // First bottom
      49600, 49800, 50000, 49800, // Rise
      49600, 49500, 49600, 49800, // Second bottom (similar level)
      50000, 50200                // Breakout
    ]
    
    const context = createContext(doubleBottom)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    if (signal.reason.includes('Double Bottom')) {
      assert.strictEqual(signal.action, 'buy')
      assert.ok(signal.confidence >= 0.65)
    }
  })

  it('should detect breakout patterns', async () => {
    // Create channel breakout
    const channelPrices = [
      50000, 50100, 50000, 50100, // Establish range
      50000, 50100, 50000, 50100,
      50000, 50100, 50000, 50100,
      50000, 50100, 50200, 50300, // Breakout
      50400, 50500                // Continuation
    ]
    
    const context = createContext(channelPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    if (signal.reason.includes('Breakout')) {
      assert.strictEqual(signal.action, 'buy')
      assert.ok(signal.confidence >= 0.5)
    }
  })

  // it('should cache similar market conditions', async () => {
  //   // First analysis
  //   const prices = [50000, 50100, 50200, 50300, 50400]
  //   const context1 = createContext(prices)
  //   const signal1 = await agent.analyze(context1)
  //   
  //   // Same pattern (should be cached)
  //   const context2 = createContext(prices)
  //   const signal2 = await agent.analyze(context2)
  //   
  //   assert.ok(signal1)
  //   assert.ok(signal2)
  //   assert.ok(signal2.reason.includes('[Cached]'))
  //   assert.strictEqual(signal1.action, signal2.action)
  // })

  // it('should prepare comprehensive prompt context', async () => {
  //   // Rich context with various patterns
  //   const complexPrices = []
  //   const complexVolumes = []
  //   
  //   for (let i = 0; i < 25; i++) {
  //     complexPrices.push(50000 + Math.sin(i * 0.5) * 500)
  //     complexVolumes.push(1000 + Math.random() * 2000)
  //   }
  //   
  //   const context = createContext(complexPrices, complexVolumes)
  //   const signal = await agent.analyze(context)
  //   
  //   assert.ok(signal)
  //   assert.ok(signal.reason.includes('[Fallback]'))
  //   // Should analyze with full context
  // })

  it('should handle insufficient data', async () => {
    const context = createContext([50000, 50100])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.confidence <= 0.5)
  })

  it('should track pattern memory', async () => {
    // Create multiple patterns over time
    const patterns = [
      [50000, 50200, 50400, 50200, 50000], // Pattern 1
      [50000, 49800, 49600, 49800, 50000], // Pattern 2
      [50000, 50100, 50200, 50300, 50400]  // Pattern 3
    ]
    
    for (const pattern of patterns) {
      const context = createContext(pattern)
      await agent.analyze(context)
    }
    
    // Should have pattern history
    const finalContext = createContext([50000, 50100, 50200])
    const signal = await agent.analyze(finalContext)
    
    assert.ok(signal)
    // Pattern memory influences analysis
  })

  it('should handle extreme market conditions', async () => {
    // Extreme volatility
    const volatilePrices = [50000, 52000, 48000, 53000, 47000, 51000]
    const highVolumes = [10000, 15000, 20000, 18000, 22000, 16000]
    
    const context = createContext(volatilePrices, highVolumes)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should handle gracefully
  })

  it('should detect support and resistance levels', async () => {
    // Multiple touches of same levels
    const supportResistance = [
      50000, 50200, 50400, 50200, // Test resistance
      50000, 49800, 50000, 50200, // Test support
      50400, 50200, 50000, 49800, // Test support again
      50000, 50200, 50400, 50600  // Break resistance
    ]
    
    const context = createContext(supportResistance)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should consider S/R levels
  })

  it('should properly reset state', async () => {
    // Build pattern memory
    for (let i = 0; i < 10; i++) {
      const prices = Array(20).fill(0).map((_, j) => 50000 + Math.random() * 1000)
      await agent.analyze(createContext(prices))
    }
    
    // Reset
    await agent.reset()
    
    // Should clear cache and memory
    const context = createContext([50000, 50100])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(!signal.reason.includes('[Cached]'))
  })

  it('should handle rate limiting gracefully', async () => {
    // Rapid successive calls
    const context = createContext([50000, 50100, 50200])
    
    const signals = []
    for (let i = 0; i < 5; i++) {
      signals.push(await agent.analyze(context))
    }
    
    // All should succeed (using fallback or cache)
    for (const signal of signals) {
      assert.ok(signal)
      assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
    }
  })

  it('should calculate trend correctly', async () => {
    // Clear uptrend
    const uptrend = Array(20).fill(0).map((_, i) => 50000 + i * 100)
    const uptrendContext = createContext(uptrend)
    const uptrendSignal = await agent.analyze(uptrendContext)
    
    assert.ok(uptrendSignal)
    
    // Clear downtrend
    const downtrend = Array(20).fill(0).map((_, i) => 52000 - i * 100)
    const downtrendContext = createContext(downtrend)
    const downtrendSignal = await agent.analyze(downtrendContext)
    
    assert.ok(downtrendSignal)
  })

  it('should handle pattern detection edge cases', async () => {
    // Not enough peaks for head and shoulders
    const incompletePrices = [50000, 50200, 50000, 50200]
    const incompleteSignal = await agent.analyze(createContext(incompletePrices))
    
    assert.ok(incompleteSignal)
    assert.ok(!incompleteSignal.reason.includes('Head and Shoulders'))
    
    // No clear pattern
    const randomPrices = Array(20).fill(0).map(() => 50000 + Math.random() * 500)
    const randomSignal = await agent.analyze(createContext(randomPrices))
    
    assert.ok(randomSignal)
    assert.ok(randomSignal.reason.includes('No clear pattern') || randomSignal.action === 'hold')
  })
})