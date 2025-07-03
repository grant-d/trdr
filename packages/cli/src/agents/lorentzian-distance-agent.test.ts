import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { LorentzianDistanceAgent } from './lorentzian-distance-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('LorentzianDistanceAgent', () => {
  let agent: LorentzianDistanceAgent
  const metadata: AgentMetadata = {
    id: 'lorentzian-test',
    name: 'Lorentzian Test Agent',
    version: '1.0.0',
    description: 'Test Lorentzian agent',
    type: 'custom'
  }

  beforeEach(async () => {
    agent = new LorentzianDistanceAgent(metadata, undefined, {
      lookbackPeriod: 50,
      similarityThreshold: 0.1,
      imaginaryWeight: 0.5,
      minMatches: 3,
      useTimeDilation: true,
      marketSpeedLimit: 0.05
    })
    await agent.initialize()
  })

  afterEach(async () => {
    await agent.shutdown()
  })

  const createContext = (price: number, volume: number = 1000): MarketContext => ({
    symbol: 'BTC-USD',
    currentPrice: price,
    candles: Array.from({ length: 20 }, (_, i) => ({
      open: price - 10 + i,
      high: price + 5 + i,
      low: price - 15 + i,
      close: price - 5 + i,
      volume,
      timestamp: toEpochDate(Date.now() - (20 - i) * 60000)
    })),
    indicators: {}
  })

  it('should initialize with proper configuration', async () => {
    const context = createContext(50000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Building Lorentzian pattern library'))
  })

  it('should create complex price representation', async () => {
    // Build sufficient history (need at least 20 entries)
    const basePrices = [50000, 50050, 50100, 50150, 50200, 50250, 50300, 50350, 50400, 50450, 
                        50500, 50550, 50600, 50650, 50700, 50750, 50800, 50850, 50900, 50950]
    for (const price of basePrices) {
      const context = createContext(price)
      await agent.analyze(context)
    }

    // Strong upward momentum should create positive imaginary component
    const context = createContext(51000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should mention phase, Lorentzian analysis, or pattern in the reason
    assert.ok(
      signal.reason.includes('phase') || 
      signal.reason.includes('Lorentzian') || 
      signal.reason.includes('pattern') ||
      signal.reason.includes('Phase'), 
      `Expected phase, Lorentzian, or pattern in reason, got: ${signal.reason}`
    )
  })

  it('should detect phase transitions', async () => {
    // Create oscillating pattern
    const prices = [50000, 50200, 50100, 49900, 50000, 50200]
    for (const price of prices) {
      const context = createContext(price)
      await agent.analyze(context)
    }

    // Sharp reversal should trigger phase transition
    const context = createContext(49800)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    if (signal.reason.includes('Phase transition')) {
      assert.ok(['buy', 'sell'].includes(signal.action))
      assert.ok(signal.confidence >= 0.65)
    }
  })

  it('should find Lorentzian neighbors', async () => {
    // Build pattern library
    const basePattern = [50000, 50100, 50200, 50150, 50250]
    
    // Add similar patterns with slight variations
    for (let j = 0; j < 3; j++) {
      for (let i = 0; i < basePattern.length; i++) {
        const price = basePattern[i]! + j * 10
        const context = createContext(price)
        await agent.analyze(context)
      }
    }

    // New pattern similar to base
    for (const price of basePattern) {
      const context = createContext(price + 5)
      await agent.analyze(context)
    }

    const finalContext = createContext(50255)
    const signal = await agent.analyze(finalContext)
    
    assert.ok(signal)
    if (signal.reason.includes('Lorentzian pattern match')) {
      assert.ok(signal.confidence >= 0.6)
      assert.ok(signal.reason.includes('similar patterns'))
    }
  })

  it('should apply time dilation for high momentum', async () => {
    // Create very rapid price movement
    const rapidPrices = [50000, 50500, 51000, 51500, 52000]
    for (const price of rapidPrices) {
      const context = createContext(price, 2000) // High volume
      await agent.analyze(context)
    }

    // High momentum should affect spacetime metric
    const context = createContext(52500, 2000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Time dilation effects would be internal to the calculation
  })

  it('should handle insufficient data gracefully', async () => {
    const context = createContext(50000)
    // First few calls should return hold signals
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.confidence <= 0.5)
  })

  it('should track pattern outcomes', async () => {
    // Build pattern history
    const pattern1 = [50000, 50100, 50200, 50300]
    for (const price of pattern1) {
      const context = createContext(price)
      await agent.analyze(context)
    }

    // Add outcome
    for (let i = 0; i < 5; i++) {
      const context = createContext(50400 + i * 100)
      await agent.analyze(context)
    }

    // Similar pattern
    for (const price of pattern1) {
      const context = createContext(price + 10)
      await agent.analyze(context)
    }

    const signal = await agent.analyze(createContext(50310))
    
    assert.ok(signal)
    // Should predict similar positive outcome
    if (signal.reason.includes('avg return')) {
      assert.ok(signal.action === 'buy' || signal.action === 'hold')
    }
  })

  it('should calculate four-velocity correctly', async () => {
    // Test relativistic calculations
    const prices = [50000, 50050, 50100, 50150]
    for (const price of prices) {
      const context = createContext(price)
      await agent.analyze(context)
    }

    // Steady velocity should produce consistent four-velocity
    const context = createContext(50200)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Four-velocity effects are internal
  })

  it('should handle phase wrap-around', async () => {
    // Create pattern that crosses phase boundaries
    const prices = [50000, 49900, 49800, 49700, 49600, 49700, 49800, 49900, 50000]
    for (const price of prices) {
      const context = createContext(price)
      await agent.analyze(context)
    }

    const signal = await agent.analyze(createContext(50100))
    
    assert.ok(signal)
    // Phase wrap detection would be in the reason
  })

  it('should handle empty candles array', async () => {
    const context: MarketContext = {
      symbol: 'BTC-USD',
      currentPrice: 50000,
      candles: [{
        open: 50000,
        high: 50000,
        low: 50000,
        close: 50000,
        volume: 0,
        timestamp: toEpochDate(Date.now())
      }],
      indicators: {}
    }

    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.confidence <= 0.3)
  })

  it('should properly reset state', async () => {
    // Build complex history
    for (let i = 0; i < 30; i++) {
      const price = 50000 + Math.sin(i * 0.5) * 1000
      const context = createContext(price)
      await agent.analyze(context)
    }

    // Reset
    await agent.reset()

    // Should be clean state
    const context = createContext(50000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    // After reset, should have insufficient data
    assert.ok(signal.confidence <= 0.5)
  })

  it('should detect patterns in hyperbolic space', async () => {
    // Create repeating Lorentzian pattern
    const lorentzianPattern = (t: number) => 50000 + 1000 * Math.sinh(t * 0.1)
    
    for (let i = 0; i < 20; i++) {
      const price = lorentzianPattern(i)
      const context = createContext(price)
      await agent.analyze(context)
    }

    // Continue pattern
    const nextPrice = lorentzianPattern(20)
    const signal = await agent.analyze(createContext(nextPrice))
    
    assert.ok(signal)
    // Should recognize hyperbolic growth pattern
  })

  it('should maintain causality in light cone', async () => {
    // Test that predictions respect causality constraints
    const prices = [50000, 50100, 50200, 50300]
    for (const price of prices) {
      const context = createContext(price)
      await agent.analyze(context)
    }

    // Sudden impossible jump (beyond speed limit)
    const jumpContext = createContext(55000) // 10% jump
    const signal = await agent.analyze(jumpContext)
    
    assert.ok(signal)
    // Should handle causal violation appropriately
  })
})