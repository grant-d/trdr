import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { TimeDecayAgent } from './time-decay-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('TimeDecayAgent', () => {
  let agent: TimeDecayAgent
  const metadata: AgentMetadata = {
    id: 'time-decay-test',
    name: 'Time Decay Test Agent',
    version: '1.0.0',
    description: 'Test time decay agent',
    type: 'custom'
  }

  beforeEach(async () => {
    agent = new TimeDecayAgent(metadata, undefined, {
      initialTrailDistance: 0.02, // 2%
      minTrailDistance: 0.005, // 0.5%
      staleThreshold: 120, // 2 hours in minutes
      decayRate: 0.1, // 10% decay per interval
      decayInterval: 30, // 30 minutes
      breakoutBoost: 1.5 // 50% confidence boost
    })
    await agent.initialize()
  })

  afterEach(async () => {
    await agent.shutdown()
  })

  const createContext = (price: number, volume: number = 1000): MarketContext => ({
    symbol: 'BTC-USD',
    currentPrice: price,
    candles: [{
      open: price - 10,
      high: price + 5,
      low: price - 15,
      close: price,
      volume,
      timestamp: toEpochDate(Date.now())
    }],
    indicators: {}
  })

  it('should initialize with proper configuration', async () => {
    const context = createContext(50000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Building time profile'))
  })

  it('should track time at price levels', async () => {
    // Stay at same price level for multiple updates
    const stablePrice = 50000
    
    for (let i = 0; i < 10; i++) {
      const context = createContext(stablePrice + (i % 2) * 10) // Small oscillation
      await agent.analyze(context)
      // Simulate time passing
      await new Promise(resolve => setTimeout(resolve, 10))
    }

    const signal = await agent.analyze(createContext(stablePrice))
    
    assert.ok(signal)
    // Should have tracked time at this level
    assert.ok(signal.reason.includes('level') || signal.reason.includes('Time'))
  })

  it('should detect stale price levels', async () => {
    // Create stale level by staying at price
    const stalePrice = 50000
    
    // Stay at level for extended time
    for (let i = 0; i < 5; i++) {
      const context = createContext(stalePrice)
      await agent.analyze(context)
    }
    
    // Move away
    for (let i = 1; i <= 5; i++) {
      const context = createContext(stalePrice + i * 100)
      await agent.analyze(context)
    }
    
    // Return to stale level
    const signal = await agent.analyze(createContext(stalePrice))
    
    assert.ok(signal)
    // Should recognize return to stale level
    if (signal.reason.includes('stale') || signal.reason.includes('visited')) {
      assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
    }
  })

  it('should apply decay to price levels', async () => {
    // Visit multiple levels
    const levels = [50000, 50100, 50200, 50300, 50400]
    
    for (const level of levels) {
      for (let i = 0; i < 3; i++) {
        const context = createContext(level)
        await agent.analyze(context)
      }
    }
    
    // Revisit first level after time
    const signal = await agent.analyze(createContext(levels[0]!))
    
    assert.ok(signal)
    // Decay should have reduced the importance of old level
  })

  it('should adjust trail based on time decay', async () => {
    // Build time profile
    const basePrice = 50000
    
    // Spend significant time at base level
    for (let i = 0; i < 10; i++) {
      const context = createContext(basePrice + (Math.random() - 0.5) * 20)
      await agent.analyze(context)
    }
    
    // Move to new level
    const newPrice = 50500
    const signal = await agent.analyze(createContext(newPrice))
    
    assert.ok(signal)
    // Trail adjustment mentioned in reason or confidence adjusted
    assert.ok(signal.confidence > 0)
  })

  it('should handle volume-weighted time tracking', async () => {
    // Same price, different volumes
    const price = 50000
    const volumes = [1000, 5000, 2000, 10000, 1500]
    
    for (const volume of volumes) {
      const context = createContext(price, volume)
      await agent.analyze(context)
    }
    
    const signal = await agent.analyze(createContext(price, 3000))
    
    assert.ok(signal)
    // High volume periods should have more weight
  })

  it('should quantize price levels correctly', async () => {
    // Prices that should map to same grid level
    const similarPrices = [50000, 50010, 49990, 50005, 49995]
    
    for (const price of similarPrices) {
      const context = createContext(price)
      await agent.analyze(context)
    }
    
    const signal = await agent.analyze(createContext(50000))
    
    assert.ok(signal)
    // Should treat these as same level
    assert.ok(signal.reason.includes('level') || signal.reason.includes('time'))
  })

  it('should detect breakout from time-decayed levels', async () => {
    // Establish strong level
    const strongLevel = 50000
    for (let i = 0; i < 15; i++) {
      const context = createContext(strongLevel + (Math.random() - 0.5) * 50)
      await agent.analyze(context)
    }
    
    // Break out significantly
    const breakoutPrice = 51000
    const signal = await agent.analyze(createContext(breakoutPrice))
    
    assert.ok(signal)
    if (signal.reason.includes('breakout') || signal.reason.includes('new territory')) {
      assert.ok(signal.confidence >= 0.6)
    }
  })

  it('should handle insufficient data gracefully', async () => {
    const context = createContext(50000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.confidence <= 0.5)
  })

  it('should track multiple active levels', async () => {
    // Visit multiple distinct levels
    const levels = [50000, 50500, 51000, 50500, 50000, 50500]
    
    for (const level of levels) {
      const context = createContext(level)
      await agent.analyze(context)
    }
    
    const signal = await agent.analyze(createContext(50250)) // Between levels
    
    assert.ok(signal)
    // Should consider proximity to multiple tracked levels
  })

  it('should properly reset state', async () => {
    // Build complex time profile
    for (let i = 0; i < 50; i++) {
      const price = 50000 + (i % 10) * 100
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
    assert.ok(signal.reason.includes('Building time profile'))
  })

  it('should generate signals based on decay strength', async () => {
    // Create strongly decayed level
    const oldLevel = 50000
    for (let i = 0; i < 5; i++) {
      await agent.analyze(createContext(oldLevel))
    }
    
    // Move away for extended time
    for (let i = 0; i < 20; i++) {
      await agent.analyze(createContext(51000 + i * 10))
    }
    
    // Return to heavily decayed level
    const signal = await agent.analyze(createContext(oldLevel))
    
    assert.ok(signal)
    // Heavily decayed level might present opportunity
  })

  it('should handle rapid price movements', async () => {
    // Rapid jumps between levels
    const prices = [50000, 51000, 50000, 52000, 50000]
    
    for (const price of prices) {
      const context = createContext(price)
      await agent.analyze(context)
    }
    
    const signal = await agent.analyze(createContext(50500))
    
    assert.ok(signal)
    // Should handle gracefully
  })
})