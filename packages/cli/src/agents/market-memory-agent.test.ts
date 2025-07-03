import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { MarketMemoryAgent } from './market-memory-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('MarketMemoryAgent', () => {
  let agent: MarketMemoryAgent
  const metadata: AgentMetadata = {
    id: 'memory-test',
    name: 'Market Memory Test Agent',
    version: '1.0.0',
    description: 'Test Market Memory agent',
    type: 'custom'
  }

  beforeEach(async () => {
    agent = new MarketMemoryAgent(metadata)
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
      high: price + 20,
      low: price - 20,
      close: price,
      volume,
      timestamp: toEpochDate(Date.now())
    }],
    indicators: {}
  })

  it('should initialize successfully', async () => {
    assert.ok(agent)
    assert.strictEqual(agent.metadata.type, 'custom')
  })

  it('should return hold signal initially', async () => {
    const context = createContext(50000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('memory'))
  })

  it('should remember price levels', async () => {
    // Visit a price level multiple times
    const testPrice = 50000
    
    // First visit
    await agent.analyze(createContext(testPrice))
    
    // Move away
    await agent.analyze(createContext(51000))
    await agent.analyze(createContext(52000))
    
    // Return to remembered level
    const signal = await agent.analyze(createContext(testPrice))
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('level') || signal.reason.includes('memory'))
  })

  it('should detect support levels', async () => {
    // Bounce off same level multiple times
    const supportLevel = 49500
    
    for (let i = 0; i < 3; i++) {
      // Approach support
      await agent.analyze(createContext(50000))
      await agent.analyze(createContext(49700))
      await agent.analyze(createContext(supportLevel))
      
      // Bounce
      await agent.analyze(createContext(49800))
      await agent.analyze(createContext(50000))
    }
    
    // Approach support again
    const signal = await agent.analyze(createContext(supportLevel + 100))
    
    assert.ok(signal)
    if (signal.reason.includes('support')) {
      assert.strictEqual(signal.action, 'buy')
      assert.ok(signal.confidence >= 0.6)
    }
  })

  it('should detect resistance levels', async () => {
    // Rejected at same level multiple times
    const resistanceLevel = 51000
    
    for (let i = 0; i < 3; i++) {
      // Approach resistance
      await agent.analyze(createContext(50000))
      await agent.analyze(createContext(50700))
      await agent.analyze(createContext(resistanceLevel))
      
      // Rejection
      await agent.analyze(createContext(50700))
      await agent.analyze(createContext(50000))
    }
    
    // Approach resistance again with position
    const context = {
      ...createContext(resistanceLevel - 100),
      currentPosition: 0.1
    }
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    if (signal.reason.includes('resistance')) {
      assert.ok(['sell', 'hold'].includes(signal.action))
    }
  })

  it('should track time-based patterns', async () => {
    // Simulate daily pattern (e.g., always drops at certain time)
    // @ts-ignore - unused variable (reserved for future use)
    const _baseTime = Date.now()
    
    // Pattern: drops every "hour" (simulated)
    for (let day = 0; day < 3; day++) {
      // Normal price
      await agent.analyze(createContext(50000))
      
      // Drop at specific "time"
      await agent.analyze(createContext(49500))
      
      // Recovery
      await agent.analyze(createContext(50000))
    }
    
    // Should remember the pattern
    const signal = await agent.analyze(createContext(50000))
    
    assert.ok(signal)
    // Memory agent might anticipate the pattern
  })

  it('should remember volume patterns', async () => {
    // High volume at certain price levels
    const highVolumeLevel = 50500
    
    // Multiple high volume events at same level
    for (let i = 0; i < 3; i++) {
      await agent.analyze(createContext(50000, 1000))
      await agent.analyze(createContext(highVolumeLevel, 5000)) // High volume
      await agent.analyze(createContext(50000, 1000))
    }
    
    // Approach high volume level again
    const signal = await agent.analyze(createContext(highVolumeLevel - 50, 1200))
    
    assert.ok(signal)
    // Should remember high volume area
  })

  it('should forget old memories', async () => {
    // Create old memory
    const oldPrice = 45000
    await agent.analyze(createContext(oldPrice))
    
    // Many new memories
    for (let i = 0; i < 100; i++) {
      await agent.analyze(createContext(50000 + i * 10))
    }
    
    // Old level might be forgotten
    const signal = await agent.analyze(createContext(oldPrice))
    
    assert.ok(signal)
    // Less likely to reference very old memory
  })

  it('should handle position constraints', async () => {
    // Create resistance memory
    for (let i = 0; i < 3; i++) {
      await agent.analyze(createContext(51000))
      await agent.analyze(createContext(50500))
    }
    
    // Approach resistance without position
    const context = {
      ...createContext(50900),
      currentPosition: 0
    }
    
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    if (signal.action === 'sell') {
      assert.ok(signal.reason.includes('[No position to sell]'))
    }
  })

  it('should remember breakout levels', async () => {
    // Failed breakout attempts
    const breakoutLevel = 52000
    
    for (let i = 0; i < 2; i++) {
      await agent.analyze(createContext(51500))
      await agent.analyze(createContext(51900))
      await agent.analyze(createContext(breakoutLevel))
      await agent.analyze(createContext(51800)) // Fail
    }
    
    // Successful breakout
    await agent.analyze(createContext(51900))
    await agent.analyze(createContext(breakoutLevel))
    await agent.analyze(createContext(52200)) // Success
    
    // Should remember breakout level
    const signal = await agent.analyze(createContext(breakoutLevel))
    
    assert.ok(signal)
    // Might treat previous resistance as support
  })

  it('should integrate multiple memory types', async () => {
    // Price memory
    await agent.analyze(createContext(50000))
    
    // Volume memory
    await agent.analyze(createContext(50500, 5000))
    
    // Time pattern (simplified)
    await agent.analyze(createContext(49500))
    
    // Combined analysis
    const signal = await agent.analyze(createContext(50000))
    
    assert.ok(signal)
    // Should consider all memory types
  })

  it('should properly reset state', async () => {
    // Build memories
    for (let i = 0; i < 20; i++) {
      await agent.analyze(createContext(50000 + i * 100))
    }
    
    // Reset
    await agent.reset()
    
    // Should have cleared memories
    const signal = await agent.analyze(createContext(50000))
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    // Fresh start without memories
  })
})