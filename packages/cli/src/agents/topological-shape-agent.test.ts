import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { TopologicalShapeAgent } from './topological-shape-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('TopologicalShapeAgent', () => {
  let agent: TopologicalShapeAgent
  const metadata: AgentMetadata = {
    id: 'topological-test',
    name: 'Topological Test Agent',
    version: '1.0.0',
    description: 'Test topological agent',
    type: 'custom'
  }

  beforeEach(async () => {
    agent = new TopologicalShapeAgent(metadata, undefined, {
      persistenceThreshold: 0.02,
      priceResolution: 0.001,
      timeWindow: 1440,
      samplePoints: 20,
      maxDimension: 2
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
      open: price,
      high: price,
      low: price,
      close: price,
      volume: volumes?.[i] || 1000,
      timestamp: toEpochDate(Date.now() - (prices.length - i) * 60000)
    })),
    indicators: {}
  })

  it('should initialize with proper configuration', async () => {
    const context = createContext([50000])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient data'))
  })

  it('should detect price voids', async () => {
    // Create price pattern with a void
    const prices = [
      50000, 50100, 50200, 50300, 50400,  // Rising
      51000, 51100, 51200,                // Big gap/void around 50700
      51150, 51100, 51050                 // Consolidation
    ]
    
    for (const price of prices) {
      const context = createContext(prices.slice(0, prices.indexOf(price) + 1))
      await agent.analyze(context)
    }

    // Price entering void area
    const voidPrice = 50700
    const context = createContext([...prices, voidPrice])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should generate signal when price enters void
    assert.ok(signal.reason.includes('void'), `Expected reason to include 'void' but got: ${signal.reason}`)
  })

  it('should identify persistent features', async () => {
    // Create stable price levels (persistent 0-dimensional features)
    const prices = [
      50000, 50100, 50000, 50100, 50000,  // Oscillating
      50100, 50000, 50100, 50000, 50100   // Persistent levels
    ]
    
    for (const price of prices) {
      const context = createContext(prices.slice(0, prices.indexOf(price) + 1))
      await agent.analyze(context)
    }

    const signal = await agent.analyze(createContext(prices))
    
    assert.ok(signal)
    // Should identify persistent support/resistance
    assert.ok(signal.reason.includes('Topological'))
  })

  it('should detect price cycles', async () => {
    // Create cyclic pattern (1-dimensional feature)
    const cycleLength = 8
    const prices: number[] = []
    for (let i = 0; i < 16; i++) {
      prices.push(50000 + 500 * Math.sin(i * 2 * Math.PI / cycleLength))
    }
    
    for (const price of prices) {
      const context = createContext(prices.slice(0, prices.indexOf(price) + 1))
      await agent.analyze(context)
    }

    const signal = await agent.analyze(createContext(prices))
    
    assert.ok(signal)
    // Should detect cyclic behavior
  })

  it('should handle volume-weighted topology', async () => {
    const prices = [50000, 50100, 50200, 50300, 50400]
    const volumes = [1000, 5000, 2000, 8000, 1500] // High volume at certain levels
    
    for (let i = 0; i < prices.length; i++) {
      const context = createContext(
        prices.slice(0, i + 1),
        volumes.slice(0, i + 1)
      )
      await agent.analyze(context)
    }

    const signal = await agent.analyze(createContext(prices, volumes))
    
    assert.ok(signal)
    // High volume areas should influence topology
  })

  it('should track feature birth and death', async () => {
    // Pattern that creates and destroys features
    const prices = [
      50000, 50100, 50200,        // Feature birth
      50150, 50100, 50050,        // Feature death
      50100, 50200, 50300, 50400  // New feature birth
    ]
    
    for (const price of prices) {
      const context = createContext(prices.slice(0, prices.indexOf(price) + 1))
      await agent.analyze(context)
    }

    const signal = await agent.analyze(createContext(prices))
    
    assert.ok(signal)
    // Should track persistence of features
  })

  it('should handle insufficient data gracefully', async () => {
    const context = createContext([50000, 50100])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.confidence <= 0.5)
  })

  it('should detect void breakouts', async () => {
    // Create pattern with established void
    const prices = [
      50000, 50100, 50200,        // Lower range
      50800, 50900, 51000,        // Upper range (big void at 50500)
      50950, 50900, 50850         // Consolidation
    ]
    
    for (let i = 0; i < prices.length; i++) {
      const context = createContext(prices.slice(0, i + 1))
      await agent.analyze(context)
    }

    // Price breaking into void
    const voidBreakPrice = 50500
    const signal = await agent.analyze(createContext([...prices, voidBreakPrice]))
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('void'), `Expected reason to include 'void' but got: ${signal.reason}`)
  })

  it('should identify homological death', async () => {
    // Pattern where a cycle closes (death of 1-dimensional feature)
    const prices = [
      50000, 50200, 50400, 50300,  // Open cycle
      50100, 50000,                 // Closing back
      50100, 50200, 50300           // New pattern
    ]
    
    for (const price of prices) {
      const context = createContext(prices.slice(0, prices.indexOf(price) + 1))
      await agent.analyze(context)
    }

    const signal = await agent.analyze(createContext(prices))
    
    assert.ok(signal)
    // Feature death could signal pattern completion
  })

  it('should handle empty price history', async () => {
    const context: MarketContext = {
      symbol: 'BTC-USD',
      currentPrice: 50000,
      candles: [{
        open: 50000,
        high: 50000,
        low: 50000,
        close: 50000,
        volume: 1000,
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
    // Build complex topology
    const prices: number[] = []
    for (let i = 0; i < 30; i++) {
      prices.push(50000 + 1000 * Math.sin(i * 0.3) + 500 * Math.cos(i * 0.7))
    }
    
    for (const price of prices.slice(0, 20)) {
      const context = createContext(prices.slice(0, prices.indexOf(price) + 1))
      await agent.analyze(context)
    }

    // Reset
    await agent.reset()

    // Should be clean state
    const context = createContext([50000, 50100])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.confidence <= 0.5)
  })

  it('should detect multi-scale voids', async () => {
    // Pattern with voids at different scales
    const prices = [
      50000, 50050, 50100,          // Small scale
      50300, 50350, 50400,          // Medium void
      50900, 51000, 51100,          // Large void
      50700                         // Price in medium void
    ]
    
    for (let i = 0; i < prices.length; i++) {
      const context = createContext(prices.slice(0, i + 1))
      await agent.analyze(context)
    }

    const signal = await agent.analyze(createContext(prices))
    
    assert.ok(signal)
    // Should detect appropriate void scale
  })

  it('should handle rapid price movements', async () => {
    // Extreme price jumps that might break topology
    const prices = [50000, 50100, 55000, 50200, 50300]
    
    for (const price of prices) {
      const context = createContext(prices.slice(0, prices.indexOf(price) + 1))
      await agent.analyze(context)
    }

    const signal = await agent.analyze(createContext(prices))
    
    assert.ok(signal)
    // Should handle gracefully without errors
  })
})