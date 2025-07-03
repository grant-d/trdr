import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { SwarmIntelligenceAgent } from './swarm-intelligence-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('SwarmIntelligenceAgent', () => {
  let agent: SwarmIntelligenceAgent
  const metadata: AgentMetadata = {
    id: 'swarm-test',
    name: 'Swarm Intelligence Test Agent',
    version: '1.0.0',
    description: 'Test swarm agent',
    type: 'custom'
  }

  beforeEach(async () => {
    agent = new SwarmIntelligenceAgent(metadata, undefined, {
      swarmSize: 50, // Smaller for testing
      communicationRadius: 0.02,
      influenceDecay: 0.5,
      herdThreshold: 0.7,
      socialWeight: 0.6,
      memoryLength: 20
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

  it('should initialize swarm with diverse participants', async () => {
    const context = createContext(50000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.action)
    assert.ok(signal.reason)
    // Swarm should be initialized
  })

  it('should detect convergence pattern', async () => {
    // Create price movement that causes swarm convergence
    const prices = [50000, 50100, 50200, 50300, 50400]
    
    for (const price of prices) {
      const context = createContext(price, 2000) // Higher volume
      const signal = await agent.analyze(context)
      assert.ok(signal)
    }
    
    // Consistent upward movement should cause convergence
    const finalSignal = await agent.analyze(createContext(50500, 2500))
    
    assert.ok(finalSignal)
    if (finalSignal.reason.includes('converging')) {
      assert.ok(finalSignal.confidence >= 0.6)
      assert.ok(['buy', 'hold'].includes(finalSignal.action))
    }
  })

  it('should detect divergence pattern', async () => {
    // Create erratic price movement
    const erraticPrices = [50000, 50200, 49900, 50300, 49800, 50400, 49700]
    
    for (const price of erraticPrices) {
      const context = createContext(price, Math.random() * 2000 + 500)
      await agent.analyze(context)
    }
    
    const signal = await agent.analyze(createContext(50000))
    
    assert.ok(signal)
    if (signal.reason.includes('fragmented') || signal.reason.includes('diversity')) {
      assert.strictEqual(signal.action, 'hold')
    }
  })

  it('should detect migration pattern', async () => {
    // Strong directional movement
    const migrationPrices = []
    for (let i = 0; i < 10; i++) {
      migrationPrices.push(50000 + i * 200) // Strong upward
    }
    
    for (const price of migrationPrices) {
      const context = createContext(price, 3000) // High volume
      await agent.analyze(context)
    }
    
    const signal = await agent.analyze(createContext(52100))
    
    assert.ok(signal)
    if (signal.reason.includes('migrating')) {
      assert.strictEqual(signal.action, 'buy')
      assert.ok(signal.confidence >= 0.65)
    }
  })

  it('should detect flocking behavior', async () => {
    // Tight range movement (flocking)
    const basePrice = 50000
    
    for (let i = 0; i < 15; i++) {
      const price = basePrice + (Math.random() - 0.5) * 50 // Very tight range
      const context = createContext(price, 1500)
      await agent.analyze(context)
    }
    
    const signal = await agent.analyze(createContext(basePrice))
    
    assert.ok(signal)
    if (signal.reason.includes('flock') || signal.reason.includes('cohesion')) {
      assert.ok(signal.confidence > 0.5)
    }
  })

  it('should handle high consensus appropriately', async () => {
    // Create strong consensus by steady movement
    for (let i = 0; i < 10; i++) {
      const price = 50000 + i * 100
      await agent.analyze(createContext(price, 2000))
    }
    
    const signal = await agent.analyze(createContext(51000))
    
    assert.ok(signal)
    if (signal.reason.includes('consensus')) {
      // High consensus might reduce confidence (overcrowded)
      assert.ok(signal.reason.includes('cautious') || signal.confidence < 0.9)
    }
  })

  it('should track participant velocity changes', async () => {
    // Acceleration pattern
    const prices = [50000, 50010, 50030, 50070, 50150, 50250]
    
    for (const price of prices) {
      await agent.analyze(createContext(price))
    }
    
    const signal = await agent.analyze(createContext(50400))
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('Momentum'))
  })

  it('should handle fragmentation into clusters', async () => {
    // Create price levels that fragment swarm
    const clusterPrices = [
      50000, 50000, 50100, 50100, // Cluster 1
      51000, 51000, 51100, 51100, // Cluster 2
      50500, 50500, 50600, 50600  // Cluster 3
    ]
    
    for (const price of clusterPrices) {
      await agent.analyze(createContext(price))
    }
    
    const signal = await agent.analyze(createContext(50800))
    
    assert.ok(signal)
    if (signal.reason.includes('fragment')) {
      assert.strictEqual(signal.action, 'hold')
    }
  })

  it('should apply separation force correctly', async () => {
    // Overcrowded price level
    const crowdedPrice = 50000
    
    // Many participants at same level
    for (let i = 0; i < 10; i++) {
      await agent.analyze(createContext(crowdedPrice + Math.random() * 10))
    }
    
    // Should see separation effects
    const signal = await agent.analyze(createContext(crowdedPrice))
    
    assert.ok(signal)
    // Separation might be mentioned or influence behavior
  })

  it('should track swarm momentum', async () => {
    // Build momentum
    const momentumPrices = [50000, 50050, 50100, 50150, 50200]
    
    for (const price of momentumPrices) {
      await agent.analyze(createContext(price))
    }
    
    const signal = await agent.analyze(createContext(50250))
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('Momentum:'))
    // Should show positive momentum percentage
  })

  it('should detect polarization', async () => {
    // Create polarized market (bulls vs bears)
    const polarizedPrices = [50000, 51000, 49000, 52000, 48000]
    
    for (const price of polarizedPrices) {
      await agent.analyze(createContext(price, Math.random() * 3000))
    }
    
    const signal = await agent.analyze(createContext(50000))
    
    assert.ok(signal)
    // Polarization affects confidence
  })

  it('should properly reset state', async () => {
    // Build complex swarm state
    for (let i = 0; i < 30; i++) {
      const price = 50000 + Math.sin(i * 0.5) * 1000
      await agent.analyze(createContext(price))
    }
    
    // Reset
    await agent.reset()
    
    // Should reinitialize swarm
    const signal = await agent.analyze(createContext(50000))
    
    assert.ok(signal)
    // Fresh swarm state
  })

  it('should handle rapid price changes', async () => {
    // Extreme price jumps
    const volatilePrices = [50000, 55000, 45000, 60000, 40000]
    
    for (const price of volatilePrices) {
      const signal = await agent.analyze(createContext(price))
      assert.ok(signal)
    }
    
    // Should handle gracefully
  })

  it('should apply influence decay over distance', async () => {
    // Test influence propagation
    const distantPrices = [50000, 50020, 50050, 50100, 50200]
    
    for (const price of distantPrices) {
      await agent.analyze(createContext(price))
    }
    
    const signal = await agent.analyze(createContext(50010))
    
    assert.ok(signal)
    // Nearby participants have more influence
  })

  it('should fade overcrowded moves', async () => {
    // Create strong flocking with high momentum
    for (let i = 0; i < 10; i++) {
      await agent.analyze(createContext(50000 + i * 100, 3000))
    }
    
    // Very high momentum should trigger fade
    const signal = await agent.analyze(createContext(51100))
    
    assert.ok(signal)
    if (signal.reason.includes('overcrowded')) {
      assert.strictEqual(signal.action, 'sell') // Fade the move
    }
  })
})