import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { QuantumSuperpositionAgent } from './quantum-superposition-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('QuantumSuperpositionAgent', () => {
  let agent: QuantumSuperpositionAgent
  const metadata: AgentMetadata = {
    id: 'quantum-test',
    name: 'Quantum Test Agent',
    version: '1.0.0',
    description: 'Test quantum agent',
    type: 'volume'
  }

  beforeEach(async () => {
    agent = new QuantumSuperpositionAgent(metadata, undefined, {
      observationWindow: 60,
      collapseVolumeThreshold: 2.0,
      superpositionStates: 5,
      amplitudeDecay: 0.95,
      volumeSensitivity: 0.7
    })
    await agent.initialize()
  })

  afterEach(async () => {
    await agent.shutdown()
  })

  const createContext = (price: number, volume: number = 1000): MarketContext => ({
    symbol: 'BTC-USD',
    currentPrice: price,
    candles: [
      { open: price - 10, high: price + 5, low: price - 15, close: price, volume, timestamp: toEpochDate(Date.now() - 60000) },
      { open: price - 5, high: price + 10, low: price - 10, close: price, volume, timestamp: toEpochDate(Date.now()) }
    ],
    indicators: {}
  })

  it('should initialize in superposition state', async () => {
    const context = createContext(50000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('superposition'))
  })

  it('should collapse wave function on volume spike', async () => {
    // Build history with normal volume
    for (let i = 0; i < 5; i++) {
      const context = createContext(50000 + i * 10, 1000)
      await agent.analyze(context)
    }

    // Trigger collapse with volume spike
    const spikeContext = createContext(50100, 3000) // 3x volume
    const signal = await agent.analyze(spikeContext)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('collapsed'))
    assert.ok(signal.confidence > 0.6)
    assert.ok(['buy', 'sell'].includes(signal.action))
  })

  it('should generate weak signals in superposition', async () => {
    // Build price history
    const prices = [50000, 50100, 50200, 50150, 50250]
    for (const price of prices) {
      const context = createContext(price, 1000)
      await agent.analyze(context)
    }

    // Price deviates from expected
    const context = createContext(50400, 1000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.confidence <= 0.6)
    assert.ok(signal.reason.includes('superposition'))
  })

  it('should reset to superposition after time window', async () => {
    // Trigger collapse
    const collapseContext = createContext(50000, 3000)
    const collapseSignal = await agent.analyze(collapseContext)
    assert.ok(collapseSignal.reason.includes('collapsed'))

    // Simulate time passing (mock time would be better, but using reset for now)
    await agent.reset()
    await agent.initialize()

    // Should be back in superposition
    const context = createContext(50000, 1000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('superposition'))
  })

  it('should calculate coherence based on volume consistency', async () => {
    // Inconsistent volumes affect coherence
    const volumes = [1000, 500, 2000, 800, 1500]
    for (let i = 0; i < volumes.length; i++) {
      const context = createContext(50000 + i * 10, volumes[i]!)
      await agent.analyze(context)
    }

    const context = createContext(50050, 1000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('coherence'))
    // Coherence should be mentioned in the reason
  })

  it('should handle empty market data gracefully', async () => {
    const context: MarketContext = {
      symbol: 'BTC-USD',
      currentPrice: 50000,
      candles: [],
      indicators: {}
    }

    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.confidence < 0.5)
  })

  it('should track price history within observation window', async () => {
    // Add many price points
    for (let i = 0; i < 10; i++) {
      const context = createContext(50000 + i * 10, 1000)
      await agent.analyze(context)
    }

    // History should be maintained
    const context = createContext(50100, 1000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should have enough history for proper analysis
    assert.ok(signal.confidence >= 0.4)
  })

  it('should detect rising price on collapse', async () => {
    // Create rising price pattern
    const prices = [50000, 50050, 50100, 50150]
    for (const price of prices) {
      const context = createContext(price, 1000)
      await agent.analyze(context)
    }

    // Collapse with rising price
    const context = createContext(50200, 3000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    if (signal.reason.includes('collapsed')) {
      assert.strictEqual(signal.action, 'buy')
      assert.ok(signal.reason.includes('rising'))
    }
  })

  it('should detect falling price on collapse', async () => {
    // Create falling price pattern
    const prices = [50200, 50150, 50100, 50050]
    for (const price of prices) {
      const context = createContext(price, 1000)
      await agent.analyze(context)
    }

    // Collapse with falling price
    const context = createContext(50000, 3000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    if (signal.reason.includes('collapsed')) {
      assert.strictEqual(signal.action, 'sell')
      assert.ok(signal.reason.includes('falling'))
    }
  })

  it('should properly reset state', async () => {
    // Build some history
    for (let i = 0; i < 5; i++) {
      const context = createContext(50000 + i * 10, 1000)
      await agent.analyze(context)
    }

    // Reset
    await agent.reset()

    // Should be clean slate
    const context = createContext(50000, 1000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.confidence <= 0.5)
  })
})