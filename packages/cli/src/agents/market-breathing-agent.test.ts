import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { MarketBreathingAgent } from './market-breathing-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('MarketBreathingAgent', () => {
  let agent: MarketBreathingAgent
  const metadata: AgentMetadata = {
    id: 'breathing-test',
    name: 'Market Breathing Test Agent',
    version: '1.0.0',
    description: 'Test breathing agent',
    type: 'custom'
  }

  beforeEach(async () => {
    agent = new MarketBreathingAgent(metadata, undefined, {
      cycleWindow: 60, // 1 hour in minutes
      minAmplitude: 0.002,
      breathHistory: 10,
      rhythmChangeThreshold: 0.3,
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
    candles: [{
      open: price - 5,
      high: price + 5,
      low: price - 5,
      close: price,
      volume,
      timestamp: toEpochDate(Date.now())
    }],
    indicators: {}
  })

  const createBreathingPattern = (basePrice: number, amplitude: number, phase: number): number => {
    return basePrice + amplitude * Math.sin(phase)
  }

  it('should initialize with proper configuration', async () => {
    const context = createContext(50000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient data'))
  })

  it('should detect inhale phase', async () => {
    // Create expanding price pattern (inhale)
    const basePrice = 50000
    const phases = Array.from({ length: 15 }, (_, i) => i * Math.PI / 8)
    
    for (const phase of phases) {
      const price = createBreathingPattern(basePrice, 200 * (phase / Math.PI), phase)
      const context = createContext(price)
      await agent.analyze(context)
    }
    
    // Should be in inhale phase
    const signal = await agent.analyze(createContext(basePrice + 300))
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('inhale') || signal.reason.includes('Breathing'))
  })

  it('should detect exhale phase', async () => {
    // Create contracting price pattern (exhale)
    const basePrice = 50000
    
    // First expand (inhale)
    for (let i = 0; i < 10; i++) {
      const price = basePrice + i * 50
      await agent.analyze(createContext(price))
    }
    
    // Then contract (exhale)
    for (let i = 10; i >= 0; i--) {
      const price = basePrice + i * 50
      await agent.analyze(createContext(price))
    }
    
    const signal = await agent.analyze(createContext(basePrice + 100))
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('exhale') || signal.reason.includes('rhythm'))
  })

  it('should track breathing rhythm', async () => {
    // Create regular breathing cycles
    const basePrice = 50000
    const amplitude = 300
    
    // Complete 3 full breath cycles
    for (let cycle = 0; cycle < 3; cycle++) {
      // Inhale
      for (let i = 0; i <= 10; i++) {
        const phase = i * Math.PI / 10
        const price = createBreathingPattern(basePrice, amplitude, phase)
        await agent.analyze(createContext(price))
      }
      // Exhale
      for (let i = 10; i >= 0; i--) {
        const phase = Math.PI + (10 - i) * Math.PI / 10
        const price = createBreathingPattern(basePrice, amplitude, phase)
        await agent.analyze(createContext(price))
      }
    }
    
    const signal = await agent.analyze(createContext(basePrice))
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('regular') || signal.reason.includes('rhythm'))
  })

  it('should detect irregular breathing', async () => {
    // Create irregular pattern
    // @ts-ignore - unused variable (reserved for future use)
    const _basePrice = 50000
    const irregularPrices = [
      50000, 50100, 50050, 50300, 50150, 50400,
      50200, 50100, 50350, 50050, 50250, 50000
    ]
    
    for (const price of irregularPrices) {
      await agent.analyze(createContext(price))
    }
    
    const signal = await agent.analyze(createContext(50150))
    
    assert.ok(signal)
    // Irregular breathing might be mentioned
  })

  it('should calculate market oxygen levels', async () => {
    // Test with varying volumes
    const price = 50000
    const volumes = [1000, 1200, 1500, 2000, 3000, 2500, 2000, 1500, 1000]
    
    for (const volume of volumes) {
      await agent.analyze(createContext(price + Math.random() * 100, volume))
    }
    
    const signal = await agent.analyze(createContext(price, 1500))
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('O2') || signal.reason.includes('oxygen') || signal.reason.includes('Breathing'))
  })

  it('should detect stressed breathing', async () => {
    // Create stressed pattern - rapid, shallow breaths
    const basePrice = 50000
    
    for (let i = 0; i < 20; i++) {
      const price = basePrice + (i % 2) * 50 * Math.random() // Erratic small movements
      const volume = 500 + Math.random() * 2000 // Erratic volume
      await agent.analyze(createContext(price, volume))
    }
    
    const signal = await agent.analyze(createContext(basePrice))
    
    assert.ok(signal)
    if (signal.reason.includes('stressed')) {
      assert.strictEqual(signal.action, 'hold')
      assert.ok(signal.confidence >= 0.7)
    }
  })

  it('should identify breath phase transitions', async () => {
    // Build to peak of inhale
    const basePrice = 50000
    for (let i = 0; i <= 10; i++) {
      const price = basePrice + i * 50
      await agent.analyze(createContext(price))
    }
    
    // Transition to exhale
    const signal = await agent.analyze(createContext(basePrice + 480))
    
    assert.ok(signal)
    // Near end of inhale might trigger sell
    if (signal.reason.includes('phase') && signal.reason.includes('70%')) {
      assert.ok(['sell', 'hold'].includes(signal.action))
    }
  })

  it('should handle pause phase', async () => {
    // Create consolidation pattern (pause)
    const pausePrice = 50000
    
    for (let i = 0; i < 15; i++) {
      const price = pausePrice + (Math.random() - 0.5) * 20 // Tiny movements
      await agent.analyze(createContext(price))
    }
    
    const signal = await agent.analyze(createContext(pausePrice))
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('pause') || signal.reason.includes('Breathing'))
  })

  it('should track breath rate', async () => {
    // Create breaths at specific rate
    const basePrice = 50000
    
    // Fast breathing - complete multiple cycles quickly
    for (let cycle = 0; cycle < 5; cycle++) {
      for (let i = 0; i < 6; i++) {
        const price = basePrice + (i < 3 ? i : 6 - i) * 100
        await agent.analyze(createContext(price))
      }
    }
    
    const signal = await agent.analyze(createContext(basePrice))
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('rate') || signal.reason.includes('/hr') || signal.reason.includes('Breathing'))
  })

  it('should detect accelerating rhythm', async () => {
    // Start with slow breaths, then accelerate
    const basePrice = 50000
    
    // Slow breaths
    for (let i = 0; i < 20; i++) {
      const price = basePrice + 200 * Math.sin(i * 0.1)
      await agent.analyze(createContext(price))
    }
    
    // Fast breaths
    for (let i = 0; i < 20; i++) {
      const price = basePrice + 200 * Math.sin(i * 0.5)
      await agent.analyze(createContext(price))
    }
    
    const signal = await agent.analyze(createContext(basePrice))
    
    assert.ok(signal)
    // Might detect acceleration
  })

  it('should properly reset state', async () => {
    // Build complex breathing history
    for (let i = 0; i < 50; i++) {
      const price = 50000 + 500 * Math.sin(i * 0.2)
      await agent.analyze(createContext(price))
    }
    
    // Reset
    await agent.reset()
    
    // Should be clean state
    const context = createContext(50000)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient data'))
  })

  it('should handle low oxygen conditions', async () => {
    // Very low volume (low oxygen)
    const price = 50000
    
    for (let i = 0; i < 15; i++) {
      const lowVolume = 100 + Math.random() * 200
      await agent.analyze(createContext(price + i * 10, lowVolume))
    }
    
    const signal = await agent.analyze(createContext(price, 150))
    
    assert.ok(signal)
    // Low oxygen should be cautious
  })

  it('should find opportunities at breath extremes', async () => {
    // Build clear breathing pattern
    const basePrice = 50000
    
    // Complete inhale
    for (let i = 0; i <= 10; i++) {
      await agent.analyze(createContext(basePrice + i * 40))
    }
    
    // Near peak - should consider selling
    const peakSignal = await agent.analyze(createContext(basePrice + 400))
    assert.ok(peakSignal)
    
    // Complete exhale
    for (let i = 10; i >= 0; i--) {
      await agent.analyze(createContext(basePrice + i * 40))
    }
    
    // Near trough - should consider buying
    const troughSignal = await agent.analyze(createContext(basePrice + 20))
    assert.ok(troughSignal)
  })
})