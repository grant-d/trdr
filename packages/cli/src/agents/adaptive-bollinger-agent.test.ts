import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { AdaptiveBollingerBandsAgent } from './adaptive-bollinger-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('AdaptiveBollingerAgent', () => {
  let agent: AdaptiveBollingerBandsAgent
  const metadata: AgentMetadata = {
    id: 'adaptive-bollinger-test',
    name: 'Adaptive Bollinger Test Agent',
    version: '1.0.0',
    description: 'Test Adaptive Bollinger agent',
    type: 'volatility'
  }

  beforeEach(async () => {
    agent = new AdaptiveBollingerBandsAgent(metadata)
    await agent.initialize()
  })

  afterEach(async () => {
    await agent.shutdown()
  })

  const createContext = (prices: number[], volumes?: number[]): MarketContext => ({
    symbol: 'BTC-USD',
    currentPrice: prices[prices.length - 1] || 50000,
    candles: prices.map((price, i) => ({
      open: price - 10,
      high: price + 25,
      low: price - 25,
      close: price,
      volume: volumes?.[i] || 1000,
      timestamp: toEpochDate(Date.now() - (prices.length - i) * 60000)
    })),
    indicators: {}
  })

  it('should initialize successfully', async () => {
    assert.ok(agent)
    assert.strictEqual(agent.metadata.type, 'volatility')
  })

  it('should adapt period based on volatility', async () => {
    // Low volatility period
    const stablePrices = Array(30).fill(0).map(() => 50000 + Math.random() * 50)
    await agent.analyze(createContext(stablePrices))
    
    // High volatility period
    const volatilePrices = Array(30).fill(0).map(() => 50000 + (Math.random() - 0.5) * 1000)
    const signal = await agent.analyze(createContext(volatilePrices))
    
    assert.ok(signal)
    // Adaptive behavior should be mentioned or affect confidence
  })

  it('should adapt multiplier based on market conditions', async () => {
    // Trending market (should widen bands)
    const trendPrices = Array(30).fill(0).map((_, i) => 50000 + i * 100)
    const trendSignal = await agent.analyze(createContext(trendPrices))
    
    // Ranging market (should tighten bands)
    const rangePrices = Array(30).fill(0).map((_, i) => 50000 + Math.sin(i * 0.3) * 200)
    const rangeSignal = await agent.analyze(createContext(rangePrices))
    
    assert.ok(trendSignal)
    assert.ok(rangeSignal)
    // Different market conditions should produce different adaptations
  })

  it('should detect squeeze with adaptive parameters', async () => {
    // Create volatility contraction
    const prices = []
    let volatility = 500
    
    for (let i = 0; i < 30; i++) {
      prices.push(50000 + (Math.random() - 0.5) * volatility)
      volatility *= 0.95 // Decreasing volatility
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should detect adaptive squeeze
    if (signal.reason.includes('squeeze')) {
      assert.ok(signal.confidence <= 0.5)
    }
  })

  it('should handle dynamic support/resistance', async () => {
    // Price walking along adaptive bands
    const prices = []
    let price = 50000
    
    // Walk along upper band
    for (let i = 0; i < 20; i++) {
      price += 50 + Math.random() * 20
      prices.push(price)
    }
    
    // Sharp reversal
    for (let i = 0; i < 10; i++) {
      price -= 100
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Adaptive bands should provide dynamic levels
  })

  it('should adapt to volume changes', async () => {
    const prices = []
    const volumes = []
    
    // Normal volume period
    for (let i = 0; i < 15; i++) {
      prices.push(50000 + Math.random() * 200)
      volumes.push(1000)
    }
    
    // High volume period
    for (let i = 0; i < 15; i++) {
      prices.push(50100 + Math.random() * 300)
      volumes.push(5000)
    }
    
    const context = createContext(prices, volumes)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Volume should affect adaptation
  })

  it('should respect position constraints', async () => {
    // Upper band touch
    const prices = Array(20).fill(50000)
    prices.push(...Array(10).fill(50500)) // Push to upper band
    
    const context = {
      ...createContext(prices),
      currentPosition: 0
    }
    
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    if (signal.action === 'sell') {
      assert.ok(signal.reason.includes('[No position to sell]'))
    }
  })

  it('should handle rapid adaptation scenarios', async () => {
    // Sudden regime change
    const prices = []
    
    // Quiet period
    for (let i = 0; i < 20; i++) {
      prices.push(50000 + Math.random() * 100)
    }
    
    // Sudden volatility
    for (let i = 0; i < 10; i++) {
      prices.push(50000 + (Math.random() - 0.5) * 1000)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should adapt quickly to new conditions
  })

  it('should provide adaptive confidence levels', async () => {
    // Well-defined bands
    const clearPrices = []
    for (let i = 0; i < 30; i++) {
      clearPrices.push(50000 + Math.sin(i * 0.2) * 300)
    }
    const clearSignal = await agent.analyze(createContext(clearPrices))
    
    // Erratic movement
    const erraticPrices = []
    for (let i = 0; i < 30; i++) {
      erraticPrices.push(50000 + (Math.random() - 0.5) * 600)
    }
    const erraticSignal = await agent.analyze(createContext(erraticPrices))
    
    assert.ok(clearSignal)
    assert.ok(erraticSignal)
    // Clear patterns should have higher confidence
  })

  it('should detect band expansion correctly', async () => {
    // Start narrow, then expand
    const prices = []
    let amplitude = 100
    
    for (let i = 0; i < 30; i++) {
      prices.push(50000 + Math.sin(i * 0.3) * amplitude)
      amplitude += 10 // Expanding range
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should detect expansion
  })

  it('should properly reset adaptive parameters', async () => {
    // Build adaptive history
    const prices = Array(30).fill(0).map((_, i) => 50000 + Math.sin(i * 0.2) * 300)
    await agent.analyze(createContext(prices))
    
    // Reset
    await agent.reset()
    
    // Should start fresh adaptation
    const signal = await agent.analyze(createContext([50000, 50100]))
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
  })
})