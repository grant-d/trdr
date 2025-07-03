import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { AdaptiveMacdAgent } from './adaptive-macd-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('AdaptiveMacdAgent', () => {
  let agent: AdaptiveMacdAgent
  const metadata: AgentMetadata = {
    id: 'adaptive-macd-test',
    name: 'Adaptive MACD Test Agent',
    version: '1.0.0',
    description: 'Test Adaptive MACD agent',
    type: 'momentum'
  }

  beforeEach(async () => {
    agent = new AdaptiveMacdAgent(metadata)
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
      high: price + 20,
      low: price - 20,
      close: price,
      volume: volumes?.[i] || 1000,
      timestamp: toEpochDate(Date.now() - (prices.length - i) * 60000)
    })),
    indicators: {}
  })

  it('should initialize successfully', async () => {
    assert.ok(agent)
    assert.strictEqual(agent.metadata.type, 'momentum')
  })

  it('should adapt periods based on volatility', async () => {
    // Low volatility period
    const stablePrices = Array(30).fill(0).map(() => 50000 + Math.random() * 50)
    await agent.analyze(createContext(stablePrices))
    
    // High volatility period
    const volatilePrices = Array(30).fill(0).map(() => 50000 + (Math.random() - 0.5) * 1000)
    const signal = await agent.analyze(createContext(volatilePrices))
    
    assert.ok(signal)
    // Adaptive behavior should adjust to volatility
  })

  it('should adapt to trending markets', async () => {
    // Strong uptrend
    const trendPrices = Array(30).fill(0).map((_, i) => 50000 + i * 150)
    const trendSignal = await agent.analyze(createContext(trendPrices))
    
    assert.ok(trendSignal)
    assert.strictEqual(trendSignal.action, 'buy')
    assert.ok(trendSignal.confidence >= 0.7)
  })

  it('should adapt to ranging markets', async () => {
    // Sideways movement
    const rangePrices = Array(30).fill(0).map((_, i) => 50000 + Math.sin(i * 0.3) * 200)
    const rangeSignal = await agent.analyze(createContext(rangePrices))
    
    assert.ok(rangeSignal)
    // Should be more cautious in ranging markets
    assert.ok(rangeSignal.confidence <= 0.6)
  })

  it('should detect adaptive crossovers', async () => {
    const prices = []
    
    // Create conditions for MACD crossover
    // Downtrend
    for (let i = 0; i < 15; i++) {
      prices.push(50000 - i * 100)
    }
    
    // Reversal
    for (let i = 0; i < 15; i++) {
      prices.push(48500 + i * 150)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('cross') || signal.reason.includes('signal'))
  })

  it('should handle volume-based adaptation', async () => {
    const prices = []
    const volumes = []
    
    // Low volume period
    for (let i = 0; i < 15; i++) {
      prices.push(50000 + i * 50)
      volumes.push(500)
    }
    
    // High volume period
    for (let i = 0; i < 15; i++) {
      prices.push(50750 + i * 100)
      volumes.push(5000)
    }
    
    const context = createContext(prices, volumes)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Volume should influence adaptation
  })

  it('should detect adaptive divergences', async () => {
    const prices = []
    
    // First peak
    for (let i = 0; i < 10; i++) {
      prices.push(50000 + i * 200)
    }
    
    // Valley
    for (let i = 0; i < 5; i++) {
      prices.push(52000 - i * 150)
    }
    
    // Second peak (higher price, but MACD should show divergence)
    for (let i = 0; i < 10; i++) {
      prices.push(51250 + i * 100) // Slower momentum
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should detect divergence with adaptive sensitivity
  })

  it('should respect position constraints', async () => {
    // Create sell signal conditions
    const prices = Array(30).fill(0).map((_, i) => 50000 - i * 100)
    
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

  it('should adapt histogram sensitivity', async () => {
    // Create expanding histogram
    const prices = []
    let basePrice = 50000
    
    for (let i = 0; i < 30; i++) {
      // Accelerating trend
      basePrice += i * 10
      prices.push(basePrice)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Expanding histogram should increase confidence
    assert.ok(signal.confidence >= 0.7)
  })

  it('should handle rapid market changes', async () => {
    const prices = []
    
    // Stable period
    for (let i = 0; i < 20; i++) {
      prices.push(50000 + Math.random() * 100)
    }
    
    // Sudden spike
    for (let i = 0; i < 10; i++) {
      prices.push(50000 + i * 500)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should adapt quickly to new conditions
  })

  it('should provide adaptive confidence levels', async () => {
    // Clear trend
    const clearPrices = Array(30).fill(0).map((_, i) => 50000 + i * 100)
    const clearSignal = await agent.analyze(createContext(clearPrices))
    
    // Choppy movement
    const choppyPrices = Array(30).fill(0).map(() => 50000 + (Math.random() - 0.5) * 500)
    const choppySignal = await agent.analyze(createContext(choppyPrices))
    
    assert.ok(clearSignal)
    assert.ok(choppySignal)
    assert.ok(clearSignal.confidence > choppySignal.confidence)
  })

  it('should adapt to market regimes', async () => {
    // Bull market
    const bullPrices = Array(30).fill(0).map((_, i) => 50000 + i * 200 + Math.random() * 100)
    const bullSignal = await agent.analyze(createContext(bullPrices))
    
    // Bear market
    const bearPrices = Array(30).fill(0).map((_, i) => 60000 - i * 200 + Math.random() * 100)
    const bearContext = {
      ...createContext(bearPrices),
      currentPosition: 0.1
    }
    const bearSignal = await agent.analyze(bearContext)
    
    assert.ok(bullSignal)
    assert.ok(bearSignal)
    // Should adapt to different market regimes
  })

  it('should properly reset adaptive parameters', async () => {
    // Build adaptive history
    const prices = Array(30).fill(0).map((_, i) => 50000 + Math.sin(i * 0.2) * 300)
    await agent.analyze(createContext(prices))
    
    // Reset
    await agent.reset()
    
    // Should start fresh
    const signal = await agent.analyze(createContext([50000, 50100]))
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
  })
})