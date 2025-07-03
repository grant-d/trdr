import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { MarketRegimeDetector } from './market-regime-detector'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('MarketRegimeDetector', () => {
  let agent: MarketRegimeDetector
  const metadata: AgentMetadata = {
    id: 'regime-test',
    name: 'Market Regime Test Agent',
    version: '1.0.0',
    description: 'Test Market Regime agent',
    type: 'regime'
  }

  beforeEach(async () => {
    agent = new MarketRegimeDetector(metadata)
    await agent.initialize()
  })

  afterEach(async () => {
    await agent.shutdown()
  })

  const createContext = (prices: number[], volumes?: number[]): MarketContext => ({
    symbol: 'BTC-USD',
    currentPrice: prices[prices.length - 1] || 50000,
    candles: prices.map((price, i) => ({
      open: price - 20,
      high: price + 30,
      low: price - 30,
      close: price,
      volume: volumes?.[i] || 1000,
      timestamp: toEpochDate(Date.now() - (prices.length - i) * 60000)
    })),
    indicators: {}
  })

  it('should initialize successfully', async () => {
    assert.ok(agent)
    assert.strictEqual(agent.metadata.type, 'regime')
  })

  it('should return hold signal with insufficient data', async () => {
    const context = createContext([50000, 50100])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient'))
  })

  it('should detect trending regime', async () => {
    // Create strong uptrend
    const prices = []
    let price = 50000
    
    for (let i = 0; i < 30; i++) {
      price += 150 + Math.random() * 50 // Consistent upward movement
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('trend') || signal.reason.includes('regime'))
    assert.strictEqual(signal.action, 'buy')
    assert.ok(signal.confidence >= 0.6)
  })

  it('should detect ranging regime', async () => {
    // Create sideways market
    const prices = []
    const basePrice = 50000
    
    for (let i = 0; i < 30; i++) {
      // Oscillate around base price
      prices.push(basePrice + Math.sin(i * 0.5) * 300)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('rang') || signal.reason.includes('sideways'))
    assert.strictEqual(signal.action, 'hold')
  })

  it('should detect volatile regime', async () => {
    // Create high volatility
    const prices = []
    
    for (let i = 0; i < 30; i++) {
      // Large random swings
      prices.push(50000 + (Math.random() - 0.5) * 2000)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('volatil') || signal.reason.includes('regime'))
    // High volatility might trigger caution
    assert.ok(signal.confidence <= 0.5)
  })

  it('should detect regime transitions', async () => {
    const prices = []
    
    // Start with trend
    let price = 50000
    for (let i = 0; i < 15; i++) {
      price += 100
      prices.push(price)
    }
    
    // Transition to range
    for (let i = 0; i < 15; i++) {
      prices.push(51500 + Math.sin(i) * 100)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should detect regime change
    assert.ok(signal.reason.includes('transition') || signal.reason.includes('chang'))
  })

  it('should detect breakout from range', async () => {
    const prices = []
    
    // Establish range
    for (let i = 0; i < 20; i++) {
      prices.push(50000 + Math.sin(i * 0.3) * 200)
    }
    
    // Breakout
    let breakoutPrice = 50000
    for (let i = 0; i < 10; i++) {
      breakoutPrice += 300
      prices.push(breakoutPrice)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('breakout') || signal.reason.includes('trend'))
    assert.strictEqual(signal.action, 'buy')
  })

  it('should adapt to volume regimes', async () => {
    const prices = []
    const volumes = []
    
    // Low volume range
    for (let i = 0; i < 15; i++) {
      prices.push(50000 + Math.random() * 200)
      volumes.push(500) // Low volume
    }
    
    // High volume trend
    let price = 50000
    for (let i = 0; i < 15; i++) {
      price += 200
      prices.push(price)
      volumes.push(3000) // High volume
    }
    
    const context = createContext(prices, volumes)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Volume change affects regime detection
  })

  it('should handle position constraints', async () => {
    // Downtrend regime
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

  it('should detect distribution/accumulation', async () => {
    const prices = []
    const volumes = []
    
    // Accumulation phase (range with increasing volume on dips)
    for (let i = 0; i < 30; i++) {
      const price = 50000 + Math.sin(i * 0.3) * 200
      prices.push(price)
      // Higher volume on lows
      volumes.push(price < 50000 ? 2000 : 1000)
    }
    
    const context = createContext(prices, volumes)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Accumulation might be bullish
  })

  it('should identify momentum regimes', async () => {
    // Strong momentum regime
    const prices = []
    let price = 50000
    let momentum = 50
    
    for (let i = 0; i < 30; i++) {
      momentum *= 1.05 // Accelerating
      price += momentum
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('momentum') || signal.reason.includes('strong'))
    assert.ok(signal.confidence >= 0.7)
  })

  it('should properly reset state', async () => {
    // Build regime history
    const trendPrices = Array(30).fill(0).map((_, i) => 50000 + i * 100)
    await agent.analyze(createContext(trendPrices))
    
    // Reset
    await agent.reset()
    
    // Should need to rebuild regime detection
    const signal = await agent.analyze(createContext([50000, 50100]))
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient'))
  })
})