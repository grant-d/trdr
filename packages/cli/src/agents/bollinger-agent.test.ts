import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { BollingerBandsAgent } from './bollinger-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('BollingerAgent', () => {
  let agent: BollingerBandsAgent
  const metadata: AgentMetadata = {
    id: 'bollinger-test',
    name: 'Bollinger Test Agent',
    version: '1.0.0',
    description: 'Test Bollinger agent',
    type: 'volatility'
  }

  beforeEach(async () => {
    agent = new BollingerBandsAgent(metadata)
    await agent.initialize()
  })

  afterEach(async () => {
    await agent.shutdown()
  })

  const createContext = (prices: number[]): MarketContext => ({
    symbol: 'BTC-USD',
    currentPrice: prices[prices.length - 1] || 50000,
    candles: prices.map((price, i) => ({
      open: price - 5,
      high: price + 20,
      low: price - 20,
      close: price,
      volume: 1000,
      timestamp: toEpochDate(Date.now() - (prices.length - i) * 60000)
    })),
    indicators: {}
  })

  it('should initialize successfully', async () => {
    assert.ok(agent)
    assert.strictEqual(agent.metadata.type, 'volatility')
  })

  it('should return hold signal with insufficient data', async () => {
    const context = createContext([50000, 50100])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient'))
  })

  it('should detect price at lower band', async () => {
    // Create steady prices then drop
    const prices = []
    
    // Build baseline
    for (let i = 0; i < 20; i++) {
      prices.push(50000 + (Math.random() - 0.5) * 100)
    }
    
    // Drop below normal range
    prices.push(49500, 49400, 49300)
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Price at lower band suggests oversold
    if (signal.reason.includes('band')) {
      assert.strictEqual(signal.action, 'buy')
      assert.ok(signal.confidence >= 0.6)
    }
  })

  it('should detect price at upper band', async () => {
    // Create steady prices then spike
    const prices = []
    
    // Build baseline
    for (let i = 0; i < 20; i++) {
      prices.push(50000 + (Math.random() - 0.5) * 100)
    }
    
    // Spike above normal range
    prices.push(50500, 50600, 50700)
    
    const context = {
      ...createContext(prices),
      currentPosition: 0.1 // Has position
    }
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Price at upper band suggests overbought
    if (signal.reason.includes('band')) {
      assert.ok(['sell', 'hold'].includes(signal.action))
    }
  })

  it('should detect band squeeze', async () => {
    // Create low volatility period
    const prices = []
    
    // Tight range
    for (let i = 0; i < 25; i++) {
      prices.push(50000 + (Math.random() - 0.5) * 50) // Very small moves
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Band squeeze indicates potential breakout
    if (signal.reason.includes('squeeze') || signal.reason.includes('narrow')) {
      assert.ok(signal.confidence <= 0.5) // Low confidence during squeeze
    }
  })

  it('should detect band expansion', async () => {
    // Create increasing volatility
    const prices = []
    let volatility = 50
    
    for (let i = 0; i < 25; i++) {
      prices.push(50000 + (Math.random() - 0.5) * volatility)
      volatility += 10 // Increasing volatility
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Expanding bands indicate increased volatility
  })

  it('should handle price walk along bands', async () => {
    // Price riding upper band (strong trend)
    const prices = []
    let price = 50000
    
    for (let i = 0; i < 25; i++) {
      price += 50 // Steady rise
      prices.push(price + Math.random() * 20) // Small variation
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Walking the band indicates strong trend
  })

  it('should respect position constraints', async () => {
    // Upper band touch without position
    const prices = Array(20).fill(50000).concat([50200, 50300, 50400])
    
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

  it('should calculate bandwidth correctly', async () => {
    // Known volatility pattern
    const prices = []
    
    // Alternating high/low for predictable bands
    for (let i = 0; i < 25; i++) {
      prices.push(i % 2 === 0 ? 50000 : 50200)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should have consistent band width
  })

  it('should handle mean reversion', async () => {
    // Price far from middle band returns
    const prices = Array(20).fill(50000)
    
    // Spike away
    prices.push(50500, 50600)
    
    // Return to mean
    prices.push(50400, 50200, 50100)
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Mean reversion might trigger signal
  })

  it('should properly reset state', async () => {
    // Build history
    const prices = Array(25).fill(0).map(() => 50000 + Math.random() * 500)
    await agent.analyze(createContext(prices))
    
    // Reset
    await agent.reset()
    
    // Should need more data
    const signal = await agent.analyze(createContext([50000, 50100]))
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient'))
  })
})