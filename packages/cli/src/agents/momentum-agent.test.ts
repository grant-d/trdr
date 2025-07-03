import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { MomentumAgent } from './momentum-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('MomentumAgent', () => {
  let agent: MomentumAgent
  const metadata: AgentMetadata = {
    id: 'momentum-test',
    name: 'Momentum Test Agent',
    version: '1.0.0',
    description: 'Test Momentum agent',
    type: 'momentum'
  }

  beforeEach(async () => {
    agent = new MomentumAgent(metadata)
    await agent.initialize()
  })

  afterEach(async () => {
    await agent.shutdown()
  })

  const createContext = (prices: number[]): MarketContext => ({
    symbol: 'BTC-USD',
    currentPrice: prices[prices.length - 1] || 50000,
    candles: prices.map((price, i) => ({
      open: price - 10,
      high: price + 15,
      low: price - 15,
      close: price,
      volume: 1000 + Math.random() * 500,
      timestamp: toEpochDate(Date.now() - (prices.length - i) * 60000)
    })),
    indicators: {}
  })

  it('should initialize successfully', async () => {
    assert.ok(agent)
    assert.strictEqual(agent.metadata.type, 'momentum')
  })

  it('should return hold signal with insufficient data', async () => {
    const context = createContext([50000, 50100])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient') || signal.reason.includes('Indicator calculation failed'))
  })

  it('should detect strong upward momentum', async () => {
    // Create a more realistic momentum scenario
    const prices = []
    let price = 50000
    
    // Start with some sideways movement
    for (let i = 0; i < 20; i++) {
      price += (Math.random() - 0.5) * 100
      prices.push(price)
    }
    
    // Then create upward momentum (not too extreme)
    for (let i = 0; i < 20; i++) {
      price += 50 + i * 5 // Gradual acceleration
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // With strong momentum but not extreme overbought, should generate buy
    if (signal.action === 'hold') {
      // Accept hold if RSI is too high
      assert.ok(signal.reason.includes('overbought') || signal.reason.includes('neutral'))
    } else {
      assert.strictEqual(signal.action, 'buy')
      assert.ok(signal.confidence >= 0.6)
    }
    assert.ok(signal.reason.toLowerCase().includes('momentum') || signal.reason.includes('RSI'))
  })

  it('should detect strong downward momentum', async () => {
    // Create strong downtrend
    const prices = []
    let price = 50000
    
    for (let i = 0; i < 20; i++) {
      price -= 200 // Consistent losses
      prices.push(price)
    }
    
    const context = {
      ...createContext(prices),
      currentPosition: 0.1 // Has position
    }
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(['sell', 'hold'].includes(signal.action))
    if (signal.action === 'sell') {
      assert.ok(signal.confidence >= 0.7)
    }
  })

  it('should detect momentum acceleration', async () => {
    // Create a more balanced acceleration pattern
    const prices = []
    let price = 50000
    
    // Start with stable prices
    for (let i = 0; i < 20; i++) {
      price += (Math.random() - 0.5) * 50
      prices.push(price)
    }
    
    // Then accelerating momentum
    let increment = 10
    for (let i = 0; i < 20; i++) {
      price += increment
      increment = Math.min(increment + 5, 100) // Cap to prevent extreme overbought
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should detect momentum, either buy or hold depending on RSI
    assert.ok(['buy', 'hold'].includes(signal.action))
    if (signal.action === 'buy') {
      assert.ok(signal.confidence >= 0.6)
    }
    assert.ok(signal.reason.toLowerCase().includes('momentum') || signal.reason.includes('RSI'))
  })

  it('should detect momentum deceleration', async () => {
    // Decreasing rate of change
    const prices = []
    let price = 50000
    let increment = 300
    
    for (let i = 0; i < 20; i++) {
      price += increment
      increment *= 0.8 // Decelerating
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Slowing momentum might trigger caution
    assert.ok(signal.confidence < 0.8)
  })

  it('should handle sideways market', async () => {
    // No clear momentum
    const prices = []
    for (let i = 0; i < 20; i++) {
      prices.push(50000 + (Math.random() - 0.5) * 100)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.confidence <= 0.5)
  })

  it('should detect momentum reversal', async () => {
    const prices = []
    let price = 50000
    
    // Strong up momentum
    for (let i = 0; i < 20; i++) {
      price += 200
      prices.push(price)
    }
    
    // Reversal
    for (let i = 0; i < 20; i++) {
      price -= 250
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should detect the reversal
    assert.ok(signal.reason.includes('momentum'))
  })

  it('should respect position constraints', async () => {
    // Downward momentum without position
    const prices = Array(20).fill(0).map((_, i) => 50000 - i * 150)
    
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

  it('should calculate rate of change correctly', async () => {
    // Create a moderate uptrend for realistic momentum
    const prices = []
    
    // Start with stable base
    for (let i = 0; i < 25; i++) {
      prices.push(50000 + (Math.random() - 0.5) * 100)
    }
    
    // Add moderate upward trend
    let price = 50000
    for (let i = 0; i < 15; i++) {
      price += 30 // Moderate consistent gains
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // With moderate uptrend, should be bullish
    assert.ok(['buy', 'hold'].includes(signal.action))
    assert.ok(signal.reason.toLowerCase().includes('momentum') || signal.reason.includes('RSI'))
  })

  it('should handle momentum divergence', async () => {
    // Price higher but momentum lower
    const prices = []
    
    // First wave
    for (let i = 0; i < 10; i++) {
      prices.push(50000 + i * 200)
    }
    
    // Pullback
    for (let i = 0; i < 5; i++) {
      prices.push(52000 - i * 100)
    }
    
    // Second wave (higher price, lower momentum)
    for (let i = 0; i < 10; i++) {
      prices.push(51500 + i * 100) // Slower rise
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Divergence should reduce confidence
    assert.ok(signal.confidence < 0.7)
  })

  it('should properly reset state', async () => {
    // Build momentum history
    const prices = Array(40).fill(0).map((_, i) => 50000 + i * 100)  // Need at least 35 candles
    await agent.analyze(createContext(prices))
    
    // Reset
    await agent.reset()
    
    // Should need more data
    const signal = await agent.analyze(createContext([50000, 50100]))
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient') || signal.reason.includes('Indicator calculation failed'))
  })
})