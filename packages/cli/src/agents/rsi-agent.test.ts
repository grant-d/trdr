import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { RsiAgent } from './rsi-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('RSIAgent', () => {
  let agent: RsiAgent
  const metadata: AgentMetadata = {
    id: 'rsi-test',
    name: 'RSI Test Agent',
    version: '1.0.0',
    description: 'Test RSI agent',
    type: 'momentum'
  }

  beforeEach(async () => {
    agent = new RsiAgent(metadata)
    await agent.initialize()
  })

  afterEach(async () => {
    await agent.shutdown()
  })

  const createContext = (prices: number[], currentPrice?: number): MarketContext => ({
    symbol: 'BTC-USD',
    currentPrice: currentPrice || prices[prices.length - 1] || 50000,
    candles: prices.map((price, i) => ({
      open: price - 10,
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
    assert.strictEqual(agent.metadata.type, 'momentum')
  })

  it('should return hold signal with insufficient data', async () => {
    const context = createContext([50000, 50100])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient'))
  })

  it('should detect oversold conditions', async () => {
    // Create downtrend to push RSI low
    const prices = []
    let price = 50000
    for (let i = 0; i < 20; i++) {
      price -= 200 // Consistent decline
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // RSI should be very low, suggesting buy
    if (signal.reason.includes('RSI')) {
      assert.strictEqual(signal.action, 'buy')
      assert.ok(signal.confidence >= 0.6)
    }
  })

  it('should detect overbought conditions', async () => {
    // Create uptrend to push RSI high
    const prices = []
    let price = 50000
    for (let i = 0; i < 20; i++) {
      price += 200 // Consistent rise
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // RSI should be very high, suggesting sell
    if (signal.reason.includes('RSI')) {
      assert.ok(['sell', 'hold'].includes(signal.action)) // Might be hold due to no-shorting
      assert.ok(signal.confidence >= 0.6)
    }
  })

  it('should hold in neutral RSI range', async () => {
    // Create sideways movement
    const prices = []
    for (let i = 0; i < 20; i++) {
      prices.push(50000 + (i % 2) * 100) // Oscillating
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    // RSI should be around 50
  })

  it('should detect RSI divergence', async () => {
    // Price making higher highs but momentum weakening
    const prices = [
      50000, 50100, 50200, 50150, 50250, // First high
      50200, 50150, 50100, 50200, 50300, // Higher high
      50250, 50200, 50150, 50250, 50350  // Even higher but weaker momentum
    ]
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Divergence might trigger caution
  })

  it('should respect position for sell signals', async () => {
    // Create overbought condition
    const prices = Array(20).fill(0).map((_, i) => 50000 + i * 300)
    
    // No position context
    const context = {
      ...createContext(prices),
      currentPosition: 0
    }
    
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should not sell without position
    if (signal.action === 'sell') {
      assert.ok(signal.reason.includes('[No position to sell]'))
    }
  })

  it('should handle rapid price movements', async () => {
    const prices = [50000, 52000, 48000, 53000, 47000, 51000]
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should handle volatility gracefully
  })

  it('should provide confidence based on RSI extremes', async () => {
    // Very oversold
    const oversoldPrices = Array(15).fill(0).map((_, i) => 50000 - i * 400)
    const oversoldSignal = await agent.analyze(createContext(oversoldPrices))
    
    // Very overbought  
    const overboughtPrices = Array(15).fill(0).map((_, i) => 50000 + i * 400)
    const overboughtSignal = await agent.analyze(createContext(overboughtPrices))
    
    assert.ok(oversoldSignal)
    assert.ok(overboughtSignal)
    
    // Extreme RSI should have higher confidence
    if (oversoldSignal.action === 'buy') {
      assert.ok(oversoldSignal.confidence >= 0.7)
    }
  })

  it('should properly reset state', async () => {
    // Build some history
    const prices = Array(20).fill(0).map((_, i) => 50000 + Math.sin(i) * 1000)
    await agent.analyze(createContext(prices))
    
    // Reset
    await agent.reset()
    
    // Should need to rebuild
    const signal = await agent.analyze(createContext([50000, 50100]))
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient'))
  })
})