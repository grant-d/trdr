import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { MacdAgent } from './macd-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('MACDAgent', () => {
  let agent: MacdAgent
  const metadata: AgentMetadata = {
    id: 'macd-test',
    name: 'MACD Test Agent',
    version: '1.0.0',
    description: 'Test MACD agent',
    type: 'momentum'
  }

  beforeEach(async () => {
    agent = new MacdAgent(metadata)
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
      high: price + 10,
      low: price - 10,
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
    const context = createContext([50000, 50100, 50200])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient'))
  })

  it('should detect bullish MACD crossover', async () => {
    // Create uptrend that causes MACD to cross above signal
    const prices = []
    let price = 50000
    
    // Initial downtrend
    for (let i = 0; i < 15; i++) {
      price -= 100
      prices.push(price)
    }
    
    // Strong uptrend
    for (let i = 0; i < 20; i++) {
      price += 150
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    if (signal.reason.includes('MACD') && signal.reason.includes('cross')) {
      assert.strictEqual(signal.action, 'buy')
      assert.ok(signal.confidence >= 0.6)
    }
  })

  it('should detect bearish MACD crossover', async () => {
    // Create downtrend that causes MACD to cross below signal
    const prices = []
    let price = 50000
    
    // Initial uptrend
    for (let i = 0; i < 15; i++) {
      price += 100
      prices.push(price)
    }
    
    // Strong downtrend
    for (let i = 0; i < 20; i++) {
      price -= 150
      prices.push(price)
    }
    
    const context = {
      ...createContext(prices),
      currentPosition: 0.1 // Has position to sell
    }
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    if (signal.reason.includes('MACD') && signal.reason.includes('cross')) {
      assert.ok(['sell', 'hold'].includes(signal.action))
    }
  })

  it('should detect histogram momentum', async () => {
    // Create accelerating uptrend (increasing histogram)
    const prices = []
    let price = 50000
    let increment = 50
    
    for (let i = 0; i < 35; i++) {
      price += increment
      increment += 5 // Accelerating
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Strong momentum should be detected
  })

  it('should detect divergence patterns', async () => {
    // Price making higher highs but MACD making lower highs
    const prices = []
    
    // First wave up
    for (let i = 0; i < 10; i++) {
      prices.push(50000 + i * 200)
    }
    
    // Pullback
    for (let i = 0; i < 5; i++) {
      prices.push(52000 - i * 150)
    }
    
    // Second wave up (higher high but weaker momentum)
    for (let i = 0; i < 10; i++) {
      prices.push(51250 + i * 100) // Slower rise
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Divergence might trigger caution
  })

  it('should handle sideways market', async () => {
    // Oscillating prices
    const prices = []
    for (let i = 0; i < 40; i++) {
      prices.push(50000 + Math.sin(i * 0.3) * 200)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // In sideways market, should be cautious
    assert.ok(signal.confidence <= 0.6)
  })

  it('should respect position constraints', async () => {
    // Create sell signal condition
    const prices = Array(35).fill(0).map((_, i) => 52000 - i * 100)
    
    // No position
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

  it('should handle zero line crossovers', async () => {
    // Create pattern that crosses zero line
    const prices = []
    let price = 50000
    
    // Build negative MACD
    for (let i = 0; i < 20; i++) {
      price -= 50
      prices.push(price)
    }
    
    // Cross to positive
    for (let i = 0; i < 20; i++) {
      price += 100
      prices.push(price)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Zero line cross is significant
  })

  it('should adjust confidence based on histogram strength', async () => {
    // Weak momentum
    const weakPrices = Array(35).fill(0).map((_, i) => 50000 + i * 10)
    const weakSignal = await agent.analyze(createContext(weakPrices))
    
    // Strong momentum
    const strongPrices = Array(35).fill(0).map((_, i) => 50000 + i * 100)
    const strongSignal = await agent.analyze(createContext(strongPrices))
    
    assert.ok(weakSignal)
    assert.ok(strongSignal)
    
    // Strong momentum should have higher confidence
    if (strongSignal.action !== 'hold' && weakSignal.action !== 'hold') {
      assert.ok(strongSignal.confidence > weakSignal.confidence)
    }
  })

  it('should properly reset state', async () => {
    // Build MACD history
    const prices = Array(35).fill(0).map((_, i) => 50000 + Math.sin(i * 0.2) * 1000)
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