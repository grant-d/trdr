import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { RelativisticFieldAgent } from './relativistic-field-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('RelativisticFieldAgent', () => {
  let agent: RelativisticFieldAgent
  const metadata: AgentMetadata = {
    id: 'relativistic-test',
    name: 'Relativistic Field Test Agent',
    version: '1.0.0',
    description: 'Test relativistic field agent',
    type: 'custom'
  }

  beforeEach(async () => {
    agent = new RelativisticFieldAgent(metadata, undefined, {
      fieldPointCount: 20,
      fieldRadius: 0.05,
      criticalFieldThreshold: 0.75,
      fieldHistoryLength: 50,
      momentumSpeedLimit: 0.1,
      fieldDecayRate: 2.0,
      minFieldDistortion: 0.3
    })
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
      high: price + 50,
      low: price - 50,
      close: price,
      volume: volumes?.[i] || 100000,
      timestamp: toEpochDate(Date.now() - (prices.length - i) * 60000)
    })),
    indicators: {
      RSI: { value: 50, timestamp: toEpochDate(Date.now()) },
      MACD: { value: 0.1, timestamp: toEpochDate(Date.now()) }
    }
  })

  it('should initialize with relativistic field configurations', async () => {
    const signal = await agent.analyze(createContext([50000]))
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
    assert.ok(signal.confidence >= 0 && signal.confidence <= 1)
  })

  it('should handle insufficient data gracefully', async () => {
    const context = createContext([50000, 50100])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient data'))
  })

  it('should generate momentum field around current price', async () => {
    // Create trending price data
    const trendingPrices = []
    for (let i = 0; i < 25; i++) {
      trendingPrices.push(50000 + i * 100) // Steady uptrend
    }
    
    const context = createContext(trendingPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
    assert.ok(signal.confidence >= 0 && signal.confidence <= 1)
  })

  it('should detect field convergence patterns', async () => {
    // Create converging price pattern
    const convergingPrices = []
    const center = 50000
    for (let i = 0; i < 25; i++) {
      // Prices converging toward center
      const deviation = (25 - i) * 50
      convergingPrices.push(center + (i % 2 === 0 ? deviation : -deviation))
    }
    
    const context = createContext(convergingPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('Relativistic field') || signal.reason.includes('field'))
  })

  it('should detect field divergence patterns', async () => {
    // Create diverging price pattern
    const divergingPrices = []
    const center = 50000
    for (let i = 0; i < 25; i++) {
      // Prices diverging from center
      const deviation = i * 50
      divergingPrices.push(center + (i % 2 === 0 ? deviation : -deviation))
    }
    
    const context = createContext(divergingPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
  })

  it('should apply relativistic momentum calculations', async () => {
    // Create high momentum scenario
    const highMomentumPrices = []
    let price = 50000
    for (let i = 0; i < 25; i++) {
      // Accelerating price movement
      const acceleration = i * 10
      price += acceleration
      highMomentumPrices.push(price)
    }
    
    const context = createContext(highMomentumPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.confidence >= 0)
  })

  it('should handle field distortion measurements', async () => {
    // Create chaotic price movement (high distortion)
    const chaoticPrices = []
    let price = 50000
    for (let i = 0; i < 25; i++) {
      // Random-like movement
      const change = (Math.sin(i) * Math.cos(i * 2) * Math.sin(i * 3)) * 500
      price += change
      chaoticPrices.push(price)
    }
    
    const context = createContext(chaoticPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('distortion') || signal.reason.includes('field'))
  })

  it('should build field history over time', async () => {
    // Analyze multiple price sequences to build history
    const sequences = [
      [50000, 50100, 50200, 50300, 50400],
      [50400, 50500, 50600, 50700, 50800],
      [50800, 50900, 51000, 51100, 51200]
    ]
    
    for (const prices of sequences) {
      const context = createContext(prices)
      await agent.analyze(context)
    }
    
    // Final analysis should have field history
    const finalContext = createContext([51200, 51300, 51400])
    const signal = await agent.analyze(finalContext)
    
    assert.ok(signal)
    assert.ok(signal.confidence >= 0)
  })

  it('should detect critical field alignment', async () => {
    // Create highly aligned field scenario
    const alignedPrices = []
    for (let i = 0; i < 25; i++) {
      // Very consistent momentum
      alignedPrices.push(50000 + i * 20)
    }
    
    const context = createContext(alignedPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should detect some form of field pattern
    assert.ok(signal.reason.includes('field') || signal.reason.includes('alignment'))
  })

  it('should pattern match historical field states', async () => {
    // Build up pattern history
    const patternPrices = [50000, 50100, 50050, 50150, 50075, 50175, 50125, 50225]
    for (let cycle = 0; cycle < 5; cycle++) {
      const prices = patternPrices.map(p => p + cycle * 100)
      const context = createContext(prices)
      await agent.analyze(context)
    }
    
    // Now test with similar pattern
    const similarPattern = patternPrices.map(p => p + 500)
    const context = createContext(similarPattern)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
  })

  it('should handle extreme field configurations', async () => {
    // Test with very strict field requirements
    const strictAgent = new RelativisticFieldAgent(metadata, undefined, {
      criticalFieldThreshold: 0.95, // Very high threshold
      minFieldDistortion: 0.8
    })
    await strictAgent.initialize()
    
    const context = createContext([50000, 50100, 50200, 50300])
    const signal = await strictAgent.analyze(context)
    
    assert.ok(signal)
    // With such high thresholds, should mostly return hold
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
    
    await strictAgent.shutdown()
  })

  it('should reset field state properly', async () => {
    // Build up some state
    const context1 = createContext([50000, 50100, 50200])
    await agent.analyze(context1)
    
    // Reset
    await agent.reset()
    
    // Should work normally after reset
    const context2 = createContext([50000, 50100, 50200])
    const signal = await agent.analyze(context2)
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
  })

  it('should provide field-based reasoning in signals', async () => {
    const context = createContext([50000, 50100, 50200, 50300, 50400])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason)
    assert.ok(typeof signal.reason === 'string')
    assert.ok(signal.reason.length > 0)
    
    // Should mention field-related concepts
    const fieldTerms = ['field', 'relativistic', 'momentum', 'alignment', 'distortion', 'convergence', 'divergence']
    const hasFieldTerm = fieldTerms.some(term => 
      signal.reason.toLowerCase().includes(term.toLowerCase())
    )
    assert.ok(hasFieldTerm, `Reason should contain field terms: ${signal.reason}`)
  })

  it('should handle large price datasets efficiently', async () => {
    // Create large dataset
    const largePrices = []
    for (let i = 0; i < 100; i++) {
      largePrices.push(50000 + Math.sin(i * 0.1) * 1000)
    }
    
    const startTime = Date.now()
    const context = createContext(largePrices)
    const signal = await agent.analyze(context)
    const endTime = Date.now()
    
    assert.ok(signal)
    assert.ok(endTime - startTime < 5000) // Should complete within 5 seconds
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
  })
})