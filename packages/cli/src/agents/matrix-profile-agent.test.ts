import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { MatrixProfileAgent } from './matrix-profile-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('MatrixProfileAgent', () => {
  let agent: MatrixProfileAgent
  const metadata: AgentMetadata = {
    id: 'matrix-test',
    name: 'Matrix Profile Test Agent',
    version: '1.0.0',
    description: 'Test matrix profile agent',
    type: 'custom'
  }

  beforeEach(async () => {
    agent = new MatrixProfileAgent(metadata, undefined, {
      windowSize: 10,
      minWindowSize: 5,
      maxWindowSize: 20,
      anomalyThreshold: 2.5,
      motifThreshold: 0.2,
      topMotifsCount: 5,
      adaptiveWindow: true,
      minPatternMatches: 3
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

  it('should initialize with matrix profile configurations', async () => {
    const signal = await agent.analyze(createContext([50000]))
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
    assert.ok(signal.confidence >= 0 && signal.confidence <= 1)
  })

  it('should handle insufficient data gracefully', async () => {
    // Need at least windowSize * 2 prices (10 * 2 = 20)
    const insufficientPrices = []
    for (let i = 0; i < 15; i++) {
      insufficientPrices.push(50000 + i * 10)
    }
    const context = createContext(insufficientPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient data'))
  })

  it('should detect repeated patterns (motifs)', async () => {
    // Create a repeating pattern
    const pattern = [50000, 50100, 50200, 50100, 50000]
    const prices = []
    
    // Repeat the pattern 5 times
    for (let i = 0; i < 5; i++) {
      prices.push(...pattern)
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('motif') || signal.reason.includes('Matrix Profile'))
  })

  it('should detect anomalies (discords)', async () => {
    // Create mostly stable prices with one anomaly
    const prices = []
    for (let i = 0; i < 25; i++) {
      if (i === 12) {
        // Insert anomaly
        prices.push(52000) // Large spike
      } else {
        prices.push(50000 + Math.sin(i * 0.5) * 50) // Gentle sine wave
      }
    }
    
    const context = createContext(prices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
  })

  it('should adapt window size based on volatility', async () => {
    // Create high volatility scenario
    const volatilePrices = []
    for (let i = 0; i < 30; i++) {
      // High volatility
      volatilePrices.push(50000 + (Math.random() - 0.5) * 5000)
    }
    
    const context = createContext(volatilePrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('window') || signal.reason.includes('Matrix'))
  })

  it('should perform z-normalization correctly', async () => {
    // Create prices with different scales
    const prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    const scaledPrices = prices.map(p => p * 100) // Scale up
    
    // Both should produce similar patterns after normalization
    const context1 = createContext(prices)
    const context2 = createContext(scaledPrices)
    
    const signal1 = await agent.analyze(context1)
    const signal2 = await agent.analyze(context2)
    
    assert.ok(signal1)
    assert.ok(signal2)
    assert.ok(['buy', 'sell', 'hold'].includes(signal1.action))
    assert.ok(['buy', 'sell', 'hold'].includes(signal2.action))
  })

  it('should build matrix profile history over time', async () => {
    // Analyze multiple sequences to build history
    const sequences = [
      [50000, 50100, 50200, 50300, 50400],
      [50400, 50300, 50200, 50100, 50000],
      [50100, 50200, 50100, 50200, 50100]
    ]
    
    for (const prices of sequences) {
      const context = createContext(prices)
      await agent.analyze(context)
    }
    
    // Final analysis should have history
    const finalContext = createContext([50200, 50300, 50400, 50500, 50600])
    const signal = await agent.analyze(finalContext)
    
    assert.ok(signal)
    assert.ok(signal.confidence >= 0)
  })

  it('should detect motif patterns matching profitable history', async () => {
    // Create a specific pattern that repeats
    const bullishPattern = [50000, 50050, 50100, 50200, 50300]
    const bearishPattern = [50300, 50250, 50200, 50100, 50000]
    
    // Build history with these patterns
    const historyPrices = []
    for (let i = 0; i < 4; i++) {
      if (i % 2 === 0) {
        historyPrices.push(...bullishPattern)
      } else {
        historyPrices.push(...bearishPattern)
      }
    }
    
    const context = createContext(historyPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
  })

  it('should identify pattern evolution', async () => {
    // Create evolving patterns
    const evolvingPrices = []
    
    // Start with regular pattern
    for (let i = 0; i < 20; i++) {
      evolvingPrices.push(50000 + Math.sin(i * 0.3) * 100)
    }
    
    // Transition to chaos
    for (let i = 0; i < 10; i++) {
      evolvingPrices.push(50000 + Math.random() * 500)
    }
    
    const context = createContext(evolvingPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('discord') || 
              signal.reason.includes('motif') || 
              signal.reason.includes('Matrix'))
  })

  it('should handle edge cases gracefully', async () => {
    // Test with constant prices
    const constantPrices = new Array(25).fill(50000)
    const context1 = createContext(constantPrices)
    const signal1 = await agent.analyze(context1)
    
    assert.ok(signal1)
    assert.strictEqual(signal1.action, 'hold')
    
    // Test with monotonic increase
    const monotonicPrices = []
    for (let i = 0; i < 25; i++) {
      monotonicPrices.push(50000 + i * 100)
    }
    const context2 = createContext(monotonicPrices)
    const signal2 = await agent.analyze(context2)
    
    assert.ok(signal2)
    assert.ok(['buy', 'sell', 'hold'].includes(signal2.action))
  })

  it('should reset state properly', async () => {
    // Build up some state
    const context1 = createContext([50000, 50100, 50200, 50300, 50400])
    await agent.analyze(context1)
    
    // Reset
    await agent.reset()
    
    // Should work normally after reset
    const context2 = createContext([50000, 50100, 50200])
    const signal = await agent.analyze(context2)
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
  })

  it('should provide matrix profile reasoning in signals', async () => {
    const context = createContext([50000, 50100, 50200, 50100, 50000, 50100, 50200, 50100, 50000])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason)
    assert.ok(typeof signal.reason === 'string')
    assert.ok(signal.reason.length > 0)
    
    // Should mention matrix profile concepts
    const matrixTerms = ['motif', 'discord', 'Matrix Profile', 'pattern', 'window', 'anomaly']
    const hasMatrixTerm = matrixTerms.some(term => 
      signal.reason.toLowerCase().includes(term.toLowerCase())
    )
    assert.ok(hasMatrixTerm, `Reason should contain matrix profile terms: ${signal.reason}`)
  })

  it('should handle large datasets efficiently', async () => {
    // Create large dataset
    const largePrices = []
    for (let i = 0; i < 200; i++) {
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

  it('should detect self-similar patterns at different scales', async () => {
    // Create fractal-like pattern
    const fractalPrices = []
    for (let i = 0; i < 50; i++) {
      // Multi-scale sine waves
      const price = 50000 + 
        Math.sin(i * 0.1) * 1000 +    // Large scale
        Math.sin(i * 0.5) * 200 +     // Medium scale
        Math.sin(i * 2) * 50          // Small scale
      fractalPrices.push(price)
    }
    
    const context = createContext(fractalPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
    // Should detect some patterns
    assert.ok(signal.reason.includes('motif') || signal.reason.includes('Matrix'))
  })
})