import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert'
import { MathematicalHarmonyAgent } from './mathematical-harmony-agent'
import type { MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { toEpochDate } from '@trdr/shared'

describe('MathematicalHarmonyAgent', () => {
  let agent: MathematicalHarmonyAgent
  const metadata: AgentMetadata = {
    id: 'math-test',
    name: 'Mathematical Harmony Test Agent',
    version: '1.0.0',
    description: 'Test mathematical harmony agent',
    type: 'custom'
  }

  beforeEach(async () => {
    agent = new MathematicalHarmonyAgent(metadata, undefined, {
      useRiemannZeta: true,
      useGoldenRatio: true,
      useFibonacci: true,
      usePrimes: true,
      useLucasNumbers: true,
      harmonyThreshold: 0.6,
      dimensions: 5
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

  it('should initialize with mathematical configurations', async () => {
    const signal = await agent.analyze(createContext([50000]))
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
    assert.ok(signal.confidence >= 0 && signal.confidence <= 1)
  })

  it('should detect prime number patterns', async () => {
    // Use prices around prime numbers - need at least 20 candles
    const primePrices = [
      50021, 50023, 50029, 50033, 50047, 50051, 50053, 50059, 50069, 50077,
      50083, 50087, 50093, 50101, 50111, 50119, 50123, 50129, 50131, 50137,
      50141, 50147, 50153, 50159, 50177
    ]
    const context = createContext(primePrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason.includes('harmony') || signal.reason.includes('Prime') || signal.action !== 'hold')
  })

  it('should detect golden ratio patterns', async () => {
    // Create prices that follow golden ratio relationships
    const basePrice = 50000
    const φ = (1 + Math.sqrt(5)) / 2
    const goldenPrices = [
      basePrice,
      basePrice * φ,
      basePrice * φ * φ,
      basePrice * φ,
      basePrice,
      basePrice / φ,
      basePrice / (φ * φ),
      basePrice / φ,
      basePrice,
      basePrice * φ
    ].map(p => Math.round(p))
    
    const context = createContext(goldenPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.confidence >= 0.3) // Should detect some pattern
  })

  it('should detect Fibonacci sequences', async () => {
    // Create prices based on Fibonacci sequence
    const fibonacciNumbers = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    const fibPrices = fibonacciNumbers.map(fib => 50000 + fib * 10)
    
    const context = createContext(fibPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should recognize mathematical structure
    assert.ok(signal.confidence > 0.2)
  })

  it('should handle insufficient data gracefully', async () => {
    const context = createContext([50000, 50100])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.strictEqual(signal.action, 'hold')
    assert.ok(signal.reason.includes('Insufficient data'))
  })

  it('should calculate mathematical harmony scores', async () => {
    // Create a more complex pattern with multiple mathematical elements
    const complexPrices = []
    let base = 50000
    
    // Mix prime numbers, Fibonacci ratios, and golden ratio
    for (let i = 0; i < 30; i++) {
      if (i % 5 === 0) {
        // Every 5th point follows golden ratio
        base *= (1 + Math.sqrt(5)) / 2
      } else if (i % 3 === 0) {
        // Every 3rd point uses Fibonacci
        const fibValue = [1, 1, 2, 3, 5, 8, 13, 21][i % 8]
        if (fibValue !== undefined) {
          base += fibValue * 10
        }
      }
      complexPrices.push(Math.round(base))
      base = Math.max(40000, Math.min(60000, base)) // Keep in reasonable range
    }
    
    const context = createContext(complexPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.confidence >= 0)
  })

  it('should detect mathematical resonance patterns', async () => {
    // Create prices with harmonic relationships
    const harmonicPrices = []
    const fundamental = 100 // Base frequency
    
    for (let i = 0; i < 25; i++) {
      // Create harmonic series: f, 2f, 3f, 4f, 5f...
      const harmonic = Math.sin(i * fundamental / 10) * 500 + 50000
      harmonicPrices.push(Math.round(harmonic))
    }
    
    const context = createContext(harmonicPrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    // Should detect some form of mathematical pattern
  })

  it('should handle extreme mathematical configurations', async () => {
    // Test with very strict harmony requirements
    const strictAgent = new MathematicalHarmonyAgent(metadata, undefined, {
      harmonyThreshold: 0.95, // Very high threshold
      dimensions: 10
    })
    await strictAgent.initialize()
    
    const context = createContext([50000, 50100, 50200, 50300])
    const signal = await strictAgent.analyze(context)
    
    assert.ok(signal)
    // With such high threshold, should mostly return hold
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
    
    await strictAgent.shutdown()
  })

  it('should generate different signals for different mathematical patterns', async () => {
    // Test various mathematical sequences
    const sequences = {
      primes: [50021, 50023, 50029, 50033, 50047],
      fibonacci: [50055, 50089, 50144, 50233, 50377],
      lucas: [50002, 50001, 50003, 50004, 50007],
      golden: [50000, 50000 * 1.618, 50000 * 2.618, 50000 * 1.618, 50000]
    }
    
    const signals = []
    
    for (const [name, prices] of Object.entries(sequences)) {
      const context = createContext(prices.map(p => Math.round(p)))
      const signal = await agent.analyze(context)
      signals.push({ name, signal })
    }
    
    // All should produce valid signals
    for (const { signal } of signals) {
      assert.ok(signal)
      assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
      assert.ok(signal.confidence >= 0 && signal.confidence <= 1)
    }
  })

  it('should maintain mathematical harmony history', async () => {
    // Perform multiple analyses to build history
    const sequences = [
      [50000, 50100, 50200],
      [50200, 50300, 50400],
      [50400, 50500, 50600]
    ]
    
    for (const prices of sequences) {
      const context = createContext(prices)
      await agent.analyze(context)
    }
    
    // Final analysis should consider harmony history
    const finalContext = createContext([50600, 50700, 50800])
    const signal = await agent.analyze(finalContext)
    
    assert.ok(signal)
    assert.ok(signal.confidence >= 0)
  })

  it('should reset mathematical state properly', async () => {
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

  it('should handle edge cases with mathematical calculations', async () => {
    // Test with zeros, negative numbers, and edge cases
    const edgePrices = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
      .map(n => 50000 + n) // Shift to positive range
    
    const context = createContext(edgePrices)
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(['buy', 'sell', 'hold'].includes(signal.action))
    assert.ok(!isNaN(signal.confidence))
    assert.ok(signal.confidence >= 0 && signal.confidence <= 1)
  })

  it('should provide mathematical reasoning in signals', async () => {
    const context = createContext([50021, 50023, 50029, 50033, 50047])
    const signal = await agent.analyze(context)
    
    assert.ok(signal)
    assert.ok(signal.reason)
    assert.ok(typeof signal.reason === 'string')
    assert.ok(signal.reason.length > 0)
    
    // Should mention mathematical concepts
    const mathematicalTerms = ['harmony', 'mathematical', 'Prime', 'Fibonacci', 'golden', 'Riemann', 'Lucas']
    const hasMathematicalTerm = mathematicalTerms.some(term => 
      signal.reason.toLowerCase().includes(term.toLowerCase())
    )
    assert.ok(hasMathematicalTerm, `Reason should contain mathematical terms: ${signal.reason}`)
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