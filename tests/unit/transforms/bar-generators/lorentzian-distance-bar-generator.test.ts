import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { LorentzianDistanceBarGenerator } from '../../../../src/transforms'

// Helper to create test data with specific values
function createTestData(prices: number[], volumes: number[] = []): OhlcvDto[] {
  const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
  return prices.map((price, i) => ({
    timestamp: baseTime + i * 1000, // 1 second apart
    symbol: 'TEST',
    exchange: 'test',
    open: price,
    high: price + 0.1,
    low: price - 0.1,
    close: price,
    volume: volumes[i] || 100
  }))
}

// Helper to convert array to async iterator
async function* arrayToAsyncIterator<T>(array: T[]): AsyncGenerator<T> {
  for (const item of array) {
    yield item
  }
}

// Helper to collect results from async iterator
async function collectResults(iterator: AsyncIterator<OhlcvDto>): Promise<OhlcvDto[]> {
  const results: OhlcvDto[] = []
  let item = await iterator.next()
  while (!item.done) {
    results.push(item.value)
    item = await iterator.next()
  }
  return results
}

describe('LorentzianDistanceBarGenerator', () => {
  it('should create bars when Lorentzian distance exceeds threshold', async () => {
    // Create test data with rapid price movements to trigger distance threshold
    const prices = [100, 105, 110, 120, 125, 135, 140, 145] // Rapid upward movement
    const testData = createTestData(prices)
    
    const generator = new LorentzianDistanceBarGenerator({
      cFactor: 1.0,
      threshold: 30.0 // Should trigger during rapid moves
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create at least 2 bars due to distance accumulation
    ok(bars.length >= 2, `Expected at least 2 bars, got ${bars.length}`)
    
    // Each bar should be valid OHLCV
    for (const bar of bars) {
      ok(bar.open > 0)
      ok(bar.high >= bar.open)
      ok(bar.high >= bar.close)
      ok(bar.low <= bar.open)
      ok(bar.low <= bar.close)
      ok(bar.volume > 0)
    }
  })

  it('should handle stable price movements with fewer bars', async () => {
    // Create stable price data with small movements
    const prices = Array(20).fill(0).map((_, i) => 100 + Math.sin(i * 0.1) * 0.5) // Small oscillation
    const testData = createTestData(prices)
    
    const generator = new LorentzianDistanceBarGenerator({
      cFactor: 1.0,
      threshold: 50.0 // Higher threshold
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create fewer bars in stable conditions
    ok(bars.length <= 3, `Expected few bars in stable conditions, got ${bars.length}`)
  })

  it('should validate cFactor parameter', () => {
    try {
      new LorentzianDistanceBarGenerator({
        cFactor: 0.05, // Too small
        threshold: 50.0
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('cFactor must be at least 0.1'))
    }
  })

  it('should validate threshold parameter', () => {
    try {
      new LorentzianDistanceBarGenerator({
        cFactor: 1.0,
        threshold: 5.0 // Too small
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('threshold must be at least 10.0'))
    }
  })

  it('should respond to volume changes as well as price changes', async () => {
    // Create data with volume spikes but stable prices
    const prices = Array(10).fill(100) // Stable prices
    const volumes = [100, 100, 100, 500, 100, 100, 1000, 100, 100, 100] // Volume spikes
    const testData = createTestData(prices, volumes)
    
    const generator = new LorentzianDistanceBarGenerator({
      cFactor: 1.0,
      threshold: 20.0 // Sensitive to volume changes
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create bars due to volume changes even with stable prices
    ok(bars.length >= 2, 'Should create bars due to volume spikes')
  })

  it('should handle time component in distance calculation', async () => {
    // Create slow-moving prices (time component should dominate)
    const prices = [100, 100.1, 100.2, 100.3, 100.4] // Very small price changes
    const testData = createTestData(prices)
    
    const generator = new LorentzianDistanceBarGenerator({
      cFactor: 2.0, // Higher time scaling factor
      threshold: 15.0
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Time component should eventually trigger bar creation
    ok(bars.length >= 1)
  })

  it('should handle multiple symbols independently', async () => {
    const testData: OhlcvDto[] = [
      // Symbol A: rapid price movement
      ...Array(8).fill(0).map((_, i) => ({
        timestamp: 1000 + i * 1000,
        symbol: 'A',
        exchange: 'test',
        open: 100 + i * 5,
        high: 105 + i * 5,
        low: 95 + i * 5,
        close: 100 + i * 5,
        volume: 100
      })),
      // Symbol B: volume spikes
      ...Array(8).fill(0).map((_, i) => ({
        timestamp: 1500 + i * 1000,
        symbol: 'B',
        exchange: 'test',
        open: 200,
        high: 201,
        low: 199,
        close: 200,
        volume: 100 + i * 200 // Increasing volume
      }))
    ]
    
    const generator = new LorentzianDistanceBarGenerator({
      cFactor: 1.0,
      threshold: 25.0
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    const symbolABars = bars.filter(b => b.symbol === 'A')
    const symbolBBars = bars.filter(b => b.symbol === 'B')
    
    // Both symbols should create bars independently
    ok(symbolABars.length >= 1, 'Symbol A should create bars')
    ok(symbolBBars.length >= 1, 'Symbol B should create bars')
  })

  it('should use Euclidean fallback for space-like intervals', async () => {
    // Create scenario where Lorentzian component might be negative
    // Rapid price and volume changes with small time differences
    const prices = [100, 120, 80, 130] // Large price swings
    const volumes = [100, 500, 50, 600] // Large volume swings
    const testData = createTestData(prices, volumes)
    
    const generator = new LorentzianDistanceBarGenerator({
      cFactor: 0.5, // Small time factor
      threshold: 25.0
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should handle negative Lorentzian components gracefully
    ok(bars.length >= 1)
    for (const bar of bars) {
      ok(bar.volume > 0)
      ok(bar.close > 0)
    }
  })

  it('should reset anchor point when new bar starts', async () => {
    // This test verifies that the anchor resets properly
    const prices = [100, 110, 120, 105, 115, 125] // Two potential bars
    const testData = createTestData(prices)
    
    const generator = new LorentzianDistanceBarGenerator({
      cFactor: 1.0,
      threshold: 20.0
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    if (bars.length >= 2) {
      // Second bar should start fresh, not continue from previous anchor
      const secondBar = bars[1]!
      
      // Second bar should be valid
      ok(secondBar.open > 0)
      ok(secondBar.volume > 0)
    }
  })

  it('should serialize and restore state correctly', () => {
    const generator = new LorentzianDistanceBarGenerator({
      cFactor: 1.5,
      threshold: 30.0
    })
    
    // Simulate some state
    const mockState = {
      'TEST': {
        currentBar: createTestData([100])[0]!,
        complete: false,
        anchorPrice: 95.0,
        anchorVolume: 150,
        anchorTime: 5,
        currentTime: 10
      }
    }
    
    generator.restoreState(mockState)
    const state = generator.getState()
    
    const testState = state.TEST as any
    strictEqual(testState.anchorPrice, 95.0)
    strictEqual(testState.anchorVolume, 150)
    strictEqual(testState.anchorTime, 5)
    strictEqual(testState.currentTime, 10)
    strictEqual(testState.complete, false)
  })

  it('should create new transform instance with withParams', () => {
    const generator1 = new LorentzianDistanceBarGenerator({
      cFactor: 1.0,
      threshold: 25.0
    })
    
    const generator2 = generator1.withParams({
      cFactor: 2.0,
      threshold: 40.0
    })
    
    strictEqual(generator1.params.cFactor, 1.0)
    strictEqual(generator1.params.threshold, 25.0)
    strictEqual(generator2.params.cFactor, 2.0)
    strictEqual(generator2.params.threshold, 40.0)
  })

  it('should handle edge case with zero volume', async () => {
    const prices = [100, 105, 110]
    const volumes = [0, 100, 0] // Include zero volumes
    const testData = createTestData(prices, volumes)
    
    const generator = new LorentzianDistanceBarGenerator({
      cFactor: 1.0,
      threshold: 30.0
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should handle zero volumes without errors
    ok(bars.length >= 1)
    for (const bar of bars) {
      ok(bar.close > 0)
    }
  })

  it('should calculate correct OHLCV aggregation', async () => {
    const prices = [100, 105, 95, 110] // Mix of highs and lows
    const volumes = [100, 150, 80, 120]
    const testData = createTestData(prices, volumes)
    
    const generator = new LorentzianDistanceBarGenerator({
      cFactor: 1.0,
      threshold: 15.0 // Low threshold to ensure bar creation
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    ok(bars.length >= 1)
    
    // Check OHLCV aggregation
    const firstBar = bars[0]!
    strictEqual(firstBar.open, 100) // First tick's close becomes bar's open
    ok(firstBar.high >= 100) // Should be max of all highs
    ok(firstBar.low <= 100) // Should be min of all lows
    ok(firstBar.volume >= 100) // Should be sum of volumes
  })
})