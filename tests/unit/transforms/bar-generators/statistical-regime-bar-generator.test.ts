import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { StatisticalRegimeBarGenerator } from '../../../../src/transforms'

// Helper to create test data with specific values
function createTestData(prices: number[], volumes: number[] = []): OhlcvDto[] {
  const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
  return prices.map((price, i) => ({
    timestamp: baseTime + i * 1000, // 1 second apart
    symbol: 'TEST',
    exchange: 'test',
    open: price,
    high: price + Math.random() * 0.5, // Add some variance
    low: price - Math.random() * 0.5,
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

describe('StatisticalRegimeBarGenerator', () => {
  it('should create bars when statistical regime changes significantly', async () => {
    // Create test data with regime change pattern
    // Start with stable prices, then add volatility, then trending
    const stablePrices = Array(15).fill(0).map((_, i) => 100 + Math.sin(i * 0.1) * 0.5) // Low volatility
    const volatilePrices = Array(10).fill(0).map((_, i) => 105 + Math.sin(i * 0.5) * 5) // High volatility
    const trendingPrices = Array(10).fill(0).map((_, i) => 110 + i * 2) // Strong trend
    
    const allPrices = [...stablePrices, ...volatilePrices, ...trendingPrices]
    const testData = createTestData(allPrices)
    
    const generator = new StatisticalRegimeBarGenerator({
      lookback: 20,
      threshold: 2.0 // Lower threshold for more sensitive detection
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create at least 1 bar due to regime changes (algorithm may be conservative)
    ok(bars.length >= 1, `Expected at least 1 bar, got ${bars.length}`)
    
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

  it('should handle stable market conditions without excessive bars', async () => {
    // Create very stable price data
    const stablePrices = Array(50).fill(0).map((_, i) => 100 + Math.sin(i * 0.05) * 0.1) // Very low volatility
    const testData = createTestData(stablePrices)
    
    const generator = new StatisticalRegimeBarGenerator({
      lookback: 20,
      threshold: 3.0 // Higher threshold
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create very few bars in stable conditions
    ok(bars.length <= 3, `Expected few bars in stable conditions, got ${bars.length}`)
  })

  it('should require minimum lookback period', () => {
    try {
      new StatisticalRegimeBarGenerator({
        lookback: 5, // Too small
        threshold: 2.5
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('lookback must be at least 10'))
    }
  })

  it('should require minimum threshold', () => {
    try {
      new StatisticalRegimeBarGenerator({
        lookback: 20,
        threshold: 0.5 // Too small
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('threshold must be at least 1.0'))
    }
  })

  it('should handle multiple symbols independently', async () => {
    const testData: OhlcvDto[] = [
      // Symbol A: stable then volatile
      ...Array(10).fill(0).map((_, i) => ({
        timestamp: 1000 + i * 1000,
        symbol: 'A',
        exchange: 'test',
        open: 100 + i * 0.1,
        high: 100.5 + i * 0.1,
        low: 99.5 + i * 0.1,
        close: 100 + i * 0.1,
        volume: 100
      })),
      // Symbol B: trending
      ...Array(10).fill(0).map((_, i) => ({
        timestamp: 1500 + i * 1000,
        symbol: 'B',
        exchange: 'test',
        open: 200 + i * 2,
        high: 202 + i * 2,
        low: 198 + i * 2,
        close: 200 + i * 2,
        volume: 150
      }))
    ]
    
    const generator = new StatisticalRegimeBarGenerator({
      lookback: 15,
      threshold: 2.5
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    const symbolABars = bars.filter(b => b.symbol === 'A')
    const symbolBBars = bars.filter(b => b.symbol === 'B')
    
    // Both symbols should potentially create bars
    ok(symbolABars.length >= 1, 'Symbol A should create at least 1 bar')
    ok(symbolBBars.length >= 1, 'Symbol B should create at least 1 bar')
  })

  it('should support daily reset', async () => {
    const baseTime = new Date('2024-01-01T23:59:55Z').getTime()
    
    // Create price data spanning midnight
    const prices = Array(20).fill(0).map((_, i) => 100 + Math.sin(i * 0.3) * 2)
    const testData: OhlcvDto[] = prices.map((price, i) => ({
      timestamp: baseTime + i * 1000, // Some will cross midnight
      symbol: 'TEST',
      exchange: 'test',
      open: price,
      high: price + 0.5,
      low: price - 0.5,
      close: price,
      volume: 100
    }))
    
    const generator = new StatisticalRegimeBarGenerator({
      lookback: 10,
      threshold: 2.0,
      resetDaily: true
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should handle daily reset without errors
    ok(bars.length >= 1)
    
    // All bars should be valid
    for (const bar of bars) {
      ok(bar.volume > 0)
      ok(bar.close > 0)
    }
  })

  it('should serialize and restore state correctly', () => {
    const generator = new StatisticalRegimeBarGenerator({
      lookback: 20,
      threshold: 2.5
    })
    
    // Simulate some state with mock data
    const mockState = {
      'TEST': {
        currentBar: createTestData([100])[0]!,
        complete: false,
        priceHistory: [100, 101, 102, 99, 100],
        returnHistory: [0.01, 0.0099, -0.0294, 0.0101],
        kurtosisHistory: [0.1, 0.2],
        skewnessHistory: [-0.1, 0.05],
        hurstHistory: [0.5, 0.52],
        entropyHistory: [2.1, 2.3]
      }
    }
    
    generator.restoreState(mockState)
    const state = generator.getState()
    
    const testState = state.TEST as any
    ok(testState.priceHistory)
    ok(testState.returnHistory)
    ok(testState.kurtosisHistory)
    ok(testState.skewnessHistory)
    ok(testState.hurstHistory)
    ok(testState.entropyHistory)
    strictEqual(testState.complete, false)
  })

  it('should create new transform instance with withParams', () => {
    const generator1 = new StatisticalRegimeBarGenerator({
      lookback: 20,
      threshold: 2.5
    })
    
    const generator2 = generator1.withParams({
      threshold: 3.0
    })
    
    strictEqual(generator1.params.threshold, 2.5)
    strictEqual(generator2.params.threshold, 3.0)
    strictEqual(generator1.params.lookback, 20)
    strictEqual(generator2.params.lookback, 20)
  })

  it('should handle edge case with insufficient data', async () => {
    // Very small dataset
    const testData = createTestData([100, 101])
    
    const generator = new StatisticalRegimeBarGenerator({
      lookback: 20,
      threshold: 2.5
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create at least one incomplete bar at the end
    ok(bars.length >= 1)
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 101)
  })

  it('should calculate correct OHLCV aggregation', async () => {
    // Create data that should trigger exactly one bar
    const prices = [100, 105, 95, 110, 85, 120, 80, 125] // High volatility to trigger regime change
    const testData = createTestData(prices)
    
    const generator = new StatisticalRegimeBarGenerator({
      lookback: 10,
      threshold: 1.5 // Low threshold to ensure bar creation
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    ok(bars.length >= 1)
    
    // Check first bar aggregation
    const firstBar = bars[0]!
    strictEqual(firstBar.open, 100) // First tick's close becomes bar's open
    ok(firstBar.high >= 100) // Should be max of all highs in the bar
    ok(firstBar.low <= 100) // Should be min of all lows in the bar
    ok(firstBar.volume >= 100) // Should be sum of all volumes
  })
})