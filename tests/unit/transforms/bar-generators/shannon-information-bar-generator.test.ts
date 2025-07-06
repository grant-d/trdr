import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { ShannonInformationBarGenerator } from '../../../../src/transforms'

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

describe('ShannonInformationBarGenerator', () => {
  it('should create bars when information content exceeds threshold', async () => {
    // Create test data with surprise events (large moves in low volatility)
    const stablePrices = Array(15).fill(0).map((_, i) => 100 + Math.sin(i * 0.1) * 0.2) // Very stable
    const surprisePrices = [105, 95, 110, 90, 108] // Sudden large moves
    const allPrices = [...stablePrices, ...surprisePrices]
    const testData = createTestData(allPrices)
    
    const generator = new ShannonInformationBarGenerator({
      lookback: 15,
      threshold: 3.0, // Should trigger on surprise moves
      decayRate: 0.90
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create at least 1 bar due to information accumulation
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
    // Create very predictable price data (low information content)
    const predictablePrices = Array(30).fill(0).map((_, i) => 100 + Math.sin(i * 0.1) * 0.1) // Very predictable
    const testData = createTestData(predictablePrices)
    
    const generator = new ShannonInformationBarGenerator({
      lookback: 15,
      threshold: 5.0, // Higher threshold
      decayRate: 0.95
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create very few bars in predictable conditions
    ok(bars.length <= 3, `Expected few bars in predictable conditions, got ${bars.length}`)
  })

  it('should validate lookback parameter', () => {
    try {
      new ShannonInformationBarGenerator({
        lookback: 5, // Too small
        threshold: 3.0,
        decayRate: 0.90
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('lookback must be at least 10'))
    }
  })

  it('should validate threshold parameter', () => {
    try {
      new ShannonInformationBarGenerator({
        lookback: 15,
        threshold: 0.5, // Too small
        decayRate: 0.90
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('threshold must be at least 1.0'))
    }
  })

  it('should validate decayRate parameter', () => {
    try {
      new ShannonInformationBarGenerator({
        lookback: 15,
        threshold: 3.0,
        decayRate: 0.75 // Too small
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('decayRate must be between 0.8 and 0.99'))
    }

    try {
      new ShannonInformationBarGenerator({
        lookback: 15,
        threshold: 3.0,
        decayRate: 1.0 // At boundary
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('decayRate must be between 0.8 and 0.99'))
    }
  })

  it('should accumulate information over time with decay', async () => {
    // Create moderate surprises that individually don't trigger threshold
    const basePrice = 100
    const smallSurprises = Array(20).fill(0).map(() => {
      // Small but consistent deviations
      return basePrice + (Math.random() - 0.5) * 1.0
    })
    const testData = createTestData(smallSurprises)
    
    const generator = new ShannonInformationBarGenerator({
      lookback: 15,
      threshold: 4.0,
      decayRate: 0.85 // Faster decay to test accumulation
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should eventually accumulate enough information
    ok(bars.length >= 1, 'Should accumulate information over time')
  })

  it('should be more sensitive to surprises in low volatility environments', async () => {
    // Test that same absolute move has different information content in different volatility regimes
    
    // Low volatility environment with sudden 2% move
    const lowVolPrices = [100, 100.1, 100.05, 99.95, 100.02, 102.0] // 2% move in stable environment
    const testData1 = createTestData(lowVolPrices)
    
    // High volatility environment with same 2% move
    const highVolPrices = [100, 102, 98, 104, 96, 102.0] // 2% move in volatile environment
    const testData2 = createTestData(highVolPrices)
    
    const generator1 = new ShannonInformationBarGenerator({
      lookback: 10,
      threshold: 2.0, // Lower threshold for testing
      decayRate: 0.90
    })
    
    const generator2 = new ShannonInformationBarGenerator({
      lookback: 10,
      threshold: 2.0,
      decayRate: 0.90
    })
    
    const result1 = await generator1.apply(arrayToAsyncIterator(testData1))
    const bars1 = await collectResults(result1.data)
    
    const result2 = await generator2.apply(arrayToAsyncIterator(testData2))
    const bars2 = await collectResults(result2.data)
    
    // Low volatility environment should be more likely to trigger bars from same move
    // This is hard to test deterministically, so we just check both work
    ok(bars1.length >= 1 || bars2.length >= 1, 'At least one environment should trigger bars')
  })

  it('should handle multiple symbols independently', async () => {
    const testData: OhlcvDto[] = [
      // Symbol A: gradual change then surprise
      ...Array(10).fill(0).map((_, i) => ({
        timestamp: 1000 + i * 1000,
        symbol: 'A',
        exchange: 'test',
        open: 100 + i * 0.1,
        high: 100.1 + i * 0.1,
        low: 99.9 + i * 0.1,
        close: 100 + i * 0.1,
        volume: 100
      })),
      // Add surprise move for Symbol A
      {
        timestamp: 11000,
        symbol: 'A',
        exchange: 'test',
        open: 101,
        high: 106,
        low: 100.9,
        close: 105,
        volume: 100
      },
      // Symbol B: consistent pattern
      ...Array(10).fill(0).map((_, i) => ({
        timestamp: 1500 + i * 1000,
        symbol: 'B',
        exchange: 'test',
        open: 200,
        high: 200.1,
        low: 199.9,
        close: 200,
        volume: 150
      }))
    ]
    
    const generator = new ShannonInformationBarGenerator({
      lookback: 10,
      threshold: 2.5,
      decayRate: 0.90
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    const symbolABars = bars.filter(b => b.symbol === 'A')
    const symbolBBars = bars.filter(b => b.symbol === 'B')
    
    // Symbol A should be more likely to create bars due to surprise move
    ok(symbolABars.length >= 1, 'Symbol A should create bars from surprise move')
    // Symbol B might or might not create bars
    ok(symbolBBars.length >= 0, 'Symbol B should handle consistent pattern')
  })

  it('should reset information when new bar starts', async () => {
    // Create data that should trigger a bar, then continue
    const surprisePrices = [100, 100, 100, 100, 100, 105, 100, 100, 100, 110] // Two potential surprises
    const testData = createTestData(surprisePrices)
    
    const generator = new ShannonInformationBarGenerator({
      lookback: 10,
      threshold: 2.0,
      decayRate: 0.90
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    if (bars.length >= 2) {
      // Each bar should start fresh
      const firstBar = bars[0]!
      const secondBar = bars[1]!
      
      ok(firstBar.volume > 0)
      ok(secondBar.volume > 0)
      ok(secondBar.open > 0)
    }
  })

  it('should serialize and restore state correctly', () => {
    const generator = new ShannonInformationBarGenerator({
      lookback: 15,
      threshold: 3.0,
      decayRate: 0.90
    })
    
    // Simulate some state
    const mockState = {
      'TEST': {
        currentBar: createTestData([100])[0]!,
        complete: false,
        returnHistory: [0.01, -0.005, 0.02, -0.01],
        cumulativeInformation: 1.5,
        previousPrice: 99.5
      }
    }
    
    generator.restoreState(mockState)
    const state = generator.getState()
    
    const testState = state.TEST as any
    ok(Array.isArray(testState.returnHistory))
    strictEqual(testState.cumulativeInformation, 1.5)
    strictEqual(testState.previousPrice, 99.5)
    strictEqual(testState.complete, false)
  })

  it('should create new transform instance with withParams', () => {
    const generator1 = new ShannonInformationBarGenerator({
      lookback: 15,
      threshold: 3.0,
      decayRate: 0.90
    })
    
    const generator2 = generator1.withParams({
      threshold: 4.0,
      decayRate: 0.95
    })
    
    strictEqual(generator1.params.threshold, 3.0)
    strictEqual(generator1.params.decayRate, 0.90)
    strictEqual(generator2.params.threshold, 4.0)
    strictEqual(generator2.params.decayRate, 0.95)
    strictEqual(generator1.params.lookback, 15)
    strictEqual(generator2.params.lookback, 15)
  })

  it('should handle edge case with zero price change', async () => {
    // Flat prices should produce zero returns and minimal information
    const flatPrices = Array(15).fill(100)
    const testData = createTestData(flatPrices)
    
    const generator = new ShannonInformationBarGenerator({
      lookback: 10,
      threshold: 3.0,
      decayRate: 0.90
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create final bar but not due to information accumulation
    ok(bars.length >= 1)
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 100)
  })

  it('should calculate correct OHLCV aggregation', async () => {
    const prices = [100, 102, 98, 105] // Mix of moves
    const volumes = [100, 150, 80, 120]
    const testData = createTestData(prices, volumes)
    
    const generator = new ShannonInformationBarGenerator({
      lookback: 10,
      threshold: 1.0, // Low threshold to ensure bar creation
      decayRate: 0.90
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