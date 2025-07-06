import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { VolumeBarGenerator } from '../../../../src/transforms'

// Helper to create test data with specific values
function createTestData(prices: number[], volumes: number[]): OhlcvDto[] {
  const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
  return prices.map((price, i) => ({
    timestamp: baseTime + i * 1000, // 1 second apart
    symbol: 'TEST',
    exchange: 'test',
    open: price,
    high: price + 0.1,
    low: price - 0.1,
    close: price,
    volume: volumes[i]!
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

describe('VolumeBarGenerator', () => {
  it('should create bars when volume threshold is reached', async () => {
    // Create test data with specific volumes
    const prices = [100, 101, 102, 103, 104, 105, 106, 107]
    const volumes = [150, 200, 100, 250, 50, 300, 150, 100]
    const testData = createTestData(prices, volumes)
    
    const generator = new VolumeBarGenerator({
      in: [],
      out: [],
      volumePerBar: 500
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Expected bars:
    // Bar 1: 150 + 200 + 100 = 450 (not complete, needs 50 more)
    //        + 250 = 700 total, so bar completes at tick 3
    // Bar 2: starts fresh at tick 4 with 50, + 300 = 350 (not complete)
    //        + 150 = 500 total, so bar completes at tick 6
    // Bar 3: starts at tick 7 with 100 (incomplete)
    
    strictEqual(bars.length, 3)
    
    // First bar: includes ticks 0-3 (total volume = 700)
    strictEqual(bars[0]!.open, 100)     // First tick
    strictEqual(bars[0]!.high, 103.1)    // Max high from all 4 ticks
    strictEqual(bars[0]!.low, 99.9)      // Min low from all 4 ticks
    strictEqual(bars[0]!.close, 103)     // Last tick
    strictEqual(bars[0]!.volume, 700)    // 150 + 200 + 100 + 250
    
    // Second bar: includes ticks 4-6 (total volume = 500)
    strictEqual(bars[1]!.open, 104)
    strictEqual(bars[1]!.high, 106.1)
    strictEqual(bars[1]!.low, 103.9)
    strictEqual(bars[1]!.close, 106)
    strictEqual(bars[1]!.volume, 500)    // 50 + 300 + 150
    
    // Third bar: incomplete, only tick 7
    strictEqual(bars[2]!.open, 107)
    strictEqual(bars[2]!.close, 107)
    strictEqual(bars[2]!.volume, 100)
  })

  it('should handle exact volume matches', async () => {
    // Test when cumulative volume exactly matches threshold
    const prices = [100, 101, 102, 103]
    const volumes = [200, 300, 250, 250]  // Exactly 500 each pair
    const testData = createTestData(prices, volumes)
    
    const generator = new VolumeBarGenerator({
      in: [],
      out: [],
      volumePerBar: 500
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create 2 complete bars
    strictEqual(bars.length, 2)
    
    // First bar: ticks 0-1 (volume = 500)
    strictEqual(bars[0]!.volume, 500)
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 101)
    
    // Second bar: ticks 2-3 (volume = 500)
    strictEqual(bars[1]!.volume, 500)
    strictEqual(bars[1]!.open, 102)
    strictEqual(bars[1]!.close, 103)
  })

  it('should handle multiple symbols independently', async () => {
    const testData: OhlcvDto[] = [
      // Symbol A
      { timestamp: 1000, symbol: 'A', exchange: 'test', open: 100, high: 100.5, low: 99.5, close: 100, volume: 300 },
      { timestamp: 2000, symbol: 'A', exchange: 'test', open: 101, high: 101.5, low: 100.5, close: 101, volume: 250 },
      { timestamp: 3000, symbol: 'A', exchange: 'test', open: 102, high: 102.5, low: 101.5, close: 102, volume: 200 },
      // Symbol B
      { timestamp: 1500, symbol: 'B', exchange: 'test', open: 200, high: 200.5, low: 199.5, close: 200, volume: 400 },
      { timestamp: 2500, symbol: 'B', exchange: 'test', open: 201, high: 201.5, low: 200.5, close: 201, volume: 150 },
      { timestamp: 3500, symbol: 'B', exchange: 'test', open: 202, high: 202.5, low: 201.5, close: 202, volume: 100 },
    ]
    
    const generator = new VolumeBarGenerator({
      in: [],
      out: [],
      volumePerBar: 500
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    const symbolABars = bars.filter(b => b.symbol === 'A')
    const symbolBBars = bars.filter(b => b.symbol === 'B')
    
    // Symbol A: 300 + 250 = 550 (1 complete bar), 200 (1 incomplete bar)
    strictEqual(symbolABars.length, 2)
    strictEqual(symbolABars[0]!.volume, 550)  // First two ticks
    strictEqual(symbolABars[1]!.volume, 200)  // Last tick
    
    // Symbol B: 400 + 150 = 550 (1 complete bar), 100 (1 incomplete bar)
    strictEqual(symbolBBars.length, 2)
    strictEqual(symbolBBars[0]!.volume, 550)
    strictEqual(symbolBBars[1]!.volume, 100)
  })

  it('should handle very small volumes', async () => {
    // Test with small volumes that take many ticks to complete a bar
    const prices = [100, 101, 102, 103, 104, 105]
    const volumes = [10, 20, 15, 25, 30, 50]
    const testData = createTestData(prices, volumes)
    
    const generator = new VolumeBarGenerator({
      in: [],
      out: [],
      volumePerBar: 100
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Bar 1: 10 + 20 + 15 + 25 + 30 = 100 (exactly)
    // Bar 2: 50 (incomplete)
    strictEqual(bars.length, 2)
    
    strictEqual(bars[0]!.volume, 100)
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 104)  // 5 ticks included
    
    strictEqual(bars[1]!.volume, 50)
    strictEqual(bars[1]!.open, 105)
    strictEqual(bars[1]!.close, 105)
  })

  it('should validate parameters', () => {
    // Zero volume
    try {
      new VolumeBarGenerator({
        in: [],
        out: [],
        volumePerBar: 0
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('volumePerBar must be greater than 0'))
    }
    
    // Negative volume
    try {
      new VolumeBarGenerator({
        in: [],
        out: [],
        volumePerBar: -100
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('volumePerBar must be greater than 0'))
    }
  })

  it('should serialize and restore state correctly', () => {
    const generator = new VolumeBarGenerator({
      in: [],
      out: [],
      volumePerBar: 1000
    })
    
    // Simulate some state
    const mockState = {
      'TEST': {
        currentBar: createTestData([100], [500])[0]!,
        accumulatedVolume: 750,
        complete: false
      }
    }
    
    generator.restoreState(mockState)
    const state = generator.getState()
    
    const testState = state.TEST as any
    strictEqual(testState.accumulatedVolume, 750)
    strictEqual(testState.complete, false)
  })
})