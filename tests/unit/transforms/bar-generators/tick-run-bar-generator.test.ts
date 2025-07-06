import { ok, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { TickRunBarGenerator } from '../../../../src/transforms'

// Helper to create test data
function createTestData(prices: number[]): OhlcvDto[] {
  const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
  return prices.map((price, i) => ({
    timestamp: baseTime + i * 1000, // 1 second apart
    symbol: 'TEST',
    exchange: 'test',
    open: price,
    high: price + 0.1,
    low: price - 0.1,
    close: price,
    volume: 100
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

describe('TickRunBarGenerator', () => {
  it('should create bars when run length is reached', async () => {
    const prices = [100, 101, 102, 103, 102, 101, 100, 99, 98] // Up run of 4, then down run
    const testData = createTestData(prices)
    
    const generator = new TickRunBarGenerator({
      in: [],
      out: [],
      runLength: 3
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create 3 bars: one after 3 up ticks, one after 3 down ticks, and incomplete bar at end
    strictEqual(bars.length, 3)
    
    // First bar should complete after 3 consecutive up ticks
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 103)
    strictEqual(bars[0]!.volume, 400) // 4 ticks * 100 volume each
    
    // Second bar should be after 3 consecutive down ticks
    strictEqual(bars[1]!.open, 102)
    strictEqual(bars[1]!.close, 100)
    strictEqual(bars[1]!.volume, 300) // 3 ticks * 100 volume each
    
    // Third bar is incomplete (only 2 down ticks)
    strictEqual(bars[2]!.open, 99)
    strictEqual(bars[2]!.close, 98)
    strictEqual(bars[2]!.volume, 200) // 2 ticks * 100 volume each
  })

  it('should reset run when direction changes', async () => {
    const prices = [100, 101, 100, 101, 102, 103] // Up, down (reset), then up run
    const testData = createTestData(prices)
    
    const generator = new TickRunBarGenerator({
      in: [],
      out: [],
      runLength: 3
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create 1 bar that completes after run of 3
    // Pattern: 100->101 (up), 100 (down-reset), 101->102->103 (3 consecutive up = complete)
    strictEqual(bars.length, 1)
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 103)
    strictEqual(bars[0]!.volume, 600) // All 6 ticks
  })

  it('should handle neutral ticks (no price change)', async () => {
    const prices = [100, 101, 101, 102, 103] // Up, neutral, up, up
    const testData = createTestData(prices)
    
    const generator = new TickRunBarGenerator({
      in: [],
      out: [],
      runLength: 3
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create 1 complete bar (neutral doesn't count but doesn't reset)
    strictEqual(bars.length, 1)
    
    // Bar completes after run of 3 non-neutral ticks in same direction
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 103)
    strictEqual(bars[0]!.volume, 500) // All 5 ticks
  })

  it('should support volume-based runs', async () => {
    const prices = [100, 101, 102, 103, 104]
    const testData = createTestData(prices)
    // Modify volumes for testing
    testData[0]!.volume = 50
    testData[1]!.volume = 100
    testData[2]!.volume = 150
    testData[3]!.volume = 200
    testData[4]!.volume = 100
    
    const generator = new TickRunBarGenerator({
      in: [],
      out: [],
      runLength: 3,
      useVolume: true
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Volume run should trigger based on volume units
    ok(bars.length >= 1)
  })

  it('should handle multiple symbols', async () => {
    const testData: OhlcvDto[] = [
      { timestamp: 1000, symbol: 'A', exchange: 'test', open: 100, high: 100.1, low: 99.9, close: 100, volume: 100 },
      { timestamp: 2000, symbol: 'B', exchange: 'test', open: 200, high: 200.1, low: 199.9, close: 200, volume: 100 },
      { timestamp: 3000, symbol: 'A', exchange: 'test', open: 101, high: 101.1, low: 100.9, close: 101, volume: 100 },
      { timestamp: 4000, symbol: 'B', exchange: 'test', open: 201, high: 201.1, low: 200.9, close: 201, volume: 100 },
      { timestamp: 5000, symbol: 'A', exchange: 'test', open: 102, high: 102.1, low: 101.9, close: 102, volume: 100 },
      { timestamp: 6000, symbol: 'B', exchange: 'test', open: 202, high: 202.1, low: 201.9, close: 202, volume: 100 },
      { timestamp: 7000, symbol: 'A', exchange: 'test', open: 103, high: 103.1, low: 102.9, close: 103, volume: 100 },
    ]
    
    const generator = new TickRunBarGenerator({
      in: [],
      out: [],
      runLength: 3
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create bars for both symbols independently
    const symbolABars = bars.filter(b => b.symbol === 'A')
    const symbolBBars = bars.filter(b => b.symbol === 'B')
    
    // Symbol A: has 4 ticks (up run of 3 completes a bar with all 4 ticks)
    strictEqual(symbolABars.length, 1)
    strictEqual(symbolABars[0]!.volume, 400) // All 4 ticks
    // Symbol B: only has 3 ticks (up run of 2, incomplete bar at end)
    strictEqual(symbolBBars.length, 1)
    strictEqual(symbolBBars[0]!.volume, 300) // All 3 ticks
  })

  it('should validate parameters', () => {
    throws(() => {
      new TickRunBarGenerator({
        in: [],
        out: [],
        runLength: 0
      })
    }, /runLength must be greater than 0/)
    
    throws(() => {
      new TickRunBarGenerator({
        in: [],
        out: [],
        runLength: -1
      })
    }, /runLength must be greater than 0/)
  })

  it('should serialize and restore state', () => {
    const generator = new TickRunBarGenerator({
      in: [],
      out: [],
      runLength: 3
    })
    
    // Simulate some state
    const mockState = {
      'TEST': {
        currentBar: createTestData([100])[0]!,
        currentRunLength: 2,
        currentRunDirection: 'up' as const,
        previousClose: 101,
        volumeInRun: 200,
        tickCount: 2,
        complete: false
      }
    }
    
    generator.restoreState(mockState)
    const state = generator.getState()
    
    const testState = state['TEST'] as any
    strictEqual(testState.currentRunLength, 2)
    strictEqual(testState.currentRunDirection, 'up')
    strictEqual(testState.previousClose, 101)
    strictEqual(testState.volumeInRun, 200)
  })

  it('should support daily reset', async () => {
    const baseTime = new Date('2024-01-01T23:59:50Z').getTime()
    const testData: OhlcvDto[] = [
      { timestamp: baseTime, symbol: 'TEST', exchange: 'test', open: 100, high: 100.1, low: 99.9, close: 100, volume: 100 },
      { timestamp: baseTime + 5000, symbol: 'TEST', exchange: 'test', open: 101, high: 101.1, low: 100.9, close: 101, volume: 100 },
      // Next day
      { timestamp: baseTime + 15000, symbol: 'TEST', exchange: 'test', open: 102, high: 102.1, low: 101.9, close: 102, volume: 100 },
      { timestamp: baseTime + 20000, symbol: 'TEST', exchange: 'test', open: 103, high: 103.1, low: 102.9, close: 103, volume: 100 },
      { timestamp: baseTime + 25000, symbol: 'TEST', exchange: 'test', open: 104, high: 104.1, low: 103.9, close: 104, volume: 100 },
    ]
    
    const generator = new TickRunBarGenerator({
      in: [],
      out: [],
      runLength: 3,
      resetDaily: true
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create 2 bars: incomplete from day 1, complete from day 2
    strictEqual(bars.length, 2)
    
    // First bar is the incomplete bar from day 1
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 101) 
    strictEqual(bars[0]!.volume, 200) // 2 ticks from day 1
    
    // Second bar completes after 3 consecutive up ticks in day 2
    strictEqual(bars[1]!.open, 102) // First tick of new day
    strictEqual(bars[1]!.close, 104)
    strictEqual(bars[1]!.volume, 300) // 3 ticks from day 2
  })
})