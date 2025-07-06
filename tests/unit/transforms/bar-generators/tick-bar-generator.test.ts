import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { TickBarGenerator } from '../../../../src/transforms'

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

describe('TickBarGenerator', () => {
  it('should create bars after specified number of ticks', async () => {
    // Create 10 ticks with increasing prices
    const prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    const testData = createTestData(prices)
    
    const generator = new TickBarGenerator({
      in: [],
      out: [],
      ticksPerBar: 3
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create 3 complete bars (9 ticks) + 1 incomplete bar (1 tick)
    strictEqual(bars.length, 4)
    
    // First bar: ticks 0-2 (prices 100, 101, 102)
    strictEqual(bars[0]!.open, 100)    // First tick open
    strictEqual(bars[0]!.high, 102.1)   // Max of all highs: max(100.1, 101.1, 102.1)
    strictEqual(bars[0]!.low, 99.9)     // Min of all lows: min(99.9, 100.9, 101.9)
    strictEqual(bars[0]!.close, 102)    // Last tick close
    strictEqual(bars[0]!.volume, 300)   // Sum: 100 + 100 + 100
    
    // Second bar: ticks 3-5 (prices 103, 104, 105)
    strictEqual(bars[1]!.open, 103)
    strictEqual(bars[1]!.high, 105.1)   // Max of all highs
    strictEqual(bars[1]!.low, 102.9)    // Min of all lows
    strictEqual(bars[1]!.close, 105)
    strictEqual(bars[1]!.volume, 300)
    
    // Third bar: ticks 6-8 (prices 106, 107, 108)
    strictEqual(bars[2]!.open, 106)
    strictEqual(bars[2]!.high, 108.1)
    strictEqual(bars[2]!.low, 105.9)
    strictEqual(bars[2]!.close, 108)
    strictEqual(bars[2]!.volume, 300)
    
    // Fourth bar: incomplete, only 1 tick (price 109)
    strictEqual(bars[3]!.open, 109)
    strictEqual(bars[3]!.close, 109)
    strictEqual(bars[3]!.volume, 100)
  })

  it('should handle multiple symbols independently', async () => {
    const testData: OhlcvDto[] = [
      // Symbol A: 5 ticks
      { timestamp: 1000, symbol: 'A', exchange: 'test', open: 100, high: 100.5, low: 99.5, close: 100, volume: 50 },
      { timestamp: 2000, symbol: 'A', exchange: 'test', open: 101, high: 101.5, low: 100.5, close: 101, volume: 60 },
      { timestamp: 3000, symbol: 'A', exchange: 'test', open: 102, high: 102.5, low: 101.5, close: 102, volume: 70 },
      { timestamp: 4000, symbol: 'A', exchange: 'test', open: 103, high: 103.5, low: 102.5, close: 103, volume: 80 },
      { timestamp: 5000, symbol: 'A', exchange: 'test', open: 104, high: 104.5, low: 103.5, close: 104, volume: 90 },
      // Symbol B: 4 ticks
      { timestamp: 1500, symbol: 'B', exchange: 'test', open: 200, high: 200.5, low: 199.5, close: 200, volume: 100 },
      { timestamp: 2500, symbol: 'B', exchange: 'test', open: 201, high: 201.5, low: 200.5, close: 201, volume: 110 },
      { timestamp: 3500, symbol: 'B', exchange: 'test', open: 202, high: 202.5, low: 201.5, close: 202, volume: 120 },
      { timestamp: 4500, symbol: 'B', exchange: 'test', open: 203, high: 203.5, low: 202.5, close: 203, volume: 130 },
    ]
    
    const generator = new TickBarGenerator({
      in: [],
      out: [],
      ticksPerBar: 3
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create bars for both symbols
    const symbolABars = bars.filter(b => b.symbol === 'A')
    const symbolBBars = bars.filter(b => b.symbol === 'B')
    
    // Symbol A: 5 ticks = 1 complete bar (3 ticks) + 1 incomplete bar (2 ticks)
    strictEqual(symbolABars.length, 2)
    
    // Symbol A first bar: ticks 0-2
    strictEqual(symbolABars[0]!.open, 100)
    strictEqual(symbolABars[0]!.close, 102)
    strictEqual(symbolABars[0]!.volume, 180) // 50 + 60 + 70
    
    // Symbol A second bar: incomplete with 2 ticks
    strictEqual(symbolABars[1]!.open, 103)
    strictEqual(symbolABars[1]!.close, 104)
    strictEqual(symbolABars[1]!.volume, 170) // 80 + 90
    
    // Symbol B: 4 ticks = 1 complete bar (3 ticks) + 1 incomplete bar (1 tick)
    strictEqual(symbolBBars.length, 2)
    
    // Symbol B first bar
    strictEqual(symbolBBars[0]!.open, 200)
    strictEqual(symbolBBars[0]!.close, 202)
    strictEqual(symbolBBars[0]!.volume, 330) // 100 + 110 + 120
  })

  it('should support daily reset', async () => {
    const baseTime = new Date('2024-01-01T23:59:58Z').getTime()
    const testData: OhlcvDto[] = [
      // Day 1: 2 ticks before midnight
      { timestamp: baseTime, symbol: 'TEST', exchange: 'test', open: 100, high: 100.1, low: 99.9, close: 100, volume: 100 },
      { timestamp: baseTime + 1000, symbol: 'TEST', exchange: 'test', open: 101, high: 101.1, low: 100.9, close: 101, volume: 100 },
      // Day 2: 4 ticks after midnight
      { timestamp: baseTime + 3000, symbol: 'TEST', exchange: 'test', open: 102, high: 102.1, low: 101.9, close: 102, volume: 100 },
      { timestamp: baseTime + 4000, symbol: 'TEST', exchange: 'test', open: 103, high: 103.1, low: 102.9, close: 103, volume: 100 },
      { timestamp: baseTime + 5000, symbol: 'TEST', exchange: 'test', open: 104, high: 104.1, low: 103.9, close: 104, volume: 100 },
      { timestamp: baseTime + 6000, symbol: 'TEST', exchange: 'test', open: 105, high: 105.1, low: 104.9, close: 105, volume: 100 },
    ]
    
    const generator = new TickBarGenerator({
      in: [],
      out: [],
      ticksPerBar: 3,
      resetDaily: true
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // With daily reset:
    // Day 1: 2 ticks = 1 incomplete bar
    // Day 2: 4 ticks = 1 complete bar (3 ticks) + 1 incomplete bar (1 tick)
    strictEqual(bars.length, 3)
    
    // First bar: incomplete from day 1
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 101)
    strictEqual(bars[0]!.volume, 200)
    
    // Second bar: complete from day 2
    strictEqual(bars[1]!.open, 102)
    strictEqual(bars[1]!.close, 104)
    strictEqual(bars[1]!.volume, 300)
    
    // Third bar: incomplete from day 2
    strictEqual(bars[2]!.open, 105)
    strictEqual(bars[2]!.close, 105)
    strictEqual(bars[2]!.volume, 100)
  })

  it('should validate parameters', () => {
    // Zero ticks per bar
    try {
      new TickBarGenerator({
        in: [],
        out: [],
        ticksPerBar: 0
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('ticksPerBar must be at least 1'))
    }
    
    // Negative ticks per bar
    try {
      new TickBarGenerator({
        in: [],
        out: [],
        ticksPerBar: -1
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('ticksPerBar must be at least 1'))
    }
  })

  it('should serialize and restore state correctly', () => {
    const generator = new TickBarGenerator({
      in: [],
      out: [],
      ticksPerBar: 3
    })
    
    // Simulate some state
    const mockState = {
      'TEST': {
        currentBar: createTestData([100])[0]!,
        tickCount: 2,
        complete: false
      }
    }
    
    generator.restoreState(mockState)
    const state = generator.getState()
    
    strictEqual(state.TEST!.tickCount, 2)
    strictEqual(state.TEST!.complete, false)
  })
})