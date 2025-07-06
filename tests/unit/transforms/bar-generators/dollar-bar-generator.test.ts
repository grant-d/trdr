import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { DollarBarGenerator } from '../../../../src/transforms'

// Helper to create test data with specific values
function createTestData(prices: number[], volumes: number[]): OhlcvDto[] {
  const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
  return prices.map((price, i) => ({
    timestamp: baseTime + i * 1000, // 1 second apart
    symbol: 'TEST',
    exchange: 'test',
    open: price,
    high: price + 0.5,
    low: price - 0.5,
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

describe('DollarBarGenerator', () => {
  it('should create bars when dollar value threshold is reached', async () => {
    // Create test data with specific prices and volumes
    // Dollar value = price * volume
    const prices = [10, 20, 30, 40, 50]
    const volumes = [100, 50, 40, 30, 20]
    // Dollar values: 1000, 1000, 1200, 1200, 1000
    const testData = createTestData(prices, volumes)
    
    const generator = new DollarBarGenerator({
      in: [],
      out: [],
      dollarValuePerBar: 2500
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Expected bars with non-overlapping behavior:
    // Tick 0: $1000 (10*100), accumulated = $1000
    // Tick 1: $1000 (20*50), accumulated = $2000
    // Tick 2: $1200 (30*40), would make $3200 > $2500, so bar 1 completes
    // Bar 1: ticks 0,1 with $2000 total
    // Bar 2 starts with tick 2
    // Tick 2: $1200, accumulated = $1200
    // Tick 3: $1200 (40*30), accumulated = $2400
    // Tick 4: $1000 (50*20), would make $3400 > $2500, so bar 2 completes
    // Bar 2: ticks 2,3 with $2400 total
    // Bar 3 starts with tick 4 (incomplete)
    
    strictEqual(bars.length, 3)
    
    // First bar: ticks 0,1
    strictEqual(bars[0]!.open, 10)
    strictEqual(bars[0]!.close, 20)
    strictEqual(bars[0]!.volume, 150)   // 100 + 50
    
    // Second bar: ticks 2,3
    strictEqual(bars[1]!.open, 30)
    strictEqual(bars[1]!.close, 40)
    strictEqual(bars[1]!.volume, 70)    // 40 + 30
    
    // Third bar: tick 4 (incomplete)
    strictEqual(bars[2]!.open, 50)
    strictEqual(bars[2]!.close, 50)
    strictEqual(bars[2]!.volume, 20)
  })

  it('should handle exact dollar value matches', async () => {
    // Test when cumulative dollar value exactly matches threshold
    const prices = [100, 50, 25, 40]
    const volumes = [10, 20, 40, 25]
    // Dollar values: 1000, 1000, 1000, 1000
    const testData = createTestData(prices, volumes)
    
    const generator = new DollarBarGenerator({
      in: [],
      out: [],
      dollarValuePerBar: 2000
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should create 2 complete bars
    strictEqual(bars.length, 2)
    
    // First bar: ticks 0-1 ($2000)
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 50)
    strictEqual(bars[0]!.volume, 30)    // 10 + 20
    
    // Second bar: ticks 2-3 ($2000)
    strictEqual(bars[1]!.open, 25)
    strictEqual(bars[1]!.close, 40)
    strictEqual(bars[1]!.volume, 65)    // 40 + 25
  })

  it('should handle different price levels correctly', async () => {
    // High price, low volume vs low price, high volume
    const prices = [1000, 1, 500, 2]
    const volumes = [1, 1000, 2, 500]
    // Dollar values: 1000, 1000, 1000, 1000
    const testData = createTestData(prices, volumes)
    
    const generator = new DollarBarGenerator({
      in: [],
      out: [],
      dollarValuePerBar: 3000
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Non-overlapping bars:
    // Tick 0: $1000, accumulated = $1000
    // Tick 1: $1000, accumulated = $2000
    // Tick 2: $1000, would make $3000, bar completes WITHOUT tick 2
    // Bar 1: ticks 0,1 with volume 1001
    // Bar 2 starts with tick 2
    // Tick 3: $1000, accumulated = $2000 (incomplete)
    strictEqual(bars.length, 2)
    
    // Price range should be extreme due to different price levels
    strictEqual(bars[0]!.high, 1000.5)  // High price tick
    strictEqual(bars[0]!.low, 0.5)       // Low price tick
    strictEqual(bars[0]!.volume, 1001)   // 1 + 1000
  })

  it('should handle multiple symbols independently', async () => {
    const testData: OhlcvDto[] = [
      // Symbol A - tech stock (high price, low volume)
      { timestamp: 1000, symbol: 'AAPL', exchange: 'test', open: 150, high: 151, low: 149, close: 150, volume: 10 },
      { timestamp: 2000, symbol: 'AAPL', exchange: 'test', open: 151, high: 152, low: 150, close: 151, volume: 15 },
      { timestamp: 3000, symbol: 'AAPL', exchange: 'test', open: 152, high: 153, low: 151, close: 152, volume: 20 },
      // Symbol B - penny stock (low price, high volume)
      { timestamp: 1500, symbol: 'PENNY', exchange: 'test', open: 0.50, high: 0.55, low: 0.45, close: 0.50, volume: 5000 },
      { timestamp: 2500, symbol: 'PENNY', exchange: 'test', open: 0.51, high: 0.56, low: 0.46, close: 0.51, volume: 6000 },
    ]
    
    const generator = new DollarBarGenerator({
      in: [],
      out: [],
      dollarValuePerBar: 3000
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    const appleBars = bars.filter(b => b.symbol === 'AAPL')
    const pennyBars = bars.filter(b => b.symbol === 'PENNY')
    
    // AAPL: 
    // Tick 0: $1500 (150*10), accumulates to $1500
    // Tick 1: $2265 (151*15), accumulates to $3765 > 3000, bar completes
    // Bar 1: ticks 0,1 ($3765 total)
    // Tick 2: $3040 (152*20), starts new bar (incomplete)
    strictEqual(appleBars.length, 2)
    
    // PENNY:
    // Tick 0: $2500 (0.50*5000), accumulates to $2500  
    // Tick 1: $3060 (0.51*6000), accumulates to $5560 > 3000, bar completes
    // Bar 1: ticks 0,1 ($5560 total)
    strictEqual(pennyBars.length, 1)
  })

  it('should calculate dollar values using close price', async () => {
    // Test that dollar value uses close price (not open/high/low)
    const testData: OhlcvDto[] = [
      { timestamp: 1000, symbol: 'TEST', exchange: 'test', open: 90, high: 110, low: 90, close: 100, volume: 25 },
      { timestamp: 2000, symbol: 'TEST', exchange: 'test', open: 110, high: 120, low: 95, close: 105, volume: 20 },
    ]
    
    const generator = new DollarBarGenerator({
      in: [],
      out: [],
      dollarValuePerBar: 2500
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Dollar values: (100*25) + (105*20) = 2500 + 2100 = 4600
    // Should create 1 complete bar
    strictEqual(bars.length, 1)
    strictEqual(bars[0]!.volume, 45)
  })

  it('should validate parameters', () => {
    // Zero dollar value
    try {
      new DollarBarGenerator({
        in: [],
        out: [],
        dollarValuePerBar: 0
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('dollarValuePerBar must be greater than 0'))
    }
    
    // Negative dollar value
    try {
      new DollarBarGenerator({
        in: [],
        out: [],
        dollarValuePerBar: -1000
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('dollarValuePerBar must be greater than 0'))
    }
  })

  it('should serialize and restore state correctly', () => {
    const generator = new DollarBarGenerator({
      in: [],
      out: [],
      dollarValuePerBar: 10000
    })
    
    // Simulate some state
    const mockState = {
      'TEST': {
        currentBar: createTestData([100], [50])[0]!,
        accumulatedValue: 7500, // $7500 accumulated so far
        complete: false
      }
    }
    
    generator.restoreState(mockState)
    const state = generator.getState()
    
    const testState = state['TEST'] as any
    strictEqual(testState.accumulatedValue, 7500)
    strictEqual(testState.complete, false)
  })
})