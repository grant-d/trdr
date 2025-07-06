import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { TickImbalanceBarGenerator } from '../../../../src/transforms'

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

describe('TickImbalanceBarGenerator', () => {
  it('should create bars when tick imbalance threshold is reached', async () => {
    // Create test data with strong directional moves
    // Buy tick: price > previous, Sell tick: price < previous
    const prices = [100, 101, 102, 103, 104, 103, 102, 101, 100, 99, 98]
    // Pattern: -, buy, buy, buy, buy, sell, sell, sell, sell, sell, sell
    const testData = createTestData(prices)
    
    const generator = new TickImbalanceBarGenerator({
      in: [],
      out: [],
      imbalanceThreshold: 3
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Expected bars:
    // Bar 1 accumulation:
    // Tick 0 (100): start, no classification
    // Tick 1 (101): buy (101 > 100), buy=1, sell=0, |1-0|=1
    // Tick 2 (102): buy (102 > 101), buy=2, sell=0, |2-0|=2
    // Tick 3 (103): buy (103 > 102), buy=3, sell=0, |3-0|=3 ✓ COMPLETES
    
    // Bar 2 accumulation (starts fresh at tick 4):
    // Tick 4 (104): buy (104 > 103), buy=1, sell=0, |1-0|=1
    // Tick 5 (103): sell (103 < 104), buy=1, sell=1, |1-1|=0
    // Tick 6 (102): sell (102 < 103), buy=1, sell=2, |1-2|=1
    // Tick 7 (101): sell (101 < 102), buy=1, sell=3, |1-3|=2
    // Tick 8 (100): sell (100 < 101), buy=1, sell=4, |1-4|=3 ✓ COMPLETES
    
    // Bar 3 (incomplete):
    // Tick 9 (99): sell (99 < 100), buy=0, sell=1, |0-1|=1
    // Tick 10 (98): sell (98 < 99), buy=0, sell=2, |0-2|=2
    
    strictEqual(bars.length, 3)
    
    // First bar: completes with buy imbalance
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 103)
    strictEqual(bars[0]!.volume, 400)   // 4 ticks * 100
    
    // Second bar: completes with sell imbalance  
    strictEqual(bars[1]!.open, 104)
    strictEqual(bars[1]!.close, 100)
    strictEqual(bars[1]!.volume, 500)   // 5 ticks * 100
    
    // Third bar: incomplete
    strictEqual(bars[2]!.open, 99)
    strictEqual(bars[2]!.close, 98)
    strictEqual(bars[2]!.volume, 200)   // 2 ticks * 100
  })

  it('should handle volume-based imbalance', async () => {
    // Test volume imbalance instead of tick count
    const prices = [100, 101, 100, 101, 102]
    const volumes = [100, 200, 150, 300, 250]
    // Pattern: -, buy(200), sell(150), buy(300), buy(250)
    const testData = createTestData(prices, volumes)
    
    const generator = new TickImbalanceBarGenerator({
      in: [],
      out: [],
      imbalanceThreshold: 500,
      useVolume: true
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Volume imbalance calculation:
    // Tick 0: no classification
    // Tick 1: buy, buyVol=200, sellVol=0, |200-0|=200
    // Tick 2: sell, buyVol=200, sellVol=150, |200-150|=50
    // Tick 3: buy, buyVol=500, sellVol=150, |500-150|=350
    // Tick 4: buy, buyVol=750, sellVol=150, |750-150|=600 >= 500
    
    strictEqual(bars.length, 1)
    
    // All ticks in one bar (completes at tick 4)
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 102)
    strictEqual(bars[0]!.volume, 1000)  // 100+200+150+300+250
  })

  it('should handle neutral ticks correctly', async () => {
    // Neutral ticks (no price change) should not affect imbalance
    const prices = [100, 101, 101, 102, 102, 103, 102, 101]
    // Pattern: -, buy, neutral, buy, neutral, buy, sell, sell
    const testData = createTestData(prices)
    
    const generator = new TickImbalanceBarGenerator({
      in: [],
      out: [],
      imbalanceThreshold: 2
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Imbalance progression:
    // Tick 0: neutral
    // Tick 1: buy=1, sell=0, |1-0|=1
    // Tick 2: neutral, no change
    // Tick 3: buy=2, sell=0, |2-0|=2 >= 2 (bar completes)
    // Bar 2 starts:
    // Tick 4: neutral
    // Tick 5: buy=1, sell=0, |1-0|=1
    // Tick 6: sell, buy=1, sell=1, |1-1|=0
    // Tick 7: sell, buy=1, sell=2, |1-2|=1 (never reaches threshold)
    
    strictEqual(bars.length, 2)
    
    // First bar completes at tick 3
    strictEqual(bars[0]!.open, 100)
    strictEqual(bars[0]!.close, 102)
    strictEqual(bars[0]!.volume, 400)   // 4 ticks
    
    // Second bar is incomplete
    strictEqual(bars[1]!.open, 102)
    strictEqual(bars[1]!.close, 101)
    strictEqual(bars[1]!.volume, 400)   // 4 ticks
  })

  it('should handle alternating buy/sell pattern', async () => {
    // Alternating pattern should keep imbalance low
    const prices = [100, 101, 100, 101, 100, 101, 100, 101]
    // Pattern: -, buy, sell, buy, sell, buy, sell, buy
    const testData = createTestData(prices)
    
    const generator = new TickImbalanceBarGenerator({
      in: [],
      out: [],
      imbalanceThreshold: 3
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // With alternating pattern, imbalance never exceeds 1
    // So only incomplete bar at end
    strictEqual(bars.length, 1)
    strictEqual(bars[0]!.volume, 800)   // All 8 ticks
  })

  it('should handle multiple symbols independently', async () => {
    const testData: OhlcvDto[] = [
      // Symbol A: strong buy pressure
      { timestamp: 1000, symbol: 'A', exchange: 'test', open: 100, high: 100.1, low: 99.9, close: 100, volume: 100 },
      { timestamp: 2000, symbol: 'A', exchange: 'test', open: 101, high: 101.1, low: 100.9, close: 101, volume: 100 },
      { timestamp: 3000, symbol: 'A', exchange: 'test', open: 102, high: 102.1, low: 101.9, close: 102, volume: 100 },
      { timestamp: 4000, symbol: 'A', exchange: 'test', open: 103, high: 103.1, low: 102.9, close: 103, volume: 100 },
      // Symbol B: strong sell pressure
      { timestamp: 1500, symbol: 'B', exchange: 'test', open: 200, high: 200.1, low: 199.9, close: 200, volume: 100 },
      { timestamp: 2500, symbol: 'B', exchange: 'test', open: 199, high: 199.1, low: 198.9, close: 199, volume: 100 },
      { timestamp: 3500, symbol: 'B', exchange: 'test', open: 198, high: 198.1, low: 197.9, close: 198, volume: 100 },
      { timestamp: 4500, symbol: 'B', exchange: 'test', open: 197, high: 197.1, low: 196.9, close: 197, volume: 100 },
    ]
    
    const generator = new TickImbalanceBarGenerator({
      in: [],
      out: [],
      imbalanceThreshold: 3
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    const symbolABars = bars.filter(b => b.symbol === 'A')
    const symbolBBars = bars.filter(b => b.symbol === 'B')
    
    // Symbol A: 3 consecutive buys = imbalance of 3, completes at tick 3
    strictEqual(symbolABars.length, 1) // All 4 ticks in one bar
    strictEqual(symbolABars[0]!.volume, 400)
    
    // Symbol B: 3 consecutive sells = imbalance of 3, completes at tick 3  
    strictEqual(symbolBBars.length, 1) // All 4 ticks in one bar
    strictEqual(symbolBBars[0]!.volume, 400)
  })

  it('should validate parameters', () => {
    // Zero threshold
    try {
      new TickImbalanceBarGenerator({
        in: [],
        out: [],
        imbalanceThreshold: 0
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('imbalanceThreshold must be greater than 0'))
    }
    
    // Negative threshold
    try {
      new TickImbalanceBarGenerator({
        in: [],
        out: [],
        imbalanceThreshold: -1
      })
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('imbalanceThreshold must be greater than 0'))
    }
  })

  it('should serialize and restore state correctly', () => {
    const generator = new TickImbalanceBarGenerator({
      in: [],
      out: [],
      imbalanceThreshold: 5
    })
    
    // Simulate some state
    const mockState = {
      'TEST': {
        currentBar: createTestData([100])[0]!,
        buyTicks: 3,
        sellTicks: 1,
        buyVolume: 300,
        sellVolume: 100,
        previousClose: 101,
        tickImbalance: 2,
        complete: false
      }
    }
    
    generator.restoreState(mockState)
    const state = generator.getState()
    
    const testState = state['TEST'] as any
    strictEqual(testState.buyTicks, 3)
    strictEqual(testState.sellTicks, 1)
    strictEqual(testState.buyVolume, 300)
    strictEqual(testState.sellVolume, 100)
    strictEqual(testState.tickImbalance, 2)
  })
})