import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { HeikinAshiGenerator } from '../../../../src/transforms'

// Helper to create test data with specific OHLC values
function createOhlcData(data: Array<{o: number, h: number, l: number, c: number, v: number}>): OhlcvDto[] {
  const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
  return data.map((candle, i) => ({
    timestamp: baseTime + i * 60000, // 1 minute apart
    symbol: 'TEST',
    exchange: 'test',
    open: candle.o,
    high: candle.h,
    low: candle.l,
    close: candle.c,
    volume: candle.v
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

describe('HeikinAshiGenerator', () => {
  it('should calculate Heikin-Ashi values correctly', async () => {
    // Create test data with known OHLC values
    const testData = createOhlcData([
      { o: 100, h: 105, l: 98, c: 103, v: 1000 },   // Regular candle 1
      { o: 103, h: 108, l: 102, c: 106, v: 1200 },  // Regular candle 2
      { o: 106, h: 110, l: 104, c: 108, v: 1100 },  // Regular candle 3
      { o: 108, h: 112, l: 106, c: 107, v: 900 },   // Regular candle 4
    ])
    
    const generator = new HeikinAshiGenerator({
      in: [],
      out: []
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    strictEqual(bars.length, 4)
    
    // First HA candle: uses regular values
    // HA Close = (O + H + L + C) / 4 = (100 + 105 + 98 + 103) / 4 = 101.5
    // HA Open = (O + C) / 2 = (100 + 103) / 2 = 101.5 (first candle special case)
    // HA High = max(H, HA Open, HA Close) = max(105, 101.5, 101.5) = 105
    // HA Low = min(L, HA Open, HA Close) = min(98, 101.5, 101.5) = 98
    strictEqual((bars[0] as any).ha_close, 101.5)
    strictEqual((bars[0] as any).ha_open, 101.5)
    strictEqual((bars[0] as any).ha_high, 105)
    strictEqual((bars[0] as any).ha_low, 98)
    strictEqual(bars[0]!.volume, 1000)  // Volume unchanged
    
    // Second HA candle
    // HA Close = (103 + 108 + 102 + 106) / 4 = 419 / 4 = 104.75
    // HA Open = (prev HA Open + prev HA Close) / 2 = (101.5 + 101.5) / 2 = 101.5
    // HA High = max(108, 101.5, 104.75) = 108
    // HA Low = min(102, 101.5, 104.75) = 101.5
    strictEqual((bars[1] as any).ha_close, 104.75)
    strictEqual((bars[1] as any).ha_open, 101.5)
    strictEqual((bars[1] as any).ha_high, 108)
    strictEqual((bars[1] as any).ha_low, 101.5)
    
    // Third HA candle
    // HA Close = (106 + 110 + 104 + 108) / 4 = 428 / 4 = 107
    // HA Open = (101.5 + 104.75) / 2 = 103.125
    // HA High = max(110, 103.125, 107) = 110
    // HA Low = min(104, 103.125, 107) = 103.125
    strictEqual((bars[2] as any).ha_close, 107)
    strictEqual((bars[2] as any).ha_open, 103.125)
    strictEqual((bars[2] as any).ha_high, 110)
    strictEqual((bars[2] as any).ha_low, 103.125)
    
    // Fourth HA candle
    // HA Close = (108 + 112 + 106 + 107) / 4 = 433 / 4 = 108.25
    // HA Open = (103.125 + 107) / 2 = 105.0625
    // HA High = max(112, 105.0625, 108.25) = 112
    // HA Low = min(106, 105.0625, 108.25) = 105.0625
    strictEqual((bars[3] as any).ha_close, 108.25)
    strictEqual((bars[3] as any).ha_open, 105.0625)
    strictEqual((bars[3] as any).ha_high, 112)
    strictEqual((bars[3] as any).ha_low, 105.0625)
  })

  it('should smooth out price action', async () => {
    // Create volatile data to test smoothing
    const testData = createOhlcData([
      { o: 100, h: 120, l: 80, c: 110, v: 1000 },   // Big range
      { o: 110, h: 115, l: 105, c: 108, v: 800 },   // Small range
      { o: 108, h: 130, l: 90, c: 95, v: 1500 },    // Big range, bearish
      { o: 95, h: 100, l: 92, c: 98, v: 600 },      // Small recovery
    ])
    
    const generator = new HeikinAshiGenerator({
      in: [],
      out: []
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Check that HA produces averaged values (smoothing effect)
    for (let i = 0; i < bars.length; i++) {
      const ha = bars[i] as any
      
      // HA close should be the average of OHLC
      const expectedClose = (testData[i]!.open + testData[i]!.high + testData[i]!.low + testData[i]!.close) / 4
      strictEqual(ha.ha_close, expectedClose)
      
      // HA open should be averaged (except first candle)
      if (i > 0) {
        const prevHa = bars[i-1] as any
        const expectedOpen = (prevHa.ha_open + prevHa.ha_close) / 2
        strictEqual(ha.ha_open, expectedOpen)
      }
    }
  })

  it('should handle multiple symbols independently', async () => {
    const testData: OhlcvDto[] = [
      // Symbol A
      { timestamp: 1000, symbol: 'A', exchange: 'test', open: 100, high: 105, low: 95, close: 102, volume: 100 },
      { timestamp: 2000, symbol: 'A', exchange: 'test', open: 102, high: 106, low: 100, close: 104, volume: 150 },
      // Symbol B
      { timestamp: 1500, symbol: 'B', exchange: 'test', open: 200, high: 210, low: 195, close: 205, volume: 200 },
      { timestamp: 2500, symbol: 'B', exchange: 'test', open: 205, high: 215, low: 200, close: 210, volume: 250 },
    ]
    
    const generator = new HeikinAshiGenerator({
      in: [],
      out: []
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    const symbolABars = bars.filter(b => b.symbol === 'A')
    const symbolBBars = bars.filter(b => b.symbol === 'B')
    
    strictEqual(symbolABars.length, 2)
    strictEqual(symbolBBars.length, 2)
    
    // Each symbol should have independent HA calculations
    // Symbol A first candle
    const aClose1 = (100 + 105 + 95 + 102) / 4  // 100.5
    strictEqual((symbolABars[0] as any).ha_close, aClose1)
    
    // Symbol B first candle
    const bClose1 = (200 + 210 + 195 + 205) / 4  // 202.5
    strictEqual((symbolBBars[0] as any).ha_close, bClose1)
  })

  it('should show clear trends with HA candles', async () => {
    // Create trending data
    const uptrendData = createOhlcData([
      { o: 100, h: 102, l: 99, c: 101, v: 100 },
      { o: 101, h: 103, l: 100, c: 102, v: 100 },
      { o: 102, h: 104, l: 101, c: 103, v: 100 },
      { o: 103, h: 105, l: 102, c: 104, v: 100 },
    ])
    
    const generator = new HeikinAshiGenerator({
      in: [],
      out: []
    })
    
    const result = await generator.apply(arrayToAsyncIterator(uptrendData))
    const bars = await collectResults(result.data)
    
    // In uptrend, HA candles should mostly have:
    // 1. Close > Open (bullish candles)
    // 2. Small or no lower wicks (low = open in strong uptrend)
    for (let i = 1; i < bars.length; i++) {
      const ha = bars[i]!
      ok((ha as any).ha_close > (ha as any).ha_open, `HA candle ${i} should be bullish in uptrend`)
      
      // Check for small lower wick (strong uptrend characteristic)
      const lowerWick = Math.max(0, (ha as any).ha_open - (ha as any).ha_low)
      const bodySize = Math.abs((ha as any).ha_close - (ha as any).ha_open)
      
      // In strong uptrend, lower wick should be minimal
      // If body size is very small, just check that lower wick is small
      if (bodySize > 0.01) {
        ok(lowerWick <= bodySize * 0.5, `HA candle ${i} should have small lower wick`)
      } else {
        ok(lowerWick < 0.5, `HA candle ${i} should have minimal lower wick`)
      }
    }
  })

  it('should handle gaps differently than regular candles', async () => {
    // Create data with a gap
    const testData = createOhlcData([
      { o: 100, h: 102, l: 98, c: 101, v: 100 },
      { o: 110, h: 112, l: 108, c: 111, v: 200 },  // Gap up
      { o: 111, h: 113, l: 109, c: 112, v: 150 },
    ])
    
    const generator = new HeikinAshiGenerator({
      in: [],
      out: []
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // HA should smooth out the gap
    // Second HA candle open should be average of previous HA open/close
    // Not the gapped up price
    const firstHAClose = (100 + 102 + 98 + 101) / 4  // 100.25
    const firstHAOpen = (100 + 101) / 2  // 100.5
    const secondHAOpen = (firstHAOpen + firstHAClose) / 2  // ~100.375
    
    // Regular candle gaps from 101 to 110, but HA smooths it
    ok((bars[1] as any).ha_open < 105, 'HA should smooth out the gap')
    ok(Math.abs((bars[1] as any).ha_open - secondHAOpen) < 0.01, 'HA open calculation should be correct')
  })

  it('should maintain timestamp and symbol information', async () => {
    const testData = createOhlcData([
      { o: 100, h: 105, l: 95, c: 102, v: 100 },
    ])
    testData[0]!.symbol = 'CUSTOM'
    testData[0]!.exchange = 'NYSE'
    testData[0]!.timestamp = 1234567890
    
    const generator = new HeikinAshiGenerator({
      in: [],
      out: []
    })
    
    const result = await generator.apply(arrayToAsyncIterator(testData))
    const bars = await collectResults(result.data)
    
    // Should preserve metadata
    strictEqual(bars[0]!.symbol, 'CUSTOM')
    strictEqual(bars[0]!.exchange, 'NYSE')
    strictEqual(bars[0]!.timestamp, 1234567890)
  })

  it('should serialize and restore state correctly', () => {
    const generator = new HeikinAshiGenerator({
      in: [],
      out: []
    })
    
    // Simulate some state
    const mockState = {
      'TEST': {
        previousHAOpen: 105.5,
        previousHAClose: 106.25
      }
    }
    
    generator.restoreState(mockState)
    const state = generator.getState()
    
    const testState = state.TEST as any
    strictEqual(testState.previousHAOpen, 105.5)
    strictEqual(testState.previousHAClose, 106.25)
  })
})