import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { AverageTrueRange } from '../../../../src/transforms'

// Helper to create test data with OHLC
function createTestData(ohlcData: Array<[number, number, number, number]>): OhlcvDto[] {
  return ohlcData.map(([open, high, low, close], i) => ({
    timestamp: Date.now() + i * 60000,
    symbol: 'TEST',
    exchange: 'test',
    open,
    high,
    low,
    close,
    volume: 1000
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

describe('Average True Range', () => {
  it('should calculate ATR correctly', async () => {
    // Test data with known ATR values
    const ohlcData: Array<[number, number, number, number]> = [
      [100, 105, 95, 102],  // TR = 10 (high-low)
      [102, 108, 100, 106], // TR = 8 (high-low) or |108-102| = 6 or |100-102| = 2, max = 8
      [106, 110, 103, 108], // TR = 7 (high-low) or |110-106| = 4 or |103-106| = 3, max = 7
      [108, 112, 105, 110], // TR = 7
      [110, 115, 108, 113], // TR = 7
    ]
    const testData = createTestData(ohlcData)
    
    const atr = new AverageTrueRange({
      in: ['high', 'low', 'close'],
      out: ['atr'],
      period: 3
    })
    
    const result = await atr.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    strictEqual(transformed.length, 5)
    
    // First two items shouldn't have ATR (need period data)
    ok(!('atr' in transformed[0]!))
    ok(!('atr' in transformed[1]!))
    
    // Third item should have initial ATR (average of first 3 TRs)
    // TR values: 10, 8, 7
    // Initial ATR = (10 + 8 + 7) / 3 = 8.33...
    const expectedInitialAtr = (10 + 8 + 7) / 3
    ok(Math.abs((transformed[2] as any).atr - expectedInitialAtr) < 0.01)
    
    // Fourth item uses Wilder's smoothing
    // ATR = ((Previous ATR * 2) + Current TR) / 3
    const expectedAtr4 = ((expectedInitialAtr * 2) + 7) / 3
    ok(Math.abs((transformed[3] as any).atr - expectedAtr4) < 0.01)
  })

  it('should handle gaps correctly', async () => {
    // Test data with price gaps
    const ohlcData: Array<[number, number, number, number]> = [
      [100, 105, 95, 100],
      [110, 115, 108, 112], // Gap up from 100 to 110
      [105, 108, 102, 104], // Gap down from 112 to 105
      [104, 106, 102, 105],
    ]
    const testData = createTestData(ohlcData)
    
    const atr = new AverageTrueRange({
      in: ['high', 'low', 'close'],
      out: ['atr'],
      period: 2
    })
    
    const result = await atr.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // Check that gaps are included in TR calculation
    // Second bar: TR = max(115-108=7, |115-100|=15, |108-100|=8) = 15
    // Third bar: TR = max(108-102=6, |108-112|=4, |102-112|=10) = 10
    ok('atr' in transformed[1]!)
    
    // ATR should reflect the increased volatility from gaps
    ok((transformed[2] as any).atr > 5) // Should be higher due to gaps
  })

  it('should validate parameters', () => {
    // Should require exactly 1 output field
    try {
      const atr = new AverageTrueRange({
        in: ['high', 'low', 'close'],
        out: ['atr1', 'atr2'], // Too many output fields
        period: 14
      })
      atr.validate()
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('requires exactly 1 output field'))
    }
  })

  it('should handle first data point correctly', async () => {
    const ohlcData: Array<[number, number, number, number]> = [
      [100, 110, 90, 105],
      [105, 115, 95, 110],
    ]
    const testData = createTestData(ohlcData)
    
    const atr = new AverageTrueRange({
      in: ['high', 'low', 'close'],
      out: ['atr'],
      period: 1
    })
    
    const result = await atr.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // With period=1, first item should have ATR = high - low
    strictEqual((transformed[0] as any).atr, 20) // 110 - 90
    
    // Second item should use true range calculation
    // TR = max(115-95=20, |115-105|=10, |95-105|=10) = 20
    strictEqual((transformed[1] as any).atr, 20)
  })

  it('should smooth volatility over time', async () => {
    // Create data with spike in volatility
    const ohlcData: Array<[number, number, number, number]> = [
      [100, 102, 98, 100],   // TR = 4
      [100, 103, 97, 101],   // TR = 6
      [101, 120, 90, 110],   // TR = 30 (spike)
      [110, 112, 108, 110],  // TR = 4
      [110, 113, 107, 111],  // TR = 6
    ]
    const testData = createTestData(ohlcData)
    
    const atr = new AverageTrueRange({
      in: ['high', 'low', 'close'],
      out: ['atr'],
      period: 3
    })
    
    const result = await atr.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // ATR should smooth out the spike
    const atr3 = (transformed[2] as any).atr // Should include the spike
    const atr4 = (transformed[3] as any).atr // Should be smoothed
    const atr5 = (transformed[4] as any).atr // Should continue smoothing
    
    // ATR should decrease after spike but not immediately
    ok(atr4 < atr3)
    ok(atr5 < atr4)
    ok(atr5 > 5) // Should still be elevated due to smoothing
  })
})