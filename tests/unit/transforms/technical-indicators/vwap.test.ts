import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { VolumeWeightedAveragePrice } from '../../../../src/transforms'

// Helper to create test data with OHLCV
function createTestData(ohlcvData: Array<[number, number, number, number, number, number?]>): OhlcvDto[] {
  const baseTime = new Date('2024-01-01T09:00:00Z').getTime()
  return ohlcvData.map(([open, high, low, close, volume, hourOffset = 0], i) => ({
    timestamp: baseTime + (i * 60000) + (hourOffset * 3600000),
    symbol: 'TEST',
    exchange: 'test',
    open,
    high,
    low,
    close,
    volume
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

describe('Volume Weighted Average Price', () => {
  it('should calculate VWAP correctly', async () => {
    // Simple test data
    const ohlcvData: Array<[number, number, number, number, number]> = [
      [100, 105, 95, 100, 1000],  // Typical price = 100, PV = 100,000
      [100, 110, 90, 105, 2000],  // Typical price = 101.67, PV = 203,333
      [105, 115, 100, 110, 3000], // Typical price = 108.33, PV = 325,000
    ]
    const testData = createTestData(ohlcvData)
    
    const vwap = new VolumeWeightedAveragePrice({
      in: [],
      out: ['vwap'],
      anchorPeriod: 'session'
    })
    
    const result = await vwap.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    strictEqual(transformed.length, 3)
    
    // First VWAP = 100,000 / 1000 = 100
    strictEqual(transformed[0]!.vwap, 100)
    
    // Second VWAP = (100,000 + 203,333) / (1000 + 2000) = 303,333 / 3000 = 101.11
    const vwap2 = transformed[1]!.vwap!
    ok(Math.abs((vwap2 as number) - 101.111) < 0.01)
    
    // Third VWAP = (100,000 + 203,333 + 325,000) / (1000 + 2000 + 3000) = 628,333 / 6000 = 104.72
    const vwap3 = transformed[2]!.vwap!
    ok(Math.abs((vwap3 as number) - 104.722) < 0.01)
  })

  it('should reset on new day', async () => {
    // Data spanning two days
    const ohlcvData: Array<[number, number, number, number, number, number]> = [
      [100, 105, 95, 100, 1000, 0],   // Day 1, 9:00
      [100, 110, 90, 105, 2000, 0],   // Day 1, 9:01
      [105, 115, 100, 110, 3000, 24], // Day 2, 9:00 (24 hours later)
      [110, 120, 105, 115, 4000, 24], // Day 2, 9:01
    ]
    const testData = createTestData(ohlcvData)
    
    const vwap = new VolumeWeightedAveragePrice({
      in: [],
      out: ['vwap'],
      anchorPeriod: 'day'
    })
    
    const result = await vwap.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // Check Day 1 VWAP
    strictEqual(transformed[0]!.vwap, 100)
    const day1Vwap2 = transformed[1]!.vwap!
    ok(Math.abs((day1Vwap2 as number) - 101.111) < 0.01)
    
    // Day 2 should reset - VWAP should be based only on Day 2 data
    // First bar of Day 2: typical = (115+100+110)/3 = 108.33, VWAP = 108.33
    const day2Vwap1 = transformed[2]!.vwap!
    ok(Math.abs((day2Vwap1 as number) - 108.333) < 0.01)
    
    // Second bar of Day 2
    const typicalPrice4 = (120 + 105 + 115) / 3 // 113.33
    const expectedVwap4 = (108.333 * 3000 + typicalPrice4 * 4000) / 7000
    const day2Vwap2 = transformed[3]!.vwap!
    ok(Math.abs((day2Vwap2 as number) - expectedVwap4) < 0.01)
  })

  it('should handle different anchor periods', async () => {
    const ohlcvData: Array<[number, number, number, number, number]> = [
      [100, 105, 95, 100, 1000],
      [100, 110, 90, 105, 2000],
      [105, 115, 100, 110, 3000],
    ]
    const testData = createTestData(ohlcvData)
    
    // Test session anchor
    const vwapSession = new VolumeWeightedAveragePrice({
      in: [],
      out: ['vwap_session'],
      anchorPeriod: 'session'
    })
    
    const sessionResult = await vwapSession.apply(arrayToAsyncIterator(testData))
    const sessionTransformed = await collectResults(sessionResult.data)
    
    // All values should accumulate
    ok(sessionTransformed[2]!.vwap_session! > sessionTransformed[1]!.vwap_session!)
  })

  it('should validate parameters', () => {
    // Should require exactly 1 output field
    try {
      const vwap = new VolumeWeightedAveragePrice({
        in: [],
        out: ['vwap1', 'vwap2'], // Too many output fields
        anchorPeriod: 'day'
      })
      vwap.validate()
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('requires exactly 1 output field'))
    }
    
    // Should validate anchor period
    try {
      const vwap = new VolumeWeightedAveragePrice({
        in: [],
        out: ['vwap'],
        anchorPeriod: 'invalid' as any
      })
      vwap.validate()
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('Anchor period must be one of'))
    }
  })

  it('should handle zero volume correctly', async () => {
    const ohlcvData: Array<[number, number, number, number, number]> = [
      [100, 105, 95, 100, 1000],
      [100, 110, 90, 105, 0],    // Zero volume
      [105, 115, 100, 110, 2000],
    ]
    const testData = createTestData(ohlcvData)
    
    const vwap = new VolumeWeightedAveragePrice({
      in: [],
      out: ['vwap'],
      anchorPeriod: 'session'
    })
    
    const result = await vwap.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // VWAP should still be calculated but zero volume bar doesn't contribute
    strictEqual(transformed[0]!.vwap, 100)
    strictEqual(transformed[1]!.vwap, 100) // Same as previous (no volume contribution)
    
    // Third bar should update normally
    const expectedVwap3 = (100 * 1000 + 108.333 * 2000) / 3000
    const vwap3Value = transformed[2]!.vwap!
    ok(Math.abs((vwap3Value as number) - expectedVwap3) < 0.01)
  })

  it('should calculate typical price correctly', async () => {
    // Test with specific values to verify typical price calculation
    const ohlcvData: Array<[number, number, number, number, number]> = [
      [100, 120, 80, 100, 1000], // Typical = (120+80+100)/3 = 100
      [100, 110, 90, 110, 1000], // Typical = (110+90+110)/3 = 103.33
    ]
    const testData = createTestData(ohlcvData)
    
    const vwap = new VolumeWeightedAveragePrice({
      in: [],
      out: ['vwap'],
      anchorPeriod: 'session'
    })
    
    const result = await vwap.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // First VWAP = 100
    strictEqual(transformed[0]!.vwap, 100)
    
    // Second VWAP = (100*1000 + 103.333*1000) / 2000 = 101.667
    const vwap2Value = transformed[1]!.vwap!
    ok(Math.abs((vwap2Value as number) - 101.667) < 0.01)
  })

  it('should support rolling VWAP with time-based period', async () => {
    // Create data with specific timestamps to test rolling period
    const baseTime = new Date('2024-01-01T09:00:00Z').getTime()
    const testData: OhlcvDto[] = [
      {
        timestamp: baseTime,
        symbol: 'TEST',
        exchange: 'test',
        open: 100, high: 105, low: 95, close: 100,
        volume: 1000
      },
      {
        timestamp: baseTime + 30 * 60000, // 30 minutes later
        symbol: 'TEST',
        exchange: 'test',
        open: 100, high: 110, low: 90, close: 105,
        volume: 2000
      },
      {
        timestamp: baseTime + 60 * 60000, // 60 minutes later (should still be in window)
        symbol: 'TEST',
        exchange: 'test',
        open: 105, high: 115, low: 100, close: 110,
        volume: 3000
      },
      {
        timestamp: baseTime + 90 * 60000, // 90 minutes later (should trigger reset)
        symbol: 'TEST',
        exchange: 'test',
        open: 110, high: 120, low: 105, close: 115,
        volume: 1500
      }
    ]
    
    const vwap = new VolumeWeightedAveragePrice({
      in: [],
      out: ['vwap'],
      anchorPeriod: 'rolling',
      rollingPeriod: 3600000 // 1 hour rolling period
    })
    
    const result = await vwap.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // First bar: only one data point
    strictEqual(transformed[0]!.vwap, 100)
    
    // Second bar: average of first two (still within 1 hour)
    const expectedVwap2 = (100 * 1000 + 101.667 * 2000) / 3000
    const rollingVwap2 = transformed[1]!.vwap!
    ok(Math.abs((rollingVwap2 as number) - expectedVwap2) < 0.01)
    
    // Third bar: average of all three (at 60 minutes, still within window)
    const expectedVwap3 = (100 * 1000 + 101.667 * 2000 + 108.333 * 3000) / 6000
    const rollingVwap3 = transformed[2]!.vwap!
    ok(Math.abs((rollingVwap3 as number) - expectedVwap3) < 0.01)
    
    // Fourth bar: at 90 minutes, should reset (exceeds 1 hour from start)
    const expectedVwap4 = 113.333 // Only this bar's typical price
    const rollingVwap4 = transformed[3]!.vwap!
    ok(Math.abs((rollingVwap4 as number) - expectedVwap4) < 0.01)
  })

  it('should validate rolling period', async () => {
    try {
      const vwap = new VolumeWeightedAveragePrice({
        in: [],
        out: ['vwap'],
        anchorPeriod: 'rolling',
        rollingPeriod: 0
      })
      vwap.validate()
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      strictEqual(error.message, 'Rolling period must be positive')
    }
  })
})