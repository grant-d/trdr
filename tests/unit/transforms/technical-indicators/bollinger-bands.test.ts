import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { BollingerBands } from '../../../../src/transforms'

// Helper to create test data
function createTestData(values: number[]): OhlcvDto[] {
  return values.map((close, i) => ({
    timestamp: Date.now() + i * 60000,
    symbol: 'TEST',
    exchange: 'test',
    open: close,
    high: close + 5,
    low: close - 5,
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

describe('Bollinger Bands', () => {
  it('should calculate bands correctly', async () => {
    const values = [20, 22, 21, 23, 24, 25, 20, 30, 15, 35]
    const testData = createTestData(values)
    
    const bb = new BollingerBands({
      in: ['close'],
      out: ['bb_middle', 'bb_upper', 'bb_lower'],
      period: 5,
      stdDev: 2
    })
    
    const result = await bb.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    strictEqual(transformed.length, 10)
    
    // First 4 items shouldn't have bands
    for (let i = 0; i < 4; i++) {
      ok(!('bb_middle' in transformed[i]!))
      ok(!('bb_upper' in transformed[i]!))
      ok(!('bb_lower' in transformed[i]!))
    }
    
    // From 5th item onwards should have bands
    const item5 = transformed[4]!
    ok('bb_middle' in item5)
    ok('bb_upper' in item5)
    ok('bb_lower' in item5)
    
    // Middle band should be SMA
    strictEqual(item5.bb_middle, 22) // (20+22+21+23+24)/5
    
    // Upper band should be above middle
    ok((item5 as any).bb_upper > (item5 as any).bb_middle)
    
    // Lower band should be below middle
    ok((item5 as any).bb_lower < (item5 as any).bb_middle)
    
    // Bands should widen as volatility increases
    const lastItem = transformed[9]!
    ok((lastItem as any).bb_upper - (lastItem as any).bb_lower > (item5 as any).bb_upper - (item5 as any).bb_lower)
  })

  it('should respect standard deviation multiplier', async () => {
    const values = [50, 52, 48, 51, 49, 53, 47, 52, 50, 51]
    const testData = createTestData(values)
    
    // Test with different stdDev values
    const bb1 = new BollingerBands({
      in: ['close'],
      out: ['bb_middle_1', 'bb_upper_1', 'bb_lower_1'],
      period: 3,
      stdDev: 1
    })
    
    const bb2 = new BollingerBands({
      in: ['close'],
      out: ['bb_middle_2', 'bb_upper_2', 'bb_lower_2'],
      period: 3,
      stdDev: 2
    })
    
    const result1 = await bb1.apply(arrayToAsyncIterator(testData))
    const transformed1 = await collectResults(result1.data)
    
    const result2 = await bb2.apply(arrayToAsyncIterator(testData))
    const transformed2 = await collectResults(result2.data)
    
    // Compare band widths
    const item1 = transformed1[2]!
    const item2 = transformed2[2]!
    
    // Middle bands should be the same (both are SMA)
    strictEqual(item1.bb_middle_1, item2.bb_middle_2)
    
    // 2 stdDev bands should be wider than 1 stdDev
    const width1 = (item1 as any).bb_upper_1 - (item1 as any).bb_lower_1
    const width2 = (item2 as any).bb_upper_2 - (item2 as any).bb_lower_2
    strictEqual(width2, width1 * 2)
  })

  it('should handle low volatility correctly', async () => {
    // Very low volatility data
    const values = [100, 100.1, 99.9, 100, 100.1, 99.9, 100, 100.1, 99.9, 100]
    const testData = createTestData(values)
    
    const bb = new BollingerBands({
      in: ['close'],
      out: ['bb_middle', 'bb_upper', 'bb_lower'],
      period: 5,
      stdDev: 2
    })
    
    const result = await bb.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // Check that bands are very tight
    const lastItem = transformed[9]!
    const bandwidth = (lastItem as any).bb_upper - (lastItem as any).bb_lower
    ok(bandwidth < 1) // Bands should be tight for low volatility
  })

  it('should validate output fields', () => {
    // Should require exactly 3 output fields
    try {
      const bb = new BollingerBands({
        in: ['close'],
        out: ['bb_middle', 'bb_upper'], // Missing third field
        period: 20,
        stdDev: 2
      })
      bb.validate()
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('requires exactly 3 output fields'))
    }
  })

  it('should calculate standard deviation correctly', async () => {
    // Known values for easy verification
    const values = [2, 4, 4, 4, 5, 5, 7, 9]
    const testData = createTestData(values)
    
    const bb = new BollingerBands({
      in: ['close'],
      out: ['bb_middle', 'bb_upper', 'bb_lower'],
      period: 8,
      stdDev: 1
    })
    
    const result = await bb.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // At index 7, we have all 8 values
    const lastItem = transformed[7]!
    
    // Mean = (2+4+4+4+5+5+7+9)/8 = 40/8 = 5
    strictEqual(lastItem.bb_middle, 5)
    
    // Standard deviation = 2
    // Upper = 5 + 1*2 = 7
    // Lower = 5 - 1*2 = 3
    strictEqual(lastItem.bb_upper, 7)
    strictEqual(lastItem.bb_lower, 3)
  })
})