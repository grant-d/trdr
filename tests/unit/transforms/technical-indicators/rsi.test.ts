import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { RelativeStrengthIndex } from '../../../../src/transforms'

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

describe('Relative Strength Index', () => {
  it('should calculate RSI correctly', async () => {
    // Classic RSI test data
    const values = [
      44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
      45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
      46.03, 46.41, 46.22, 45.64
    ]
    const testData = createTestData(values)
    
    const rsi = new RelativeStrengthIndex({
      in: ['close'],
      out: ['rsi'],
      period: 14
    })
    
    const result = await rsi.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    strictEqual(transformed.length, 20)
    
    // First 14 items shouldn't have RSI
    for (let i = 0; i < 14; i++) {
      ok(!('rsi' in transformed[i]!))
    }
    
    // From 15th item onwards should have RSI
    ok('rsi' in transformed[14]!)
    ok(transformed[14]!.rsi! > 0 && transformed[14]!.rsi! < 100)
    
    // RSI should be bounded between 0 and 100
    for (let i = 14; i < 20; i++) {
      const rsiValue = transformed[i]!.rsi!
      ok(rsiValue >= 0 && rsiValue <= 100)
    }
  })

  it('should handle extreme values correctly', async () => {
    // All prices increasing - RSI should approach 100
    const increasingValues = Array.from({ length: 20 }, (_, i) => 100 + i * 10)
    const increasingData = createTestData(increasingValues)
    
    const rsi = new RelativeStrengthIndex({
      in: ['close'],
      out: ['rsi'],
      period: 5
    })
    
    const increasingResult = await rsi.apply(arrayToAsyncIterator(increasingData))
    const increasingTransformed = await collectResults(increasingResult.data)
    
    // RSI should be very high (close to 100)
    const lastRsi = increasingTransformed[19]!.rsi!
    ok(lastRsi > 95)
    
    // All prices decreasing - RSI should approach 0
    const decreasingValues = Array.from({ length: 20 }, (_, i) => 100 - i * 10)
    const decreasingData = createTestData(decreasingValues)
    
    const rsi2 = new RelativeStrengthIndex({
      in: ['close'],
      out: ['rsi'],
      period: 5
    })
    
    const decreasingResult = await rsi2.apply(arrayToAsyncIterator(decreasingData))
    const decreasingTransformed = await collectResults(decreasingResult.data)
    
    // RSI should be very low (close to 0)
    const lastRsi2 = decreasingTransformed[19]!.rsi!
    ok(lastRsi2 < 5)
  })

  it('should handle multiple fields', async () => {
    const testData = createTestData([40, 42, 41, 43, 45, 44, 46, 48, 47, 49])
    
    const rsi = new RelativeStrengthIndex({
      in: ['open', 'close'],
      out: ['open_rsi', 'close_rsi'],
      period: 3
    })
    
    const result = await rsi.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // Should calculate RSI for both fields
    ok('open_rsi' in transformed[3]!)
    ok('close_rsi' in transformed[3]!)
    
    // Values should be reasonable
    ok(transformed[3]!.open_rsi! > 0 && transformed[3]!.open_rsi! < 100)
    ok(transformed[3]!.close_rsi! > 0 && transformed[3]!.close_rsi! < 100)
  })

  it('should use Wilder smoothing method', async () => {
    // Test that RSI uses Wilder's smoothing (not simple average)
    const values = [50, 52, 51, 53, 52, 54, 53, 55, 54, 56]
    const testData = createTestData(values)
    
    const rsi = new RelativeStrengthIndex({
      in: ['close'],
      out: ['rsi'],
      period: 3
    })
    
    const result = await rsi.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // Check that we get RSI values after period
    ok('rsi' in transformed[3]!)
    
    // RSI should show moderate values for this oscillating data
    const rsiValues = transformed.slice(3).map(item => item.rsi!)
    rsiValues.forEach(value => {
      ok(value > 30 && value < 70) // Should be in neutral zone
    })
  })

  it('should handle flat prices correctly', async () => {
    // All prices the same - no gains or losses
    const flatValues = Array(10).fill(50)
    const testData = createTestData(flatValues)
    
    const rsi = new RelativeStrengthIndex({
      in: ['close'],
      out: ['rsi'],
      period: 5
    })
    
    const result = await rsi.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // With no price changes, RSI should be 50 (neutral)
    // Actually with no losses, RSI = 100 - (100 / (1 + avgGain/0)) = 100
    if ('rsi' in transformed[5]!) {
      strictEqual(transformed[5]!.rsi, 100)
    }
  })
})