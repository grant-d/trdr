import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { SimpleMovingAverage, ExponentialMovingAverage } from '../../../../src/transforms'

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
function arrayToAsyncIterator<T>(array: T[]): AsyncIterator<T> {
  let index = 0
  return {
    async next(): Promise<IteratorResult<T, any>> {
      if (index < array.length) {
        return { done: false, value: array[index++]! }
      }
      return { done: true, value: undefined }
    }
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

describe('Simple Moving Average', () => {
  it('should calculate SMA correctly', async () => {
    const values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    const testData = createTestData(values)
    
    const sma = new SimpleMovingAverage({
      in: ['close'],
      out: ['sma'],
      period: 3
    })
    
    const result = await sma.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    strictEqual(transformed.length, 10)
    
    // First two items shouldn't have SMA (insufficient data)
    ok(!('sma' in transformed[0]!))
    ok(!('sma' in transformed[1]!))
    
    // From third item onwards, should have SMA
    strictEqual(transformed[2]!.sma, 20) // (10+20+30)/3
    strictEqual(transformed[3]!.sma, 30) // (20+30+40)/3
    strictEqual(transformed[4]!.sma, 40) // (30+40+50)/3
    strictEqual(transformed[9]!.sma, 90) // (80+90+100)/3
  })

  it('should handle multiple fields', async () => {
    const testData = createTestData([10, 20, 30, 40, 50])
    
    const sma = new SimpleMovingAverage({
      in: ['open', 'close'],
      out: ['open_sma', 'close_sma'],
      period: 2
    })
    
    const result = await sma.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // Check that both fields are calculated
    ok(!('open_sma' in transformed[0]!))
    strictEqual(transformed[1]!.open_sma, 15) // (10+20)/2
    strictEqual(transformed[1]!.close_sma, 15) // (10+20)/2
    strictEqual(transformed[4]!.open_sma, 45) // (40+50)/2
    strictEqual(transformed[4]!.close_sma, 45) // (40+50)/2
  })

  it('should validate parameters', () => {
    const sma = new SimpleMovingAverage({
      in: ['close'],
      out: ['sma'],
      period: 0
    })
    
    try {
      sma.validate()
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('Period must be at least 1'))
    }
  })
})

describe('Exponential Moving Average', () => {
  it('should calculate EMA correctly', async () => {
    const values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    const testData = createTestData(values)
    
    const ema = new ExponentialMovingAverage({
      in: ['close'],
      out: ['ema'],
      period: 3
    })
    
    const result = await ema.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    strictEqual(transformed.length, 10)
    
    // First two items shouldn't have EMA
    ok(!('ema' in transformed[0]!))
    ok(!('ema' in transformed[1]!))
    
    // Third item should have EMA equal to SMA
    strictEqual(transformed[2]!.ema, 20) // Initial EMA = SMA = (10+20+30)/3
    
    // Fourth item onwards should use EMA formula
    const multiplier = 2 / (3 + 1) // 0.5
    const ema3 = (40 - 20) * multiplier + 20
    strictEqual(transformed[3]!.ema, ema3) // 30
    
    const ema4 = (50 - ema3) * multiplier + ema3
    strictEqual(transformed[4]!.ema, ema4) // 40
  })

  it('should handle multiple fields independently', async () => {
    const testData = createTestData([10, 20, 30, 40, 50])
    
    const ema = new ExponentialMovingAverage({
      in: ['open', 'close'],
      out: ['open_ema', 'close_ema'],
      period: 2
    })
    
    const result = await ema.apply(arrayToAsyncIterator(testData))
    const transformed = await collectResults(result.data)
    
    // First item shouldn't have EMA
    ok(!('open_ema' in transformed[0]!))
    
    // Second item should have initial EMA (SMA)
    strictEqual(transformed[1]!.open_ema, 15) // (10+20)/2
    strictEqual(transformed[1]!.close_ema, 15) // (10+20)/2
    
    // Third item should use EMA formula
    const multiplier = 2 / 3 // ~0.667
    const openEma2 = (30 - 15) * multiplier + 15
    const closeEma2 = (30 - 15) * multiplier + 15
    strictEqual(transformed[2]!.open_ema, openEma2) // 25
    strictEqual(transformed[2]!.close_ema, closeEma2) // 25
  })

  it('should produce smoother results than SMA', async () => {
    // Create data with a spike - longer series for smoother EMA
    const values = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 100, 10, 10, 10, 10]
    const testData = createTestData(values)
    
    const sma = new SimpleMovingAverage({
      in: ['close'],
      out: ['sma'],
      period: 3
    })
    
    const ema = new ExponentialMovingAverage({
      in: ['close'],
      out: ['ema'],
      period: 10
    })
    
    const smaResult = await sma.apply(arrayToAsyncIterator(testData))
    const smaTransformed = await collectResults(smaResult.data)
    
    const emaResult = await ema.apply(arrayToAsyncIterator(testData))
    const emaTransformed = await collectResults(emaResult.data)
    
    // At the spike (index 10)
    strictEqual(smaTransformed[10]!.sma, 40) // (10+10+100)/3
    
    // EMA with longer period should be smoother (less affected by spike)
    ok((emaTransformed[10] as any).ema < (smaTransformed[10] as any).sma)
    
    // After spike, when SMA drops back to normal, EMA should still be higher (gradual decay)
    ok((emaTransformed[13] as any).ema > (smaTransformed[13] as any).sma)
  })
})