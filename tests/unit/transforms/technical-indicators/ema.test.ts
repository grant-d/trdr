import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import { ExponentialMovingAverage, SimpleMovingAverage } from '../../../../src/transforms'
import { DataBuffer } from '../../../../src/utils'
import { DataSlice } from '../../../../src/utils'

describe('Exponential Moving Average', () => {
  function createTestBuffer(values: number[]): DataBuffer {
    const buffer = new DataBuffer({
      columns: {
        timestamp: { index: 0 },
        open: { index: 1 },
        high: { index: 2 },
        low: { index: 3 },
        close: { index: 4 },
        volume: { index: 5 }
      }
    })

    const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
    values.forEach((close, i) => {
      buffer.push({
        timestamp: baseTime + i * 60000,
        open: close,
        high: close + 5,
        low: close - 5,
        close,
        volume: 1000
      })
    })

    return buffer
  }

  it('should calculate EMA correctly', () => {
    const values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    const buffer = createTestBuffer(values)

    const slice = new DataSlice(buffer, 0, buffer.length())
    const ema = new ExponentialMovingAverage(
      {
        tx: {
          in: 'close',
          out: 'ema',
          window: 3
        }
      },
      slice
    )

    // Process all data
    ema.next(0, buffer.length())

    const outputBuffer = ema.outputBuffer
    strictEqual(outputBuffer.length(), buffer.length())

    // First two items shouldn't have EMA
    const row0 = outputBuffer.getRow(0)
    const row1 = outputBuffer.getRow(1)
    ok(row0 && !('ema' in row0))
    ok(row1 && !('ema' in row1))

    // Third item should have EMA equal to SMA
    const row2 = outputBuffer.getRow(2)
    strictEqual(row2?.ema, 20) // Initial EMA = SMA = (10+20+30)/3

    // Fourth item onwards should use EMA formula
    const multiplier = 2 / (3 + 1) // 0.5
    const row3 = outputBuffer.getRow(3)
    const ema3 = (40 - 20) * multiplier + 20
    strictEqual(row3?.ema, ema3) // 30

    const row4 = outputBuffer.getRow(4)
    const ema4 = (50 - ema3) * multiplier + ema3
    strictEqual(row4?.ema, ema4) // 40
  })

  it('should handle multiple EMA configurations', () => {
    const testData = [10, 20, 30, 40, 50]
    const buffer = createTestBuffer(testData)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const ema = new ExponentialMovingAverage(
      {
        tx: [
          { in: 'close', out: 'ema_2', window: 2 },
          { in: 'close', out: 'ema_3', window: 3 },
          { in: 'volume', out: 'vol_ema', window: 2 }
        ]
      },
      slice
    )

    ema.next(0, buffer.length())

    const outputBuffer = ema.outputBuffer

    // First item shouldn't have EMA
    const row0 = outputBuffer.getRow(0)
    ok(row0 && !('ema_2' in row0))

    // Second item should have initial EMA (SMA) for window 2
    const row1 = outputBuffer.getRow(1)
    strictEqual(row1?.ema_2, 15) // (10+20)/2
    strictEqual(row1?.vol_ema, 1000) // (1000+1000)/2

    // Third item should use EMA formula
    const multiplier = 2 / 3 // ~0.667
    const row2 = outputBuffer.getRow(2)
    const ema2_2 = (30 - 15) * multiplier + 15
    strictEqual(row2?.ema_2, ema2_2) // 25
    strictEqual(row2?.ema_3, 20) // Initial EMA for window 3 = (10+20+30)/3
  })

  it('should produce smoother results than SMA', () => {
    // Create data with a spike
    const values = [
      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 100, 10, 10, 10, 10
    ]
    const buffer = createTestBuffer(values)
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Create SMA for comparison
    const sma = new SimpleMovingAverage(
      {
        tx: {
          in: 'close',
          out: 'sma',
          window: 3
        }
      },
      slice
    )

    const ema = new ExponentialMovingAverage(
      {
        tx: {
          in: 'close',
          out: 'ema',
          window: 10
        }
      },
      slice
    )

    // Process both
    sma.next(0, buffer.length())
    ema.next(0, buffer.length())

    const smaBuffer = sma.outputBuffer
    const emaBuffer = ema.outputBuffer

    // At the spike (index 10)
    const smaRow10 = smaBuffer.getRow(10)
    const emaRow10 = emaBuffer.getRow(10)
    strictEqual(smaRow10?.sma, 40) // (10+10+100)/3

    // EMA with longer period should be smoother (less affected by spike)
    ok(emaRow10!.ema! < smaRow10.sma)

    // After spike, EMA should decay gradually
    const emaRow13 = emaBuffer.getRow(13)
    const smaRow13 = smaBuffer.getRow(13)
    ok(emaRow13!.ema! > smaRow13!.sma!)
  })

  it('should handle batch processing correctly', () => {
    const values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    const buffer = createTestBuffer(values)

    // For batch processing, the transform should be initialized with the full data
    const slice = new DataSlice(buffer, 0, buffer.length())
    const ema = new ExponentialMovingAverage(
      {
        tx: {
          in: 'close',
          out: 'ema',
          window: 3
        }
      },
      slice
    )

    // Process all data at once (transforms process their internal slice)
    ema.next(0, buffer.length())

    // Results should be consistent
    const outputBuffer = ema.outputBuffer
    const row9 = outputBuffer.getRow(9)
    ok(row9?.ema !== undefined)
    ok(row9.ema > 80) // Should be weighted towards recent values
  })

  it('should validate window parameter', () => {
    const buffer = createTestBuffer([10, 20, 30])

    // Test window too small
    try {
      const initSlice = new DataSlice(buffer, 0, buffer.length())
      new ExponentialMovingAverage(
        {
          tx: {
            in: 'close',
            out: 'ema',
            window: 1
          }
        },
        initSlice
      )
      ok(false, 'Should have thrown')
    } catch (error: any) {
      ok(error.message.includes('at least 2'))
    }

    // Test window too large
    try {
      const initSlice = new DataSlice(buffer, 0, buffer.length())
      new ExponentialMovingAverage(
        {
          tx: {
            in: 'close',
            out: 'ema',
            window: 10000
          }
        },
        initSlice
      )
      ok(false, 'Should have thrown')
    } catch (error: any) {
      ok(error.message.includes('9999'))
    }
  })

  it('should validate column names', () => {
    const buffer = createTestBuffer([10, 20, 30])

    // Test invalid input column
    try {
      const initSlice = new DataSlice(buffer, 0, buffer.length())
      new ExponentialMovingAverage(
        {
          tx: {
            in: 'invalid-column!',
            out: 'ema',
            window: 3
          }
        },
        initSlice
      )
      ok(false, 'Should have thrown')
    } catch (error: any) {
      ok(error.message.includes('alphanumeric'))
    }

    // Test non-existent input column
    try {
      const initSlice = new DataSlice(buffer, 0, buffer.length())
      new ExponentialMovingAverage(
        {
          tx: {
            in: 'nonexistent',
            out: 'ema',
            window: 3
          }
        },
        initSlice
      )
      ok(false, 'Should have thrown')
    } catch (error: any) {
      ok(error.message.includes('not found'))
    }
  })

  it('should ensure unique output names', () => {
    const buffer = createTestBuffer([10, 20, 30])

    try {
      const initSlice = new DataSlice(buffer, 0, buffer.length())
      new ExponentialMovingAverage(
        {
          tx: [
            { in: 'close', out: 'ema', window: 3 },
            { in: 'open', out: 'ema', window: 3 }
          ]
        },
        initSlice
      )
      ok(false, 'Should have thrown')
    } catch (error: any) {
      ok(error.message.includes('unique'))
    }
  })

  it('should be ready after processing enough data', () => {
    const buffer = createTestBuffer([10, 20, 30, 40, 50])

    // Initialize with partial data to test readiness
    const slice1 = new DataSlice(buffer, 0, 2)
    const ema1 = new ExponentialMovingAverage(
      {
        tx: {
          in: 'close',
          out: 'ema',
          window: 3
        }
      },
      slice1
    )

    // Not ready initially
    strictEqual(ema1.isReady, false)

    // Process the data
    ema1.next(0, 1)
    strictEqual(ema1.isReady, false) // Still not ready with only 2 rows

    // Create new instance with enough data
    const slice2 = new DataSlice(buffer, 0, 3)
    const ema2 = new ExponentialMovingAverage(
      {
        tx: {
          in: 'close',
          out: 'ema',
          window: 3
        }
      },
      slice2
    )

    ema2.next(0, 3)
    strictEqual(ema2.isReady, true)
  })

  it('should handle different input columns', () => {
    const buffer = createTestBuffer([10, 20, 30, 40, 50])

    const initSlice = new DataSlice(buffer, 0, buffer.length())
    const ema = new ExponentialMovingAverage(
      {
        tx: [
          { in: 'high', out: 'high_ema', window: 2 },
          { in: 'low', out: 'low_ema', window: 2 }
        ]
      },
      initSlice
    )

    ema.next(0, buffer.length())

    const outputBuffer = ema.outputBuffer
    const row1 = outputBuffer.getRow(1)

    // high values are close + 5, low values are close - 5
    strictEqual(row1?.high_ema, 20) // (15+25)/2 initial SMA
    strictEqual(row1?.low_ema, 10) // (5+15)/2 initial SMA

    // Check EMA calculation on next row
    const row2 = outputBuffer.getRow(2)
    const multiplier = 2 / 3
    strictEqual(row2?.high_ema, (35 - 20) * multiplier + 20) // 30
    strictEqual(row2?.low_ema, (25 - 10) * multiplier + 10) // 20
  })
})
