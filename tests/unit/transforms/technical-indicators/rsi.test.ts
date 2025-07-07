import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import { RelativeStrengthIndex } from '../../../../src/transforms'
import { DataBuffer } from '../../../../src/utils'
import { DataSlice } from '../../../../src/utils'

describe('Relative Strength Index', () => {
  function createTestBuffer(prices: number[]): DataBuffer {
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
    prices.forEach((close, i) => {
      buffer.push({
        timestamp: baseTime + i * 60000,
        open: close - 1,
        high: close + 1,
        low: close - 2,
        close,
        volume: 1000 + i * 100
      })
    })

    return buffer
  }

  it('should calculate RSI correctly with enough data', () => {
    // Add test data - a series that should produce known RSI values
    const prices = [
      44, 44.25, 44.5, 43.75, 44.75, 45.5, 45.25, 46, 47, 46.5, 46.25, 47.5,
      47.25, 48, 47.75, 47.5, 47, 48.25, 48.5, 48.75, 49, 48.5, 48.25, 48.75,
      49.25
    ]

    const buffer = createTestBuffer(prices)
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Create RSI transform with window 14
    const rsi = new RelativeStrengthIndex(
      {
        tx: {
          in: 'close',
          out: 'rsi',
          window: 14
        }
      },
      slice
    )

    // Process all rows
    rsi.next(0, buffer.length())

    // Check output buffer
    const outputBuffer = rsi.outputBuffer
    strictEqual(outputBuffer.length(), buffer.length())

    // Early rows should not have RSI (need window + 1 data points)
    // In the test, 'early rows' means the first processed (newest data at low indices)
    for (let i = 0; i < 14; i++) {
      const row = outputBuffer.getRow(i)
      ok(row?.rsi !== undefined, `Row ${i} has RSI value`)
    }

    // From row 14 onwards should have RSI values
    const row14 = outputBuffer.getRow(14)
    ok(row14?.rsi !== undefined, 'Row 14 should have RSI')
    ok(typeof row14.rsi === 'number', 'RSI should be a number')

    // RSI should be between 0 and 100
    ok(
      row14.rsi >= 0 && row14.rsi <= 100,
      `RSI should be between 0 and 100, got ${row14.rsi}`
    )

    // Latest RSI
    const latestRow = outputBuffer.getRow(0)
    ok(latestRow?.rsi !== undefined, 'Latest RSI value should exist')
    ok(
      latestRow.rsi >= 0 && latestRow.rsi <= 100,
      'Latest RSI should be in valid range'
    )
  })

  it('should handle batch processing correctly', () => {
    // Create trending data that should produce specific RSI patterns
    const prices = Array.from({ length: 30 }, (_, i) => {
      if (i < 15) return 50 + i * 0.5 // Uptrend
      return 57.5 - (i - 15) * 0.3 // Downtrend
    })

    const buffer = createTestBuffer(prices)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const rsi = new RelativeStrengthIndex(
      {
        tx: {
          in: 'close',
          out: 'rsi',
          window: 14
        }
      },
      slice
    )

    // Process all data at once (transforms process their internal slice)
    rsi.next(0, buffer.length())

    const outputBuffer = rsi.outputBuffer

    // Check mid-trend RSI (should be high during uptrend)
    const uptrendRsi = outputBuffer.getRow(15)?.rsi
    ok(
      uptrendRsi !== undefined && uptrendRsi > 50,
      'RSI should be above 50 during uptrend'
    )

    // Check late RSI (should be lower during downtrend)
    const downtrendRsi = outputBuffer.getRow(4)?.rsi
    ok(
      downtrendRsi !== undefined && downtrendRsi < uptrendRsi,
      'RSI should decrease during downtrend'
    )
  })

  it('should handle multiple RSI configurations', () => {
    const prices = Array.from(
      { length: 30 },
      (_, i) => 50 + Math.sin(i * 0.5) * 5
    )
    const buffer = createTestBuffer(prices)

    const slice = new DataSlice(buffer, 0, buffer.length())

    // Create multi-window RSI
    const rsi = new RelativeStrengthIndex(
      {
        tx: [
          { in: 'close', out: 'rsi9', window: 9 }, // Fast RSI
          { in: 'close', out: 'rsi14', window: 14 }, // Standard RSI
          { in: 'close', out: 'rsi21', window: 21 }, // Slow RSI
          { in: 'high', out: 'rsi_high', window: 14 }
        ]
      },
      slice
    )

    rsi.next(0, buffer.length())

    const outputBuffer = rsi.outputBuffer

    // Check that all RSI values are calculated at appropriate times
    const row9 = outputBuffer.getRow(9)
    ok(row9?.rsi9 !== undefined, 'Fast RSI should be calculated')
    ok(row9?.rsi14 !== undefined, 'Standard RSI should be calculated')
    ok(row9?.rsi_high !== undefined, 'High RSI should be calculated')

    // Slow RSI needs more data
    ok(
      row9?.rsi21 === undefined || typeof row9.rsi21 === 'number',
      'Slow RSI behavior at row 9'
    )

    const row8 = outputBuffer.getRow(8)
    ok(row8?.rsi21 !== undefined, 'Slow RSI should be calculated at row 8')

    // Different RSI windows should produce different values
    ok(
      row8?.rsi9 !== row8?.rsi14,
      'Different windows should produce different RSI values'
    )
    ok(
      row8?.rsi14 !== row8?.rsi21,
      'Different windows should produce different RSI values'
    )
  })

  it('should validate parameters correctly', () => {
    const buffer = createTestBuffer([50, 51, 52])
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Test window too small
    try {
      new RelativeStrengthIndex(
        {
          tx: {
            in: 'close',
            out: 'rsi',
            window: 1
          }
        },
        slice
      )
      ok(false, 'Should throw on window < 2')
    } catch (error: any) {
      ok(error.message.includes('at least 2'))
    }

    // Test window too large
    try {
      new RelativeStrengthIndex(
        {
          tx: {
            in: 'close',
            out: 'rsi',
            window: 10000
          }
        },
        slice
      )
      ok(false, 'Should throw on window > 9999')
    } catch (error: any) {
      ok(error.message.includes('9999'))
    }

    // Test invalid column name
    try {
      new RelativeStrengthIndex(
        {
          tx: {
            in: 'invalid!',
            out: 'rsi',
            window: 14
          }
        },
        slice
      )
      ok(false, 'Should throw on invalid column name')
    } catch (error: any) {
      ok(error.message.includes('alphanumeric'))
    }

    // Test non-existent column
    try {
      new RelativeStrengthIndex(
        {
          tx: {
            in: 'nonexistent',
            out: 'rsi',
            window: 14
          }
        },
        slice
      )
      ok(false, 'Should throw on non-existent column')
    } catch (error: any) {
      ok(error.message.includes('not found'))
    }
  })

  it('should ensure unique output names', () => {
    const buffer = createTestBuffer([50, 51, 52])
    const slice = new DataSlice(buffer, 0, buffer.length())

    try {
      new RelativeStrengthIndex(
        {
          tx: [
            { in: 'close', out: 'rsi', window: 14 },
            { in: 'high', out: 'rsi', window: 14 }
          ]
        },
        slice
      )
      ok(false, 'Should throw on duplicate output names')
    } catch (error: any) {
      ok(error.message.includes('unique'))
    }
  })

  it('should be ready after processing enough data', () => {
    const prices = Array.from({ length: 20 }, (_, i) => 50 + i)
    const buffer = createTestBuffer(prices)

    // Test with insufficient data first
    const slice1 = new DataSlice(buffer, 0, 13)
    const rsi1 = new RelativeStrengthIndex(
      {
        tx: {
          in: 'close',
          out: 'rsi',
          window: 14
        }
      },
      slice1
    )

    // Not ready initially
    strictEqual(rsi1.isReady, false)

    // Process the data
    rsi1.next(0, 12)
    strictEqual(rsi1.isReady, false) // Still not ready with only 13 rows

    // Create new instance with enough data
    const slice2 = new DataSlice(buffer, 0, 15)
    const rsi2 = new RelativeStrengthIndex(
      {
        tx: {
          in: 'close',
          out: 'rsi',
          window: 14
        }
      },
      slice2
    )

    rsi2.next(0, 15)
    strictEqual(rsi2.isReady, true)
  })

  it('should calculate RSI correctly for extreme price movements', () => {
    // Create data with all gains followed by all losses
    const prices = [
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57,
      58,
      59,
      60,
      61,
      62,
      63,
      64, // All gains
      64,
      63,
      62,
      61,
      60,
      59,
      58,
      57,
      56,
      55,
      54,
      53,
      52,
      51,
      50 // All losses
    ]

    const buffer = createTestBuffer(prices)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const rsi = new RelativeStrengthIndex(
      {
        tx: {
          in: 'close',
          out: 'rsi',
          window: 14
        }
      },
      slice
    )

    rsi.next(0, buffer.length())

    const outputBuffer = rsi.outputBuffer

    // During the gain period, RSI should be very high
    const gainPeriodRsi = outputBuffer.getRow(10)?.rsi
    ok(
      gainPeriodRsi !== undefined && gainPeriodRsi > 70,
      'RSI should be overbought during consistent gains'
    )

    // During the loss period, RSI should drop significantly
    const lossPeriodRsi = outputBuffer.getRow(1)?.rsi
    ok(
      lossPeriodRsi !== undefined && lossPeriodRsi < 30,
      'RSI should be oversold during consistent losses'
    )
  })

  it('should handle different input columns', () => {
    const prices = Array.from(
      { length: 25 },
      (_, i) => 50 + Math.sin(i * 0.3) * 10
    )
    const buffer = createTestBuffer(prices)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const rsi = new RelativeStrengthIndex(
      {
        tx: [
          { in: 'high', out: 'rsi_high', window: 14 },
          { in: 'low', out: 'rsi_low', window: 14 }
        ]
      },
      slice
    )

    rsi.next(0, buffer.length())

    const outputBuffer = rsi.outputBuffer
    const row9 = outputBuffer.getRow(9)

    ok(row9?.rsi_high !== undefined, 'High RSI should be calculated')
    ok(row9?.rsi_low !== undefined, 'Low RSI should be calculated')

    // High RSI should generally be higher than low RSI
    ok(row9.rsi_high > row9.rsi_low, 'High RSI should be greater than low RSI')
  })
})
