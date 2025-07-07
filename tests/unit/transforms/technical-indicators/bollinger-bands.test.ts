import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import { BollingerBands } from '../../../../src/transforms'
import { DataBuffer } from '../../../../src/utils'
import { DataSlice } from '../../../../src/utils'

// Helper to create test buffer
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

describe('Bollinger Bands', () => {
  it('should calculate bands correctly', () => {
    const values = [20, 22, 21, 23, 24, 25, 20, 30, 15, 35]
    const buffer = createTestBuffer(values)
    const initSlice = new DataSlice(buffer, 0, buffer.length())

    const bb = new BollingerBands(
      {
        tx: {
          in: 'close',
          out: {
            upper: 'bb_upper',
            middle: 'bb_middle',
            lower: 'bb_lower'
          },
          window: 5,
          std: 2
        }
      },
      initSlice
    )

    new DataSlice(buffer, 0, buffer.length())
    const result = bb.next(0, buffer.length())

    const outputBuffer = bb.outputBuffer
    strictEqual(outputBuffer.length(), 10)

    // Result should indicate valid data starts from row 5 (index 4)
    strictEqual(result.from, 4, 'Valid data should start from row 5')
    strictEqual(result.to, buffer.length(), 'Valid data should go to end')

    // First 4 items shouldn't have bands (or have default values)
    for (let i = 0; i < 4; i++) {
      const row = outputBuffer.getRow(i)
      ok(row?.bb_middle === 0 || row?.bb_middle === undefined)
      ok(row?.bb_upper === 0 || row?.bb_upper === undefined)
      ok(row?.bb_lower === 0 || row?.bb_lower === undefined)
    }

    // From 5th item onwards should have bands
    const item5 = outputBuffer.getRow(4)
    ok(item5?.bb_middle !== undefined && item5.bb_middle !== 0)
    ok(item5?.bb_upper !== undefined && item5.bb_upper !== 0)
    ok(item5?.bb_lower !== undefined && item5.bb_lower !== 0)

    // Middle band should be SMA
    strictEqual(item5?.bb_middle, 22) // (20+22+21+23+24)/5

    // Upper band should be above middle
    ok(item5.bb_upper > item5.bb_middle)

    // Lower band should be below middle
    ok(item5.bb_lower < item5.bb_middle)

    // Bands should widen as volatility increases
    const lastItem = outputBuffer.getRow(9)
    ok(lastItem && item5 && lastItem.bb_upper && lastItem.bb_lower && item5.bb_upper && item5.bb_lower)
    if (lastItem && item5 && lastItem.bb_upper && lastItem.bb_lower && item5.bb_upper && item5.bb_lower) {
      ok(lastItem.bb_upper - lastItem.bb_lower > item5.bb_upper - item5.bb_lower)
    }
  })

  it('should respect standard deviation multiplier', () => {
    const values = [50, 52, 48, 51, 49, 53, 47, 52, 50, 51]
    const buffer = createTestBuffer(values)
    const initSlice = new DataSlice(buffer, 0, buffer.length())

    // Test with different stdDev values
    const bb1 = new BollingerBands(
      {
        tx: {
          in: 'close',
          out: {
            upper: 'bb_upper_1',
            middle: 'bb_middle_1',
            lower: 'bb_lower_1'
          },
          window: 3,
          std: 1
        }
      },
      initSlice
    )

    const bb2 = new BollingerBands(
      {
        tx: {
          in: 'close',
          out: {
            upper: 'bb_upper_2',
            middle: 'bb_middle_2',
            lower: 'bb_lower_2'
          },
          window: 3,
          std: 2
        }
      },
      new DataSlice(buffer, 0, buffer.length())
    )

    new DataSlice(buffer, 0, buffer.length())
    bb1.next(0, buffer.length())
    bb2.next(0, buffer.length())

    const outputBuffer1 = bb1.outputBuffer
    const outputBuffer2 = bb2.outputBuffer

    // Compare band widths at row 2 (third item)
    const item1 = outputBuffer1.getRow(2)
    const item2 = outputBuffer2.getRow(2)

    // Middle bands should be the same (both are SMA)
    strictEqual(item1?.bb_middle_1, item2?.bb_middle_2)

    // 2 stdDev bands should be wider than 1 stdDev
    ok(item1?.bb_upper_1 !== undefined && item1.bb_lower_1 !== undefined)
    ok(item2?.bb_upper_2 !== undefined && item2.bb_lower_2 !== undefined)
    const width1 = item1.bb_upper_1 - item1.bb_lower_1
    const width2 = item2.bb_upper_2 - item2.bb_lower_2
    // Use approximate equality due to floating point precision
    ok(
      Math.abs(width2 - width1 * 2) < 0.0001,
      `Width2 (${width2}) should be approximately 2x width1 (${width1})`
    )
  })

  it('should handle low volatility correctly', () => {
    // Very low volatility data
    const values = [100, 100.1, 99.9, 100, 100.1, 99.9, 100, 100.1, 99.9, 100]
    const buffer = createTestBuffer(values)
    const initSlice = new DataSlice(buffer, 0, buffer.length())

    const bb = new BollingerBands(
      {
        tx: {
          in: 'close',
          out: {
            upper: 'bb_upper',
            middle: 'bb_middle',
            lower: 'bb_lower'
          },
          window: 5,
          std: 2
        }
      },
      initSlice
    )

    new DataSlice(buffer, 0, buffer.length())
    bb.next(0, buffer.length())

    const outputBuffer = bb.outputBuffer

    // Check that bands are very tight
    const lastItem = outputBuffer.getRow(9)
    ok(lastItem?.bb_upper !== undefined && lastItem.bb_lower !== undefined)
    const bandwidth = lastItem.bb_upper - lastItem.bb_lower
    ok(bandwidth < 1) // Bands should be tight for low volatility
  })

  it('should validate output fields', () => {
    const buffer = createTestBuffer([100, 101, 102])
    const initSlice = new DataSlice(buffer, 0, buffer.length())

    // Should validate schema properly
    try {
      new BollingerBands(
        {
          tx: {
            in: 'close',
            // @ts-expect-error Negative test
            out: {
              upper: 'bb_upper'
              // Missing middle and lower
            },
            window: 20,
            std: 2
          }
        },
        initSlice
      )
      ok(false, 'Should have thrown')
    } catch (error: any) {
      ok(error instanceof Error)
      ok(
        error.message.includes('Required') || error.message.includes('missing')
      )
    }
  })

  it('should calculate standard deviation correctly', () => {
    // Known values for easy verification
    const values = [2, 4, 4, 4, 5, 5, 7, 9]
    const buffer = createTestBuffer(values)
    const initSlice = new DataSlice(buffer, 0, buffer.length())

    const bb = new BollingerBands(
      {
        tx: {
          in: 'close',
          out: {
            upper: 'bb_upper',
            middle: 'bb_middle',
            lower: 'bb_lower'
          },
          window: 8,
          std: 1
        }
      },
      initSlice
    )

    new DataSlice(buffer, 0, buffer.length())
    bb.next(0, buffer.length())

    const outputBuffer = bb.outputBuffer

    // At index 7, we have all 8 values
    const lastItem = outputBuffer.getRow(7)

    // Mean = (2+4+4+4+5+5+7+9)/8 = 40/8 = 5
    strictEqual(lastItem?.bb_middle, 5)

    // Standard deviation = 2
    // Upper = 5 + 1*2 = 7
    // Lower = 5 - 1*2 = 3
    strictEqual(lastItem?.bb_upper, 7)
    strictEqual(lastItem?.bb_lower, 3)
  })

  it('should handle multiple Bollinger Bands configurations', () => {
    const values = [50, 52, 48, 51, 49, 53, 47, 52, 50, 51]
    const buffer = createTestBuffer(values)
    const initSlice = new DataSlice(buffer, 0, buffer.length())

    const bb = new BollingerBands(
      {
        tx: [
          {
            in: 'close',
            out: {
              upper: 'bb1_upper',
              middle: 'bb1_middle',
              lower: 'bb1_lower'
            },
            window: 3,
            std: 1
          },
          {
            in: 'close',
            out: {
              upper: 'bb2_upper',
              middle: 'bb2_middle',
              lower: 'bb2_lower'
            },
            window: 3,
            std: 2
          }
        ]
      },
      initSlice
    )

    new DataSlice(buffer, 0, buffer.length())
    bb.next(0, buffer.length())

    const outputBuffer = bb.outputBuffer
    const row3 = outputBuffer.getRow(3)

    // Both sets of bands should be calculated
    ok(row3?.bb1_middle !== undefined)
    ok(row3?.bb2_middle !== undefined)

    // Middle bands should be the same (same SMA)
    ok(row3?.bb1_middle !== undefined && row3.bb2_middle !== undefined)
    strictEqual(row3.bb1_middle, row3.bb2_middle)

    // BB2 should be wider than BB1
    ok(row3.bb1_upper !== undefined && row3.bb1_lower !== undefined)
    ok(row3.bb2_upper !== undefined && row3.bb2_lower !== undefined)
    const width1 = row3.bb1_upper - row3.bb1_lower
    const width2 = row3.bb2_upper - row3.bb2_lower
    // Use approximate equality due to floating point precision
    ok(
      Math.abs(width2 - width1 * 2) < 0.0001,
      `Width2 (${width2}) should be approximately 2x width1 (${width1})`
    )
  })
})
