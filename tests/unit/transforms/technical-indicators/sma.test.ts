import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import { SimpleMovingAverage } from '../../../../src/transforms'
import { DataBuffer } from '../../../../src/utils'
import { DataSlice } from '../../../../src/utils'

describe('Simple Moving Average', () => {
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

  it('should calculate SMA correctly', () => {
    const values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    const buffer = createTestBuffer(values)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const sma = new SimpleMovingAverage(
      {
        tx: [
          {
            in: 'close',
            out: 'sma',
            window: 3
          }
        ]
      },
      slice
    )

    // Process all data
    const result = sma.next(0, buffer.length())

    // Check the output buffer
    const outputBuffer = sma.outputBuffer
    strictEqual(outputBuffer.length(), buffer.length())

    // First two items shouldn't have valid SMA (insufficient data)
    // The result slice should indicate that only rows from index 2 onwards have valid data
    strictEqual(result.from, 2, 'Valid data should start from row 2')
    strictEqual(result.to, buffer.length(), 'Valid data should go to end')

    // Note: getRow now uses forward indexing - 0 is oldest
    const row0 = outputBuffer.getRow(0) // First item (oldest)
    const row1 = outputBuffer.getRow(1) // Second item
    ok(row0, 'Row 0 should exist')
    ok(row1, 'Row 1 should exist')
    strictEqual(row0.sma, 0, 'First row should have uninitialized SMA')
    strictEqual(row1.sma, 0, 'Second row should have uninitialized SMA')

    // From third item onwards, should have SMA
    const row2 = outputBuffer.getRow(2) // Third item from start
    const row3 = outputBuffer.getRow(3) // Fourth item
    const row4 = outputBuffer.getRow(4) // Fifth item
    const row9 = outputBuffer.getRow(9) // Last item (most recent)

    strictEqual(row2?.sma, 20) // (10+20+30)/3
    strictEqual(row3?.sma, 30) // (20+30+40)/3
    strictEqual(row4?.sma, 40) // (30+40+50)/3
    strictEqual(row9?.sma, 90) // (80+90+100)/3
  })

  it('should handle multiple SMA configurations', () => {
    const testData = [10, 20, 30, 40, 50]
    const buffer = createTestBuffer(testData)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const sma = new SimpleMovingAverage(
      {
        tx: [
          { in: 'close', out: 'sma_2', window: 2 },
          { in: 'close', out: 'sma_3', window: 3 },
          { in: 'volume', out: 'vol_sma', window: 2 }
        ]
      },
      slice
    )

    // Process all data
    sma.next(0, buffer.length())

    const outputBuffer = sma.outputBuffer

    // Check SMA with window 2
    const row1 = outputBuffer.getRow(1) // Second item from start
    strictEqual(row1?.sma_2, 15) // (10+20)/2
    strictEqual(row1?.vol_sma, 1000) // (1000+1000)/2

    // Check SMA with window 3
    const row2 = outputBuffer.getRow(2) // Third item from start
    strictEqual(row2?.sma_3, 20) // (10+20+30)/3

    const row4 = outputBuffer.getRow(4) // Last item (most recent)
    strictEqual(row4?.sma_2, 45) // (40+50)/2
    strictEqual(row4?.sma_3, 40) // (30+40+50)/3
  })

  it('should handle batch processing correctly', () => {
    const values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    const buffer = createTestBuffer(values)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const sma = new SimpleMovingAverage(
      {
        tx: [
          {
            in: 'close',
            out: 'sma',
            window: 3
          }
        ]
      },
      slice
    )

    // Process all data at once (transforms process their internal slice)
    sma.next(0, buffer.length())

    // Results should be the same as processing all at once
    const outputBuffer = sma.outputBuffer
    const row9 = outputBuffer.getRow(9) // Last item (index 9)
    strictEqual(row9?.sma, 90) // (80+90+100)/3
  })

  it('should validate window parameter', () => {
    const buffer = createTestBuffer([10, 20, 30])
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Test window too small
    try {
      new SimpleMovingAverage(
        {
          tx: [
            {
              in: 'close',
              out: 'sma',
              window: 1
            }
          ]
        },
        slice
      )
      ok(false, 'Should have thrown')
    } catch (error: any) {
      ok(error.message.includes('at least 2'))
    }

    // Test window too large
    try {
      new SimpleMovingAverage(
        {
          tx: [
            {
              in: 'close',
              out: 'sma',
              window: 10000
            }
          ]
        },
        slice
      )
      ok(false, 'Should have thrown')
    } catch (error: any) {
      ok(error.message.includes('9999'))
    }
  })

  it('should validate column names', () => {
    const buffer = createTestBuffer([10, 20, 30])
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Test invalid input column
    try {
      new SimpleMovingAverage(
        {
          tx: [
            {
              in: 'invalid-column!',
              out: 'sma',
              window: 3
            }
          ]
        },
        slice
      )
      ok(false, 'Should have thrown')
    } catch (error: any) {
      ok(error.message.includes('alphanumeric'))
    }

    // Test non-existent input column
    try {
      new SimpleMovingAverage(
        {
          tx: {
            in: 'nonexistent',
            out: 'sma',
            window: 3
          }
        },
        slice
      )
      ok(false, 'Should have thrown')
    } catch (error: any) {
      ok(error.message.includes('not found'))
    }
  })

  it('should ensure unique output names', () => {
    const buffer = createTestBuffer([10, 20, 30])
    const slice = new DataSlice(buffer, 0, buffer.length())

    try {
      new SimpleMovingAverage(
        {
          tx: [
            { in: 'close', out: 'sma', window: 3 },
            { in: 'open', out: 'sma', window: 3 }
          ]
        },
        slice
      )
      ok(false, 'Should have thrown')
    } catch (error: any) {
      ok(error.message.includes('unique'))
    }
  })

  it('should be ready after processing enough data', () => {
    const buffer = createTestBuffer([10, 20, 30, 40, 50])

    // Test with insufficient data first
    const slice1 = new DataSlice(buffer, 0, 2)
    const sma1 = new SimpleMovingAverage(
      {
        tx: {
          in: 'close',
          out: 'sma',
          window: 3
        }
      },
      slice1
    )

    // Not ready initially
    strictEqual(sma1.isReady, false)

    // Process the data
    sma1.next(0, 1)
    strictEqual(sma1.isReady, false) // Still not ready with only 2 rows

    // Create new instance with enough data
    const slice2 = new DataSlice(buffer, 0, 3)
    const sma2 = new SimpleMovingAverage(
      {
        tx: {
          in: 'close',
          out: 'sma',
          window: 3
        }
      },
      slice2
    )

    sma2.next(0, 2)
    strictEqual(sma2.isReady, true)
  })

  it('should handle different input columns', () => {
    const buffer = createTestBuffer([10, 20, 30, 40, 50])
    const slice = new DataSlice(buffer, 0, buffer.length())

    const sma = new SimpleMovingAverage(
      {
        tx: [
          { in: 'high', out: 'high_sma', window: 2 },
          { in: 'low', out: 'low_sma', window: 2 }
        ]
      },
      slice
    )

    sma.next(0, buffer.length())

    const outputBuffer = sma.outputBuffer
    const row1 = outputBuffer.getRow(1) // Second item from start

    // high values are close + 5, low values are close - 5
    strictEqual(row1?.high_sma, 20) // (15+25)/2
    strictEqual(row1?.low_sma, 10) // (5+15)/2
  })
})
