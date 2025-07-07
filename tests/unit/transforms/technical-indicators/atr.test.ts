import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import { AverageTrueRange } from '../../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../../src/utils'

describe('Average True Range', () => {
  function createTestBuffer(ohlcData: Array<[number, number, number, number]>): DataBuffer {
    const buffer = new DataBuffer({
      columns: {
        timestamp: { index: 0 },
        symbol: { index: 1 },
        exchange: { index: 2 },
        open: { index: 3 },
        high: { index: 4 },
        low: { index: 5 },
        close: { index: 6 },
        volume: { index: 7 }
      }
    })

    const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
    ohlcData.forEach(([open, high, low, close], i) => {
      buffer.push({
        timestamp: baseTime + i * 60000,
        open,
        high,
        low,
        close,
        volume: 1000
      })
    })

    return buffer
  }

  it('should calculate ATR correctly', () => {
    // Test data with known ATR values
    const ohlcData: Array<[number, number, number, number]> = [
      [100, 105, 95, 102], // TR = 10 (high-low)
      [102, 108, 100, 106], // TR = 8 (high-low) or |108-102| = 6 or |100-102| = 2, max = 8
      [106, 110, 103, 108], // TR = 7 (high-low) or |110-106| = 4 or |103-106| = 3, max = 7
      [108, 112, 105, 110], // TR = 7
      [110, 115, 108, 113] // TR = 7
    ]
    const buffer = createTestBuffer(ohlcData)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const atr = new AverageTrueRange({
      in: { high: 'high', low: 'low', close: 'close' },
      tx: { out: 'atr', window: 3 }
    }, slice)

    // Process all data
    new DataSlice(buffer, 0, buffer.length())
    const result = atr.next(0, buffer.length())

    // Check the output buffer
    const outputBuffer = atr.outputBuffer
    strictEqual(outputBuffer.length(), buffer.length())

    // ATR calculates from first row
    strictEqual(result.from, 0, 'Valid ATR data should start from row 0')
    strictEqual(result.to, buffer.length(), 'Valid data should go to end')

    // Check ATR values
    const row0 = outputBuffer.getRow(0)
    const row1 = outputBuffer.getRow(1)
    const row2 = outputBuffer.getRow(2)
    const row3 = outputBuffer.getRow(3)
    outputBuffer.getRow(4)

    // First items should have partial ATR (running average before full window)
    strictEqual(row0?.atr, 10, 'First row should have initial TR as ATR')  // TR = 10
    strictEqual(row1?.atr, 9, 'Second row should have average of first 2 TRs')  // (10+8)/2 = 9

    // Third item should have initial ATR (average of first 3 TRs)
    // TR values: 10, 8, 7
    // Initial ATR = (10 + 8 + 7) / 3 = 8.33...
    const expectedInitialAtr = (10 + 8 + 7) / 3
    ok(Math.abs((row2?.atr || 0) - expectedInitialAtr) < 0.01, `Expected ATR ${expectedInitialAtr}, got ${row2?.atr}`)

    // Fourth item uses Wilder's smoothing
    // ATR = ((Previous ATR * (period-1)) + Current TR) / period
    const expectedAtr4 = (expectedInitialAtr * 2 + 7) / 3
    ok(Math.abs((row3?.atr || 0) - expectedAtr4) < 0.01, `Expected ATR ${expectedAtr4}, got ${row3?.atr}`)
  })

  it('should handle gaps correctly', () => {
    // Test data with price gaps
    const ohlcData: Array<[number, number, number, number]> = [
      [100, 105, 95, 100],
      [110, 115, 108, 112], // Gap up from 100 to 110
      [112, 120, 110, 118]
    ]
    const buffer = createTestBuffer(ohlcData)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const atr = new AverageTrueRange({
      in: { high: 'high', low: 'low', close: 'close' },
      tx: { out: 'atr', window: 3 }
    }, slice)

    // Process all data
    new DataSlice(buffer, 0, buffer.length())
    atr.next(0, buffer.length())

    const outputBuffer = atr.outputBuffer
    
    // Should handle the gap correctly - TR for second bar should be max of:
    // high-low = 115-108 = 7
    // |high-prev_close| = |115-100| = 15
    // |low-prev_close| = |108-100| = 8
    // So TR = 15 (the gap)
    
    // First row: TR = 105-95 = 10
    // Second row: TR = max(7, 15, 8) = 15
    // Third row: TR = max(120-110, |120-112|, |110-112|) = max(10, 8, 2) = 10
    
    // We need at least 3 values to calculate ATR, so only the third row should have a value
    const row2 = outputBuffer.getRow(2)
    ok(row2?.atr !== undefined && row2.atr > 0, 'Third row should have valid ATR accounting for gap')
  })

  it('should validate parameters', () => {
    const buffer = createTestBuffer([[100, 105, 95, 102]])
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Should throw on invalid window
    let threwError = false
    try {
      new AverageTrueRange({
        in: { high: 'high', low: 'low', close: 'close' },
        tx: { out: 'atr', window: 1 }
      }, slice)
    } catch (error) {
      threwError = true
      ok(error instanceof Error)
    }
    strictEqual(threwError, true, 'Should throw error for window < 2')

    // Should throw on missing columns
    threwError = false
    try {
      new AverageTrueRange({
        in: { high: 'missing', low: 'low', close: 'close' },
        tx: { out: 'atr', window: 14 }
      }, slice)
    } catch (error) {
      threwError = true
      ok(error instanceof Error)
    }
    strictEqual(threwError, true, 'Should throw error for missing column')
  })

  it('should handle first data point correctly', () => {
    // Single data point
    const ohlcData: Array<[number, number, number, number]> = [
      [100, 105, 95, 102]
    ]
    const buffer = createTestBuffer(ohlcData)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const atr = new AverageTrueRange({
      in: { high: 'high', low: 'low', close: 'close' },
      tx: { out: 'atr', window: 3 }
    }, slice)

    new DataSlice(buffer, 0, buffer.length())
    const result = atr.next(0, buffer.length())
    
    // Should produce TR as ATR for single data point
    const outputBuffer = atr.outputBuffer
    const row0 = outputBuffer.getRow(0)
    strictEqual(row0?.atr, 10, 'Single data point should produce TR as ATR (105-95=10)')
    strictEqual(result.from, 0, 'Should have valid output row for single data point')
  })

  it('should smooth volatility over time', () => {
    // Test that ATR smooths out volatility spikes
    const ohlcData: Array<[number, number, number, number]> = [
      [100, 105, 95, 102],   // TR = 10
      [102, 104, 100, 103],  // TR = 4 (low volatility)
      [103, 130, 100, 125],  // TR = 30 (high volatility spike)
      [125, 128, 123, 126],  // TR = 5 (back to normal)
      [126, 129, 124, 127]   // TR = 5
    ]
    const buffer = createTestBuffer(ohlcData)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const atr = new AverageTrueRange({
      in: { high: 'high', low: 'low', close: 'close' },
      tx: { out: 'atr', window: 3 }
    }, slice)

    new DataSlice(buffer, 0, buffer.length())
    atr.next(0, buffer.length())

    const outputBuffer = atr.outputBuffer
    const row2 = outputBuffer.getRow(2) // First ATR
    outputBuffer.getRow(3) // ATR after spike
    const row4 = outputBuffer.getRow(4) // ATR smoothing down

    // First ATR should be average: (10 + 4 + 30) / 3 = 14.67
    const expectedFirstAtr = (10 + 4 + 30) / 3
    ok(Math.abs((row2?.atr || 0) - expectedFirstAtr) < 0.01)

    // ATR should be smoothing - not as high as the spike, not as low as current TR
    const currentAtr = row4?.atr || 0
    ok(currentAtr > 5, 'ATR should be higher than current low TR due to smoothing')
    ok(currentAtr < 30, 'ATR should be lower than the volatility spike')
  })
})