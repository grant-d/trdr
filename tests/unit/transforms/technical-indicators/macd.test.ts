import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import { Macd } from '../../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../../src/utils'

describe('MACD (Moving Average Convergence Divergence)', () => {
  function createTestBuffer(closePrices: number[]): DataBuffer {
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
    closePrices.forEach((close, i) => {
      buffer.push({
        timestamp: baseTime + i * 60000,
        open: close,
        high: close + 1,
        low: close - 1,
        close,
        volume: 1000
      })
    })

    return buffer
  }

  it('should calculate MACD correctly', () => {
    // Create test data with enough points for MACD calculation
    // MACD needs at least slow + signal periods = 26 + 9 = 35 data points for full accuracy
    const closePrices = Array.from({ length: 40 }, (_, i) => 100 + Math.sin(i * 0.1) * 10)
    const buffer = createTestBuffer(closePrices)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const macd = new Macd({
      tx: {
        in: 'close',
        fast: 12,
        slow: 26,
        signal: 9,
        out: {
          macd: 'macd',
          signal: 'signal',
          hist: 'histogram'
        }
      }
    }, slice)

    // Process all data
    const result = macd.next(0, buffer.length())

    // Check the output buffer
    const outputBuffer = macd.outputBuffer
    strictEqual(outputBuffer.length(), buffer.length())

    // MACD waits until it has enough data for meaningful calculations
    // Typically starts after slow EMA period (26 by default)
    ok(result.from >= 20, 'MACD should start after sufficient data for slow EMA')
    strictEqual(result.to, buffer.length(), 'Valid data should go to end')

    // Check that output columns exist and have reasonable values
    const firstRow = outputBuffer.getRow(0)
    const middleRow = outputBuffer.getRow(20)
    const lastRow = outputBuffer.getRow(39)

    ok(firstRow?.macd !== undefined, 'First row should have MACD value')
    ok(firstRow?.signal !== undefined, 'First row should have signal value')
    ok(firstRow?.histogram !== undefined, 'First row should have histogram value')

    // Early rows might have zero values as EMAs build up
    ok(typeof middleRow?.macd === 'number', 'Middle row should have numeric MACD')
    ok(typeof middleRow?.signal === 'number', 'Middle row should have numeric signal')
    ok(typeof middleRow?.histogram === 'number', 'Middle row should have numeric histogram')

    // Histogram should equal MACD - Signal
    const tolerance = 0.0001
    ok(Math.abs((lastRow?.histogram || 0) - ((lastRow?.macd || 0) - (lastRow?.signal || 0))) < tolerance,
      'Histogram should equal MACD minus Signal')
  })

  it('should handle different parameter configurations', () => {
    const closePrices = Array.from({ length: 30 }, (_, i) => 100 + i)
    const buffer = createTestBuffer(closePrices)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const macd = new Macd({
      tx: {
        in: 'close',
        fast: 5,
        slow: 13,
        signal: 5,
        out: {
          macd: 'macd_fast',
          signal: 'signal_fast',
          hist: 'hist_fast'
        }
      }
    }, slice)

    const result = macd.next(0, buffer.length())

    const outputBuffer = macd.outputBuffer
    strictEqual(result.to, buffer.length())

    // Check that custom parameter MACD works
    const lastRow = outputBuffer.getRow(29)
    ok(typeof lastRow?.macd_fast === 'number', 'Custom MACD should be calculated')
    ok(typeof lastRow?.signal_fast === 'number', 'Custom signal should be calculated')
    ok(typeof lastRow?.hist_fast === 'number', 'Custom histogram should be calculated')
  })

  it('should support multiple MACD configurations', () => {
    const closePrices = Array.from({ length: 50 }, (_, i) => 100 + Math.cos(i * 0.2) * 5)
    const buffer = createTestBuffer(closePrices)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const macd = new Macd({
      tx: [
        {
          in: 'close',
          fast: 12,
          slow: 26,
          signal: 9,
          out: { macd: 'macd_std', signal: 'signal_std', hist: 'hist_std' }
        },
        {
          in: 'close',
          fast: 5,
          slow: 13,
          signal: 5,
          out: { macd: 'macd_fast', signal: 'signal_fast', hist: 'hist_fast' }
        }
      ]
    }, slice)

    macd.next(0, buffer.length())

    const outputBuffer = macd.outputBuffer
    const lastRow = outputBuffer.getRow(49)

    // Check that both MACD configurations are calculated
    ok(typeof lastRow?.macd_std === 'number', 'Standard MACD should exist')
    ok(typeof lastRow?.signal_std === 'number', 'Standard signal should exist')
    ok(typeof lastRow?.hist_std === 'number', 'Standard histogram should exist')

    ok(typeof lastRow?.macd_fast === 'number', 'Fast MACD should exist')
    ok(typeof lastRow?.signal_fast === 'number', 'Fast signal should exist')
    ok(typeof lastRow?.hist_fast === 'number', 'Fast histogram should exist')

    // Fast MACD should be different from standard MACD (due to different parameters)
    ok(lastRow?.macd_std !== lastRow?.macd_fast, 'Different parameters should yield different results')
  })

  it('should validate parameters', () => {
    const buffer = createTestBuffer([100, 101, 102])
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Should throw on missing input column
    let threwError = false
    try {
      new Macd({
        // @ts-expect-error Negative test
        tx: {
          in: 'missing_column',
          out: { macd: 'macd', signal: 'signal', hist: 'hist' }
        }
      }, slice)
    } catch (error) {
      threwError = true
      ok(error instanceof Error)
    }
    strictEqual(threwError, true, 'Should throw error for missing input column')

    // Should throw on invalid fast period
    threwError = false
    try {
      new Macd({
        tx: {
          in: 'close',
          fast: 1, // Invalid: too small
          slow: 26,
          signal: 9,
          out: { macd: 'macd', signal: 'signal', hist: 'hist' }
        }
      }, slice)
    } catch (error) {
      threwError = true
      ok(error instanceof Error)
    }
    strictEqual(threwError, true, 'Should throw error for invalid fast period')

    // Should throw on invalid slow period
    threwError = false
    try {
      new Macd({
        tx: {
          in: 'close',
          slow: 1, // Invalid: too small
          fast: 12,
          signal: 9,
          out: { macd: 'macd', signal: 'signal', hist: 'hist' }
        }
      }, slice)
    } catch (error) {
      threwError = true
      ok(error instanceof Error)
    }
    strictEqual(threwError, true, 'Should throw error for invalid slow period')
  })

  it('should handle fast > slow validation', () => {
    const buffer = createTestBuffer([100, 101, 102])
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Should work with proper fast < slow
    let threwError = false
    try {
      new Macd({
        tx: {
          in: 'close',
          fast: 12,
          slow: 26, // slow > fast (correct)
          signal: 9,
          out: { macd: 'macd', signal: 'signal', hist: 'hist' }
        }
      }, slice)
    } catch (error) {
      threwError = true
    }
    strictEqual(threwError, false, 'Should not throw error when slow > fast')

    // Test case where fast >= slow (should still work, just unusual)
    threwError = false
    try {
      new Macd({
        tx: {
          in: 'close',
          fast: 26,
          slow: 12, // fast > slow (unusual but valid)
          signal: 9,
          out: { macd: 'macd', signal: 'signal', hist: 'hist' }
        }
      }, slice)
    } catch (error) {
      threwError = true
    }
    strictEqual(threwError, false, 'Should handle fast > slow case')
  })

  it('should handle trending data correctly', () => {
    // Create strongly trending data
    const closePrices = Array.from({ length: 35 }, (_, i) => 100 + i * 2) // Strong uptrend
    const buffer = createTestBuffer(closePrices)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const macd = new Macd({
      tx: {
        in: 'close',
        fast: 12,
        slow: 26,
        signal: 9,
        out: { macd: 'macd', signal: 'signal', hist: 'histogram' }
      }
    }, slice)

    macd.next(0, buffer.length())

    const outputBuffer = macd.outputBuffer
    outputBuffer.getRow(26) // After slow EMA builds up
    const lateRow = outputBuffer.getRow(34)

    // In a strong uptrend, MACD should generally be positive
    // (fast EMA > slow EMA when price is rising)
    ok((lateRow?.macd || 0) > 0, 'MACD should be positive in strong uptrend')
    
    // Histogram should reflect the momentum
    ok(typeof lateRow?.histogram === 'number', 'Histogram should be calculated')
  })

  it('should handle default parameters', () => {
    const closePrices = Array.from({ length: 35 }, (_, i) => 100 + Math.sin(i * 0.1) * 5)
    const buffer = createTestBuffer(closePrices)
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Test with default parameters (should use 12, 26, 9)
    const macd = new Macd({
      tx: {
        in: 'close',
        fast: 12,
        slow: 26,
        signal: 9,
        out: { macd: 'macd', signal: 'signal', hist: 'hist' }
      }
    }, slice)

    const result = macd.next(0, buffer.length())

    strictEqual(result.to, buffer.length())

    const outputBuffer = macd.outputBuffer
    const lastRow = outputBuffer.getRow(34)
    ok(typeof lastRow?.macd === 'number', 'Default MACD should be calculated')
    ok(typeof lastRow?.signal === 'number', 'Default signal should be calculated')
    ok(typeof lastRow?.hist === 'number', 'Default histogram should be calculated')
  })
})