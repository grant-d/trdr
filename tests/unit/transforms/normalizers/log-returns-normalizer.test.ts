import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import { LogReturnsNormalizer } from '../../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../../src/utils'

describe('LogReturnsNormalizer', () => {
  // Helper to create test buffer
  const createTestBuffer = (prices: number[]): DataBuffer => {
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
    
    prices.forEach((price, i) => {
      buffer.push({
        timestamp: new Date('2024-01-01').getTime() + i * 60000,
        open: price - 1,
        high: price + 1,
        low: price - 2,
        close: price,
        volume: 1000
      })
    })
    
    return buffer
  }

  describe('constructor and validation', () => {
    it('should create instance with default parameters', () => {
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
      const slice = new DataSlice(buffer, 0, 0)
      const normalizer = new LogReturnsNormalizer({
        tx: { in: 'close', out: 'returns', base: 'ln' }
      }, slice)
      ok(normalizer)
      strictEqual(normalizer.type, 'logReturns')
      strictEqual(normalizer.name, 'LogReturns')
    })

    it('should create instance with custom base', () => {
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
      const slice = new DataSlice(buffer, 0, 0)
      const normalizer = new LogReturnsNormalizer({ 
        tx: { in: 'close', out: 'returns', base: 'log10' }
      }, slice)
      strictEqual(
        normalizer.description,
        'Calculates logarithmic returns from price data'
      )
    })
  })

  describe('log returns calculation', () => {
    it('should calculate natural log returns for close prices by default', () => {
      const buffer = createTestBuffer([100, 105, 110])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new LogReturnsNormalizer({
        tx: { in: 'close', out: 'close_returns', base: 'ln' }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      strictEqual(outputBuffer.length(), 3)

      // First row has no previous value, so returns 0
      const row0 = outputBuffer.getRow(0)
      strictEqual(row0?.close_returns, 0)

      // Calculate expected log returns: ln(105/100) and ln(110/105)
      const expectedReturn1 = Math.log(105 / 100)
      const expectedReturn2 = Math.log(110 / 105)

      const row1 = outputBuffer.getRow(1)
      ok(Math.abs(row1!.close_returns! - expectedReturn1) < 0.0001)
      
      const row2 = outputBuffer.getRow(2)
      ok(Math.abs(row2!.close_returns! - expectedReturn2) < 0.0001)
    })

    it('should calculate log10 returns when specified', () => {
      const buffer = createTestBuffer([100, 110])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new LogReturnsNormalizer({
        tx: { in: 'close', out: 'close_log10', base: 'log10' }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      strictEqual(outputBuffer.length(), 2)

      const expectedReturn = Math.log10(110 / 100)
      const row1 = outputBuffer.getRow(1)
      ok(Math.abs(row1!.close_log10! - expectedReturn) < 0.0001)
    })

    it('should use specified price field', () => {
      const buffer = createTestBuffer([100, 105])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new LogReturnsNormalizer({
        tx: { in: 'open', out: 'open_returns', base: 'log10' }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      strictEqual(outputBuffer.length(), 2)

      // Open values are price - 1, so 99 -> 104
      const expectedReturn = Math.log(104 / 99)
      const row1 = outputBuffer.getRow(1)
      ok(Math.abs(row1!.open_returns! - expectedReturn) < 0.0001)
    })

    it('should calculate returns for multiple fields', () => {
      const buffer = createTestBuffer([100, 105, 110])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new LogReturnsNormalizer({
        tx: [
          { in: 'open', out: 'open_returns', base: 'ln' },
          { in: 'close', out: 'close_returns', base: 'ln' },
          { in: 'high', out: 'high_returns', base: 'ln' }
        ]
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      strictEqual(outputBuffer.length(), 3)

      // Check that all fields have log returns
      const row1 = outputBuffer.getRow(1)
      ok(row1?.open_returns !== undefined)
      ok(row1?.close_returns !== undefined)
      ok(row1?.high_returns !== undefined)
      ok(row1?.low_returns === undefined) // Not included

      // Verify calculations
      const expectedCloseReturn = Math.log(105 / 100)
      ok(Math.abs(row1.close_returns - expectedCloseReturn) < 0.0001)
    })
  })

  describe('edge cases', () => {
    it('should handle zero prices', () => {
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
      
      buffer.push({
        timestamp: 1000,
        open: 100,
        high: 101,
        low: 99,
        close: 100,
        volume: 1000
      })
      buffer.push({
        timestamp: 2000,
        open: 0,
        high: 0,
        low: 0,
        close: 0,
        volume: 0
      })
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new LogReturnsNormalizer({
        tx: { in: 'close', out: 'close_returns', base: 'log10' }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      // Should return 0 for invalid price transitions
      const row1 = outputBuffer.getRow(1)
      strictEqual(row1?.close_returns, 0)
    })

    it('should handle negative prices', () => {
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
      
      buffer.push({
        timestamp: 1000,
        open: 100,
        high: 101,
        low: 99,
        close: 100,
        volume: 1000
      })
      buffer.push({
        timestamp: 2000,
        open: -100,
        high: -99,
        low: -101,
        close: -100,
        volume: 1000
      })
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new LogReturnsNormalizer({
        tx: { in: 'close', out: 'close_returns', base: 'ln' }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      // Should return 0 for invalid price transitions
      const row1 = outputBuffer.getRow(1)
      strictEqual(row1?.close_returns, 0)
    })

    it('should handle very large price changes', () => {
      const buffer = createTestBuffer([1, 1000000])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new LogReturnsNormalizer({
        tx: { in: 'close', out: 'close_returns', base: 'log10' }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      const expectedReturn = Math.log(1000000 / 1)
      const row1 = outputBuffer.getRow(1)
      ok(Math.abs(row1!.close_returns! - expectedReturn) < 0.0001)
    })

    it('should handle empty buffer', () => {
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
      const slice = new DataSlice(buffer, 0, 0)
      const normalizer = new LogReturnsNormalizer({
        tx: { in: 'close', out: 'close_returns', base: 'ln' }
      }, slice)

      const result = normalizer.next(0, 0)
      strictEqual(result.to - result.from, 0)
    })

    it('should handle single data point', () => {
      const buffer = createTestBuffer([100])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new LogReturnsNormalizer({
        tx: { in: 'close', out: 'close_returns', base: 'ln' }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      strictEqual(outputBuffer.length(), 1)
      // First value always returns 0
      const row0 = outputBuffer.getRow(0)
      strictEqual(row0?.close_returns, 0)
    })
  })

  describe('readiness', () => {
    it('should be ready immediately', () => {
      const buffer = createTestBuffer([100, 105])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new LogReturnsNormalizer({
        tx: { in: 'close', out: 'close_returns', base: 'log10' }
      }, slice)
      
      // Should be ready even before processing
      strictEqual(normalizer.isReady, true)
      
      normalizer.next(0, buffer.length())
      
      // Still ready after processing
      strictEqual(normalizer.isReady, true)
    })
  })

  describe('multiple configurations', () => {
    it('should handle array of configurations', () => {
      const buffer = createTestBuffer([100, 110, 120])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new LogReturnsNormalizer({
        tx: [
          { in: 'close', out: 'close_ln', base: 'ln' },
          { in: 'close', out: 'close_log10', base: 'log10' },
          { in: 'open', out: 'open_returns', base: 'ln' }
        ]
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      const row1 = outputBuffer.getRow(1)
      
      // Check natural log
      const expectedLn = Math.log(110 / 100)
      ok(Math.abs(row1!.close_ln! - expectedLn) < 0.0001)
      
      // Check log10
      const expectedLog10 = Math.log10(110 / 100)
      ok(Math.abs(row1!.close_log10! - expectedLog10) < 0.0001)
      
      // Check open returns (natural log by default)
      const expectedOpen = Math.log(109 / 99) // open values are price - 1
      ok(Math.abs(row1!.open_returns! - expectedOpen) < 0.0001)
    })
  })
})