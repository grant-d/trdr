import { ok, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import { MinMaxNormalizer } from '../../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../../src/utils'

describe('MinMaxNormalizer', () => {
  // Helper to create test buffer
  const createTestBuffer = (values: number[]): DataBuffer => {
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
    
    values.forEach((value, i) => {
      buffer.push({
        timestamp: new Date('2024-01-01').getTime() + i * 60000,
        open: value,
        high: value + 10,
        low: value - 10,
        close: value + 5,
        volume: value * 10
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
      const normalizer = new MinMaxNormalizer({ tx: { in: 'close', out: 'close_norm', window: 10, min: 0, max: 1 } }, slice)
      ok(normalizer)
      strictEqual(normalizer.type, 'minMax')
      strictEqual(normalizer.name, 'MinMaxNormalizer')
    })

    it('should create instance with custom range', () => {
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
      const normalizer = new MinMaxNormalizer({ tx: { in: 'close', out: 'close_norm', window: 10, min: -1, max: 1 } }, slice)
      ok(normalizer)
    })

    it('should validate window size', () => {
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
      throws(() => {
        new MinMaxNormalizer({ tx: { in: 'close', out: 'close_norm', window: 1, min: 0, max: 1 } }, slice)
      }, /Window size must be at least 2/)
    })

    it('should validate target range', () => {
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
      throws(() => {
        new MinMaxNormalizer({ tx: { in: 'close', out: 'close_norm', window: 10, min: 1, max: 0 } }, slice)
      }, /targetMax must be greater than targetMin/)
    })
  })

  describe('rolling window min-max normalization', () => {
    it('should normalize data to [0, 1] range by default', () => {
      // Values: 0, 50, 100
      const buffer = createTestBuffer([0, 50, 100])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'open',
          out: 'open_norm',
          window: 3,
          min: 0,
          max: 1
        }
      }, slice)

      normalizer.next(0, buffer.length())
      
      // Check the output buffer
      const outputBuffer = normalizer.outputBuffer
      strictEqual(outputBuffer.length(), 3)
      
      // Third item has full window [0, 50, 100]: min=0, max=100, value=100
      const row = outputBuffer.getRow(2)
      strictEqual(row?.open_norm, 1) // (100-0)/(100-0) = 1
    })

    it('should normalize to custom range', () => {
      const buffer = createTestBuffer([10, 20, 30])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'open',
          out: 'open_norm',
          min: -1,
          max: 1,
          window: 3
        }
      }, slice)

      normalizer.next(0, buffer.length())
      
      const outputBuffer = normalizer.outputBuffer
      strictEqual(outputBuffer.length(), 3)
      
      // Third item has window [10, 20, 30]: min=10, max=30, value=30
      // Formula: (x-min)/range * targetRange + targetMin = (30-10)/20 * 2 + (-1) = 1
      const row = outputBuffer.getRow(2)
      strictEqual(row?.open_norm, 1)
    })

    it('should handle multiple fields', () => {
      const buffer = createTestBuffer([100, 200])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: [
          { in: 'open', out: 'open_norm', window: 2, min: 0, max: 1 },
          { in: 'close', out: 'close_norm', window: 2, min: 0, max: 1 },
          { in: 'volume', out: 'volume_norm', window: 2, min: 0, max: 1 }
        ]
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer

      strictEqual(outputBuffer.length(), 2)

      // Check the second row has all normalized values
      const row = outputBuffer.getRow(1)
      ok(row?.open_norm !== undefined)
      ok(row?.close_norm !== undefined)
      ok(row?.volume_norm !== undefined)
      ok(row?.high_norm === undefined)
      ok(row?.low_norm === undefined)
    })

    it('should handle constant values (range=0)', () => {
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
      
      // Add 5 rows with constant values
      for (let i = 0; i < 5; i++) {
        buffer.push({
          timestamp: 1000 + i * 1000,
          open: 100,
          high: 100,
          low: 100,
          close: 100,
          volume: 1000
        })
      }
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'close',
          out: 'close_norm',
          window: 3,
          min: 0,
          max: 1
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      strictEqual(outputBuffer.length(), 5)

      // When all values in window are the same, should use middle of target range
      // Check rows starting from window-1 (index 2)
      for (let i = 2; i < 5; i++) {
        const row = outputBuffer.getRow(i)
        strictEqual(row?.close_norm, 0.5) // Middle of [0, 1]
      }
    })
  })

  describe('rolling window behavior validation', () => {
    it('should normalize using rolling window', () => {
      const buffer = createTestBuffer([10, 20, 30, 25, 15])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'open',
          out: 'open_norm',
          window: 3,
          min: 0,
          max: 1
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      strictEqual(outputBuffer.length(), 5)

      // Check normalized values starting from window-1
      // Window [10, 20, 30]: min=10, max=30, value=30
      const row0 = outputBuffer.getRow(2)
      strictEqual(row0?.open_norm, 1) // (30-10)/(30-10) = 1

      // Window [20, 30, 25]: min=20, max=30, value=25
      const row1 = outputBuffer.getRow(3)
      const expected1 = (25 - 20) / (30 - 20) // 0.5
      strictEqual(row1?.open_norm, expected1)

      // Window [30, 25, 15]: min=15, max=30, value=15
      const row2 = outputBuffer.getRow(4)
      const expected2 = (15 - 15) / (30 - 15) // 0
      strictEqual(row2?.open_norm, expected2)
    })

    it('should handle insufficient data gracefully', () => {
      const buffer = createTestBuffer([100, 105, 110])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'open',
          out: 'open_norm',
          window: 5,
          min: 0,
          max: 1
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer

      // With only 3 data points but window size 5, values before index 4 will be NaN
      strictEqual(outputBuffer.length(), 3)
      
      // Check that all values are NaN since we don't have enough data
      for (let i = 0; i < 3; i++) {
        const row = outputBuffer.getRow(i)
        ok(row?.open_norm === undefined || isNaN(row.open_norm))
      }
    })
  })

  describe('column-driven configuration', () => {
    it('should use custom output column names', () => {
      const buffer = createTestBuffer([10, 20, 30])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'close',
          out: 'close_scaled',
          window: 3,
          min: 0,
          max: 1
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      const row = outputBuffer.getRow(2)
      ok(row?.close_scaled !== undefined)
      ok(row?.close_norm === undefined)
    })

    it('should overwrite original columns when using same names', () => {
      const buffer = createTestBuffer([10, 20, 30])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'volume',
          out: 'volume',
          window: 3,
          min: 0,
          max: 1
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      // Check the third row
      const row = outputBuffer.getRow(2)
      // Volume: 300, window volumes: [100, 200, 300], min=100, max=300, range=200
      // (300-100)/200 = 1
      strictEqual(row?.volume, 1)
    })

    // MinMaxNormalizer doesn't support null output columns - test removed
  })

  describe('edge cases', () => {
    it('should handle zero variance windows', () => {
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

      // Test that all constant values result in midpoint
      for (let i = 0; i < 5; i++) {
        buffer.push({
          timestamp: 1000 + i * 1000,
          open: 100,
          high: 100,
          low: 100,
          close: 100,
          volume: 100
        })
      }
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'close',
          out: 'close_norm',
          window: 3,
          min: 0,
          max: 1
        }
      }, slice)
      
      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      // All values should be 0.5 (midpoint of [0,1])
      for (let i = 2; i < 5; i++) {
        const row = outputBuffer.getRow(i)
        strictEqual(row?.close_norm, 0.5)
      }
    })
  })

  // Test removed - getOutputFields and getRequiredFields not part of new API

  // Test removed - withParams not part of new API

  describe('readiness', () => {
    it('should track readiness based on window size', () => {
      const buffer = createTestBuffer([1, 2, 3, 4, 5, 6])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'open',
          out: 'open_norm',
          window: 3,
          min: 0,
          max: 1
        }
      }, slice)
      
      strictEqual(normalizer.isReady, false)
      
      normalizer.next(0, buffer.length())
      
      // Should be ready after processing window size rows
      strictEqual(normalizer.isReady, true)
    })
  })

  describe('additional edge cases', () => {
    it('should handle single data point', () => {
      const buffer = createTestBuffer([100, 105])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'close',
          out: 'close_norm',
          window: 2,
          min: 0,
          max: 1
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      strictEqual(outputBuffer.length(), 2)
      // Second row should have normalized value
      const row = outputBuffer.getRow(1)
      ok(typeof row?.close_norm === 'number')
    })

    it('should handle empty data stream', () => {
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
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'close',
          out: 'close_norm',
          window: 5,
          min: 0,
          max: 1
        }
      }, slice)

      const result = normalizer.next(0, 0)
      strictEqual(result.to - result.from, 0)
    })

    it('should handle all zeros', () => {
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
      
      // Add all zeros
      for (let i = 0; i < 5; i++) {
        buffer.push({
          timestamp: 1000 + i * 1000,
          open: 0,
          high: 0,
          low: 0,
          close: 0,
          volume: 0
        })
      }
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: [
          { in: 'close', out: 'close_norm', window: 3, min: 0, max: 1 },
          { in: 'volume', out: 'volume_norm', window: 3, min: 0, max: 1 }
        ]
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      // All values same (0), should use middle of range
      for (let i = 2; i < 5; i++) {
        const row = outputBuffer.getRow(i)
        strictEqual(row?.close_norm, 0.5)
        strictEqual(row?.volume_norm, 0.5)
      }
    })

    it('should handle negative values', () => {
      const buffer = createTestBuffer([-100, -50, -200])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'open',
          out: 'open_norm',
          window: 3,
          min: 0,
          max: 1
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      // Window [-100, -50, -200]: min=-200, max=-50, value=-200
      // (-200-(-200))/(-50-(-200)) = 0/150 = 0
      const row = outputBuffer.getRow(2)
      strictEqual(row?.open_norm, 0)
    })

    it('should handle very large values', () => {
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
        open: 1e15,
        high: 1e15 + 1e10,
        low: 1e15 - 1e10,
        close: 1e15,
        volume: 1e18
      })
      
      buffer.push({
        timestamp: 2000,
        open: 1e15 + 1e14,
        high: 1e15 + 1e14 + 1e10,
        low: 1e15 + 1e14 - 1e10,
        close: 1e15 + 1e14,
        volume: 1e18 + 1e17
      })
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new MinMaxNormalizer({
        tx: {
          in: 'open',
          out: 'open_norm',
          window: 2,
          min: 0,
          max: 1
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      // Second item should have normalized value of 1
      const row = outputBuffer.getRow(1)
      strictEqual(row?.open_norm, 1)
    })
  })
})
