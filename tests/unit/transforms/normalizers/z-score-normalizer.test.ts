import { ok, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import { ZScoreNormalizer } from '../../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../../src/utils'

describe('ZScoreNormalizer', () => {
  // Helper to create test buffer with predictable values
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
        close: value + 2,
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
      const normalizer = new ZScoreNormalizer({ tx: { in: 'close', out: 'close_z', window: 20 } }, slice)
      ok(normalizer)
      strictEqual(normalizer.type, 'zScore')
      strictEqual(normalizer.name, 'ZScore')
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
        new ZScoreNormalizer({ tx: { in: 'close', out: 'close_z', window: 1 } }, slice)
      }, /Window size must be at least 2/)
    })
  })

  describe('rolling window z-score normalization', () => {
    it('should normalize data to mean=0, std=1', () => {
      // Create data with known values for window calculation
      const buffer = createTestBuffer([100, 110, 120, 130, 140])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_zscore',
          window: 3
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      strictEqual(outputBuffer.length(), 5)

      // Check the third row (index 2) which has a full window
      // Window [102, 112, 122]: close values from testData
      const window1 = [102, 112, 122]
      const mean1 = window1.reduce((a, b) => a + b) / 3 // 112
      const variance1 =
        window1.reduce((sum, val) => sum + Math.pow(val - mean1, 2), 0) / 3
      const std1 = Math.sqrt(variance1)
      const expected2 = (122 - mean1) / std1

      const row = outputBuffer.getRow(2)
      ok(Math.abs(row!.close_zscore! - expected2) < 0.0001)
    })

    it('should handle multiple fields', () => {
      const buffer = createTestBuffer([100, 110, 120])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: [
          { in: 'open', out: 'open_zscore', window: 3 },
          { in: 'close', out: 'close_zscore', window: 3 },
          { in: 'volume', out: 'volume_zscore', window: 3 }
        ]
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer

      strictEqual(outputBuffer.length(), 3)

      // Check that all fields have z-score versions
      const row = outputBuffer.getRow(2)
      ok(row?.open_zscore !== undefined)
      ok(row?.close_zscore !== undefined)
      ok(row?.volume_zscore !== undefined)
      ok(row?.high_zscore === undefined)
      ok(row?.low_zscore === undefined)
    })

    it('should handle constant values (std=0)', () => {
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
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_zscore',
          window: 3
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      strictEqual(outputBuffer.length(), 5)

      // When std=0 (all values same), all z-scores should be 0
      for (let i = 2; i < 5; i++) {
        const row = outputBuffer.getRow(i)
        strictEqual(row?.close_zscore, 0)
      }
    })

    it('should process data correctly with rolling window', () => {
      const buffer = createTestBuffer([100, 110, 120])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_zscore',
          window: 3
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      strictEqual(outputBuffer.length(), 3)

      // Check that output column exists
      const row = outputBuffer.getRow(2)
      ok(row?.close_zscore !== undefined)

      // Item should have a calculated z-score
      ok(typeof row.close_zscore === 'number')
    })
  })

  describe('rolling window behavior validation', () => {
    it('should normalize using rolling window', () => {
      const buffer = createTestBuffer([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_zscore',
          window: 3
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      strictEqual(outputBuffer.length(), 10)

      // Check a specific window (items 2, 3, 4 with closes 112, 122, 132)
      // Window for item at index 4 would be [122, 132, 142]
      const window = [122, 132, 142]
      const mean = window.reduce((a, b) => a + b, 0) / 3 // 132
      const variance =
        window.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / 3
      const std = Math.sqrt(variance)

      // Item at index 4 has value 142, should be normalized based on this window
      const expectedZ = (142 - mean) / std
      const row = outputBuffer.getRow(4)
      ok(Math.abs(row!.close_zscore! - expectedZ) < 0.01)
    })

    it('should handle insufficient data gracefully', () => {
      const buffer = createTestBuffer([100, 110, 120])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_zscore',
          window: 5
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer

      // With only 3 data points but window size 5, values before index 4 will have incomplete windows
      strictEqual(outputBuffer.length(), 3)
      
      // All values should still be calculated with available data
      for (let i = 0; i < 3; i++) {
        const row = outputBuffer.getRow(i)
        ok(typeof row?.close_zscore === 'number')
      }
    })
  })

  describe('column-driven configuration', () => {
    it('should use custom output column names', () => {
      const buffer = createTestBuffer([100, 110, 120])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_z',
          window: 3
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      const row = outputBuffer.getRow(2)
      ok(row?.close_z !== undefined)
      ok(row?.close_zscore === undefined)
    })

    it('should overwrite original columns when using same names', () => {
      const buffer = createTestBuffer([100, 110, 120])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'volume',
          out: 'volume',
          window: 3
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      
      strictEqual(outputBuffer.length(), 3)

      // Item should have normalized value (z-score)
      const row = outputBuffer.getRow(2)
      ok(typeof row?.volume === 'number')
      // Volume should be transformed to z-score, not original value
      ok(row?.volume !== 1200) // Original would be 1200 (120 * 10)
    })

    // ZScoreNormalizer doesn't support null output columns - test removed
  })

  // Test removed - getOutputFields and getRequiredFields not part of new API

  // Test removed - withParams not part of new API

  describe('readiness', () => {
    it('should not be ready before processing data', () => {
      const buffer = createTestBuffer([100, 110, 120, 130, 140])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_zscore',
          window: 5
        }
      }, slice)
      strictEqual(normalizer.isReady, false)
    })

    it('should be ready after processing enough data', () => {
      const buffer = createTestBuffer([100, 110, 120, 130, 140, 150])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_zscore',
          window: 3
        }
      }, slice)

      // Check readiness before processing
      strictEqual(normalizer.isReady, false)

      normalizer.next(0, buffer.length())
      
      // Should be ready after processing window size rows
      strictEqual(normalizer.isReady, true)
    })
  })

  describe('edge cases', () => {
    it('should handle single data point', () => {
      const buffer = createTestBuffer([100, 105, 110])
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_zscore',
          window: 3
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer

      strictEqual(outputBuffer.length(), 3)
      const row = outputBuffer.getRow(2)
      ok(typeof row?.close_zscore === 'number')
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
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_zscore',
          window: 3
        }
      }, slice)

      const result = normalizer.next(0, 0)
      strictEqual(result.to - result.from, 0)
    })

    it('should handle negative values', () => {
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
      
      // Add negative values
      buffer.push({
        timestamp: 1000,
        open: -100,
        high: -95,
        low: -105,
        close: -98,
        volume: 1000
      })
      buffer.push({
        timestamp: 2000,
        open: -50,
        high: -45,
        low: -55,
        close: -52,
        volume: 1100
      })
      buffer.push({
        timestamp: 3000,
        open: -200,
        high: -195,
        low: -205,
        close: -198,
        volume: 1200
      })
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_zscore',
          window: 3
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      strictEqual(outputBuffer.length(), 3)

      // Should have calculated z-score
      const row = outputBuffer.getRow(2)
      ok(typeof row?.close_zscore === 'number')
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
      buffer.push({
        timestamp: 3000,
        open: 1e15 + 2e14,
        high: 1e15 + 2e14 + 1e10,
        low: 1e15 + 2e14 - 1e10,
        close: 1e15 + 2e14,
        volume: 1e18 + 2e17
      })
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const normalizer = new ZScoreNormalizer({
        tx: {
          in: 'close',
          out: 'close_zscore',
          window: 3
        }
      }, slice)

      normalizer.next(0, buffer.length())
      const outputBuffer = normalizer.outputBuffer
      strictEqual(outputBuffer.length(), 3)

      // Should handle large numbers without issues
      const row = outputBuffer.getRow(2)
      ok(typeof row?.close_zscore === 'number') // Should calculate properly
    })
  })
})