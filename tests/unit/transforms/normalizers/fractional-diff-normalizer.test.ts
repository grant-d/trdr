import { doesNotThrow, ok, strictEqual, throws } from 'node:assert/strict'
import { describe, it } from 'node:test'
import { FractionalDiffNormalizer } from '../../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../../src/utils'

describe('FractionalDiffNormalizer', () => {
  const createTestBuffer = (count = 4): DataBuffer => {
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
    
    const baseTime = Date.parse('2024-01-01T00:00:00Z')
    const timestamps = [baseTime, baseTime + 3600000, baseTime + 7200000, baseTime + 10800000]
    const opens = [100, 105, 110, 115]
    const highs = [110, 115, 120, 125]
    const lows = [95, 100, 105, 110]
    const closes = [105, 110, 115, 120]
    const volumes = [1000, 1200, 1500, 1800]
    
    for (let i = 0; i < count; i++) {
      buffer.push({
        timestamp: timestamps[i]!,
        open: opens[i]!,
        high: highs[i]!,
        low: lows[i]!,
        close: closes[i]!,
        volume: volumes[i]!
      })
    }
    
    return buffer
  }

  describe('validation', () => {
    it('should require d parameter in tx config', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, 0)
      
      throws(
        // @ts-expect-error Negative test
        () => new FractionalDiffNormalizer({ tx: { in: 'close', out: 'close_fd' } }, slice),
        { message: /Required/ }
      )
    })

    it('should reject d values outside valid range', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, 0)
      
      throws(() => new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: -0.5, maxWeights: 100, minWeight: 1e-5 } 
      }, slice), {
        message: /Number must be greater than or equal to 0/
      })

      throws(() => new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: 2.5, maxWeights: 100, minWeight: 1e-5 } 
      }, slice), {
        message: /Number must be less than or equal to 2/
      })
    })

    it('should accept valid d values', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, 0)
      
      doesNotThrow(() => new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: 0.5, maxWeights: 100, minWeight: 1e-5 } 
      }, slice))
      doesNotThrow(() => new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: 0, maxWeights: 100, minWeight: 1e-5 } 
      }, slice))
      doesNotThrow(() => new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: 1, maxWeights: 100, minWeight: 1e-5 } 
      }, slice))
      doesNotThrow(() => new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: 2, maxWeights: 100, minWeight: 1e-5 } 
      }, slice))
    })
  })

  describe('weight calculation', () => {
    it('should calculate correct weights for d=0.5', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, 0)
      const transform = new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: 0.5, maxWeights: 100, minWeight: 1e-5 } 
      }, slice)

      // Initialize the transform to calculate weights
      transform.next(0, 1)

      // Access weights through transform state
      const state = (transform as any)._state.values().next().value
      const weights = state.weights

      // First few weights for d=0.5
      // w_0 = 1
      // w_1 = -w_0 * (d - 0) / 1 = -1 * 0.5 / 1 = -0.5
      // w_2 = -w_1 * (d - 1) / 2 = -(-0.5) * (0.5 - 1) / 2 = 0.5 * (-0.5) / 2 = -0.125
      // w_3 = -w_2 * (d - 2) / 3 = -(-0.125) * (0.5 - 2) / 3 = 0.125 * (-1.5) / 3 = -0.0625
      strictEqual(weights[0], 1)
      strictEqual(weights[1], -0.5)
      strictEqual(weights[2], -0.125)
      strictEqual(weights[3], -0.0625)
    })

    it('should respect maxWeights parameter', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, 0)
      const transform = new FractionalDiffNormalizer({
        tx: { in: 'close', out: 'close_fd', d: 0.5, maxWeights: 5, minWeight: 1e-5 }
      }, slice)

      // Initialize the transform
      transform.next(0, 1)
      
      const state = (transform as any)._state.values().next().value
      const weights = state.weights
      ok(weights.length <= 5)
    })

    it('should respect minWeight threshold', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, 0)
      const transform = new FractionalDiffNormalizer({
        tx: { in: 'close', out: 'close_fd', d: 0.5, maxWeights: 100, minWeight: 0.1 }
      }, slice)

      // Initialize the transform
      transform.next(0, 1)
      
      const state = (transform as any)._state.values().next().value
      const weights = state.weights
      
      // All weights should have magnitude >= minWeight (except possibly the last)
      for (let i = 0; i < weights.length - 1; i++) {
        ok(Math.abs(weights[i]) >= 0.1)
      }
    })
  })

  describe('transformation', () => {
    it('should apply fractional differentiation by default', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, buffer.length())
      
      const transform = new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: 0.5, maxWeights: 100, minWeight: 1e-5 } 
      }, slice)
      
      // Process all data
      const result = transform.next(0, buffer.length())
      
      strictEqual(result.from, 0)
      strictEqual(result.to, buffer.length())

      // First data point uses only w_0 = 1
      strictEqual(result.underlyingBuffer.getValue(0, result.underlyingBuffer.getColumn('close_fd')!.index), 105)

      // Second data point: w_0 * current + w_1 * previous
      // For close: 1 * 110 + (-0.5) * 105 = 57.5
      strictEqual(result.underlyingBuffer.getValue(1, result.underlyingBuffer.getColumn('close_fd')!.index), 57.5)
    })

    it('should handle multiple columns with array config', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, buffer.length())
      
      const transform = new FractionalDiffNormalizer({
        tx: [
          { in: 'close', out: 'close_fd', d: 0.5, maxWeights: 100, minWeight: 1e-5 },
          { in: 'volume', out: 'volume_fd', d: 0.5, maxWeights: 100, minWeight: 1e-5 }
        ]
      }, slice)

      // Process all data
      const result = transform.next(0, buffer.length())

      // Check that both columns were created and transformed
      const closeFdCol = result.underlyingBuffer.getColumn('close_fd')
      const volumeFdCol = result.underlyingBuffer.getColumn('volume_fd')
      
      ok(closeFdCol)
      ok(volumeFdCol)

      // Verify first values
      strictEqual(result.underlyingBuffer.getValue(0, closeFdCol.index), 105)
      strictEqual(result.underlyingBuffer.getValue(0, volumeFdCol.index), 1000)
    })

    it('should handle d=0 (no differentiation)', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, buffer.length())
      
      const transform = new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: 0, maxWeights: 100, minWeight: 1e-5 } 
      }, slice)

      // Process all data
      const result = transform.next(0, buffer.length())

      // With d=0, output should be identical to input
      for (let i = 0; i < buffer.length(); i++) {
        strictEqual(
          result.underlyingBuffer.getValue(i, result.underlyingBuffer.getColumn('close_fd')!.index),
          buffer.getValue(i, buffer.getColumn('close')!.index)
        )
      }
    })

    it('should handle d=1 (standard differentiation)', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, buffer.length())
      
      const transform = new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: 1, maxWeights: 100, minWeight: 1e-5 } 
      }, slice)

      // Process all data
      const result = transform.next(0, buffer.length())
      const closeFdCol = result.underlyingBuffer.getColumn('close_fd')!.index

      // With d=1, this is standard first differencing
      // w_0 = 1, w_1 = -1, all other weights = 0
      strictEqual(result.underlyingBuffer.getValue(0, closeFdCol), 105) // First value unchanged
      strictEqual(result.underlyingBuffer.getValue(1, closeFdCol), 5) // 110 - 105 = 5
      strictEqual(result.underlyingBuffer.getValue(2, closeFdCol), 5) // 115 - 110 = 5
      strictEqual(result.underlyingBuffer.getValue(3, closeFdCol), 5) // 120 - 115 = 5
    })

    it('should process batches incrementally', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, buffer.length())
      
      const transform = new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: 0.5, maxWeights: 100, minWeight: 1e-5 } 
      }, slice)

      // Process in batches
      const result1 = transform.next(0, 2)
      strictEqual(result1.from, 0)
      strictEqual(result1.to, 2)

      const result2 = transform.next(2, 4)
      strictEqual(result2.from, 2)
      strictEqual(result2.to, 4)

      // Verify continuity - the third value should use history from first two
      const closeFdCol = result2.underlyingBuffer.getColumn('close_fd')!.index
      // Third value: w_0 * 115 + w_1 * 110 + w_2 * 105
      // = 1 * 115 + (-0.5) * 110 + (-0.125) * 105
      // = 115 - 55 - 13.125 = 46.875
      strictEqual(result2.underlyingBuffer.getValue(2, closeFdCol), 46.875)
    })
  })

  describe('getters', () => {
    it('should always be ready', () => {
      const buffer = createTestBuffer()
      const slice = new DataSlice(buffer, 0, 0)
      const transform = new FractionalDiffNormalizer({ 
        tx: { in: 'close', out: 'close_fd', d: 0.5, maxWeights: 100, minWeight: 1e-5 } 
      }, slice)
      
      strictEqual(transform.isReady, true)
    })
  })
})