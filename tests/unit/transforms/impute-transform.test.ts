import { strict as assert } from 'node:assert'
import { test } from 'node:test'
import type { ImputeParams } from '../../../src/transforms'
import { ImputeTransform } from '../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../src/utils'

// Helper to create test data buffer
function createTestDataBuffer(): DataBuffer {
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

  // Add test data with some invalid values
  buffer.push({
    timestamp: 1640995200000,
    open: 100,
    high: 110,
    low: 95,
    close: 105,
    volume: 1000
  })
  
  buffer.push({
    timestamp: 1640995260000,
    open: NaN, // Intentionally invalid
    high: 115,
    low: null as any, // Intentionally invalid
    close: 110,
    volume: undefined as any // Intentionally invalid
  })
  
  buffer.push({
    timestamp: 1640995320000,
    open: 108,
    high: 118,
    low: 105,
    close: 115,
    volume: 1200
  })

  return buffer
}

test('ImputeTransform - Forward Fill Strategy', () => {
  const params: ImputeParams = {
    tx: {
      in: 'open',
      out: 'open',
      strategy: 'forward',
      fillValue: 0
    }
  }

  const buffer = createTestDataBuffer()
  const slice = new DataSlice(buffer, 0, buffer.length())
  const handler = new ImputeTransform(params, slice)

  const result = handler.next(0, buffer.length())
  const outputBuffer = result.underlyingBuffer

  assert.equal(outputBuffer.length(), 3)

  // First item should be unchanged
  assert.equal(outputBuffer.getValue(0, 1), 100) // open
  assert.equal(outputBuffer.getValue(0, 3), 95) // low
  assert.equal(outputBuffer.getValue(0, 5), 1000) // volume

  // Second item should have open filled from first
  assert.equal(outputBuffer.getValue(1, 1), 100) // open filled from previous
  assert.equal(outputBuffer.getValue(1, 2), 115) // high original value kept
  assert.equal(outputBuffer.getValue(1, 4), 110) // close original value kept

  // Third item should be unchanged
  assert.equal(outputBuffer.getValue(2, 1), 108) // open
  assert.equal(outputBuffer.getValue(2, 5), 1200) // volume
})

test('ImputeTransform - Backward Fill Strategy', () => {
  const params: ImputeParams = {
    tx: {
      in: 'open',
      out: 'open',
      strategy: 'interpolate',
      fillValue: 0
    }
  }

  const buffer = createTestDataBuffer()
  const slice = new DataSlice(buffer, 0, buffer.length())
  const handler = new ImputeTransform(params, slice)

  const result = handler.next(0, buffer.length())
  const outputBuffer = result.underlyingBuffer

  assert.equal(outputBuffer.length(), 3)

  // Second item should have open filled from third (next valid)
  assert.equal(outputBuffer.getValue(1, 1), 108) // open filled from next
})

test('ImputeTransform - Custom Value Strategy', () => {
  const params: ImputeParams = {
    tx: {
      in: 'open',
      out: 'open',
      strategy: 'value',
      fillValue: 0
    }
  }

  const buffer = createTestDataBuffer()
  const slice = new DataSlice(buffer, 0, buffer.length())
  const handler = new ImputeTransform(params, slice)

  const result = handler.next(0, buffer.length())
  const outputBuffer = result.underlyingBuffer

  // Missing open value should be filled with 0
  assert.equal(outputBuffer.getValue(1, 1), 0) // open
})

test('ImputeTransform - Multiple Fields', () => {
  const params: ImputeParams = {
    tx: [
      { in: 'open', out: 'open', strategy: 'forward', fillValue: 0 },
      { in: 'close', out: 'close', strategy: 'forward', fillValue: 0 }
    ]
  }

  const buffer = createTestDataBuffer()
  // Add missing close value to second row
  buffer.updateValue(1, 4, NaN) // close column index is 4
  
  const slice = new DataSlice(buffer, 0, buffer.length())
  const handler = new ImputeTransform(params, slice)

  const result = handler.next(0, buffer.length())
  const outputBuffer = result.underlyingBuffer

  // Both open and close should be filled
  assert.equal(outputBuffer.getValue(1, 1), 100) // open filled
  assert.equal(outputBuffer.getValue(1, 4), 105) // close filled
})

test('ImputeTransform - Basic Implementation', () => {
  const params: ImputeParams = {
    tx: {
      in: 'open',
      out: 'open',
      strategy: 'forward',
      fillValue: 0
    }
  }

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

  // Add test data with missing values
  buffer.push({
    timestamp: 1640995200000,
    open: 100,
    high: 110,
    low: 95,
    close: 105,
    volume: 1000
  })

  buffer.push({
    timestamp: 1640995200000,
    open: NaN,
    high: 115,
    low: 100,
    close: 110,
    volume: 1200
  })

  buffer.push({
    timestamp: 1640995260000,
    open: 108,
    high: 118,
    low: 105,
    close: 115,
    volume: 1400
  })

  const slice = new DataSlice(buffer, 0, buffer.length())
  const handler = new ImputeTransform(params, slice)

  const result = handler.next(0, buffer.length())
  const outputBuffer = result.underlyingBuffer

  // Second item's open should be filled from first
  assert.equal(outputBuffer.getValue(1, 1), 100) // open filled from previous
})

test('ImputeTransform - Validation', () => {
  const buffer = new DataBuffer({
    columns: {
      timestamp: { index: 0 },
      open: { index: 1 },
      close: { index: 2 }
    }
  })
  const slice = new DataSlice(buffer, 0, 0)

  // Should throw when using 'value' strategy without fillValue
  assert.throws(() => {
    new ImputeTransform({
      tx: {
        in: 'open',
        out: 'open',
        strategy: 'value',
        fillValue: undefined as any // Force invalid value
      }
    }, slice)
  }, /fillValue must be provided/)

  // Should throw for invalid maxFillGap
  assert.throws(() => {
    new ImputeTransform({
      tx: {
        in: 'open',
        out: 'open',
        strategy: 'forward',
        fillValue: 0
      }
    }, slice)
  }, /maxFillGap must be at least 1/)
})

test('ImputeTransform - Type and Name', () => {
  const buffer = new DataBuffer({
    columns: {
      timestamp: { index: 0 },
      open: { index: 1 },
      close: { index: 2 }
    }
  })
  const slice = new DataSlice(buffer, 0, 0)
  
  const handler = new ImputeTransform({
    tx: {
      in: 'open',
      out: 'open',
      strategy: 'forward',
      fillValue: 0
    }
  }, slice)

  assert.equal(handler.type, 'impute')
  assert.equal(handler.name, 'Impute Transform')
})
