import { strict as assert } from 'node:assert'
import { test } from 'node:test'
import type { BaseTransformParams } from '../../../src/interfaces'
import { DataSlice } from '../../../src/utils'
import { BaseTransform } from '../../../src/transforms'
import { DataBuffer } from '../../../src/utils'

// Test implementation of BaseTransform
interface TestParams extends BaseTransformParams {
  multiplier: number;
}

class TestTransform extends BaseTransform<TestParams> {
  constructor(params: TestParams, inputSlice: DataSlice) {
    super('priceCalc', 'Test Transform', 'A test transform', params, inputSlice)
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    // Get the close column index
    const closeCol = this.inputSlice.getColumn('close')
    if (!closeCol) {
      throw new Error('Close column not found')
    }

    // Process each row in the batch
    for (let i = from; i <= to; i++) {
      const closeValue = this.inputSlice.underlyingBuffer.getValue(i, closeCol.index)
      if (typeof closeValue === 'number') {
        // Multiply the close value
        this.inputSlice.underlyingBuffer.updateValue(i, closeCol.index, closeValue * this.params.multiplier)
      }
    }

    return { from, to }
  }
}

test('BaseTransform - Constructor and Properties', () => {
  const buffer = new DataBuffer({
    columns: {
      timestamp: { index: 0 },
      close: { index: 1 }
    }
  })
  
  const slice = new DataSlice(buffer, 0, 0)
  const params: TestParams = { multiplier: 2 }
  const transform = new TestTransform(params, slice)

  assert.equal(transform.type, 'priceCalc')
  assert.equal(transform.name, 'Test Transform')
  assert.equal(transform.description, 'A test transform')
  assert.deepEqual(transform.params, params)
})

test('BaseTransform - Process Transform', () => {
  const buffer = new DataBuffer({
    columns: {
      timestamp: { index: 0 },
      close: { index: 1 }
    }
  })

  // Add test data
  buffer.push({
    timestamp: 1640995200000,
    close: 105
  })

  const slice = new DataSlice(buffer, 0, 0)
  const params: TestParams = { multiplier: 2 }
  const transform = new TestTransform(params, slice)

  // Process the data
  transform.next(0, 0)

  // Check the result
  const transformedValue = buffer.getValue(0, 1) // close column
  assert.equal(transformedValue, 210) // 105 * 2
})

test('BaseTransform - Batch Processing', () => {
  const buffer = new DataBuffer({
    columns: {
      timestamp: { index: 0 },
      close: { index: 1 }
    }
  })

  // Add multiple rows
  buffer.push({ timestamp: 1640995200000, close: 100 })
  buffer.push({ timestamp: 1640995260000, close: 110 })
  buffer.push({ timestamp: 1640995320000, close: 120 })

  const slice = new DataSlice(buffer, 0, 2)
  const params: TestParams = { multiplier: 3 }
  const transform = new TestTransform(params, slice)

  // Process all data
  transform.next(0, 2)

  // Check all values were transformed
  assert.equal(buffer.getValue(0, 1), 300) // 100 * 3
  assert.equal(buffer.getValue(1, 1), 330) // 110 * 3
  assert.equal(buffer.getValue(2, 1), 360) // 120 * 3
})