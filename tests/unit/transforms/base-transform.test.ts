import { test } from 'node:test'
import { strict as assert } from 'node:assert'
import { BaseTransform } from '../../../src/transforms/base-transform'
import type { OhlcvDto } from '../../../src/models'
import type { Transform, BaseTransformParams } from '../../../src/interfaces/transform.interface'

// Test implementation of BaseTransform
interface TestParams extends BaseTransformParams {
  multiplier: number
}

class TestTransform extends BaseTransform<TestParams> {
  constructor(params: TestParams) {
    super('priceCalc', 'Test Transform', 'A test transform', params)
  }
  
  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let item = await data.next()
    while (!item.done) {
      const value = item.value
      
      yield {
        ...value,
        close: value.close * this.params.multiplier
      }
      item = await data.next()
    }
  }
  
  public getOutputFields(): string[] {
    return ['close'] // Modified existing field
  }
  
  public getRequiredFields(): string[] {
    return ['close']
  }
  
  public withParams(params: Partial<TestParams>): Transform<TestParams> {
    return new TestTransform({ ...this.params, ...params })
  }

  // Expose protected methods for testing
  public testArrayToAsyncIterator<T>(data: T[]): AsyncIterator<T> {
    return this.arrayToAsyncIterator(data)
  }

  public testCollectAsyncIterator<T>(iterator: AsyncIterator<T>): Promise<T[]> {
    return this.collectAsyncIterator(iterator)
  }
}

test('BaseTransform - Constructor and Properties', () => {
  const params: TestParams = { multiplier: 2 }
  const transform = new TestTransform(params)
  
  assert.equal(transform.type, 'priceCalc')
  assert.equal(transform.name, 'Test Transform')
  assert.equal(transform.description, 'A test transform')
  assert.deepEqual(transform.params, params)
})

test('BaseTransform - Apply Transform', async () => {
  const params: TestParams = { multiplier: 2 }
  const transform = new TestTransform(params)
  
  const testData: OhlcvDto[] = [
    {
      timestamp: 1640995200000,
      symbol: 'BTCUSD',
      exchange: 'test',
      open: 100,
      high: 110,
      low: 95,
      close: 105,
      volume: 1000
    }
  ]
  
  // Apply the transform
  const input = transform.testArrayToAsyncIterator(testData)
  const result = await transform.apply(input)
  
  // Collect results
  const transformed: OhlcvDto[] = []
  let item = await result.data.next()
  while (!item.done) {
    transformed.push(item.value)
    item = await result.data.next()
  }
  
  assert.equal(transformed.length, 1)
  assert.equal(transformed[0]!.close, 210) // 105 * 2
})

test('BaseTransform - Validation', () => {
  // Create a transform that overrides validation
  class ValidatingTransform extends TestTransform {
    public validate(): void {
      super.validate()
      if (this.params.multiplier <= 0) {
        throw new Error('Multiplier must be positive')
      }
    }
  }
  
  assert.throws(
    () => {
      const transform = new ValidatingTransform({ multiplier: -1 })
      transform.validate()
    },
    /Multiplier must be positive/
  )
})

test('BaseTransform - Column Validation', () => {
  // Test duplicate output columns
  assert.throws(
    () => {
      const transform = new TestTransform({ 
        multiplier: 2,
        in: ['close', 'open'],
        out: ['result', 'result'] // Duplicate output
      })
      transform.validate()
    },
    /Output columns must be unique/
  )
  
  // Test that input duplicates are allowed
  assert.doesNotThrow(() => {
    const transform = new TestTransform({ 
      multiplier: 2,
      in: ['close', 'close'], // Duplicate input - should be allowed
      out: ['result1', 'result2'] // Unique outputs
    })
    transform.validate()
  })
  
  // Test null outputs are handled correctly
  assert.doesNotThrow(() => {
    const transform = new TestTransform({ 
      multiplier: 2,
      in: ['close', 'open', 'high'],
      out: ['result', null, 'result2'] // null is allowed
    })
    transform.validate()
  })
  
  // Test invalid column names
  assert.throws(
    () => {
      const transform = new TestTransform({ 
        multiplier: 2,
        in: ['close-invalid'], // Invalid character
        out: ['result']
      })
      transform.validate()
    },
    /Invalid input column name/
  )
})


test('BaseTransform - WithParams Creates New Instance', () => {
  const transform = new TestTransform({ multiplier: 2 })
  const newTransform = transform.withParams({ multiplier: 3 }) as TestTransform
  
  assert.notEqual(transform, newTransform)
  assert.equal(transform.params.multiplier, 2)
  assert.equal(newTransform.params.multiplier, 3)
})

test('BaseTransform - Readiness', () => {
  const transform = new TestTransform({ multiplier: 2 })
  
  // Most transforms are ready immediately by default
  assert.equal(transform.isReady(), true)
})

test('BaseTransform - Helper Methods', async () => {
  const transform = new TestTransform({ multiplier: 1 })

  // Test arrayToAsyncIterator
  const data = [1, 2, 3]
  const iterator = transform.testArrayToAsyncIterator(data)
  const collected: number[] = []

  let item = await iterator.next()
  while (!item.done) {
    collected.push(item.value)
    item = await iterator.next()
  }
  
  assert.deepEqual(collected, data)
  
  // Test collectAsyncIterator
  const iterator2 = transform.testArrayToAsyncIterator([4, 5, 6])
  const result = await transform.testCollectAsyncIterator(iterator2)
  
  assert.deepEqual(result, [4, 5, 6])
})