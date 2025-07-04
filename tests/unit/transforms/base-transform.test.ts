import { test } from 'node:test'
import { strict as assert } from 'node:assert'
import { BaseTransform } from '../../../src/transforms/base-transform'
import type { OhlcvDto } from '../../../src/models'
import type { Transform, TransformType, BaseTransformParams } from '../../../src/interfaces/transform.interface'

// Test implementation of BaseTransform
interface TestParams extends BaseTransformParams {
  multiplier: number
}

class TestTransform extends BaseTransform<TestParams> {
  constructor(params: TestParams) {
    super('priceCalc', 'Test Transform', 'A test transform', params, true)
  }
  
  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let item = await data.next()
    while (!item.done) {
      const value = item.value
      // Store coefficients on first item
      if (!this.getCoefficients()) {
        this.setCoefficients(value.symbol, { multiplier: this.params.multiplier })
      }
      
      yield {
        ...value,
        close: value.close * this.params.multiplier
      }
      item = await data.next()
    }
  }
  
  // Expose getCoefficients for testing
  public getCoefficients() {
    return super.getCoefficients()
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
  
  public async *reverse(
    data: AsyncIterator<OhlcvDto>,
    coefficients: any
  ): AsyncGenerator<OhlcvDto> {
    const multiplier = coefficients.values.multiplier
    
    let item = await data.next()
    while (!item.done) {
      yield {
        ...item.value,
        close: item.value.close / multiplier
      }
      item = await data.next()
    }
  }
}

test('BaseTransform - Constructor and Properties', () => {
  const params: TestParams = { multiplier: 2 }
  const transform = new TestTransform(params)
  
  assert.equal(transform.type, 'priceCalc')
  assert.equal(transform.name, 'Test Transform')
  assert.equal(transform.description, 'A test transform')
  assert.equal(transform.isReversible, true)
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
  const input = transform['arrayToAsyncIterator'](testData)
  const result = await transform.apply(input)
  
  // Collect results - this will trigger coefficient setting
  const transformed: OhlcvDto[] = []
  let item = await result.data.next()
  while (!item.done) {
    transformed.push(item.value)
    item = await result.data.next()
  }
  
  assert.equal(transformed.length, 1)
  assert.equal(transformed[0]!.close, 210) // 105 * 2
  
  // Check coefficients after transform has been consumed
  const coefficients = transform.getCoefficients()
  assert.ok(coefficients, 'Coefficients should be set after transform')
  assert.equal(coefficients!.type, 'priceCalc')
  assert.equal(coefficients!.symbol, 'BTCUSD')
  assert.equal(coefficients!.values.multiplier, 2)
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

test('BaseTransform - Reversible Transform', async () => {
  const params: TestParams = { multiplier: 3 }
  const transform = new TestTransform(params)
  
  const testData: OhlcvDto[] = [
    {
      timestamp: 1640995200000,
      symbol: 'BTCUSD',
      exchange: 'test',
      open: 100,
      high: 110,
      low: 95,
      close: 300, // Already multiplied
      volume: 1000
    }
  ]
  
  const input = transform['arrayToAsyncIterator'](testData)
  const coefficients = {
    type: 'priceCalc' as TransformType,
    timestamp: Date.now(),
    symbol: 'BTCUSD',
    values: { multiplier: 3 }
  }
  
  const reversed = transform.reverse!(input, coefficients)
  
  const result: OhlcvDto[] = []
  for await (const item of reversed) {
    result.push(item)
  }
  
  assert.equal(result.length, 1)
  assert.equal(result[0]!.close, 100) // 300 / 3
})

test('BaseTransform - Non-Reversible Transform', () => {
  class NonReversibleTransform extends BaseTransform<TestParams> {
    constructor(params: TestParams) {
      super('priceCalc', 'Non-Reversible', 'Cannot be reversed', params, false)
    }
    
    protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
      let item = await data.next()
      while (!item.done) {
        yield item.value
        item = await data.next()
      }
    }
    
    public getOutputFields(): string[] {
      return []
    }
    
    public getRequiredFields(): string[] {
      return []
    }
    
    public withParams(params: Partial<TestParams>): Transform<TestParams> {
      return new NonReversibleTransform({ ...this.params, ...params })
    }
  }
  
  const transform = new NonReversibleTransform({ multiplier: 1 })
  const coefficients = {
    type: 'priceCalc' as TransformType,
    timestamp: Date.now(),
    symbol: 'TEST',
    values: {}
  }
  
  assert.throws(
    () => transform.reverse!(transform['arrayToAsyncIterator']([]), coefficients),
    /Transform Non-Reversible is not reversible/
  )
})

test('BaseTransform - WithParams Creates New Instance', () => {
  const transform = new TestTransform({ multiplier: 2 })
  const newTransform = transform.withParams({ multiplier: 3 }) as TestTransform
  
  assert.notEqual(transform, newTransform)
  assert.equal(transform.params.multiplier, 2)
  assert.equal(newTransform.params.multiplier, 3)
})

test('BaseTransform - Helper Methods', async () => {
  const transform = new TestTransform({ multiplier: 1 })
  
  // Test arrayToAsyncIterator
  const data = [1, 2, 3]
  const iterator = transform['arrayToAsyncIterator'](data)
  const collected: number[] = []
  
  for await (const item of iterator) {
    collected.push(item)
  }
  
  assert.deepEqual(collected, data)
  
  // Test collectAsyncIterator
  const iterator2 = transform['arrayToAsyncIterator']([4, 5, 6])
  const result = await transform['collectAsyncIterator'](iterator2)
  
  assert.deepEqual(result, [4, 5, 6])
})