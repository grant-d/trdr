import { test } from 'node:test'
import { strict as assert } from 'node:assert'
import { MissingValueHandler } from '../../../src/transforms/missing-value-handler'
import type { OhlcvDto } from '../../../src/models/ohlcv.dto'
import type { MissingValueParams } from '../../../src/transforms/transform-params'

// Helper to create test data
function createTestData(): OhlcvDto[] {
  return [
    {
      timestamp: 1640995200000,
      symbol: 'BTCUSD',
      exchange: 'test',
      open: 100,
      high: 110,
      low: 95,
      close: 105,
      volume: 1000
    },
    {
      timestamp: 1640995260000,
      symbol: 'BTCUSD',
      exchange: 'test',
      open: NaN, // Intentionally invalid
      high: 115,
      low: null as any, // Intentionally invalid
      close: 110,
      volume: undefined as any // Intentionally invalid
    },
    {
      timestamp: 1640995320000,
      symbol: 'BTCUSD',
      exchange: 'test',
      open: 108,
      high: 118,
      low: 105,
      close: 115,
      volume: 1200
    }
  ]
}

// Helper to convert array to async iterator
async function* arrayToAsyncIterator<T>(array: T[]): AsyncGenerator<T> {
  for (const item of array) {
    yield item
  }
}

test('MissingValueHandler - Forward Fill Strategy', async () => {
  const params: MissingValueParams = {
    strategy: 'forward'
  }
  
  const handler = new MissingValueHandler(params)
  const testData = createTestData()
  const input = arrayToAsyncIterator(testData)
  
  const result = await handler.apply(input)
  const processed: OhlcvDto[] = []
  
  let item = await result.data.next()
  while (!item.done) {
    processed.push(item.value)
    item = await result.data.next()
  }
  
  assert.equal(processed.length, 3)
  
  // First item should be unchanged
  assert.equal(processed[0]!.open, 100)
  assert.equal(processed[0]!.low, 95)
  assert.equal(processed[0]!.volume, 1000)
  
  // Second item should have values filled from first
  assert.equal(processed[1]!.open, 100) // Filled from previous
  assert.equal(processed[1]!.high, 115) // Original value kept
  assert.equal(processed[1]!.low, 95) // Filled from previous
  assert.equal(processed[1]!.close, 110) // Original value kept
  assert.equal(processed[1]!.volume, 1000) // Filled from previous
  
  // Third item should be unchanged
  assert.equal(processed[2]!.open, 108)
  assert.equal(processed[2]!.volume, 1200)
})

test('MissingValueHandler - Backward Fill Strategy', async () => {
  const params: MissingValueParams = {
    strategy: 'backward'
  }
  
  const handler = new MissingValueHandler(params)
  const testData = createTestData()
  const input = arrayToAsyncIterator(testData)
  
  const result = await handler.apply(input)
  const processed: OhlcvDto[] = []
  
  let item = await result.data.next()
  while (!item.done) {
    processed.push(item.value)
    item = await result.data.next()
  }
  
  assert.equal(processed.length, 3)
  
  // Second item should have values filled from third (next valid)
  assert.equal(processed[1]!.open, 108) // Filled from next
  assert.equal(processed[1]!.low, 105) // Filled from next
  assert.equal(processed[1]!.volume, 1200) // Filled from next
})

test('MissingValueHandler - Custom Value Strategy', async () => {
  const params: MissingValueParams = {
    strategy: 'value',
    fillValue: 0
  }
  
  const handler = new MissingValueHandler(params)
  const testData = createTestData()
  const input = arrayToAsyncIterator(testData)
  
  const result = await handler.apply(input)
  const processed: OhlcvDto[] = []
  
  let item = await result.data.next()
  while (!item.done) {
    processed.push(item.value)
    item = await result.data.next()
  }
  
  // Missing values should be filled with 0
  assert.equal(processed[1]!.open, 0)
  assert.equal(processed[1]!.low, 0)
  assert.equal(processed[1]!.volume, 0)
})

test('MissingValueHandler - Specific Fields Only', async () => {
  const params: MissingValueParams = {
    strategy: 'forward',
    fields: ['open', 'close'] // Only process these fields
  }
  
  const handler = new MissingValueHandler(params)
  const testData = createTestData()
  testData[1]!.close = NaN // Add missing close value
  
  const input = arrayToAsyncIterator(testData)
  const result = await handler.apply(input)
  const processed: OhlcvDto[] = []
  
  let item = await result.data.next()
  while (!item.done) {
    processed.push(item.value)
    item = await result.data.next()
  }
  
  // Only open and close should be filled
  assert.equal(processed[1]!.open, 100) // Filled
  assert.equal(processed[1]!.close, 105) // Filled
  assert.equal(processed[1]!.low, null) // Not processed
  assert.equal(processed[1]!.volume, undefined) // Not processed
})

test('MissingValueHandler - Multi-Symbol Handling', async () => {
  const params: MissingValueParams = {
    strategy: 'forward'
  }
  
  const handler = new MissingValueHandler(params)
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
    },
    {
      timestamp: 1640995200000,
      symbol: 'ETHUSD',
      exchange: 'test',
      open: 3000,
      high: 3100,
      low: 2950,
      close: 3050,
      volume: 500
    },
    {
      timestamp: 1640995260000,
      symbol: 'BTCUSD',
      exchange: 'test',
      open: NaN,
      high: 115,
      low: NaN,
      close: 110,
      volume: NaN
    },
    {
      timestamp: 1640995260000,
      symbol: 'ETHUSD',
      exchange: 'test',
      open: NaN,
      high: 3150,
      low: NaN,
      close: 3100,
      volume: NaN
    }
  ]
  
  const input = arrayToAsyncIterator(testData)
  const result = await handler.apply(input)
  const processed: OhlcvDto[] = []
  
  let item = await result.data.next()
  while (!item.done) {
    processed.push(item.value)
    item = await result.data.next()
  }
  
  // Each symbol should use its own last valid values
  assert.equal(processed[2]!.open, 100) // BTC filled from BTC
  assert.equal(processed[2]!.volume, 1000)
  assert.equal(processed[3]!.open, 3000) // ETH filled from ETH
  assert.equal(processed[3]!.volume, 500)
})

test('MissingValueHandler - Validation', () => {
  // Should throw when using 'value' strategy without fillValue
  assert.throws(
    () => {
      const handler = new MissingValueHandler({ name: 'mv1',
        strategy: 'value'
      })
      handler.validate()
    },
    /fillValue must be provided/
  )
  
  // Should throw for invalid maxFillGap
  assert.throws(
    () => {
      const handler = new MissingValueHandler({ name: 'mv1',
        strategy: 'forward',
        maxFillGap: 0
      })
      handler.validate()
    },
    /maxFillGap must be at least 1/
  )
})

test('MissingValueHandler - Required and Output Fields', () => {
  const handler1 = new MissingValueHandler({ name: 'mv1',
    strategy: 'forward'
  })
  
  assert.deepEqual(handler1.getOutputFields(), [])
  assert.deepEqual(handler1.getRequiredFields(), [])
  
  const handler2 = new MissingValueHandler({ name: 'mv1',
    strategy: 'forward',
    fields: ['open', 'close']
  })
  
  assert.deepEqual(handler2.getRequiredFields(), ['open', 'close'])
})

test('MissingValueHandler - WithParams', () => {
  const handler = new MissingValueHandler({ name: 'mv1',
    strategy: 'forward'
  })
  
  const newHandler = handler.withParams({
    strategy: 'backward'
  }) as MissingValueHandler
  
  assert.notEqual(handler, newHandler)
  assert.equal(handler.params.strategy, 'forward')
  assert.equal(newHandler.params.strategy, 'backward')
})