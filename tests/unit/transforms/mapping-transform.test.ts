import { test } from 'node:test'
import { strict as assert } from 'node:assert'
import { MappingTransform } from '../../../src/transforms/mapping-transform'
import type { OhlcvDto } from '../../../src/models'

// Helper to convert array to async iterator
async function* arrayToAsyncIterator<T>(array: T[]): AsyncIterator<T> {
  for (const item of array) {
    yield item
  }
}

// Helper to collect async iterator results
async function collectResults<T>(iterator: AsyncIterator<T>): Promise<T[]> {
  const results: T[] = []
  let item = await iterator.next()
  while (!item.done) {
    results.push(item.value)
    item = await iterator.next()
  }
  return results
}

test('MappingTransform', async (t) => {
  await t.test('should copy single field to new name', async () => {
    const transform = new MappingTransform({
      in: ['close'],
      out: ['adj_close']
    })

    const testData: OhlcvDto[] = [
      {
        timestamp: Date.now(),
        exchange: 'test',
        symbol: 'BTC/USD',
        open: 100,
        high: 110,
        low: 90,
        close: 105,
        volume: 1000
      }
    ]

    const result = await transform.apply(arrayToAsyncIterator(testData))
    const outputs = await collectResults(result.data)

    assert.equal(outputs.length, 1)
    assert.equal(outputs[0]!.close, 105)
    assert.equal((outputs[0] as any)['adj_close'], 105)
  })

  await t.test('should copy multiple fields', async () => {
    const transform = new MappingTransform({
      in: ['close', 'volume'],
      out: ['adj_close', 'adj_volume']
    })

    const testData: OhlcvDto[] = [
      {
        timestamp: Date.now(),
        exchange: 'test',
        symbol: 'BTC/USD',
        open: 100,
        high: 110,
        low: 90,
        close: 105,
        volume: 1000
      }
    ]

    const result = await transform.apply(arrayToAsyncIterator(testData))
    const outputs = await collectResults(result.data)

    assert.equal(outputs.length, 1)
    assert.equal((outputs[0] as any)['adj_close'], 105)
    assert.equal((outputs[0] as any)['adj_volume'], 1000)
  })

  await t.test('should validate parameters', async () => {
    assert.throws(() => {
      new MappingTransform({
        in: [],
        out: []
      }).validate()
    }, /inputColumns cannot be empty/)

    assert.throws(() => {
      new MappingTransform({
        in: ['close'],
        out: []
      }).validate()
    }, /inputColumns and outputColumns must have the same length/)

    assert.throws(() => {
      new MappingTransform({
        in: ['close'],
        out: ['adj_close', 'adj_volume']
      }).validate()
    }, /inputColumns and outputColumns must have the same length/)
  })

  await t.test('should be always ready', async () => {
    const transform = new MappingTransform({
      in: ['close'],
      out: ['adj_close']
    })

    assert.equal(transform.isReady(), true)
  })

  await t.test('should have correct type', async () => {
    const transform = new MappingTransform({
      in: ['close'],
      out: ['adj_close']
    })

    assert.equal(transform.type, 'map')
  })
})