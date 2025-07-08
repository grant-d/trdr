import { strict as assert } from 'node:assert'
import { describe, it } from 'node:test'
import { MapTransform } from '../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../src/utils'

describe('MapTransform', () => {
  it('should copy single field to new name', () => {
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
      timestamp: Date.now(),
      open: 100,
      high: 110,
      low: 90,
      close: 105,
      volume: 1000
    })

    const slice = new DataSlice(buffer, 0, 0)
    const transform = new MapTransform({
      tx: {
        in: 'close',
        out: 'adj_close'
      }
    }, slice)

    transform.next(0, 1)
    const outputBuffer = transform.outputBuffer

    assert.equal(outputBuffer.length(), 1)
    const row = outputBuffer.getRow(0)
    assert.ok(row, 'Row should exist')
    assert.equal(row.close, 105)
    assert.equal((row as any).adj_close, 105)
  })

  it('should handle multiple mappings', () => {
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
      timestamp: Date.now(),
      open: 100,
      high: 110,
      low: 90,
      close: 105,
      volume: 1000
    })

    const slice = new DataSlice(buffer, 0, 0)
    const transform = new MapTransform({
      tx: [
        { in: 'close', out: 'price' },
        { in: 'volume', out: 'vol' }
      ]
    }, slice)

    transform.next(0, 1)
    const outputBuffer = transform.outputBuffer

    assert.equal(outputBuffer.length(), 1)
    const row = outputBuffer.getRow(0)
    assert.equal((row as any).price, 105)
    assert.equal((row as any).vol, 1000)
  })

  it('should handle basic functionality without errors', () => {
    const buffer = new DataBuffer({
      columns: {
        timestamp: { index: 0 },
        close: { index: 1 }
      }
    })

    buffer.push({
      timestamp: Date.now(),
      close: 100
    })

    const slice = new DataSlice(buffer, 0, 0)
    const transform = new MapTransform({
      tx: {
        in: 'close',
        out: 'mapped_close'
      }
    }, slice)

    transform.next(0, 1)
    const outputBuffer = transform.outputBuffer

    assert.equal(outputBuffer.length(), 1)
  })
})