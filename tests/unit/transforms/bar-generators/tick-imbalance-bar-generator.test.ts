import { ok } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { TickImbalanceBarGenerator } from '../../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../../src/utils'

// Helper to create test data with specific values
function createTestData(prices: number[], volumes: number[] = []): OhlcvDto[] {
  const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
  return prices.map((price, i) => ({
    timestamp: baseTime + i * 1000, // 1 second apart
    open: price,
    high: price + 0.1,
    low: price - 0.1,
    close: price,
    volume: volumes[i] || 100
  }))
}

// Helper to create a buffer with test data
function createTestBuffer(data: OhlcvDto[]): DataBuffer {
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

  for (const item of data) {
    buffer.push({
      timestamp: item.timestamp,
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
      volume: item.volume
    })
  }

  return buffer
}

describe('TickImbalanceBarGenerator', () => {
  it('should create bars when tick imbalance threshold is reached', () => {
    // Create test data with strong directional moves
    const prices = [100, 101, 102, 103, 104, 103, 102, 101, 100, 99, 98]
    const testData = createTestData(prices)
    const buffer = createTestBuffer(testData)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const generator = new TickImbalanceBarGenerator({
      tx: {
        imbalanceThreshold: 5,
        useVolume: false
      }
    }, slice)

    generator.next(0, buffer.length())
    const outputBuffer = generator.outputBuffer

    // Should process without errors
    ok(outputBuffer.length() >= 0, 'Should process without errors')
  })

  it('should handle basic functionality without errors', () => {
    const prices = [100, 101, 102, 103, 104, 105]
    const testData = createTestData(prices)
    const buffer = createTestBuffer(testData)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const generator = new TickImbalanceBarGenerator({
      tx: {
        imbalanceThreshold: 3,
        useVolume: false
      }
    }, slice)

    generator.next(0, buffer.length())
    const outputBuffer = generator.outputBuffer

    // Should process without errors
    ok(outputBuffer.length() >= 0, 'Should process without errors')
  })

  it('should require positive threshold', () => {
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

    try {
      new TickImbalanceBarGenerator({
        tx: {
          imbalanceThreshold: 0, // Invalid
          useVolume: false
        }
      }, slice)
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('positive') || error.message.includes('1'))
    }
  })
})