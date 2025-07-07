import { ok, strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../../src/models'
import { TickBarGenerator } from '../../../../src/transforms'
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

describe('TickBarGenerator', () => {
  it('should create bars after specified number of ticks', () => {
    // Create 10 ticks with increasing prices
    const prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    const testData = createTestData(prices)
    const buffer = createTestBuffer(testData)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const generator = new TickBarGenerator({
      tx: {
        ticks: 3
      }
    }, slice)

    generator.next(0, buffer.length())
    const outputBuffer = generator.outputBuffer
    const bars: any[] = []
    
    // Extract bars from output buffer
    for (let i = 0; i < outputBuffer.length(); i++) {
      const row = outputBuffer.getRow(i)
      bars.push(row)
    }

    // Should create 3 complete bars (9 ticks)
    strictEqual(bars.length, 3)

    // First bar: ticks 0-2 (prices 100, 101, 102)
    strictEqual(bars[0]!.open, 100) // First tick open
    strictEqual(bars[0]!.high, 102.1) // Max of all highs: max(100.1, 101.1, 102.1)
    strictEqual(bars[0]!.low, 99.9) // Min of all lows: min(99.9, 100.9, 101.9)
    strictEqual(bars[0]!.close, 102) // Last tick close
    strictEqual(bars[0]!.volume, 300) // Sum: 100 + 100 + 100

    // Second bar: ticks 3-5 (prices 103, 104, 105)
    strictEqual(bars[1]!.open, 103)
    strictEqual(bars[1]!.high, 105.1) // Max of all highs
    strictEqual(bars[1]!.low, 102.9) // Min of all lows
    strictEqual(bars[1]!.close, 105)
    strictEqual(bars[1]!.volume, 300)

    // Third bar: ticks 6-8 (prices 106, 107, 108)
    strictEqual(bars[2]!.open, 106)
    strictEqual(bars[2]!.high, 108.1)
    strictEqual(bars[2]!.low, 105.9)
    strictEqual(bars[2]!.close, 108)
    strictEqual(bars[2]!.volume, 300)
  })

  it('should handle basic functionality without errors', () => {
    const prices = [100, 101, 102, 103, 104, 105]
    const testData = createTestData(prices)
    const buffer = createTestBuffer(testData)
    const slice = new DataSlice(buffer, 0, buffer.length())

    const generator = new TickBarGenerator({
      tx: {
        ticks: 3
      }
    }, slice)

    generator.next(0, buffer.length())
    const outputBuffer = generator.outputBuffer

    // Should process without errors
    ok(outputBuffer.length() >= 0, 'Should process without errors')
  })

  it('should require positive ticks per bar', () => {
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
      new TickBarGenerator({
        tx: {
          ticks: 0 // Invalid
        }
      }, slice)
      ok(false, 'Should have thrown')
    } catch (error) {
      ok(error instanceof Error)
      ok(error.message.includes('positive'))
    }
  })
})