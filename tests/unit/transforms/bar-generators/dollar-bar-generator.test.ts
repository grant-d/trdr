import { strictEqual } from 'node:assert'
import { describe, it } from 'node:test'
import { DollarBarGenerator } from '../../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../../src/utils'

describe('DollarBarGenerator', () => {
  it('should create bars when dollar value threshold is reached', () => {
    // Create test data buffer
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

    // Add test data with specific prices and volumes
    // Dollar value = price * volume
    const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
    const testData = [
      {
        timestamp: baseTime,
        open: 10,
        high: 10.5,
        low: 9.5,
        close: 10,
        volume: 100
      }, // $1000
      {
        timestamp: baseTime + 1000,
        open: 20,
        high: 20.5,
        low: 19.5,
        close: 20,
        volume: 50
      }, // $1000
      {
        timestamp: baseTime + 2000,
        open: 30,
        high: 30.5,
        low: 29.5,
        close: 30,
        volume: 40
      }, // $1200
      {
        timestamp: baseTime + 3000,
        open: 40,
        high: 40.5,
        low: 39.5,
        close: 40,
        volume: 30
      }, // $1200
      {
        timestamp: baseTime + 4000,
        open: 50,
        high: 50.5,
        low: 49.5,
        close: 50,
        volume: 20
      } // $1000
    ]

    for (const row of testData) {
      buffer.push(row)
    }

    // Create DataSlice for the generator
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Create generator with $2500 threshold
    const generator = new DollarBarGenerator(
      {
        tx: {
          threshold: 2500,
          priceField: 'close'
        }
      },
      slice
    )

    // Process batch using the transform's next() method
    generator.next(0, slice.length())

    // Should create 2 bars:
    // Bar 1: ticks 0-2 (accumulated = $3200 >= $2500)
    // Bar 2: ticks 3-4 (accumulated = $2200, incomplete)

    // Check output buffer
    const outputBuffer = generator.outputBuffer
    strictEqual(outputBuffer.length(), 2, 'Should create 2 bars')

    // Check first bar values
    const bar1 = outputBuffer.getRow(0)
    if (bar1) {
      strictEqual(
        bar1.timestamp,
        baseTime + 2000,
        'First bar should end at tick 2'
      )
      strictEqual(bar1.open, 10, 'First bar open should be first tick open')
      strictEqual(bar1.close, 30, 'First bar close should be last tick close')
      strictEqual(
        bar1.volume,
        190,
        'First bar volume should be sum of volumes'
      )
    }
  })

  it('should handle multiple dollar value thresholds', () => {
    // Create test data buffer
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

    // Add test data
    const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
    const testData = [
      {
        timestamp: baseTime,
        open: 100,
        high: 105,
        low: 95,
        close: 100,
        volume: 50
      }, // $5000
      {
        timestamp: baseTime + 1000,
        open: 100,
        high: 105,
        low: 95,
        close: 100,
        volume: 100
      }, // $10000
      {
        timestamp: baseTime + 2000,
        open: 100,
        high: 105,
        low: 95,
        close: 100,
        volume: 150
      } // $15000
    ]

    for (const row of testData) {
      buffer.push(row)
    }

    // Create DataSlice for the generator
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Create generator with multiple thresholds
    const generator = new DollarBarGenerator(
      {
        tx: [
          { threshold: 10000, priceField: 'close' },
          { threshold: 20000, priceField: 'close' }
        ]
      },
      slice
    )

    // Process batch
    generator.next(0, slice.length())

    // Check output buffer - should have 3 bars
    const outputBuffer = generator.outputBuffer
    strictEqual(
      outputBuffer.length(),
      3,
      'Should create 3 bars (2 for 10k threshold, 1 for 20k threshold)'
    )
  })

  it('should handle different price field options', () => {
    // Create test data buffer
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

    // Add test data with different OHLC values
    const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
    buffer.push({
      timestamp: baseTime,
      open: 90,
      high: 110,
      low: 90,
      close: 100,
      volume: 100
    })

    // Create DataSlice for the generator
    const slice = new DataSlice(buffer, 0, buffer.length())

    // Test with typical price (high + low + close) / 3 = (110 + 90 + 100) / 3 = 100
    const generator = new DollarBarGenerator(
      {
        tx: {
          threshold: 5000,
          priceField: 'typical'
        }
      },
      slice
    )

    generator.next(0, slice.length())

    // Typical price = 100, volume = 100, so dollar value = 10000
    // Should create one bar since 10000 >= 5000
    const outputBuffer = generator.outputBuffer
    strictEqual(
      outputBuffer.length(),
      1,
      'Should create 1 bar with typical price'
    )
  })
})
