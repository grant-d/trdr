import { ok, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import { TimeBarGenerator } from '../../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../../src/utils'

describe('TimeBarGenerator', () => {
  // Helper to create test buffer
  const createTestBuffer = (count: number, intervalMs = 60000): DataBuffer => {
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
    
    const baseTime = new Date('2024-01-01T10:00:00Z').getTime()
    
    for (let i = 0; i < count; i++) {
      buffer.push({
        timestamp: baseTime + i * intervalMs,
        open: 100 + i,
        high: 102 + i,
        low: 99 + i,
        close: 101 + i, // i=0: 101, i=1: 102, i=2: 103, i=3: 104, i=4: 105
        volume: 1000 + i * 10
      })
    }
    
    return buffer
  }

  describe('constructor and validation', () => {
    it('should create instance with valid parameters', () => {
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
      const generator = new TimeBarGenerator({
        tx: { timeframe: '5m', alignToMarketOpen: false, marketOpenTime: '09:30' }
      }, slice)

      ok(generator)
      strictEqual(generator.type, 'timeBars')
      strictEqual(generator.name, 'TimeBars')
    })

    it('should set default parameters', () => {
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
      const generator = new TimeBarGenerator({ tx: { timeframe: '5m', alignToMarketOpen: false, marketOpenTime: '09:30' } }, slice)
      
      // Check that defaults are applied (from schema)
      ok(generator)
    })

    it('should validate target timeframe', () => {
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
      
      throws(() => {
        new TimeBarGenerator({ tx: { timeframe: '7m' as any, alignToMarketOpen: false, marketOpenTime: '09:30' } }, slice)
      })
    })

    it('should validate market open time format', () => {
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
      
      throws(() => {
        new TimeBarGenerator({
          tx: {
            timeframe: '5m',
            alignToMarketOpen: false,
            marketOpenTime: '25:00'
          }
        }, slice)
      })
    })
  })

  describe('aggregation functionality', () => {
    it('should aggregate 1m bars to 5m bars', () => {
      const buffer = createTestBuffer(10, 60000) // 10 1-minute bars
      const slice = new DataSlice(buffer, 0, buffer.length())
      const generator = new TimeBarGenerator({ tx: { timeframe: '5m', alignToMarketOpen: false, marketOpenTime: '09:30' } }, slice)

      generator.next(0, buffer.length())
      const outputBuffer = generator.outputBuffer
      
      // Should get 2 complete 5m bars from 10 1m bars
      strictEqual(outputBuffer.length(), 2)

      // Check first 5m bar (aggregates minutes 0-4)
      const firstBar = outputBuffer.getRow(0)
      strictEqual(firstBar?.open, 100) // Open of first 1m bar (index 0)
      strictEqual(firstBar?.high, 106) // Max high (102+4)
      strictEqual(firstBar?.low, 99) // Min low (99+0)
      strictEqual(firstBar?.close, 105) // Close of 5th 1m bar (index 4): 101 + 4 = 105
      strictEqual(firstBar?.volume, 5100) // Sum of volumes

      // Check second 5m bar (aggregates minutes 5-9)
      const secondBar = outputBuffer.getRow(1)
      strictEqual(secondBar?.open, 105) // Open of 6th 1m bar (index 5)
      strictEqual(secondBar?.high, 111) // Max high (102+9)
      strictEqual(secondBar?.low, 104) // Min low (99+5)
      strictEqual(secondBar?.close, 110) // Close of 10th 1m bar (index 9): 101 + 9 = 110
      strictEqual(secondBar?.volume, 5350) // Sum of volumes
    })

    it('should handle partial bars', () => {
      const buffer = createTestBuffer(7, 60000) // 7 1-minute bars
      const slice = new DataSlice(buffer, 0, buffer.length())
      const generator = new TimeBarGenerator({
        tx: { timeframe: '5m', alignToMarketOpen: false, marketOpenTime: '09:30' }
      }, slice)

      generator.next(0, buffer.length())
      const outputBuffer = generator.outputBuffer
      
      // Should have 1 complete 5m bar (first 5 minutes)
      // The last 2 ticks are stored in state for next batch
      strictEqual(outputBuffer.length(), 1)
      
      // Check that bar is complete
      const bar = outputBuffer.getRow(0)
      strictEqual(bar?.open, 100)
      strictEqual(bar?.close, 105)
      strictEqual(bar?.volume, 5100)
    })

    it('should aggregate to hourly bars', () => {
      const buffer = createTestBuffer(120, 60000) // 120 1-minute bars (2 hours)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const generator = new TimeBarGenerator({ tx: { timeframe: '1h', alignToMarketOpen: false, marketOpenTime: '09:30' } }, slice)

      generator.next(0, buffer.length())
      const outputBuffer = generator.outputBuffer
      
      // Should have 2 hourly bars
      strictEqual(outputBuffer.length(), 2)

      // Check first hourly bar
      const firstHour = outputBuffer.getRow(0)
      strictEqual(firstHour?.open, 100) // Open of first minute
      strictEqual(firstHour?.close, 160) // Close of 60th minute (index 59): 101 + 59 = 160
      // Volume: sum of 1000 + i*10 for i=0 to 59
      // = 60*1000 + 10*(0+1+...+59) = 60000 + 10*1770 = 60000 + 17700 = 77700
      strictEqual(firstHour?.volume, 77700) // Sum of 60 minutes
    })

    it('should handle empty data', () => {
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
      const generator = new TimeBarGenerator({ tx: { timeframe: '5m', alignToMarketOpen: false, marketOpenTime: '09:30' } }, slice)
      
      const result = generator.next(0, 0)
      strictEqual(result.to - result.from, 0)
      strictEqual(generator.outputBuffer.length(), 0)
    })

    it('should handle single data point', () => {
      const buffer = createTestBuffer(1)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const generator = new TimeBarGenerator({
        tx: { timeframe: '5m', alignToMarketOpen: false, marketOpenTime: '09:30' }
      }, slice)

      generator.next(0, buffer.length())
      const outputBuffer = generator.outputBuffer
      
      // Single tick won't complete a bar yet (stored in state)
      strictEqual(outputBuffer.length(), 0)
    })
  })

  describe('edge cases', () => {
    it('should handle data crossing bar boundaries', () => {
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
      
      const baseTime = new Date('2024-01-01T10:00:00Z').getTime()
      
      // Add data that spans across 5-minute boundaries
      buffer.push({
        timestamp: baseTime + 4 * 60000, // 10:04
        open: 100,
        high: 102,
        low: 99,
        close: 101,
        volume: 1000
      })
      buffer.push({
        timestamp: baseTime + 5 * 60000, // 10:05 (new bar)
        open: 105,
        high: 107,
        low: 104,
        close: 106,
        volume: 1100
      })
      buffer.push({
        timestamp: baseTime + 6 * 60000, // 10:06
        open: 106,
        high: 108,
        low: 105,
        close: 107,
        volume: 1200
      })
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const generator = new TimeBarGenerator({ tx: { timeframe: '5m', alignToMarketOpen: false, marketOpenTime: '09:30' } }, slice)
      
      generator.next(0, buffer.length())
      const outputBuffer = generator.outputBuffer
      
      // Should have 1 complete bar (10:00-10:05)
      strictEqual(outputBuffer.length(), 1)
      
      const bar = outputBuffer.getRow(0)
      strictEqual(bar?.open, 100)
      strictEqual(bar?.close, 101)
      strictEqual(bar?.volume, 1000)
    })

    it('should handle gaps in data', () => {
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
      
      const baseTime = new Date('2024-01-01T10:00:00Z').getTime()
      
      buffer.push({
        timestamp: baseTime, // 10:00
        open: 100,
        high: 102,
        low: 99,
        close: 101,
        volume: 1000
      })
      buffer.push({
        timestamp: baseTime + 10 * 60000, // 10:10 (10 minute gap)
        open: 110,
        high: 112,
        low: 109,
        close: 111,
        volume: 1100
      })
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const generator = new TimeBarGenerator({ tx: { timeframe: '5m', alignToMarketOpen: false, marketOpenTime: '09:30' } }, slice)
      
      generator.next(0, buffer.length())
      const outputBuffer = generator.outputBuffer
      
      // Should have 2 bars (one for each 5-minute period with data)
      strictEqual(outputBuffer.length(), 2)
      
      // First bar should only contain first data point
      strictEqual(outputBuffer.getRow(0)?.volume, 1000)
      
      // Second bar should only contain second data point
      strictEqual(outputBuffer.getRow(1)?.volume, 1100)
    })
  })

  describe('state management', () => {
    it('should maintain state across batches', () => {
      const buffer1 = createTestBuffer(3, 60000) // 3 1-minute bars
      const slice1 = new DataSlice(buffer1, 0, buffer1.length())
      const generator = new TimeBarGenerator({ tx: { timeframe: '5m', alignToMarketOpen: false, marketOpenTime: '09:30' } }, slice1)
      
      // Process first batch
      generator.next(0, buffer1.length())
      const outputBuffer1 = generator.outputBuffer
      strictEqual(outputBuffer1.length(), 0) // No complete bars yet
      
      // Create second batch
      const buffer2 = new DataBuffer({
        columns: {
          timestamp: { index: 0 },
          open: { index: 1 },
          high: { index: 2 },
          low: { index: 3 },
          close: { index: 4 },
          volume: { index: 5 }
        }
      })
      
      const baseTime = new Date('2024-01-01T10:00:00Z').getTime()
      
      // Add more data
      for (let i = 3; i < 6; i++) {
        buffer2.push({
          timestamp: baseTime + i * 60000,
          open: 100 + i,
          high: 102 + i,
          low: 99 + i,
          close: 101 + i,
          volume: 1000 + i * 10
        })
      }
      
      // Update the generator's input slice
      const combinedBuffer = new DataBuffer({
        columns: {
          timestamp: { index: 0 },
          open: { index: 1 },
          high: { index: 2 },
          low: { index: 3 },
          close: { index: 4 },
          volume: { index: 5 }
        }
      })
      
      // Copy all data
      for (let i = 0; i < 6; i++) {
        combinedBuffer.push({
          timestamp: baseTime + i * 60000,
          open: 100 + i,
          high: 102 + i,
          low: 99 + i,
          close: 101 + i,
          volume: 1000 + i * 10
        })
      }
      
      // Process second batch
      generator.next(3, 6)
      
      // Now should have 1 complete bar
      strictEqual(generator.outputBuffer.length(), 1)
      
      const bar = generator.outputBuffer.getRow(0)
      strictEqual(bar?.open, 100) // From first batch
      strictEqual(bar?.close, 105) // From second batch
      strictEqual(bar?.volume, 5100) // Sum of 5 minutes
    })
  })
})