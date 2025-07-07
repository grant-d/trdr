import { strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import { PriceCalculations } from '../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../src/utils'

// Helper function to create test buffer
function createTestBuffer(
  open: number,
  high: number,
  low: number,
  close: number,
  volume = 1000
): DataBuffer {
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
    open,
    high,
    low,
    close,
    volume
  })
  
  return buffer
}

describe('PriceCalculations', () => {
  describe('constructor and validation', () => {
    it('should create instance with required parameters', () => {
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

      const calc = new PriceCalculations({ in: {}, tx: { calc: 'hlc3', out: 'hlc3' } }, slice)
      strictEqual(calc.type, 'priceCalc')
      strictEqual(calc.name, 'Price Calculations')
    })

    it('should validate custom calculation requires formula', () => {
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
        new PriceCalculations({ in: {}, tx: { calc: 'custom', out: 'custom' } }, slice)
      }, /formula/)
    })

    it('should validate formula only with custom calculation', () => {
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
        new PriceCalculations({
          in: {},
          tx: {
            calc: 'hlc3',
            formula: '(high + low) / 2',
            out: 'hlc3'
          }
        }, slice)
      }, /formula/)
    })
  })

  describe('HLC3 calculation', () => {
    it('should calculate HLC3 correctly', () => {
      const buffer = createTestBuffer(100, 110, 90, 105)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({ in: {}, tx: { calc: 'hlc3', out: 'hlc3' } }, slice)

      calc.next(0, buffer.length())
      
      // Check the output buffer
      const outputBuffer = calc.outputBuffer
      strictEqual(outputBuffer.length(), 1)
      const row = outputBuffer.getRow(0)
      strictEqual(row?.hlc3, (110 + 90 + 105) / 3) // 101.667
      strictEqual(row?.hlc3?.toFixed(3), '101.667')
    })

    it('should use custom output field', () => {
      const buffer = createTestBuffer(100, 110, 90, 105)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({
        in: {},
        tx: {
          calc: 'hlc3',
          out: 'typical'
        }
      }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      const row = outputBuffer.getRow(0)
      strictEqual(row?.typical, (110 + 90 + 105) / 3)
      strictEqual(row?.hlc3, undefined)
    })
  })

  describe('OHLC4 calculation', () => {
    it('should calculate OHLC4 correctly', () => {
      const buffer = createTestBuffer(100, 110, 90, 105)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({ in: {}, tx: { calc: 'ohlc4', out: 'ohlc4' } }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      strictEqual(outputBuffer.length(), 1)
      const row = outputBuffer.getRow(0)
      strictEqual(row?.ohlc4, (100 + 110 + 90 + 105) / 4) // 101.25
    })
  })

  describe('Typical price calculation', () => {
    it('should calculate typical price (same as HLC3)', () => {
      const buffer = createTestBuffer(100, 110, 90, 105)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({ in: {}, tx: { calc: 'typical', out: 'typical_price' } }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      strictEqual(outputBuffer.length(), 1)
      const row = outputBuffer.getRow(0)
      strictEqual(row?.typical_price, (110 + 90 + 105) / 3)
    })
  })

  describe('Weighted close calculation', () => {
    it('should calculate weighted close correctly', () => {
      const buffer = createTestBuffer(100, 110, 90, 105)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({ in: {}, tx: { calc: 'weighted', out: 'weighted_close' } }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      strictEqual(outputBuffer.length(), 1)
      const row = outputBuffer.getRow(0)
      // (H + L + C + C) / 4
      strictEqual(row?.weighted_close, (110 + 90 + 105 + 105) / 4) // 102.5
    })
  })

  describe('Median price calculation', () => {
    it('should calculate median price correctly', () => {
      const buffer = createTestBuffer(100, 110, 90, 105)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({ in: {}, tx: { calc: 'median', out: 'median_price' } }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      strictEqual(outputBuffer.length(), 1)
      const row = outputBuffer.getRow(0)
      strictEqual(row?.median_price, (110 + 90) / 2) // 100
    })
  })

  describe('Custom formula calculation', () => {
    it('should evaluate simple custom formula', () => {
      const buffer = createTestBuffer(100, 110, 90, 105)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({
        in: {},
        tx: {
          calc: 'custom',
          formula: '(high + low) / 2',
          out: 'custom_price'
        }
      }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      strictEqual(outputBuffer.length(), 1)
      const row = outputBuffer.getRow(0)
      strictEqual(row?.custom_price, 100) // (110 + 90) / 2
    })

    it('should evaluate complex custom formula', () => {
      const buffer = createTestBuffer(100, 110, 90, 105, 1000)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({
        in: {},
        tx: {
          calc: 'custom',
          formula: '(open + 2 * close) / 3 + volume * 0.001',
          out: 'custom_price'
        }
      }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      strictEqual(outputBuffer.length(), 1)
      const row = outputBuffer.getRow(0)
      // (100 + 2 * 105) / 3 + 1000 * 0.001 = 103.333 + 1 = 104.333
      strictEqual(
        row?.custom_price?.toFixed(3),
        '104.333'
      )
    })

    it('should handle formula with all OHLCV fields', () => {
      const buffer = createTestBuffer(100, 110, 90, 105, 2000)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({
        in: {},
        tx: {
          calc: 'custom',
          formula: '(open + high + low + close) / 4 * (volume / 1000)',
          out: 'volume_weighted_avg'
        }
      }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      const row = outputBuffer.getRow(0)
      // ((100 + 110 + 90 + 105) / 4) * (2000 / 1000) = 101.25 * 2 = 202.5
      strictEqual(row?.volume_weighted_avg, 202.5)
    })

    it('should reject invalid formula characters', () => {
      const buffer = createTestBuffer(100, 110, 90, 105)
      const slice = new DataSlice(buffer, 0, buffer.length())
      
      throws(() => {
        new PriceCalculations({
          in: {},
          tx: {
            calc: 'custom',
            formula: 'high; alert("hack")',
            out: 'custom'
          }
        }, slice)
      })
    })

    it('should handle division by zero', () => {
      const buffer = createTestBuffer(100, 110, 90, 105, 0)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({
        in: {},
        tx: {
          calc: 'custom',
          formula: 'close / volume',
          out: 'result'
        }
      }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      const row = outputBuffer.getRow(0)
      // Division by zero should fallback to close price
      strictEqual(row?.result, 105)
    })
  })

  describe('original fields', () => {
    it('should preserve original OHLC fields', () => {
      const buffer = createTestBuffer(100, 110, 90, 105)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({ in: {}, tx: { calc: 'hlc3', out: 'hlc3' } }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      const row = outputBuffer.getRow(0)
      
      // Original fields remain in input buffer
      strictEqual(buffer.getRow(0)?.open, 100)
      strictEqual(buffer.getRow(0)?.high, 110)
      strictEqual(buffer.getRow(0)?.low, 90)
      strictEqual(buffer.getRow(0)?.close, 105)
      
      // New field is added
      strictEqual(row?.hlc3, (110 + 90 + 105) / 3)
    })
  })

  describe('multiple data points', () => {
    it('should process multiple bars correctly', () => {
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
      
      // Add multiple rows
      buffer.push({ timestamp: Date.now(), open: 100, high: 110, low: 90, close: 105, volume: 1000 })
      buffer.push({ timestamp: Date.now() + 1000, open: 105, high: 115, low: 95, close: 110, volume: 1000 })
      buffer.push({ timestamp: Date.now() + 2000, open: 110, high: 120, low: 100, close: 115, volume: 1000 })
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({ in: {}, tx: { calc: 'hlc3', out: 'hlc3' } }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      strictEqual(outputBuffer.length(), 3)
      strictEqual(outputBuffer.getRow(0)?.hlc3?.toFixed(3), '101.667') // (110+90+105)/3
      strictEqual(outputBuffer.getRow(1)?.hlc3?.toFixed(3), '106.667') // (115+95+110)/3
      strictEqual(outputBuffer.getRow(2)?.hlc3?.toFixed(3), '111.667') // (120+100+115)/3
    })
  })

  describe('multiple calculations', () => {
    it('should support multiple calculations in one transform', () => {
      const buffer = createTestBuffer(100, 110, 90, 105)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({
        in: {},
        tx: [
          { calc: 'hlc3', out: 'typical' },
          { calc: 'ohlc4', out: 'average' },
          { calc: 'hl2', out: 'midpoint' }
        ]
      }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      const row = outputBuffer.getRow(0)
      
      strictEqual(row?.typical, (110 + 90 + 105) / 3)
      strictEqual(row?.average, (100 + 110 + 90 + 105) / 4)
      strictEqual(row?.midpoint, (110 + 90) / 2)
    })
  })

  describe('custom calculations with input mapping', () => {
    it('should use custom input column names', () => {
      const buffer = new DataBuffer({
        columns: {
          timestamp: { index: 0 },
          o: { index: 1 },
          h: { index: 2 },
          l: { index: 3 },
          c: { index: 4 },
          v: { index: 5 }
        }
      })
      
      buffer.push({ timestamp: Date.now(), o: 100, h: 110, l: 90, c: 105, v: 1000 })
      
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({
        in: { open: 'o', high: 'h', low: 'l', close: 'c' },
        tx: { calc: 'ohlc4', out: 'avg' }
      }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      const row = outputBuffer.getRow(0)
      strictEqual(row?.avg, (100 + 110 + 90 + 105) / 4)
    })
  })

  describe('edge cases', () => {
    it('should handle zero prices', () => {
      const buffer = createTestBuffer(0, 0, 0, 0, 0)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({ in: {}, tx: { calc: 'hlc3', out: 'hlc3' } }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      strictEqual(outputBuffer.getRow(0)?.hlc3, 0)
    })

    it('should handle negative prices', () => {
      const buffer = createTestBuffer(-10, -5, -20, -15)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({ in: {}, tx: { calc: 'ohlc4', out: 'ohlc4' } }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      strictEqual(outputBuffer.getRow(0)?.ohlc4, (-10 + -5 + -20 + -15) / 4) // -12.5
    })

    it('should handle very large prices', () => {
      const buffer = createTestBuffer(1e10, 1.1e10, 0.9e10, 1.05e10)
      const slice = new DataSlice(buffer, 0, buffer.length())
      const calc = new PriceCalculations({ in: {}, tx: { calc: 'hlc3', out: 'hlc3' } }, slice)

      calc.next(0, buffer.length())
      
      const outputBuffer = calc.outputBuffer
      // (1.1e10 + 0.9e10 + 1.05e10) / 3
      const expected = (1.1e10 + 0.9e10 + 1.05e10) / 3
      strictEqual(outputBuffer.getRow(0)?.hlc3, expected)
    })

    it('should handle empty buffer', () => {
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
      const calc = new PriceCalculations({ in: {}, tx: { calc: 'hlc3', out: 'hlc3' } }, slice)

      const result = calc.next(0, 0)
      strictEqual(result.to - result.from, 0)
    })
  })
})
