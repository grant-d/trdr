import { describe, it } from 'node:test'
import { deepStrictEqual, strictEqual, throws, ok } from 'node:assert'
import type { OhlcvDto } from '../../../src/models'
import { PriceCalculations } from '../../../src/transforms/price-calculations'

// Helper function to create test data
function createTestData(
  open: number,
  high: number,
  low: number,
  close: number,
  volume = 1000
): OhlcvDto[] {
  return [{
    timestamp: Date.now(),
    symbol: 'TEST',
    exchange: 'test',
    open,
    high,
    low,
    close,
    volume,
  }]
}

// Helper to convert array to async iterator
async function* arrayToAsyncIterator<T>(array: T[]): AsyncGenerator<T> {
  for (const item of array) {
    yield item
  }
}

// Helper to collect results from async iterator
async function collectResults(iterator: AsyncIterator<OhlcvDto>): Promise<OhlcvDto[]> {
  const results: OhlcvDto[] = []
  let item = await iterator.next()
  while (!item.done) {
    results.push(item.value)
    item = await iterator.next()
  }
  return results
}

describe('PriceCalculations', () => {
  describe('constructor and validation', () => {
    it('should create instance with required parameters', () => {
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'hlc3' })
      strictEqual(calc.type, 'priceCalc')
      strictEqual(calc.name, 'Price Calculations')
      ok(!calc.isReversible)
    })

    it('should validate custom calculation requires formula', () => {
      throws(
        () => {
          const calc = new PriceCalculations({ name: 'pc1', calculation: 'custom' })
          calc.validate()
        },
        /Custom formula is required/
      )
    })

    it('should validate formula only with custom calculation', () => {
      throws(
        () => {
          const calc = new PriceCalculations({ name: 'pc1', 
            calculation: 'hlc3',
            customFormula: '(high + low) / 2' 
          })
          calc.validate()
        },
        /Custom formula should only be provided when calculation type is "custom"/
      )
    })
  })

  describe('HLC3 calculation', () => {
    it('should calculate HLC3 correctly', async () => {
      const testData = createTestData(100, 110, 90, 105)
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'hlc3' })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      strictEqual(transformed[0]!.hlc3, (110 + 90 + 105) / 3) // 101.667
      strictEqual(transformed[0]!.hlc3.toFixed(3), '101.667')
    })

    it('should use custom output field', async () => {
      const testData = createTestData(100, 110, 90, 105)
      const calc = new PriceCalculations({ name: 'pc1', 
        calculation: 'hlc3',
        outputField: 'typical' 
      })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed[0]!.typical, (110 + 90 + 105) / 3)
      strictEqual(transformed[0]!.hlc3, undefined)
    })
  })

  describe('OHLC4 calculation', () => {
    it('should calculate OHLC4 correctly', async () => {
      const testData = createTestData(100, 110, 90, 105)
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'ohlc4' })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      strictEqual(transformed[0]!.ohlc4, (100 + 110 + 90 + 105) / 4) // 101.25
    })
  })

  describe('Typical price calculation', () => {
    it('should calculate typical price (same as HLC3)', async () => {
      const testData = createTestData(100, 110, 90, 105)
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'typical' })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      strictEqual(transformed[0]!.typical_price, (110 + 90 + 105) / 3)
    })
  })

  describe('Weighted close calculation', () => {
    it('should calculate weighted close correctly', async () => {
      const testData = createTestData(100, 110, 90, 105)
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'weighted' })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      // (H + L + C + C) / 4
      strictEqual(transformed[0]!.weighted_close, (110 + 90 + 105 + 105) / 4) // 102.5
    })
  })

  describe('Median price calculation', () => {
    it('should calculate median price correctly', async () => {
      const testData = createTestData(100, 110, 90, 105)
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'median' })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      strictEqual(transformed[0]!.median_price, (110 + 90) / 2) // 100
    })
  })

  describe('Custom formula calculation', () => {
    it('should evaluate simple custom formula', async () => {
      const testData = createTestData(100, 110, 90, 105)
      const calc = new PriceCalculations({ name: 'pc1', 
        calculation: 'custom',
        customFormula: '(high + low) / 2' 
      })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      strictEqual(transformed[0]!.custom_price, 100) // (110 + 90) / 2
    })

    it('should evaluate complex custom formula', async () => {
      const testData = createTestData(100, 110, 90, 105, 1000)
      const calc = new PriceCalculations({ name: 'pc1', 
        calculation: 'custom',
        customFormula: '(open + 2 * close) / 3 + volume * 0.001' 
      })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 1)
      // (100 + 2 * 105) / 3 + 1000 * 0.001 = 103.333 + 1 = 104.333
      strictEqual((transformed[0]!.custom_price as number).toFixed(3), '104.333')
    })

    it('should handle formula with all OHLCV fields', async () => {
      const testData = createTestData(100, 110, 90, 105, 2000)
      const calc = new PriceCalculations({ name: 'pc1', 
        calculation: 'custom',
        customFormula: '(open + high + low + close) / 4 * (volume / 1000)',
        outputField: 'volume_weighted_avg'
      })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // ((100 + 110 + 90 + 105) / 4) * (2000 / 1000) = 101.25 * 2 = 202.5
      strictEqual(transformed[0]!.volume_weighted_avg, 202.5)
    })

    it('should reject invalid formula characters', async () => {
      const testData = createTestData(100, 110, 90, 105)
      const calc = new PriceCalculations({ name: 'pc1', 
        calculation: 'custom',
        customFormula: 'high; alert("hack")' 
      })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      
      try {
        await collectResults(result.data)
        throw new Error('Expected error was not thrown')
      } catch (error) {
        ok(error instanceof Error)
        ok(error.message.includes('Invalid characters in formula'))
      }
    })

    it('should handle division by zero', async () => {
      const testData = createTestData(100, 110, 90, 105, 0)
      const calc = new PriceCalculations({ name: 'pc1', 
        calculation: 'custom',
        customFormula: 'close / volume' 
      })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      
      try {
        await collectResults(result.data)
        throw new Error('Expected error was not thrown')
      } catch (error) {
        ok(error instanceof Error)
        ok(error.message.includes('Formula did not produce a valid number'))
      }
    })
  })

  describe('keepOriginal option', () => {
    it('should keep original OHLC fields by default', async () => {
      const testData = createTestData(100, 110, 90, 105)
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'hlc3' })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed[0]!.open, 100)
      strictEqual(transformed[0]!.high, 110)
      strictEqual(transformed[0]!.low, 90)
      strictEqual(transformed[0]!.close, 105)
      ok(transformed[0]!.hlc3)
    })

    it('should remove original fields when keepOriginal is false', async () => {
      const testData = createTestData(100, 110, 90, 105)
      const calc = new PriceCalculations({ name: 'pc1', 
        calculation: 'hlc3',
        keepOriginal: false 
      })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed[0]!.open, undefined)
      strictEqual(transformed[0]!.high, undefined)
      strictEqual(transformed[0]!.low, undefined)
      strictEqual(transformed[0]!.close, undefined)
      strictEqual(transformed[0]!.hlc3, 101.66666666666667)
      strictEqual(transformed[0]!.volume, 1000) // Volume should remain
    })
  })

  describe('multiple data points', () => {
    it('should process multiple bars correctly', async () => {
      const testData = [
        createTestData(100, 110, 90, 105)[0]!,
        createTestData(105, 115, 95, 110)[0]!,
        createTestData(110, 120, 100, 115)[0]!,
      ]
      
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'hlc3' })
      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 3)
      strictEqual((transformed[0]!.hlc3 as number).toFixed(3), '101.667') // (110+90+105)/3
      strictEqual((transformed[1]!.hlc3 as number).toFixed(3), '106.667') // (115+95+110)/3
      strictEqual((transformed[2]!.hlc3 as number).toFixed(3), '111.667') // (120+100+115)/3
    })
  })

  describe('getOutputFields and getRequiredFields', () => {
    it('should return correct output fields', () => {
      const calc1 = new PriceCalculations({ name: 'pc1', calculation: 'hlc3' })
      deepStrictEqual(calc1.getOutputFields(), ['hlc3'])

      const calc2 = new PriceCalculations({ name: 'pc1', 
        calculation: 'ohlc4',
        outputField: 'avg_price' 
      })
      deepStrictEqual(calc2.getOutputFields(), ['avg_price'])
    })

    it('should return required OHLC fields', () => {
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'hlc3' })
      deepStrictEqual(calc.getRequiredFields(), ['open', 'high', 'low', 'close'])
    })
  })

  describe('withParams', () => {
    it('should create new instance with updated params', () => {
      const calc1 = new PriceCalculations({ name: 'pc1', calculation: 'hlc3' })
      const calc2 = calc1.withParams({ calculation: 'ohlc4' }) as PriceCalculations

      strictEqual(calc1.params.calculation, 'hlc3')
      strictEqual(calc2.params.calculation, 'ohlc4')
    })
  })

  describe('edge cases', () => {
    it('should handle zero prices', async () => {
      const testData = createTestData(0, 0, 0, 0, 0)
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'hlc3' })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed[0]!.hlc3, 0)
    })

    it('should handle negative prices', async () => {
      const testData = createTestData(-10, -5, -20, -15)
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'ohlc4' })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      strictEqual(transformed[0]!.ohlc4, (-10 + -5 + -20 + -15) / 4) // -12.5
    })

    it('should handle very large prices', async () => {
      const testData = createTestData(1e10, 1.1e10, 0.9e10, 1.05e10)
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'hlc3' })

      const result = await calc.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)

      // (1.1e10 + 0.9e10 + 1.05e10) / 3
      const expected = (1.1e10 + 0.9e10 + 1.05e10) / 3
      strictEqual(transformed[0]!.hlc3, expected)
    })

    it('should handle empty data stream', async () => {
      const calc = new PriceCalculations({ name: 'pc1', calculation: 'hlc3' })
      const result = await calc.apply(arrayToAsyncIterator([]))
      const transformed = await collectResults(result.data)

      strictEqual(transformed.length, 0)
    })
  })
})