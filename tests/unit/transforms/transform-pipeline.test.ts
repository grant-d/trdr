import { deepStrictEqual, ok, strictEqual, throws } from 'node:assert'
import { beforeEach, describe, it } from 'node:test'
import type { TransformCoefficients } from '../../../src/interfaces'
import type { OhlcvDto } from '../../../src/models'
import {
  createPipeline,
  LogReturnsNormalizer,
  MinMaxNormalizer,
  PriceCalculations,
  TransformPipeline,
  ZScoreNormalizer
} from '../../../src/transforms'

// Helper to create test data
function createTestData(count: number): OhlcvDto[] {
  const data: OhlcvDto[] = []
  const baseTime = Date.now()
  
  for (let i = 0; i < count; i++) {
    data.push({
      timestamp: baseTime + i * 60000,
      symbol: 'TEST',
      exchange: 'test',
      open: 100 + i,
      high: 110 + i,
      low: 90 + i,
      close: 105 + i,
      volume: 1000 + i * 100
    })
  }
  
  return data
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

describe('TransformPipeline', () => {
  let logReturns: LogReturnsNormalizer
  let zScore: ZScoreNormalizer
  let minMax: MinMaxNormalizer
  let priceCalc: PriceCalculations

  beforeEach(() => {
    logReturns = new LogReturnsNormalizer({ outputField: 'returns' })
    zScore = new ZScoreNormalizer({ fields: ['returns'] })
    minMax = new MinMaxNormalizer({ fields: ['close'] })
    priceCalc = new PriceCalculations({ calculation: 'hlc3' })
  })

  describe('constructor and basic operations', () => {
    it('should create empty pipeline', () => {
      const pipeline = new TransformPipeline({ transforms: [] })
      strictEqual(pipeline.type, 'pipeline')
      strictEqual(pipeline.name, 'Transform Pipeline')
      strictEqual(pipeline.description, 'Pipeline of 0 transforms')
      strictEqual(pipeline.isReversible, true) // Empty pipeline is reversible
      deepStrictEqual(pipeline.getTransforms(), [])
    })

    it('should create pipeline with transforms', () => {
      const pipeline = new TransformPipeline({ 
        transforms: [logReturns, zScore],
        name: 'Test Pipeline'
      })
      
      strictEqual(pipeline.name, 'Test Pipeline')
      strictEqual(pipeline.description, 'Pipeline of 2 transforms')
      strictEqual(pipeline.isReversible, true)
      strictEqual(pipeline.getTransforms().length, 2)
    })

    it('should detect non-reversible pipeline', () => {
      const pipeline = new TransformPipeline({ 
        transforms: [priceCalc, zScore] // priceCalc is not reversible
      })
      
      strictEqual(pipeline.isReversible, false)
    })
  })

  describe('add, remove, insert, move operations', () => {
    it('should add transform to pipeline', () => {
      const pipeline = new TransformPipeline({ transforms: [logReturns] })
      const newPipeline = pipeline.add(zScore)
      
      // Original unchanged
      strictEqual(pipeline.getTransforms().length, 1)
      
      // New pipeline has both transforms
      strictEqual(newPipeline.getTransforms().length, 2)
      strictEqual(newPipeline.getTransforms()[1], zScore)
    })

    it('should remove transform from pipeline', () => {
      const pipeline = new TransformPipeline({ 
        transforms: [logReturns, zScore, minMax] 
      })
      const newPipeline = pipeline.remove(1) // Remove zScore
      
      strictEqual(newPipeline.getTransforms().length, 2)
      strictEqual(newPipeline.getTransforms()[0], logReturns)
      strictEqual(newPipeline.getTransforms()[1], minMax)
    })

    it('should throw on invalid remove index', () => {
      const pipeline = new TransformPipeline({ transforms: [logReturns] })
      
      throws(
        () => pipeline.remove(-1),
        /Invalid index/
      )
      
      throws(
        () => pipeline.remove(1),
        /Invalid index/
      )
    })

    it('should insert transform at index', () => {
      const pipeline = new TransformPipeline({ 
        transforms: [logReturns, minMax] 
      })
      const newPipeline = pipeline.insert(1, zScore)
      
      strictEqual(newPipeline.getTransforms().length, 3)
      strictEqual(newPipeline.getTransforms()[0], logReturns)
      strictEqual(newPipeline.getTransforms()[1], zScore)
      strictEqual(newPipeline.getTransforms()[2], minMax)
    })

    it('should move transform within pipeline', () => {
      const pipeline = new TransformPipeline({ 
        transforms: [logReturns, zScore, minMax] 
      })
      const newPipeline = pipeline.move(0, 2) // Move logReturns to end
      
      strictEqual(newPipeline.getTransforms()[0], zScore)
      strictEqual(newPipeline.getTransforms()[1], minMax)
      strictEqual(newPipeline.getTransforms()[2], logReturns)
    })

    it('should clear all transforms', () => {
      const pipeline = new TransformPipeline({ 
        transforms: [logReturns, zScore, minMax] 
      })
      const newPipeline = pipeline.clear()
      
      strictEqual(newPipeline.getTransforms().length, 0)
    })
  })

  describe('apply method', () => {
    it('should apply empty pipeline (passthrough)', async () => {
      const testData = createTestData(3)
      const pipeline = new TransformPipeline({ transforms: [] })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      deepStrictEqual(transformed, testData)
      strictEqual(result.coefficients, undefined)
    })

    it('should apply single transform', async () => {
      const testData = createTestData(3)
      const pipeline = new TransformPipeline({ transforms: [priceCalc] })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 3)
      ok(transformed[0]!.hlc3)
      strictEqual(result.coefficients, undefined) // No coefficients from priceCalc
    })

    it('should apply transform with coefficients', async () => {
      const testData = createTestData(3)
      const pipeline = new TransformPipeline({ transforms: [minMax] })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 3)
      ok(transformed[0]!.close_norm !== undefined)
      ok(result.coefficients, `Expected coefficients but got: ${JSON.stringify(result.coefficients)}`)
    })

    it('should chain multiple transforms', async () => {
      const testData = createTestData(5)
      
      // Pipeline: calculate HLC3 -> normalize with min-max
      const pipeline = new TransformPipeline({ 
        transforms: [
          new PriceCalculations({ calculation: 'hlc3' }),
          new MinMaxNormalizer({ fields: ['hlc3'] })
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 5)
      
      // Check that both transforms were applied
      ok(transformed[0]!.hlc3) // PriceCalc output
      ok(transformed[0]!.hlc3_norm !== undefined) // MinMax output
      
      // Check coefficients were collected (only MinMax produces coefficients)
      ok(result.coefficients, `Expected coefficients but got: ${JSON.stringify(result.coefficients)}`)
      ok(result.coefficients.values.t1_hlc3_min !== undefined)
      ok(result.coefficients.values.t1_hlc3_max !== undefined)
    })

    it('should collect and aggregate coefficients', async () => {
      const testData = createTestData(10)
      
      // Pipeline with two normalizers that produce coefficients
      const pipeline = new TransformPipeline({ 
        transforms: [
          new MinMaxNormalizer({ fields: ['close'] }),
          new ZScoreNormalizer({ fields: ['open'] })
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 10)
      
      // Check aggregated coefficients
      ok(result.coefficients)
      strictEqual(result.coefficients.type as any, 'pipeline')
      
      // MinMax coefficients (transform 0)
      ok(result.coefficients.values.t0_close_min !== undefined)
      ok(result.coefficients.values.t0_close_max !== undefined)
      
      // ZScore coefficients (transform 1)
      ok(result.coefficients.values.t1_open_mean !== undefined)
      ok(result.coefficients.values.t1_open_std !== undefined)
    })

    it('should apply complex pipeline', async () => {
      const testData = createTestData(20)
      
      // Complex pipeline: returns -> z-score -> min-max
      const pipeline = new TransformPipeline({ 
        transforms: [
          new LogReturnsNormalizer({ outputField: 'returns' }),
          new ZScoreNormalizer({ fields: ['returns'], windowSize: 10 }),
          new MinMaxNormalizer({ fields: ['returns_zscore'], targetMin: -1, targetMax: 1 })
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      // LogReturns includes all data points (first point has 0 return)
      strictEqual(transformed.length, 20)
      
      // Check all transforms were applied
      ok(transformed[0]!.returns !== undefined)
      ok(transformed[0]!.returns_zscore !== undefined)
      ok(transformed[0]!.returns_zscore_norm !== undefined)
      
      // Check coefficients from all transforms
      ok(result.coefficients)
      // LogReturns coefficients
      ok(result.coefficients.values.t0_priceField !== undefined)
      ok(result.coefficients.values.t0_base !== undefined)
      // ZScore doesn't produce global coefficients with window
      // MinMax coefficients  
      ok(result.coefficients.values.t2_returns_zscore_min !== undefined)
    })
  })

  describe('validate method', () => {
    it('should validate empty pipeline', () => {
      const pipeline = new TransformPipeline({ transforms: [] })
      pipeline.validate() // Should not throw
    })

    it('should validate all transforms', () => {
      // Create invalid transform
      const invalidPriceCalc = new PriceCalculations({ 
        calculation: 'custom'
        // Missing customFormula
      })
      
      const pipeline = new TransformPipeline({ 
        transforms: [logReturns, invalidPriceCalc] 
      })
      
      throws(
        () => pipeline.validate(),
        /Transform 1 \(Price Calculations\) validation failed: Custom formula is required/
      )
    })

    it('should include transform index in validation errors', () => {
      const invalid1 = new PriceCalculations({ calculation: 'custom' })
      const invalid2 = new PriceCalculations({ calculation: 'custom' })
      
      const pipeline = new TransformPipeline({ 
        transforms: [logReturns, invalid1, zScore, invalid2] 
      })
      
      throws(
        () => pipeline.validate(),
        /Transform 1/
      )
    })
  })

  describe('getOutputFields and getRequiredFields', () => {
    it('should aggregate output fields from all transforms', () => {
      const pipeline = new TransformPipeline({ 
        transforms: [
          new PriceCalculations({ calculation: 'hlc3' }),
          new PriceCalculations({ calculation: 'ohlc4' }),
          new MinMaxNormalizer({ fields: ['close'] })
        ] 
      })
      
      const fields = pipeline.getOutputFields()
      ok(fields.includes('hlc3'))
      ok(fields.includes('ohlc4'))
      ok(fields.includes('close_norm'))
    })

    it('should return required fields from first transform', () => {
      const pipeline = new TransformPipeline({ 
        transforms: [
          new LogReturnsNormalizer({}), // Requires close
          new ZScoreNormalizer({ fields: ['returns'] }) // Requires returns
        ] 
      })
      
      const fields = pipeline.getRequiredFields()
      deepStrictEqual(fields, ['close'])
    })

    it('should return empty required fields for empty pipeline', () => {
      const pipeline = new TransformPipeline({ transforms: [] })
      deepStrictEqual(pipeline.getRequiredFields(), [])
    })
  })

  describe('reverse method', () => {
    it('should throw if pipeline is not reversible', async () => {
      const pipeline = new TransformPipeline({ 
        transforms: [priceCalc] // Not reversible
      })
      
      const testData = createTestData(1)
      const iterator = arrayToAsyncIterator(testData)
      const dummyCoeff: TransformCoefficients = {
        type: 'pipeline' as any,
        timestamp: Date.now(),
        symbol: 'TEST',
        values: {}
      }
      
      try {
        const generator = pipeline.reverse(iterator, dummyCoeff)
        await generator.next()
        throw new Error('Expected error was not thrown')
      } catch (error) {
        ok(error instanceof Error)
        ok(error.message.includes('Pipeline contains non-reversible transforms'))
      }
    })

    it('should reverse single transform pipeline', async () => {
      const testData = createTestData(5)
      const pipeline = new TransformPipeline({ 
        transforms: [new MinMaxNormalizer({ fields: ['close'] })] 
      })
      
      // Apply forward
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      // Reverse
      const reversed = pipeline.reverse(
        arrayToAsyncIterator(transformed), 
        result.coefficients!
      )
      const reversedData = await collectResults(reversed)
      
      strictEqual(reversedData.length, 5)
      // Check close values are restored (approximately due to floating point)
      for (let i = 0; i < 5; i++) {
        const diff = Math.abs(reversedData[i]!.close - testData[i]!.close)
        ok(diff < 0.0001)
      }
    })

    it('should reverse multi-transform pipeline', async () => {
      const testData = createTestData(10)
      const pipeline = new TransformPipeline({ 
        transforms: [
          new MinMaxNormalizer({ fields: ['close'] }),
          new ZScoreNormalizer({ fields: ['open'] })
        ] 
      })
      
      // Apply forward
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      // Reverse
      const reversed = pipeline.reverse(
        arrayToAsyncIterator(transformed), 
        result.coefficients!
      )
      const reversedData = await collectResults(reversed)
      
      strictEqual(reversedData.length, 10)
      
      // Check values are restored
      for (let i = 0; i < 10; i++) {
        const closeDiff = Math.abs(reversedData[i]!.close - testData[i]!.close)
        const openDiff = Math.abs(reversedData[i]!.open - testData[i]!.open)
        ok(closeDiff < 0.0001)
        ok(openDiff < 0.0001)
      }
    })
  })

  describe('withParams', () => {
    it('should create new instance with updated params', () => {
      const pipeline1 = new TransformPipeline({ 
        transforms: [logReturns],
        name: 'Pipeline 1'
      })
      
      const pipeline2 = pipeline1.withParams({ 
        name: 'Pipeline 2'
      }) as TransformPipeline
      
      strictEqual(pipeline1.name, 'Pipeline 1')
      strictEqual(pipeline2.name, 'Pipeline 2')
      strictEqual(pipeline2.getTransforms().length, 1)
    })
  })

  describe('createPipeline helper', () => {
    it('should create pipeline from array', () => {
      const pipeline = createPipeline([logReturns, zScore])
      
      strictEqual(pipeline.type, 'pipeline')
      strictEqual(pipeline.getTransforms().length, 2)
    })

    it('should create pipeline with custom name', () => {
      const pipeline = createPipeline([logReturns], 'Custom Pipeline')
      
      strictEqual(pipeline.name, 'Custom Pipeline')
    })
  })

  describe('edge cases', () => {
    it('should handle empty data stream', async () => {
      const pipeline = new TransformPipeline({ 
        transforms: [priceCalc, minMax] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator([]))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 0)
    })

    it('should handle single data point', async () => {
      const testData = createTestData(1)
      const pipeline = new TransformPipeline({ 
        transforms: [
          new PriceCalculations({ calculation: 'hlc3' }),
          new MinMaxNormalizer({ fields: ['hlc3'] })
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 1)
      ok(transformed[0]!.hlc3)
      strictEqual(transformed[0]!.hlc3_norm, 0.5) // Single value normalizes to middle
    })

    it('should handle transform that filters data', async () => {
      const testData = createTestData(5)
      
      // LogReturns filters out first data point
      const pipeline = new TransformPipeline({ 
        transforms: [
          new LogReturnsNormalizer({}),
          new MinMaxNormalizer({ fields: ['returns'] })
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 5) // All data points preserved with LogReturns
    })

    it('should maintain data integrity through pipeline', async () => {
      const testData = createTestData(3)
      const pipeline = new TransformPipeline({ 
        transforms: [
          new PriceCalculations({ calculation: 'hlc3' }),
          new MinMaxNormalizer({ fields: ['volume'] }) // Normalize different field
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      // Check original fields are preserved
      for (let i = 0; i < 3; i++) {
        strictEqual(transformed[i]!.symbol, testData[i]!.symbol)
        strictEqual(transformed[i]!.timestamp, testData[i]!.timestamp)
        strictEqual(transformed[i]!.open, testData[i]!.open)
      }
    })
  })
})