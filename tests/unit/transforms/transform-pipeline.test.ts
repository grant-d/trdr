import { deepStrictEqual, ok, strictEqual, throws } from 'node:assert'
import { beforeEach, describe, it } from 'node:test'
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
    logReturns = new LogReturnsNormalizer({ in: ['close'], out: ['returns'] })
    zScore = new ZScoreNormalizer({ in: ['returns'], out: ['returns_z'], windowSize: 3 })
    minMax = new MinMaxNormalizer({ in: ['close'], out: ['close_norm'], windowSize: 3 })
    priceCalc = new PriceCalculations({ calculation: 'hlc3' })
  })

  describe('constructor and basic operations', () => {
    it('should create empty pipeline', () => {
      const pipeline = new TransformPipeline({ transforms: [] })
      strictEqual(pipeline.type, 'pipeline')
      strictEqual(pipeline.name, 'Transform Pipeline')
      strictEqual(pipeline.description, 'Pipeline of 0 transforms')
      deepStrictEqual(pipeline.getTransforms(), [])
    })

    it('should create pipeline with transforms', () => {
      const pipeline = new TransformPipeline({ 
        transforms: [logReturns, zScore],
        description: 'Test Pipeline'
      })
      
      strictEqual(pipeline.name, 'Transform Pipeline')
      strictEqual(pipeline.description, 'Test Pipeline')
      strictEqual(pipeline.getTransforms().length, 2)
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
    })

    it('should apply single transform', async () => {
      const testData = createTestData(3)
      const pipeline = new TransformPipeline({ transforms: [priceCalc] })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 3)
      ok(transformed[0]!.hlc3)
    })

    it('should apply transform', async () => {
      const testData = createTestData(3)
      const pipeline = new TransformPipeline({ transforms: [minMax] })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 1)
      ok(transformed[0]!.close_norm !== undefined)
    })

    it('should chain multiple transforms', async () => {
      const testData = createTestData(5)
      
      // Pipeline: calculate HLC3 -> normalize with min-max
      const pipeline = new TransformPipeline({ 
        transforms: [
          new PriceCalculations({  calculation: 'hlc3' }),
          new MinMaxNormalizer({ in: ['hlc3'], out: ['hlc3_norm'], windowSize: 3 })
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 3)
      
      // MinMaxNormalizer with windowSize=3 yields 3 results from 5 input items
      ok(transformed[0]!.hlc3_norm !== undefined)
      ok(transformed[1]!.hlc3_norm !== undefined)
      ok(transformed[2]!.hlc3_norm !== undefined)
    })

    it('should collect', async () => {
      const testData = createTestData(10)
      
      // Pipeline with two normalizers
      const pipeline = new TransformPipeline({ 
        transforms: [
          new MinMaxNormalizer({ in: ['close'], out: ['close_norm'], windowSize: 5 }),
          new ZScoreNormalizer({ in: ['open'], out: ['open_z'], windowSize: 3 })
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 4) // MinMax yields 6 items, ZScore with windowSize=3 yields 4 from those 6
      ok(transformed[0]!.close_norm !== undefined)
      ok(transformed[0]!.open_z !== undefined)
    })

    it('should apply complex pipeline', async () => {
      const testData = createTestData(20)
      
      // Complex pipeline: returns -> z-score -> min-max
      const pipeline = new TransformPipeline({ 
        transforms: [
          new LogReturnsNormalizer({ in: ['close'], out: ['returns'] }),
          new ZScoreNormalizer({ in: ['returns'], out: ['returns_zscore'], windowSize: 5 }),
          new MinMaxNormalizer({ in: ['returns_zscore'], out: ['returns_zscore_norm'], min: -1, max: 1, windowSize: 3 })
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      // LogReturns yields 19 items, ZScore with windowSize=5 yields 15 items, MinMax with windowSize=3 yields 13 items
      strictEqual(transformed.length, 13)
      ok(transformed[0]!.returns_zscore_norm !== undefined)
      ok(typeof transformed[0]!.returns_zscore_norm === 'number')
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
      const invalid1 = new PriceCalculations({  calculation: 'custom' })
      const invalid2 = new PriceCalculations({  calculation: 'custom' })
      
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
          new MinMaxNormalizer({ in: ['close'], out: ['close_norm'] })
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
          new LogReturnsNormalizer({ in: ['close'], out: ['returns'] }), // Requires close
          new ZScoreNormalizer({ in: ['returns'], out: ['returns_z'] }) // Requires returns
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

  describe('withParams', () => {
    it('should create new instance with updated params', () => {
      const pipeline1 = new TransformPipeline({ 
        transforms: [logReturns],
        description: 'Pipeline 1'
      })
      
      const pipeline2 = pipeline1.withParams({ 
        description: 'Pipeline 2'
      }) as TransformPipeline
      
      strictEqual(pipeline1.name, 'Transform Pipeline')
      strictEqual(pipeline2.name, 'Transform Pipeline')
      strictEqual(pipeline1.description, 'Pipeline 1')
      strictEqual(pipeline2.description, 'Pipeline 2')
      strictEqual(pipeline2.getTransforms().length, 1)
    })
  })

  describe('createPipeline helper', () => {
    it('should create pipeline from array', () => {
      const pipeline = createPipeline([logReturns, zScore])
      
      strictEqual(pipeline.type, 'pipeline')
      strictEqual(pipeline.getTransforms().length, 2)
    })

    it('should create pipeline with custom description', () => {
      const pipeline = createPipeline([logReturns], 'Custom Pipeline')
      
      strictEqual(pipeline.name, 'Transform Pipeline')
      strictEqual(pipeline.description, 'Custom Pipeline')
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

    it('should handle insufficient data for transform requirements', async () => {
      const testData = createTestData(2) // 2 data points
      const pipeline = new TransformPipeline({ 
        transforms: [
          new PriceCalculations({ calculation: 'hlc3' }),
          new MinMaxNormalizer({ in: ['hlc3'], out: ['hlc3_norm'], windowSize: 3 })
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 0) // MinMaxNormalizer needs 3 data points, only have 2
    })

    it('should handle transform that filters data', async () => {
      const testData = createTestData(5)
      
      // LogReturns yields 4 items from 5 input items (starts from 2nd item)
      const pipeline = new TransformPipeline({ 
        transforms: [
          new LogReturnsNormalizer({ in: ['close'], out: ['returns'] }),
          new MinMaxNormalizer({ in: ['returns'], out: ['returns_norm'], windowSize: 3 })
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      // LogReturns yields 4 items, MinMaxNormalizer with windowSize=3 yields 2 items (4-3+1=2)
      strictEqual(transformed.length, 2)
      
      // Verify both transforms were applied
      ok('returns' in transformed[0]!)
      ok('returns_norm' in transformed[0]!)
      ok(typeof transformed[0]!.returns === 'number')
      ok(typeof transformed[0]!.returns_norm === 'number')
    })

    it('should maintain data integrity through pipeline', async () => {
      const testData = createTestData(5)
      const pipeline = new TransformPipeline({ 
        transforms: [
          new PriceCalculations({ calculation: 'hlc3' }),
          new MinMaxNormalizer({ in: ['volume'], out: ['volume_norm'], windowSize: 3 }) // Normalize different field
        ] 
      })
      
      const result = await pipeline.apply(arrayToAsyncIterator(testData))
      const transformed = await collectResults(result.data)
      
      strictEqual(transformed.length, 3) // MinMaxNormalizer with windowSize=3 yields 3 results from 5 items
      ok(transformed[0]!.hlc3 !== undefined) // HLC3 field added by PriceCalculations
      ok(transformed[0]!.volume_norm !== undefined) // Volume normalized by MinMaxNormalizer
    })
  })
})