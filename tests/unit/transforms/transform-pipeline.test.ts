import { deepStrictEqual, ok, strictEqual, throws } from 'node:assert'
import { beforeEach, describe, it } from 'node:test'
import { 
  LogReturnsNormalizer, 
  MinMaxNormalizer, 
  PriceCalculations, 
  TransformPipeline, 
  ZScoreNormalizer 
} from '../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../src/utils'

// Helper to create test buffer
function createTestBuffer(count: number): DataBuffer {
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
  
  const baseTime = Date.now()
  
  for (let i = 0; i < count; i++) {
    buffer.push({
      timestamp: baseTime + i * 60000,
      open: 100 + i,
      high: 110 + i,
      low: 90 + i,
      close: 105 + i,
      volume: 1000 + i * 100
    })
  }
  
  return buffer
}

describe('TransformPipeline', () => {
  let buffer: DataBuffer
  let slice: DataSlice
  let logReturns: LogReturnsNormalizer
  let zScore: ZScoreNormalizer
  let minMax: MinMaxNormalizer

  beforeEach(() => {
    buffer = createTestBuffer(10)
    slice = new DataSlice(buffer, 0, buffer.length())
    
    logReturns = new LogReturnsNormalizer({ 
      tx: { in: 'close', out: 'returns', base: 'ln' } 
    }, slice)
    
    zScore = new ZScoreNormalizer({
      tx: { in: 'returns', out: 'returns_z', window: 3 }
    }, slice)
    
    minMax = new MinMaxNormalizer({
      tx: { in: 'close', out: 'close_norm', window: 3, min: 0, max: 1 }
    }, slice)
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

      throws(() => pipeline.remove(-1), /Invalid index/)
      throws(() => pipeline.remove(1), /Invalid index/)
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

  describe('next method (buffer-based processing)', () => {
    it('should handle empty pipeline', () => {
      const pipeline = new TransformPipeline({ transforms: [] })
      
      throws(() => pipeline.outputBuffer, /Pipeline has no transforms/)
    })

    it('should apply single transform', () => {
      const testBuffer = createTestBuffer(3)
      const testSlice = new DataSlice(testBuffer, 0, testBuffer.length())
      const priceCalc = new PriceCalculations({ 
        in: {},
        tx: { calc: 'hlc3', out: 'hlc3' } 
      }, testSlice)
      
      const pipeline = new TransformPipeline({ transforms: [priceCalc] })
      
      pipeline.next(0, testBuffer.length())
      const outputBuffer = pipeline.outputBuffer
      
      strictEqual(outputBuffer.length(), 3)
      ok(outputBuffer.getRow(0)?.hlc3)
    })

    it('should apply transform with window', () => {
      const testBuffer = createTestBuffer(5)
      const testSlice = new DataSlice(testBuffer, 0, testBuffer.length())
      const minMax = new MinMaxNormalizer({
        tx: { in: 'close', out: 'close_norm', window: 3, min: 0, max: 1 }
      }, testSlice)
      
      const pipeline = new TransformPipeline({ transforms: [minMax] })
      
      pipeline.next(0, testBuffer.length())
      const outputBuffer = pipeline.outputBuffer
      
      strictEqual(outputBuffer.length(), 5)
      // Check that normalization was applied
      ok(outputBuffer.getRow(2)?.close_norm !== undefined)
    })

    it('should chain multiple transforms', () => {
      const testBuffer = createTestBuffer(5)
      const testSlice = new DataSlice(testBuffer, 0, testBuffer.length())
      
      // Create transforms that will be chained
      const priceCalc = new PriceCalculations({ 
        in: {},
        tx: { calc: 'hlc3', out: 'hlc3' } 
      }, testSlice)
      
      // MinMax will use the output from priceCalc
      const minMaxSlice = new DataSlice(priceCalc.outputBuffer, 0, priceCalc.outputBuffer.length())
      const minMax = new MinMaxNormalizer({
        tx: { in: 'hlc3', out: 'hlc3_norm', window: 3, min: 0, max: 1 }
      }, minMaxSlice)
      
      const pipeline = new TransformPipeline({
        transforms: [priceCalc, minMax]
      })
      
      pipeline.next(0, testBuffer.length())
      const outputBuffer = pipeline.outputBuffer
      
      strictEqual(outputBuffer.length(), 5)
      // Check that both transforms were applied
      ok(outputBuffer.getRow(2)?.hlc3 !== undefined)
      ok(outputBuffer.getRow(2)?.hlc3_norm !== undefined)
    })

    it('should handle complex pipeline', () => {
      const testBuffer = createTestBuffer(20)
      const testSlice = new DataSlice(testBuffer, 0, testBuffer.length())
      
      // Create chained transforms
      const logReturns = new LogReturnsNormalizer({ 
        tx: { in: 'close', out: 'returns', base: 'ln' } 
      }, testSlice)
      
      const zScoreSlice = new DataSlice(logReturns.outputBuffer, 0, logReturns.outputBuffer.length())
      const zScore = new ZScoreNormalizer({
        tx: { in: 'returns', out: 'returns_zscore', window: 5 }
      }, zScoreSlice)
      
      const minMax = new MinMaxNormalizer({
        tx: { 
          in: 'returns_zscore', 
          out: 'returns_zscore_norm', 
          min: -1, 
          max: 1, 
          window: 3 
        }
      }, new DataSlice(zScore.outputBuffer, 0, zScore.outputBuffer.length()))
      
      const pipeline = new TransformPipeline({
        transforms: [logReturns, zScore, minMax]
      })
      
      pipeline.next(0, testBuffer.length())
      const outputBuffer = pipeline.outputBuffer
      
      // All transforms should have processed the data
      strictEqual(outputBuffer.length(), 20)
      
      // Check that final output has all expected fields
      const lastRow = outputBuffer.getRow(19)
      ok(lastRow?.returns !== undefined)
      ok(lastRow?.returns_zscore !== undefined)
      ok(lastRow?.returns_zscore_norm !== undefined)
      ok(typeof lastRow?.returns_zscore_norm === 'number')
    })
  })

  describe('readiness', () => {
    it('should track readiness of all transforms', () => {
      const testBuffer = createTestBuffer(2)
      const testSlice = new DataSlice(testBuffer, 0, testBuffer.length())
      
      // MinMax with window=3 won't be ready with only 2 data points
      const minMax = new MinMaxNormalizer({
        tx: { in: 'close', out: 'close_norm', window: 3, min: 0, max: 1 }
      }, testSlice)
      
      const pipeline = new TransformPipeline({ transforms: [minMax] })
      
      // Pipeline should not be ready before processing
      strictEqual(pipeline.isReady, false)
      
      // Process the data
      pipeline.next(0, testBuffer.length())
      
      // Still not ready because we don't have enough data for window
      strictEqual(pipeline.isReady, false)
      
      // Add more data and process
      const testBuffer2 = createTestBuffer(5)
      const testSlice2 = new DataSlice(testBuffer2, 0, testBuffer2.length())
      const minMax2 = new MinMaxNormalizer({
        tx: { in: 'close', out: 'close_norm', window: 3, min: 0, max: 1 }
      }, testSlice2)
      
      const pipeline2 = new TransformPipeline({ transforms: [minMax2] })
      pipeline2.next(0, testBuffer2.length())
      
      // Now should be ready
      strictEqual(pipeline2.isReady, true)
    })
  })
})