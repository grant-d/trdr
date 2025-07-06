import { deepStrictEqual, ok, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import type {
  LogReturnsParams,
  TransformPipeline
} from '../../../src/transforms'
import {
  LogReturnsNormalizer,
  MinMaxNormalizer,
  PriceCalculations,
  ZScoreNormalizer,
  createPipeline
} from '../../../src/transforms'
import {
  TransformSerializer,
  type SerializedPipeline,
  type SerializedTransform
} from '../../../src/transforms/transform-serializer'

describe('Transform Serializer', () => {
  describe('serializeTransform', () => {
    it('should serialize a basic transform', () => {
      const transform = new LogReturnsNormalizer({
        in: ['close'],
        out: ['returns']
      })

      const serialized = TransformSerializer.serializeTransform(transform)

      strictEqual(serialized.type, 'logReturns')
      deepStrictEqual(serialized.params, {
        in: ['close'],
        out: ['returns']
      })
    })

    it('should serialize transforms with complex parameters', () => {
      const transform = new ZScoreNormalizer({
        in: ['open', 'close', 'volume'],
        out: ['open_z', 'close_z', 'volume_z'],
        windowSize: 20
      })

      const serialized = TransformSerializer.serializeTransform(transform)

      deepStrictEqual(serialized.params.in, ['open', 'close', 'volume'])
      deepStrictEqual(serialized.params.out, ['open_z', 'close_z', 'volume_z'])
      strictEqual(serialized.params.windowSize, 20)
    })
  })

  describe('deserializeTransform', () => {
    it('should deserialize a basic transform', () => {
      const serialized: SerializedTransform = {
        type: 'logReturns',
        params: {
          in: ['close'],
          out: ['log_returns']
        }
      }

      const transform = TransformSerializer.deserializeTransform(serialized)

      strictEqual(transform.type, 'logReturns')
      const params = transform.params as Partial<LogReturnsParams>
      deepStrictEqual(params.in, ['close'])
      deepStrictEqual(params.out, ['log_returns'])
    })

    it('should deserialize transform with complex parameters', () => {
      const serialized: SerializedTransform = {
        type: 'minMax',
        params: {
          in: ['close', 'volume'],
          out: ['close_norm', 'volume_norm'],
          min: -1,
          max: 1
        }
      }

      const transform = TransformSerializer.deserializeTransform(serialized)

      strictEqual(transform.type, 'minMax')
      strictEqual((transform.params as any).min, -1)
      strictEqual((transform.params as any).max, 1)
      deepStrictEqual(transform.params.in, ['close', 'volume'])
      deepStrictEqual(transform.params.out, ['close_norm', 'volume_norm'])
    })

    it('should throw error for unknown transform type', () => {
      const serialized: SerializedTransform = {
        type: 'unknownTransform' as any,
        params: {}
      }

      throws(() => {
        TransformSerializer.deserializeTransform(serialized)
      }, /Unknown transform type: unknownTransform/)
    })
  })

  describe('serializePipeline', () => {
    it('should serialize a transform pipeline', () => {
      const pipeline = createPipeline([
        new LogReturnsNormalizer({ in: ['close'], out: ['returns'] }),
        new ZScoreNormalizer({ in: ['returns'], out: ['returns_z'] }),
        new MinMaxNormalizer({ in: ['returns_z'], out: ['returns_norm'] })
      ], 'Test Pipeline')

      const serialized = TransformSerializer.serializePipeline(pipeline)

      strictEqual(serialized.description, 'Test Pipeline')
      strictEqual(serialized.transforms.length, 3)
      strictEqual(serialized.transforms[0]!.type, 'logReturns')
      strictEqual(serialized.transforms[1]!.type, 'zScore')
      strictEqual(serialized.transforms[2]!.type, 'minMax')
      ok(serialized.metadata?.createdAt)
      strictEqual(serialized.metadata?.version, '1.0.0')
    })
  })

  describe('deserializePipeline', () => {
    it('should deserialize a transform pipeline', () => {
      const serialized: SerializedPipeline = {
        transforms: [
          {
            type: 'priceCalc',
            params: { calculation: 'hlc3' }
          },
          {
            type: 'zScore',
            params: { in: ['hlc3'], out: ['hlc3_z'] }
          }
        ],
        metadata: {
          createdAt: '2024-01-01T00:00:00Z',
          version: '1.0.0'
        }
      }

      const pipeline = TransformSerializer.deserializePipeline(serialized)

      strictEqual(pipeline.name, 'Transform Pipeline')
      const transforms = pipeline.getTransforms()
      strictEqual(transforms.length, 2)
      strictEqual(transforms[0]!.type, 'priceCalc')
      strictEqual(transforms[1]!.type, 'zScore')
    })
  })

  describe('toJSON and fromJSON', () => {
    it('should round-trip serialize a transform', () => {
      const original = new ZScoreNormalizer({
        in: ['open', 'close'],
        out: ['open_z', 'close_z'],
        windowSize: 50
      })

      const json = TransformSerializer.toJSON(original)
      const deserialized = TransformSerializer.fromJSON(json)

      strictEqual(deserialized.type, original.type)
      deepStrictEqual(deserialized.params, original.params)
    })

    it('should round-trip serialize a pipeline', () => {
      const original = createPipeline([
        new LogReturnsNormalizer({ in: ['close'], out: ['returns'] }),
        new MinMaxNormalizer({ in: ['returns'], out: ['returns_norm'], min: -1, max: 1 })
      ], 'Round-trip Pipeline')

      const json = TransformSerializer.toJSON(original)
      const deserialized = TransformSerializer.fromJSON(json) as TransformPipeline

      strictEqual(deserialized.name, original.name)
      const transforms = deserialized.getTransforms()
      strictEqual(transforms.length, 2)
      strictEqual(transforms[0]!.type, 'logReturns')
      strictEqual(transforms[1]!.type, 'minMax')
    })

    it('should produce formatted JSON', () => {
      const transform = new PriceCalculations({ calculation: 'ohlc4' })
      const json = TransformSerializer.toJSON(transform)

      // Check that it's properly formatted
      ok(json.includes('\n'))
      ok(json.includes('  ')) // Indentation
    })
  })



  describe('validateSerialized', () => {
    it('should validate correct serialized transform', () => {
      const valid: SerializedTransform = {
        type: 'logReturns',
        params: { in: ['close'], out: ['returns'] }
      }

      ok(TransformSerializer.validateSerialized(valid))
    })

    it('should reject invalid structures', () => {
      strictEqual(TransformSerializer.validateSerialized(null), false)
      strictEqual(TransformSerializer.validateSerialized('string'), false)
      strictEqual(TransformSerializer.validateSerialized({}), false)
      strictEqual(TransformSerializer.validateSerialized({ type: 'logReturns' }), false)
      strictEqual(TransformSerializer.validateSerialized({ params: {} }), false)
      strictEqual(TransformSerializer.validateSerialized({ 
        type: 'unknownType',
        params: {}
      }), false)
    })
  })

  describe('cloneParams', () => {
    it('should create a deep copy of parameters', () => {
      const original = {
        fields: ['open', 'close'],
        nested: {
          value: 100,
          array: [1, 2, 3]
        }
      }

      const cloned = TransformSerializer.cloneParams(original)

      // Modify cloned
      cloned.fields.push('volume')
      cloned.nested.value = 200
      cloned.nested.array.push(4)

      // Original should be unchanged
      strictEqual(original.fields.length, 2)
      strictEqual(original.nested.value, 100)
      strictEqual(original.nested.array.length, 3)

      // Cloned should have modifications
      strictEqual(cloned.fields.length, 3)
      strictEqual(cloned.nested.value, 200)
      strictEqual(cloned.nested.array.length, 4)
    })
  })

  describe('integration tests', () => {
    it('should handle complex pipeline serialization and deserialization', () => {
      // Create a complex pipeline
      const logReturns = new LogReturnsNormalizer({ in: ['close'], out: ['returns'] })
      
      const minMax = new MinMaxNormalizer({
        in: ['returns'],
        out: ['returns_norm'],
        min: -1,
        max: 1
      })

      const zScore = new ZScoreNormalizer({
        in: ['close', 'volume'],
        out: ['close_z', 'volume_z'],
        windowSize: 30
      })

      const original = createPipeline([logReturns, minMax, zScore], 'Complex Pipeline')

      // Serialize to JSON
      const json = TransformSerializer.toJSON(original)
      
      // Deserialize back
      const deserialized = TransformSerializer.fromJSON(json) as TransformPipeline

      // Verify structure
      strictEqual(deserialized.name, 'Transform Pipeline')
      const transforms = deserialized.getTransforms()
      strictEqual(transforms.length, 3)

      // Verify transform types and parameters were preserved
      strictEqual(transforms[0]!.type, 'logReturns')
      strictEqual(transforms[1]!.type, 'minMax')
      strictEqual(transforms[2]!.type, 'zScore')
      
      // Check that parameters are preserved
      const deserializedMinMax = transforms[1] as MinMaxNormalizer
      strictEqual(deserializedMinMax.params.min, -1)
      strictEqual(deserializedMinMax.params.max, 1)
      
      const deserializedZScore = transforms[2] as ZScoreNormalizer
      strictEqual(deserializedZScore.params.windowSize, 30)
    })
  })
})