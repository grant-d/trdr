import { deepStrictEqual, ok, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import type { LogReturnsParams, TransformPipeline } from '../../../src/transforms'
import { createPipeline, LogReturnsNormalizer, MinMaxNormalizer, PriceCalculations, ZScoreNormalizer } from '../../../src/transforms'
import { type SerializedPipeline, type SerializedTransform, TransformSerializer } from '../../../src/transforms'
import { DataBuffer, DataSlice } from '../../../src/utils'

describe('Transform Serializer', () => {
  describe('serializeTransform', () => {
    it('should serialize a basic transform', () => {
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
      
      const transform = new LogReturnsNormalizer({
        tx: { in: 'close', out: 'returns', base: 'ln' }
      }, slice)

      const serialized = TransformSerializer.serializeTransform(transform)

      strictEqual(serialized.type, 'logReturns')
      deepStrictEqual(serialized.params.tx, {
        in: 'close',
        out: 'returns',
        base: 'ln'
      })
    })

    it('should serialize transforms with complex parameters', () => {
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
      
      const transform = new ZScoreNormalizer({
        tx: [
          { in: 'open', out: 'open_z', window: 20 },
          { in: 'close', out: 'close_z', window: 20 },
          { in: 'volume', out: 'volume_z', window: 20 }
        ]
      }, slice)

      const serialized = TransformSerializer.serializeTransform(transform)

      ok(Array.isArray(serialized.params.tx))
      strictEqual(serialized.params.tx.length, 3)
      strictEqual(serialized.params.tx[0].window, 20)
    })
  })

  describe('deserializeTransform', () => {
    it('should deserialize a basic transform', () => {
      const serialized: SerializedTransform = {
        type: 'logReturns',
        params: {
          tx: {
            in: 'close',
            out: 'log_returns',
            base: 'ln'
          }
        }
      }

      const transform = TransformSerializer.deserializeTransform(serialized)

      strictEqual(transform.type, 'logReturns')
      const params = transform.params as Partial<LogReturnsParams>
      deepStrictEqual((params as any).tx, {
        in: 'close',
        out: 'log_returns',
        base: 'ln'
      })
    })

    it('should deserialize transform with complex parameters', () => {
      const serialized: SerializedTransform = {
        type: 'minMax',
        params: {
          tx: {
            in: 'close',
            out: 'close_norm',
            min: -1,
            max: 1,
            window: 20
          }
        }
      }

      const transform = TransformSerializer.deserializeTransform(serialized)

      strictEqual(transform.type, 'minMax')
      const tx = (transform.params as any).tx
      strictEqual(tx.min, -1)
      strictEqual(tx.max, 1)
      strictEqual(tx.in, 'close')
      strictEqual(tx.out, 'close_norm')
      strictEqual(tx.window, 20)
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
      const buffer = new DataBuffer({
        columns: {
          timestamp: { index: 0 },
          open: { index: 1 },
          high: { index: 2 },
          low: { index: 3 },
          close: { index: 4 },
          volume: { index: 5 },
          returns: { index: 6 },
          returns_z: { index: 7 }
        }
      })
      const slice = new DataSlice(buffer, 0, 0)
      
      const pipeline = createPipeline(
        [
          new LogReturnsNormalizer({ tx: { in: 'close', out: 'returns', base: 'ln' } }, slice),
          new ZScoreNormalizer({ tx: { in: 'returns', out: 'returns_z', window: 20 } }, slice),
          new MinMaxNormalizer({ tx: { in: 'returns_z', out: 'returns_norm', min: -1, max: 1, window: 20 } }, slice)
        ],
        'Test Pipeline'
      )

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
            params: { tx: { calc: 'hlc3', out: 'hlc3' } }
          },
          {
            type: 'zScore',
            params: { tx: { in: 'hlc3', out: 'hlc3_z', window: 20 } }
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
      
      const original = new ZScoreNormalizer({
        tx: [
          { in: 'open', out: 'open_z', window: 50 },
          { in: 'close', out: 'close_z', window: 50 }
        ]
      }, slice)

      const json = TransformSerializer.toJSON(original)
      const deserialized = TransformSerializer.fromJSON(json)

      strictEqual(deserialized.type, original.type)
      deepStrictEqual(deserialized.params, original.params)
    })

    it('should round-trip serialize a pipeline', () => {
      const buffer = new DataBuffer({
        columns: {
          timestamp: { index: 0 },
          open: { index: 1 },
          high: { index: 2 },
          low: { index: 3 },
          close: { index: 4 },
          volume: { index: 5 },
          returns: { index: 6 }
        }
      })
      const slice = new DataSlice(buffer, 0, 0)
      
      const original = createPipeline(
        [
          new LogReturnsNormalizer({ tx: { in: 'close', out: 'returns', base: 'ln' } }, slice),
          new MinMaxNormalizer({
            tx: {
              in: 'returns',
              out: 'returns_norm',
              min: -1,
              max: 1,
              window: 20
            }
          }, slice)
        ],
        'Round-trip Pipeline'
      )

      const json = TransformSerializer.toJSON(original)
      const deserialized = TransformSerializer.fromJSON(
        json
      ) as TransformPipeline

      strictEqual(deserialized.name, original.name)
      const transforms = deserialized.getTransforms()
      strictEqual(transforms.length, 2)
      strictEqual(transforms[0]!.type, 'logReturns')
      strictEqual(transforms[1]!.type, 'minMax')
    })

    it('should produce formatted JSON', () => {
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
      
      const transform = new PriceCalculations({ 
        in: { open: 'open', high: 'high', low: 'low', close: 'close' },
        tx: { calc: 'ohlc4', out: 'ohlc4' } 
      }, slice)
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
      strictEqual(
        TransformSerializer.validateSerialized({ type: 'logReturns' }),
        false
      )
      strictEqual(
        TransformSerializer.validateSerialized({ params: {} }),
        false
      )
      strictEqual(
        TransformSerializer.validateSerialized({
          type: 'unknownType',
          params: {}
        }),
        false
      )
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

      const logReturns = new LogReturnsNormalizer({
        tx: { in: 'close', out: 'returns', base: 'ln' }
      }, slice)

      const minMax = new MinMaxNormalizer({
        tx: {
          in: 'returns',
          out: 'returns_norm',
          min: -1,
          max: 1,
          window: 20
        }
      }, slice)

      const zScore = new ZScoreNormalizer({
        tx: [
          { in: 'close', out: 'close_z', window: 30 },
          { in: 'volume', out: 'volume_z', window: 30 }
        ]
      }, slice)

      const original = createPipeline(
        [logReturns, minMax, zScore],
        'Complex Pipeline'
      )

      // Serialize to JSON
      const json = TransformSerializer.toJSON(original)

      // Deserialize back
      const deserialized = TransformSerializer.fromJSON(
        json
      ) as TransformPipeline

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
      strictEqual((deserializedMinMax.params.tx as any).min, -1)
      strictEqual((deserializedMinMax.params.tx as any).max, 1)

      const deserializedZScore = transforms[2] as ZScoreNormalizer
      strictEqual(Array.isArray(deserializedZScore.params.tx), true)
      strictEqual((deserializedZScore.params.tx as any[])[0].window, 30)
    })
  })
})
