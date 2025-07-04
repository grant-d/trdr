import { describe, it } from 'node:test'
import { strictEqual, deepStrictEqual, ok, throws } from 'node:assert'
import { 
  TransformSerializer,
  type SerializedTransform,
  type SerializedPipeline
} from '../../../src/transforms/transform-serializer'
import {
  LogReturnsNormalizer,
  MinMaxNormalizer,
  ZScoreNormalizer,
  PriceCalculations,
  TransformPipeline,
  createPipeline
} from '../../../src/transforms'
import type { TransformCoefficients } from '../../../src/interfaces'

describe('Transform Serializer', () => {
  describe('serializeTransform', () => {
    it('should serialize a basic transform', () => {
      const transform = new LogReturnsNormalizer({
        outputField: 'returns',
        priceField: 'close'
      })

      const serialized = TransformSerializer.serializeTransform(transform)

      strictEqual(serialized.type, 'logReturns')
      strictEqual(serialized.name, 'Log Returns Normalizer')
      deepStrictEqual(serialized.params, {
        outputField: 'returns',
        priceField: 'close'
      })
      strictEqual(serialized.coefficients, undefined)
    })

    it('should serialize transform with coefficients', () => {
      const transform = new MinMaxNormalizer({
        fields: ['close'],
        targetMin: 0,
        targetMax: 1
      })

      // Simulate coefficients being set
      ;(transform as any).setCoefficients('BTC-USD', {
        close_min: 30000,
        close_max: 40000
      })

      const serialized = TransformSerializer.serializeTransform(transform)

      ok(serialized.coefficients)
      strictEqual(serialized.coefficients.type, 'minMax')
      strictEqual(serialized.coefficients.symbol, 'BTC-USD')
      strictEqual(serialized.coefficients.values.close_min, 30000)
      strictEqual(serialized.coefficients.values.close_max, 40000)
    })

    it('should serialize transforms with complex parameters', () => {
      const transform = new ZScoreNormalizer({
        fields: ['open', 'close', 'volume'],
        windowSize: 20,
        suffix: '_normalized',
        addSuffix: true
      })

      const serialized = TransformSerializer.serializeTransform(transform)

      deepStrictEqual(serialized.params.fields, ['open', 'close', 'volume'])
      strictEqual(serialized.params.windowSize, 20)
      strictEqual(serialized.params.suffix, '_normalized')
      strictEqual(serialized.params.addSuffix, true)
    })
  })

  describe('deserializeTransform', () => {
    it('should deserialize a basic transform', () => {
      const serialized: SerializedTransform = {
        type: 'logReturns',
        params: {
          outputField: 'log_returns',
          priceField: 'close'
        },
        name: 'Log Returns'
      }

      const transform = TransformSerializer.deserializeTransform(serialized)

      strictEqual(transform.type, 'logReturns')
      strictEqual(transform.params.outputField, 'log_returns')
      strictEqual(transform.params.priceField, 'close')
    })

    it('should deserialize transform with coefficients', () => {
      const serialized: SerializedTransform = {
        type: 'minMax',
        params: {
          fields: ['close', 'volume']
        },
        coefficients: {
          type: 'minMax',
          timestamp: Date.now(),
          symbol: 'ETH-USD',
          values: {
            close_min: 1000,
            close_max: 2000,
            volume_min: 100,
            volume_max: 1000
          }
        }
      }

      const transform = TransformSerializer.deserializeTransform(serialized)

      strictEqual(transform.type, 'minMax')
      
      // Check that coefficients were restored
      const coefficients = (transform as any).getCoefficients()
      ok(coefficients)
      strictEqual(coefficients.symbol, 'ETH-USD')
      strictEqual(coefficients.values.close_min, 1000)
      strictEqual(coefficients.values.close_max, 2000)
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
        new LogReturnsNormalizer({ outputField: 'returns' }),
        new ZScoreNormalizer({ fields: ['returns'] }),
        new MinMaxNormalizer({ fields: ['returns_zscore'] })
      ], 'Test Pipeline')

      const serialized = TransformSerializer.serializePipeline(pipeline)

      strictEqual(serialized.name, 'Test Pipeline')
      strictEqual(serialized.transforms.length, 3)
      strictEqual(serialized.transforms[0]!.type, 'logReturns')
      strictEqual(serialized.transforms[1]!.type, 'zScore')
      strictEqual(serialized.transforms[2]!.type, 'minMax')
      ok(serialized.metadata?.createdAt)
      strictEqual(serialized.metadata?.version, '1.0.0')
    })

    it('should serialize pipeline with transform coefficients', () => {
      const minMax = new MinMaxNormalizer({ fields: ['close'] })
      ;(minMax as any).setCoefficients('BTC-USD', {
        close_min: 30000,
        close_max: 40000
      })

      const pipeline = createPipeline([
        new LogReturnsNormalizer({ outputField: 'returns' }),
        minMax
      ])

      const serialized = TransformSerializer.serializePipeline(pipeline)

      const minMaxSerialized = serialized.transforms[1]!
      ok(minMaxSerialized.coefficients)
      strictEqual(minMaxSerialized.coefficients.values.close_min, 30000)
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
            params: { fields: ['hlc3'] }
          }
        ],
        name: 'Deserialized Pipeline',
        metadata: {
          createdAt: '2024-01-01T00:00:00Z',
          version: '1.0.0'
        }
      }

      const pipeline = TransformSerializer.deserializePipeline(serialized)

      strictEqual(pipeline.name, 'Deserialized Pipeline')
      const transforms = pipeline.getTransforms()
      strictEqual(transforms.length, 2)
      strictEqual(transforms[0]!.type, 'priceCalc')
      strictEqual(transforms[1]!.type, 'zScore')
    })
  })

  describe('toJSON and fromJSON', () => {
    it('should round-trip serialize a transform', () => {
      const original = new ZScoreNormalizer({
        fields: ['open', 'close'],
        windowSize: 50,
        suffix: '_z'
      })

      const json = TransformSerializer.toJSON(original)
      const deserialized = TransformSerializer.fromJSON(json)

      strictEqual(deserialized.type, original.type)
      deepStrictEqual(deserialized.params, original.params)
    })

    it('should round-trip serialize a pipeline', () => {
      const original = createPipeline([
        new LogReturnsNormalizer({ outputField: 'returns' }),
        new MinMaxNormalizer({ fields: ['returns'], targetMin: -1, targetMax: 1 })
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

  describe('coefficientsToRepositoryFormat', () => {
    it('should convert transform coefficients to repository format', () => {
      const coefficients: TransformCoefficients = {
        type: 'minMax',
        timestamp: Date.now(),
        symbol: 'BTC-USD',
        values: {
          close_min: 30000,
          close_max: 40000,
          volume_min: 1000000,
          volume_max: 5000000
        }
      }

      const repositoryData = TransformSerializer.coefficientsToRepositoryFormat(
        coefficients,
        'price_normalizer'
      )

      strictEqual(repositoryData.length, 4)
      
      const closeMin = repositoryData.find(d => d.name.includes('close_min'))!
      strictEqual(closeMin.name, 'price_normalizer_minMax_close_min')
      strictEqual(closeMin.value, 30000)
      strictEqual(closeMin.symbol, 'BTC-USD')
      strictEqual(closeMin.metadata?.transformType, 'minMax')
      strictEqual(closeMin.metadata?.coefficientKey, 'close_min')
    })
  })

  describe('repositoryToCoefficients', () => {
    it('should convert repository data back to transform coefficients', () => {
      const repositoryData = [
        {
          name: 'normalizer_minMax_close_min',
          value: 30000,
          timestamp: Date.now(),
          symbol: 'BTC-USD',
          metadata: {
            transformType: 'minMax',
            coefficientKey: 'close_min',
            transformName: 'normalizer'
          }
        },
        {
          name: 'normalizer_minMax_close_max',
          value: 40000,
          timestamp: Date.now(),
          symbol: 'BTC-USD',
          metadata: {
            transformType: 'minMax',
            coefficientKey: 'close_max',
            transformName: 'normalizer'
          }
        }
      ]

      const coefficients = TransformSerializer.repositoryToCoefficients(
        repositoryData,
        'minMax'
      )

      ok(coefficients)
      strictEqual(coefficients.type, 'minMax')
      strictEqual(coefficients.symbol, 'BTC-USD')
      strictEqual(coefficients.values.close_min, 30000)
      strictEqual(coefficients.values.close_max, 40000)
    })

    it('should return null for empty data', () => {
      const coefficients = TransformSerializer.repositoryToCoefficients([], 'minMax')
      strictEqual(coefficients, null)
    })

    it('should filter by transform type', () => {
      const repositoryData = [
        {
          name: 'normalizer_minMax_value',
          value: 100,
          timestamp: Date.now(),
          symbol: 'BTC-USD',
          metadata: {
            transformType: 'minMax',
            coefficientKey: 'value',
            transformName: 'normalizer'
          }
        },
        {
          name: 'other_zScore_mean',
          value: 50,
          timestamp: Date.now(),
          symbol: 'BTC-USD',
          metadata: {
            transformType: 'zScore',
            coefficientKey: 'mean',
            transformName: 'other'
          }
        }
      ]

      const coefficients = TransformSerializer.repositoryToCoefficients(
        repositoryData,
        'minMax'
      )

      ok(coefficients)
      strictEqual(Object.keys(coefficients.values).length, 1)
      strictEqual(coefficients.values.value, 100)
    })
  })

  describe('validateSerialized', () => {
    it('should validate correct serialized transform', () => {
      const valid: SerializedTransform = {
        type: 'logReturns',
        params: { outputField: 'returns' }
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
      // Create a complex pipeline with coefficients
      const logReturns = new LogReturnsNormalizer({ outputField: 'returns' })
      
      const minMax = new MinMaxNormalizer({ 
        fields: ['returns'],
        targetMin: -1,
        targetMax: 1
      })
      ;(minMax as any).setCoefficients('ETH-USD', {
        returns_min: -0.1,
        returns_max: 0.15
      })

      const zScore = new ZScoreNormalizer({
        fields: ['close', 'volume'],
        windowSize: 30
      })
      ;(zScore as any).setCoefficients('ETH-USD', {
        close_mean: 2000,
        close_std: 100,
        volume_mean: 1000000,
        volume_std: 50000
      })

      const original = createPipeline([logReturns, minMax, zScore], 'Complex Pipeline')

      // Serialize to JSON
      const json = TransformSerializer.toJSON(original)
      
      // Deserialize back
      const deserialized = TransformSerializer.fromJSON(json) as TransformPipeline

      // Verify structure
      strictEqual(deserialized.name, 'Complex Pipeline')
      const transforms = deserialized.getTransforms()
      strictEqual(transforms.length, 3)

      // Verify coefficients were preserved
      const deserializedMinMax = transforms[1] as any
      const minMaxCoeffs = deserializedMinMax.getCoefficients()
      ok(minMaxCoeffs)
      strictEqual(minMaxCoeffs.values.returns_min, -0.1)
      strictEqual(minMaxCoeffs.values.returns_max, 0.15)

      const deserializedZScore = transforms[2] as any
      const zScoreCoeffs = deserializedZScore.getCoefficients()
      ok(zScoreCoeffs)
      strictEqual(zScoreCoeffs.values.close_mean, 2000)
      strictEqual(zScoreCoeffs.values.volume_std, 50000)
    })
  })
})