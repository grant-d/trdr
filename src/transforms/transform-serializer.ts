import type { Transform, TransformConfig, TransformCoefficients, TransformType } from '../interfaces'
import type { CoefficientData } from '../repositories'
import { LogReturnsNormalizer } from './log-returns-normalizer'
import { MinMaxNormalizer } from './min-max-normalizer'
import { ZScoreNormalizer } from './z-score-normalizer'
import { PriceCalculations } from './price-calculations'
import { MissingValueHandler } from './missing-value-handler'
import { TimeframeAggregator } from './timeframe-aggregator'
import { TransformPipeline } from './transform-pipeline'

/**
 * Serialized representation of a transform
 */
export interface SerializedTransform {
  /** Type of the transform */
  type: TransformType
  /** Transform parameters */
  params: Record<string, any>
  /** Optional name override */
  name?: string
  /** Coefficients if this is a reversible transform */
  coefficients?: TransformCoefficients
}

/**
 * Serialized representation of a transform pipeline
 */
export interface SerializedPipeline {
  /** Array of serialized transforms in the pipeline */
  transforms: SerializedTransform[]
  /** Pipeline name */
  name?: string
  /** Pipeline metadata */
  metadata?: {
    createdAt?: string
    updatedAt?: string
    version?: string
    description?: string
  }
}

/**
 * Registry of transform constructors for deserialization
 */
const TRANSFORM_REGISTRY: Record<string, (params: any) => Transform> = {
  logReturns: (params) => new LogReturnsNormalizer(params),
  minMax: (params) => new MinMaxNormalizer(params),
  zScore: (params) => new ZScoreNormalizer(params),
  priceCalc: (params) => new PriceCalculations(params),
  missingValues: (params) => new MissingValueHandler(params),
  timeframeAggregation: (params) => new TimeframeAggregator(params),
}

/**
 * Utility class for serializing and deserializing transforms
 */
export class TransformSerializer {
  /**
   * Serialize a transform to a JSON-compatible object
   */
  public static serializeTransform(transform: Transform): SerializedTransform {
    const serialized: SerializedTransform = {
      type: transform.type,
      params: { ...transform.params },
      name: transform.name
    }

    // Include coefficients if available (for reversible transforms)
    if (transform.isReversible && 'getCoefficients' in transform) {
      const coefficients = (transform as any).getCoefficients()
      if (coefficients) {
        serialized.coefficients = coefficients
      }
    }

    return serialized
  }

  /**
   * Serialize a transform pipeline
   */
  public static serializePipeline(pipeline: TransformPipeline): SerializedPipeline {
    const transforms = pipeline.getTransforms()
    
    return {
      transforms: transforms.map(t => this.serializeTransform(t)),
      name: pipeline.name,
      metadata: {
        createdAt: new Date().toISOString(),
        version: '1.0.0'
      }
    }
  }

  /**
   * Deserialize a transform from a serialized representation
   */
  public static deserializeTransform(serialized: SerializedTransform): Transform {
    const constructor = TRANSFORM_REGISTRY[serialized.type]
    
    if (!constructor) {
      throw new Error(`Unknown transform type: ${serialized.type}`)
    }

    const transform = constructor(serialized.params)

    // Restore coefficients if present
    if (serialized.coefficients && 'setCoefficients' in transform) {
      const { symbol, values } = serialized.coefficients
      ;(transform as any).setCoefficients(symbol, values)
    }

    return transform
  }

  /**
   * Deserialize a transform pipeline
   */
  public static deserializePipeline(serialized: SerializedPipeline): TransformPipeline {
    const transforms = serialized.transforms.map(t => this.deserializeTransform(t))
    
    return new TransformPipeline({
      transforms,
      name: serialized.name
    })
  }

  /**
   * Convert transform configuration to serialized form
   * This is useful for saving configurations
   */
  public static configToSerialized(config: TransformConfig): SerializedTransform {
    return {
      type: config.type,
      params: config.params,
      name: config.params.name
    }
  }

  /**
   * Convert TransformCoefficients to CoefficientData for repository storage
   */
  public static coefficientsToRepositoryFormat(
    coefficients: TransformCoefficients,
    transformName: string
  ): CoefficientData[] {
    const results: CoefficientData[] = []
    
    for (const [key, value] of Object.entries(coefficients.values)) {
      results.push({
        name: `${transformName}_${coefficients.type}_${key}`,
        value,
        timestamp: coefficients.timestamp,
        symbol: coefficients.symbol,
        metadata: {
          transformType: coefficients.type,
          coefficientKey: key,
          transformName
        }
      })
    }

    return results
  }

  /**
   * Convert CoefficientData array back to TransformCoefficients
   */
  public static repositoryToCoefficients(
    data: CoefficientData[],
    transformType: TransformType
  ): TransformCoefficients | null {
    if (data.length === 0) return null

    const values: Record<string, number> = {}
    let symbol = ''
    let timestamp = 0

    for (const item of data) {
      const metadata = item.metadata as any
      if (metadata?.transformType === transformType && metadata?.coefficientKey) {
        values[metadata.coefficientKey] = item.value
        symbol = item.symbol || symbol
        timestamp = item.timestamp
      }
    }

    if (Object.keys(values).length === 0) return null

    return {
      type: transformType,
      timestamp,
      symbol,
      values
    }
  }

  /**
   * Save transform configuration to JSON string
   */
  public static toJSON(transform: Transform | TransformPipeline): string {
    if (transform instanceof TransformPipeline) {
      return JSON.stringify(this.serializePipeline(transform), null, 2)
    } else {
      return JSON.stringify(this.serializeTransform(transform), null, 2)
    }
  }

  /**
   * Load transform from JSON string
   */
  public static fromJSON(json: string): Transform | TransformPipeline {
    const parsed = JSON.parse(json)
    
    // Check if it's a pipeline
    if ('transforms' in parsed && Array.isArray(parsed.transforms)) {
      return this.deserializePipeline(parsed)
    } else {
      return this.deserializeTransform(parsed)
    }
  }

  /**
   * Validate a serialized transform structure
   */
  public static validateSerialized(serialized: any): serialized is SerializedTransform {
    if (!serialized || typeof serialized !== 'object') {
      return false
    }

    if (!serialized.type || typeof serialized.type !== 'string') {
      return false
    }

    if (!serialized.params || typeof serialized.params !== 'object') {
      return false
    }

    // Validate known transform types
    if (!(serialized.type in TRANSFORM_REGISTRY) && serialized.type !== 'pipeline') {
      return false
    }

    return true
  }

  /**
   * Create a deep copy of transform parameters
   * Useful for ensuring immutability
   */
  public static cloneParams<T extends Record<string, any>>(params: T): T {
    return JSON.parse(JSON.stringify(params))
  }
}