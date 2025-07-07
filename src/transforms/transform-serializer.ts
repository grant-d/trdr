import type { Transform, TransformConfig, TransformType } from '../interfaces'
import { DataBuffer, DataSlice } from '../utils'
import { DollarBarGenerator, TickBarGenerator, TickImbalanceBarGenerator, TickRunBarGenerator, TimeBarGenerator, VolumeBarGenerator } from './bar-generators'
import { LogReturnsNormalizer, MinMaxNormalizer, ZScoreNormalizer } from './normalizers'
import { PriceCalculations } from './price-calculations'
import {
  AverageTrueRange,
  BollingerBands,
  ExponentialMovingAverage,
  ImputeTransform,
  Macd,
  RelativeStrengthIndex,
  SimpleMovingAverage,
  VolumeWeightedAveragePrice
} from './technical-indicators'
import { TransformPipeline } from './transform-pipeline'

/**
 * Serialized representation of a transform
 */
export interface SerializedTransform {
  /** Type of the transform */
  type: TransformType;
  /** Transform parameters */
  params: Record<string, any>;
}

/**
 * Serialized representation of a transform pipeline
 */
export interface SerializedPipeline {
  /** Array of serialized transforms in the pipeline */
  transforms: SerializedTransform[];
  /** Pipeline description */
  description?: string;
  /** Pipeline metadata */
  metadata?: {
    createdAt?: string;
    updatedAt?: string;
    version?: string;
  };
}

/**
 * Registry of transform constructors for deserialization
 */
// Note: Transform serialization needs to be updated for the new buffer-based architecture
// For now, create a dummy buffer and slice for deserialization
const createDummyBuffer = () => {
  return new DataBuffer({
    columns: {
      timestamp: { index: 0 },
      symbol: { index: 1 },
      exchange: { index: 2 },
      open: { index: 3 },
      high: { index: 4 },
      low: { index: 5 },
      close: { index: 6 },
      volume: { index: 7 }
    }
  })
}

const createDummySlice = () => {
  const buffer = createDummyBuffer()
  return new DataSlice(buffer, 0, buffer.length())
}

const TRANSFORM_REGISTRY: Readonly<Record<string, (params: any) => Transform>> =
  {
    logReturns: (params) =>
      new LogReturnsNormalizer(params, createDummySlice()),
    minMax: (params) => new MinMaxNormalizer(params, createDummySlice()),
    zScore: (params) => new ZScoreNormalizer(params, createDummySlice()),
    priceCalc: (params) => new PriceCalculations(params, createDummySlice()),
    impute: (params) => new ImputeTransform(params, createDummySlice()),
    timeBars: (params) => new TimeBarGenerator(params, createDummySlice()),
    sma: (params) => new SimpleMovingAverage(params, createDummySlice()),
    ema: (params) => new ExponentialMovingAverage(params, createDummySlice()),
    rsi: (params) => new RelativeStrengthIndex(params, createDummySlice()),
    bollinger: (params) => new BollingerBands(params, createDummySlice()),
    macd: (params) => new Macd(params, createDummySlice()),
    atr: (params) => new AverageTrueRange(params, createDummySlice()),
    vwap: (params) =>
      new VolumeWeightedAveragePrice(params, createDummySlice()),
    tickBars: (params) => new TickBarGenerator(params, createDummySlice()),
    volumeBars: (params) => new VolumeBarGenerator(params, createDummySlice()),
    dollarBars: (params) => new DollarBarGenerator(params, createDummySlice()),
    tickImbalanceBars: (params) =>
      new TickImbalanceBarGenerator(params, createDummySlice()),
    tickRunBars: (params) =>
      new TickRunBarGenerator(params, createDummySlice())
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
      params: { ...transform.params }
    }

    return serialized
  }

  /**
   * Serialize a transform pipeline
   */
  public static serializePipeline(
    pipeline: TransformPipeline
  ): SerializedPipeline {
    const transforms = pipeline.getTransforms()

    return {
      transforms: transforms.map((t) => this.serializeTransform(t)),
      description: pipeline.description,
      metadata: {
        createdAt: new Date().toISOString(),
        version: '1.0.0'
      }
    }
  }

  /**
   * Deserialize a transform from a serialized representation
   */
  public static deserializeTransform(
    serialized: SerializedTransform
  ): Transform {
    const constructor = TRANSFORM_REGISTRY[serialized.type]

    if (!constructor) {
      throw new Error(`Unknown transform type: ${serialized.type}`)
    }

    const transform = constructor(serialized.params)

    return transform
  }

  /**
   * Deserialize a transform pipeline
   */
  public static deserializePipeline(
    serialized: SerializedPipeline
  ): TransformPipeline {
    const transforms = serialized.transforms.map((t) =>
      this.deserializeTransform(t)
    )

    return new TransformPipeline({
      transforms,
      description: serialized.description
    })
  }

  /**
   * Convert transform configuration to serialized form
   * This is useful for saving configurations
   */
  public static configToSerialized(
    config: TransformConfig
  ): SerializedTransform {
    return {
      type: config.type,
      params: config.params
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
  public static validateSerialized(
    serialized: any
  ): serialized is SerializedTransform {
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
    if (
      !(serialized.type in TRANSFORM_REGISTRY) &&
      serialized.type !== 'pipeline'
    ) {
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
