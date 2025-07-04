import type { OhlcvDto } from '../models'

/**
 * Types of transformations available in the pipeline
 */
export type TransformType =
  | 'missingValues'
  | 'timeframeAggregation'
  | 'logReturns'
  | 'zScore'
  | 'minMax'
  | 'percentChange'
  | 'tickBars'
  | 'volumeBars'
  | 'dollarBars'
  | 'tickImbalanceBars'
  | 'heikinAshi'
  | 'priceCalc'
  | 'movingAverage'
  | 'rsi'
  | 'bollinger'
  | 'macd'
  | 'atr'
  | 'vwap'
  | 'volumeProfile'
  | 'rollingStats'
  | 'percentileRank'
  | 'bucket'
  | 'pipeline'

/**
 * Base parameters interface that all transform parameters extend
 */
export interface BaseTransformParams {
  /** Optional name for the transform instance */
  name?: string
}

/**
 * Configuration for a transform including its type and parameters
 */
export interface TransformConfig<T extends BaseTransformParams = BaseTransformParams> {
  /** Type of transform to apply */
  type: TransformType

  /** Transform-specific parameters */
  params: T

  /** Whether this transform is enabled */
  enabled: boolean

  /** Optional intermediate output configuration */
  output?: {
    type: 'csv' | 'jsonl' | 'sqlite'
    path?: string
    table?: string
    includeMetadata?: boolean
  }
}

/**
 * Coefficients that can be stored for reversible transforms
 */
export interface TransformCoefficients {
  /** Transform type these coefficients belong to */
  type: TransformType

  /** Timestamp when coefficients were calculated */
  timestamp: number

  /** Symbol these coefficients apply to */
  symbol: string

  /** The actual coefficient values */
  values: Record<string, number>
}

/**
 * Result of applying a transform, including any coefficients
 */
export interface TransformResult {
  /** The transformed data */
  data: AsyncIterator<OhlcvDto>

  /** Coefficients if this is a reversible transform */
  coefficients?: TransformCoefficients
}

/**
 * Interface that all transforms must implement
 * Transforms process OHLCV data streams and can add new fields or modify existing ones
 */
export interface Transform<T extends BaseTransformParams = BaseTransformParams> {
  /** Type identifier for this transform */
  readonly type: TransformType

  /** Human-readable name for the transform */
  readonly name: string

  /** Description of what this transform does */
  readonly description: string

  /** Whether this transform can be reversed using coefficients */
  readonly isReversible: boolean

  /** Current parameters for this transform */
  readonly params: T

  /**
   * Applies the transformation to a stream of OHLCV data
   * @param data Input data stream
   * @returns Transformed data stream and optional coefficients
   */
  apply(data: AsyncIterator<OhlcvDto>): Promise<TransformResult>

  /**
   * Validates that the transform can be applied with current parameters
   * @throws Error if validation fails
   */
  validate(): void

  /**
   * Gets the list of new fields this transform will add to the data
   * @returns Array of field names that will be added
   */
  getOutputFields(): string[]

  /**
   * Gets the list of fields this transform requires to be present
   * @returns Array of field names that must exist in input data
   */
  getRequiredFields(): string[]

  /**
   * Reverses the transformation using stored coefficients
   * Only applicable for reversible transforms
   * @param data Transformed data to reverse
   * @param coefficients Coefficients to use for reversal
   * @returns Original data stream
   * @throws Error if transform is not reversible
   */
  reverse?(data: AsyncIterator<OhlcvDto>, coefficients: TransformCoefficients): AsyncGenerator<OhlcvDto>

  /**
   * Creates a copy of this transform with new parameters
   * @param params New parameters to apply
   * @returns New transform instance
   */
  withParams(params: Partial<T>): Transform<T>
}
