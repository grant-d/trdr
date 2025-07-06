import type { OhlcvDto } from '../models'

/**
 * Types of transformations available in the pipeline
 */
export type TransformType =
  | 'missingValues'
  | 'timeframeAggregation'
  | 'logReturns'
  | 'map'
  | 'zScore'
  | 'minMax'
  | 'percentChange'
  | 'fractionalDiff'
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
  | 'statisticalRegime'
  | 'lorentzianDistance'
  | 'shannonInformation'

/**
 * Base parameters interface that all transform parameters extend
 */
export interface BaseTransformParams {
  /** Optional custom name for this transform instance */
  // name?: string

  /** Optional description of what this transform does */
  description?: string
  
  /** 
   * Input columns to transform. If not specified, defaults to standard OHLCV columns.
   * Can include both standard columns (open, high, low, close, volume) and 
   * columns created by previous transforms (e.g., 'o_lr', 'c_z')
   */
  in?: string[]
  
  /** 
   * Output column names. If not specified, defaults to overwriting input columns.
   * Must have same length as inputColumns. Can overwrite existing columns or create new ones.
   * Use null to drop a column from the output.
   * Example: in: ['close', 'c_lr'], out: ['close', null] 
   * would overwrite 'close' and drop 'c_lr' from the output
   */
  out?: (string | null)[]
}

/**
 * Configuration for a transform including its type and parameters
 */
export interface TransformConfig<T extends BaseTransformParams = BaseTransformParams> {
  /** Type of transform to apply */
  type: TransformType

  /** Transform-specific parameters */
  params: T

  /** Whether this transform is disabled */
  disabled?: boolean

  /** Optional intermediate output configuration */
  output?: {
    type: 'csv' | 'jsonl'
    path?: string
    table?: string
    includeMetadata?: boolean
  }
}

/**
 * Result of applying a transform
 */
export interface TransformResult {
  /** The transformed data */
  data: AsyncIterator<OhlcvDto>
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

  /** Current parameters for this transform */
  readonly params: T

  /**
   * Applies the transformation to a stream of OHLCV data
   * @param data Input data stream
   * @returns Transformed data stream
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
   * Creates a copy of this transform with new parameters
   * @param params New parameters to apply
   * @returns New transform instance
   */
  withParams(params: Partial<T>): Transform<T>

  /**
   * Indicates whether this transform is ready to emit meaningful data.
   * For example, SMA(20) needs 20 data points before it can emit valid moving averages.
   * @returns true if the transform is ready to emit, false if it needs more data
   */
  isReady(): boolean
}
