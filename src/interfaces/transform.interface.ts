import type { OhlcvDto } from '../models'
import type { DataBuffer, DataSlice } from '../utils'

/**
 * Types of transformations available in the pipeline
 */
export type TransformType =
  | 'atr'
  | 'bollinger'
  // | 'bucket'
  | 'dollarBars'
  | 'ema'
  | 'fractionalDiff'
  | 'heikinAshi'
  | 'impute'
  | 'logReturns'
  | 'lorentzianBars'
  | 'lorentzianDistance'
  | 'macd'
  | 'map'
  | 'minMax'
  | 'missingValues'
  | 'movingAverage'
  | 'percentChange'
  | 'percentileRank'
  | 'pipeline'
  | 'priceCalc'
  | 'regimeBars'
  | 'rollingStats'
  | 'rsi'
  // | 'shannonInformation'
  | 'shannonInfoBars'
  | 'sma'
  // | 'statisticalRegime'
  | 'tickBars'
  | 'tickImbalanceBars'
  | 'tickRunBars'
  | 'timeBars'
  // | 'timeframeAggregation'
  | 'volumeBars'
  | 'volumeProfile'
  | 'vwap'
  | 'vwapBars'
  | 'zScore'

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
  // in?: string[]

  /**
   * Output column names. If not specified, defaults to overwriting input columns.
   * Must have same length as inputColumns. Can overwrite existing columns or create new ones.
   * Use null to drop a column from the output.
   * Example: in: ['close', 'c_lr'], out: ['close', null]
   * would overwrite 'close' and drop 'c_lr' from the output
   */
  // out?: (string | null)[]
}

/**
 * Configuration for a transform including its type and parameters
 */
export interface TransformConfig<
  T extends BaseTransformParams = BaseTransformParams,
> {
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
export interface Transform<
  T extends BaseTransformParams = BaseTransformParams,
> {
  /** Type identifier for this transform */
  readonly type: TransformType

  /** Human-readable name for the transform */
  readonly name: string

  /** Description of what this transform does */
  readonly description: string

  /** Current parameters for this transform */
  readonly params: T

  readonly batchNumber: number

  /**
   * Process a batch of rows and return a slice of the transformed data
   * @param from Start index in the internal buffer
   * @param to End index in the internal buffer
   * @returns DataSlice containing the transformed data (may be a subset if not all rows were processed)
   */
  next(from: number, to: number): DataSlice

  /**
   * Indicates whether this transform is ready to emit meaningful data.
   * For example, SMA(20) needs 20 data points before it can emit valid moving averages.
   * @returns true if the transform is ready to emit, false if it needs more data
   */
  readonly isReady: boolean

  /**
   * Get the buffer that should be passed to the next transform.
   * For most transforms, this is the input buffer (in-place modification).
   * For reshaping transforms, this is a separate output buffer.
   */
  readonly outputBuffer: DataBuffer
}
