import type { BaseTransformParams } from '../interfaces'

export type { BaseTransformParams }

/**
 * Parameters for missing value handling
 */
export interface MissingValueParams extends BaseTransformParams {
  /** Strategy to use for filling missing values */
  strategy: 'forward' | 'backward' | 'interpolate' | 'value'

  /** Custom value to use when strategy is 'value' */
  fillValue?: number

  /** Maximum number of consecutive values to fill */
  maxFillGap?: number

  /** Fields to check for missing values (default: all numeric fields) */
  fields?: string[]
}

/**
 * Parameters for timeframe aggregation
 */
export interface TimeframeAggregationParams extends BaseTransformParams {
  /** Target timeframe (e.g., '5m', '1h', '4h', '1d') */
  targetTimeframe: string

  /** Whether to align timestamps to market open */
  alignToMarketOpen?: boolean

  /** Market open time in HH:MM format (default: '09:30') */
  marketOpenTime?: string

  /** Market timezone (default: 'America/New_York') */
  marketTimezone?: string

  /** How to handle incomplete bars at the end */
  incompleteBarBehavior?: 'emit' | 'drop'
}

/**
 * Parameters for log returns calculation
 */
export interface LogReturnsParams extends BaseTransformParams {
  /** Base for logarithm calculation */
  base?: 'natural' | 'log10'

  /** Field to calculate returns on (default: 'close') */
  // priceField?: 'open' | 'high' | 'low' | 'close'

  /** Name of the output field */
  // outputField?: string
}

/**
 * Parameters for z-score normalization
 */
export interface ZScoreParams extends BaseTransformParams {
  /** Window size for rolling z-score (null for global) */
  windowSize?: number | null

  /** Fields to normalize (default: all numeric fields) */
  // fields?: string[]
}

/**
 * Parameters for min-max normalization
 */
export interface MinMaxParams extends BaseTransformParams {
  /** Window size for rolling min-max (null for global) */
  windowSize?: number | null

  /** Target range for normalization */
  min?: number
  max?: number

  /** Fields to normalize (default: all numeric fields) */
  // fields?: string[]
}

/**
 * Parameters for price calculations
 */
export interface PriceCalcParams extends BaseTransformParams {
  /** Type of price calculation */
  calculation: 'hlc3' | 'ohlc4' | 'typical' | 'weighted' | 'median' | 'custom'

  /** Custom formula when calculation is 'custom' */
  customFormula?: string

  /** Name of the output field */
  outputField?: string

  /** Whether to keep original OHLC fields */
  keepOriginal?: boolean
}

/**
 * Parameters for percent change calculation
 */
export interface PercentChangeParams extends BaseTransformParams {
  /** Number of periods to look back */
  periods?: number

  /** Field to calculate percent change on */
  // priceField?: 'open' | 'high' | 'low' | 'close'

  /** Name of the output field */
  // outputField?: string
}
}
