import type { EpochDate } from '@trdr/shared'

/**
 * Base interface for all indicator results
 */
export interface IndicatorResult {
  /** The calculated value */
  readonly value: number
  
  /** Timestamp when indicator was calculated */
  readonly timestamp: EpochDate
  
  /** Additional values for multi-value indicators */
  readonly values?: Record<string, number>
  
  /** Any metadata about the calculation */
  readonly metadata?: Record<string, unknown>
}