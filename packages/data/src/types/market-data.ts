// Re-export market data types from @trdr/types
export type { Candle } from '@trdr/types'

// Import base types from shared
import type { PriceTick as BasePriceTick, EpochDate } from '@trdr/shared'

/**
 * Extended price tick interface for database storage
 */
export interface PriceTick extends Omit<BasePriceTick, 'timestamp'> {
  /** Timestamp as Date object */
  readonly timestamp: EpochDate
  /** Bid price */
  readonly bid?: number
  /** Ask price */
  readonly ask?: number
  /** Bid size */
  readonly bidSize?: number
  /** Ask size */
  readonly askSize?: number
}