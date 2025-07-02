import type { Candle as BaseCandle, PriceTick as BasePriceTick } from '@trdr/shared'

/**
 * Extended candle interface for database storage
 */
export interface Candle extends Omit<BaseCandle, 'timestamp'> {
  /** Trading pair symbol */
  readonly symbol: string
  /** Time interval (e.g., '1h', '4h', '1d') */
  readonly interval: string
  /** Timestamp as Date object */
  readonly timestamp: Date
  /** Opening time as Date object */
  readonly openTime: Date
  /** Closing time as Date object */
  readonly closeTime: Date
  /** Quote asset volume */
  readonly quoteVolume?: number
  /** Number of trades in this period */
  readonly tradesCount?: number
}

/**
 * Extended price tick interface for database storage
 */
export interface PriceTick extends Omit<BasePriceTick, 'timestamp'> {
  /** Timestamp as Date object */
  readonly timestamp: Date
  /** Bid price */
  readonly bid?: number
  /** Ask price */
  readonly ask?: number
  /** Bid size */
  readonly bidSize?: number
  /** Ask size */
  readonly askSize?: number
}