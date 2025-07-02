import type { EpochDate, StockSymbol } from '@trdr/shared'

/**
 * Represents a market data candle (OHLCV)
 * Use for price charting and technical analysis
 */
export interface Candle {
  /** Trading symbol (e.g. 'BTC-USD') */
  readonly symbol: StockSymbol
  /** Candle interval (e.g. '1m', '1h') */
  readonly interval: string
  /** Candle open timestamp (epoch) */
  readonly timestamp: EpochDate
  /** Candle open time (epoch) */
  readonly openTime: EpochDate
  /** Candle close time (epoch) */
  readonly closeTime: EpochDate
  /** Open price */
  readonly open: number
  /** High price */
  readonly high: number
  /** Low price */
  readonly low: number
  /** Close price */
  readonly close: number
  /** Volume traded */
  readonly volume: number
  /** Quote volume (optional) */
  readonly quoteVolume?: number
  /** Number of trades (optional) */
  readonly tradesCount?: number
  /** Taker buy volume (optional) */
  readonly takerBuyVolume?: number
  /** Taker buy quote volume (optional) */
  readonly takerBuyQuoteVolume?: number
  /** Bid price (optional) */
  readonly bid?: number
  /** Ask price (optional) */
  readonly ask?: number
}
