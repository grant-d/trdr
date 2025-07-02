import type { EpochDate } from './dates'

/**
 * Represents a single candlestick in OHLCV format.
 * Used for price chart visualization and technical analysis.
 */
export interface Candle {
  /** Unix timestamp in milliseconds */
  readonly timestamp: EpochDate
  /** Opening price at the start of the period */
  readonly open: number
  /** Highest price during the period */
  readonly high: number
  /** Lowest price during the period */
  readonly low: number
  /** Closing price at the end of the period */
  readonly close: number
  /** Total volume traded during the period */
  readonly volume: number
}

/**
 * Represents the current state of buy and sell orders in the market.
 * Critical for understanding market depth and liquidity.
 */
export interface OrderBook {
  /** Unix timestamp in milliseconds when snapshot was taken */
  readonly timestamp: EpochDate
  /** Buy orders sorted by price (highest first) */
  readonly bids: readonly OrderBookLevel[]
  /** Sell orders sorted by price (lowest first) */
  readonly asks: readonly OrderBookLevel[]
}

/**
 * Represents a single price level in the order book.
 */
export interface OrderBookLevel {
  /** Price level */
  readonly price: number
  /** Total size/quantity available at this price */
  readonly size: number
}

/**
 * Represents a completed trade execution in the market.
 * Used for analyzing market activity and trade flow.
 */
export interface Trade {
  /** Unique trade identifier from exchange */
  readonly id: string
  /** Unix timestamp in milliseconds */
  readonly timestamp: EpochDate
  /** Execution price */
  readonly price: number
  /** Trade size/quantity */
  readonly size: number
  /** Trade direction from taker's perspective */
  readonly side: 'buy' | 'sell'
  /** Whether this trade was maker (true) or taker (false) */
  readonly maker: boolean
}

/**
 * Real-time market ticker data.
 * Provides current market prices and 24h volume.
 */
export interface Ticker {
  /** Trading pair symbol (e.g., 'BTC-USD') */
  readonly symbol: string
  /** Unix timestamp in milliseconds */
  readonly timestamp: EpochDate
  /** Best bid price */
  readonly bid: number
  /** Best ask price */
  readonly ask: number
  /** Last trade price */
  readonly last: number
  /** 24-hour trading volume */
  readonly volume24h: number
}

/**
 * Complete market data snapshot at a point in time.
 * Combines all market data types for comprehensive analysis.
 */
export interface MarketDataSnapshot {
  /** Trading pair symbol */
  readonly symbol: string
  /** Unix timestamp in milliseconds */
  readonly timestamp: EpochDate
  /** Current candle data */
  readonly candle: Candle
  /** Current order book state */
  readonly orderBook: OrderBook
  /** Current ticker data */
  readonly ticker: Ticker
}

/**
 * Supported time intervals for candlestick data.
 * Used for different trading timeframes and analysis.
 */
export type TimeInterval = '1m' | '5m' | '15m' | '1h' | '4h' | '1d'

/**
 * Represents a subscription to market data updates.
 * Used to manage real-time data feeds.
 */
export interface MarketDataSubscription {
  /** Unique subscription identifier */
  readonly id: string
  /** Trading pair to subscribe to */
  readonly symbol: string
  /** Type of market data to receive */
  readonly type: 'candle' | 'orderbook' | 'trade' | 'ticker'
  /** Time interval for candle subscriptions */
  readonly interval?: TimeInterval
}

/**
 * Unified market update event for data pipeline.
 * Provides type-safe handling of different market data types.
 */
export interface MarketUpdate {
  /** Type of market data update */
  readonly type: 'candle' | 'tick' | 'orderbook' | 'trade'
  /** Trading pair symbol */
  readonly symbol: string
  /** Unix timestamp in milliseconds */
  readonly timestamp: EpochDate
  /** Actual market data payload */
  readonly data: Candle | Ticker | OrderBook | Trade
}

/**
 * Lightweight price update for real-time monitoring.
 * More efficient than full ticker for high-frequency updates.
 */
export interface PriceTick {
  /** Trading pair symbol */
  readonly symbol: string
  /** Unix timestamp in milliseconds */
  readonly timestamp: EpochDate
  /** Current price */
  readonly price: number
  /** Optional volume at this price */
  readonly volume?: number
}

/**
 * Unified interface for market data access across all modes.
 * Abstracts data source differences between live, paper, and backtest.
 */
export interface MarketDataPipeline {
  /** Subscribe to market updates with callback */
  subscribe(callback: (data: MarketUpdate) => void): void
  /** Unsubscribe from market updates */
  unsubscribe(): void
  /** Get historical candle data for a time range */
  getHistorical(from: EpochDate, to: EpochDate): Promise<Candle[]>
  /** Get current market price */
  getCurrentPrice(): Promise<number>
}
