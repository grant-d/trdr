/**
 * Coinbase-specific type definitions
 */

/**
 * Configuration options for Coinbase provider
 */
export interface CoinbaseConfig {
  /** API key for authentication */
  apiKey?: string

  /** API secret for authentication */
  apiSecret?: string

  /** Whether to use sandbox/test environment */
  sandbox?: boolean

  /** Rate limit per second */
  rateLimitPerSecond?: number

  /** Maximum number of retries for failed requests */
  maxRetries?: number

  /** Delay between retries in milliseconds */
  retryDelayMs?: number
}

/**
 * WebSocket message types
 */
export enum WebSocketMessageType {
  SUBSCRIBE = 'subscribe',
  UNSUBSCRIBE = 'unsubscribe',
  HEARTBEAT = 'heartbeat',
  TICKER = 'ticker',
  CANDLES = 'candles',
  ERROR = 'error',
}

/**
 * WebSocket subscription message
 */
export interface WebSocketSubscribeMessage {
  type: WebSocketMessageType.SUBSCRIBE
  channels: string[]
  product_ids: string[]
}

/**
 * WebSocket candle update
 */
export interface WebSocketCandleUpdate {
  type: WebSocketMessageType.CANDLES
  product_id: string
  time: string
  open: string
  high: string
  low: string
  close: string
  volume: string
}
