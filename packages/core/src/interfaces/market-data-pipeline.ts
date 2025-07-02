import type { Candle, PriceTick } from '@trdr/shared'

/**
 * Configuration for market data feed
 */
export interface DataFeedConfig {
  /** Trading symbol (e.g., 'BTC-USD') */
  readonly symbol: string
  /** Feed type identifier */
  readonly feedType: 'coinbase' | 'backtest' | 'paper'
  /** Optional API credentials */
  readonly apiKey?: string
  readonly apiSecret?: string
  /** Optional passphrase for Coinbase Pro */
  readonly passphrase?: string
  /** WebSocket endpoint override */
  readonly wsEndpoint?: string
  /** REST API endpoint override */
  readonly restEndpoint?: string
  /** Enable debug logging */
  readonly debug?: boolean
}

/**
 * Market data event types
 */
export interface MarketDataEvents {
  /** Emitted when a new candle is received */
  candle: (candle: Candle) => void
  /** Emitted when a price tick is received */
  tick: (tick: PriceTick) => void
  /** Emitted when the connection is established */
  connected: () => void
  /** Emitted when the connection is lost */
  disconnected: (reason: string) => void
  /** Emitted when an error occurs */
  error: (error: Error) => void
  /** Emitted when connection is being retried */
  reconnecting: (attempt: number) => void
}

/**
 * Historical data request parameters
 */
export interface HistoricalDataRequest {
  /** Trading symbol */
  readonly symbol: string
  /** Start time for historical data */
  readonly start: Date
  /** End time for historical data */
  readonly end: Date
  /** Candle interval (e.g., '1m', '5m', '1h') */
  readonly interval?: string
  /** Maximum number of candles to return */
  readonly limit?: number
}

/**
 * Market data pipeline interface for all data feed implementations
 */
export interface MarketDataPipeline {
  /**
   * Subscribe to real-time market data for specified symbols
   * @param symbols - Array of trading symbols to subscribe to
   */
  subscribe(symbols: string[]): Promise<void>

  /**
   * Unsubscribe from market data for specified symbols
   * @param symbols - Array of trading symbols to unsubscribe from
   */
  unsubscribe(symbols: string[]): Promise<void>

  /**
   * Get historical market data
   * @param request - Historical data request parameters
   * @returns Array of historical candles
   */
  getHistorical(request: HistoricalDataRequest): Promise<Candle[]>

  /**
   * Get current price for a symbol
   * @param symbol - Trading symbol
   * @returns Current price
   */
  getCurrentPrice(symbol: string): Promise<number>

  /**
   * Start the data feed connection
   */
  start(): Promise<void>

  /**
   * Stop the data feed connection
   */
  stop(): Promise<void>

  /**
   * Check if the data feed is connected and healthy
   */
  isHealthy(): boolean

  /**
   * Get connection statistics
   */
  getStats(): ConnectionStats
}

/**
 * Connection statistics for monitoring
 */
export interface ConnectionStats {
  /** Is currently connected */
  readonly connected: boolean
  /** Connection uptime in milliseconds */
  readonly uptime: number
  /** Number of reconnection attempts */
  readonly reconnectAttempts: number
  /** Last error message if any */
  readonly lastError?: string
  /** Timestamp of last successful message */
  readonly lastMessageTime?: Date
  /** Number of messages received */
  readonly messagesReceived: number
  /** Current subscribed symbols */
  readonly subscribedSymbols: string[]
}
