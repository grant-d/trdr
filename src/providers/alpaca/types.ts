/**
 * Alpaca-specific type definitions
 */

/**
 * Configuration options for Alpaca provider
 */
export interface AlpacaConfig {
  /** API key for authentication */
  apiKey?: string

  /** API secret for authentication */
  apiSecret?: string

  /** Whether to use paper trading (default: true based on ALPACA_PAPER env) */
  paper?: boolean

  /** Rate limit per second (default: 200) */
  rateLimitPerSecond?: number

  /** Maximum number of retries for failed requests */
  maxRetries?: number

  /** Delay between retries in milliseconds */
  retryDelayMs?: number

  /** Maximum retry delay in milliseconds */
  maxRetryDelayMs?: number

  /** Backoff multiplier for exponential backoff */
  backoffMultiplier?: number
}

/**
 * Alpaca bar data structure
 */
export interface AlpacaBar {
  /** Symbol */
  S: string

  /** Timestamp (ISO string) */
  t: string

  /** Open price */
  o: number

  /** High price */
  h: number

  /** Low price */
  l: number

  /** Close price */
  c: number

  /** Volume */
  v: number

  /** Number of trades */
  n?: number

  /** Volume weighted average price */
  vw?: number
}

/**
 * Alpaca WebSocket message types
 */
export enum AlpacaMessageType {
  SUCCESS = 'success',
  ERROR = 'error',
  SUBSCRIPTION = 'subscription',
  TRADES = 't',
  QUOTES = 'q',
  BARS = 'b',
}

/**
 * Alpaca WebSocket connection states
 */
export enum AlpacaConnectionState {
  CONNECTING = 'connecting',
  AUTHENTICATING = 'authenticating',
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  AUTH_FAILED = 'auth_failed',
}
