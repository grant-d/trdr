import type { OhlcvDto } from '../models'

/**
 * Parameters for fetching historical data
 */
export interface HistoricalParams {
  /** Trading pair symbols to fetch (e.g., ['BTC-USD', 'ETH-USD']) */
  symbols: string[]

  /** Start timestamp in Unix milliseconds (UTC) */
  start: number

  /** End timestamp in Unix milliseconds (UTC) */
  end: number

  /** Timeframe/interval for the data (e.g., '1m', '5m', '1h') */
  timeframe: string
}

/**
 * Parameters for subscribing to real-time data
 */
export interface RealtimeParams {
  /** Trading pair symbols to subscribe to */
  symbols: string[]

  /** Timeframe/interval for the data */
  timeframe: string
}

/**
 * Configuration options for data providers
 */
export type DataProviderConfig = Record<string, unknown>

/**
 * Interface that all data providers must implement
 * Provides a consistent API for fetching market data from different sources
 */
export interface DataProvider {
  /** Unique name identifier for the provider */
  readonly name: string

  /**
   * Establishes connection to the data source
   * Should validate credentials and prepare for data fetching
   * @throws Error if connection fails or credentials are invalid
   */
  connect(): Promise<void>

  /**
   * Closes connection to the data source
   * Should clean up any resources or connections
   */
  disconnect(): Promise<void>

  /**
   * Fetches historical OHLCV data for the specified parameters
   * @param params Parameters specifying what data to fetch
   * @returns Async iterator that yields OHLCV data points
   * @throws Error if parameters are invalid or fetch fails
   */
  getHistoricalData(params: HistoricalParams): AsyncIterableIterator<OhlcvDto>

  /**
   * Subscribes to real-time OHLCV data updates
   * @param params Parameters specifying what data to subscribe to
   * @returns Async iterator that yields OHLCV data as it arrives
   * @throws Error if subscription fails or is not supported
   */
  subscribeRealtime(params: RealtimeParams): AsyncIterableIterator<OhlcvDto>

  /**
   * Returns list of environment variables required by this provider
   * Used for validation before attempting to connect
   * @returns Array of environment variable names (e.g., ['COINBASE_API_KEY'])
   */
  getRequiredEnvVars(): string[]

  /**
   * Checks if the provider is currently connected
   * @returns true if connected, false otherwise
   */
  isConnected(): boolean

  /**
   * Gets the list of supported timeframes for this provider
   * @returns Array of supported timeframe strings (e.g., ['1m', '5m', '1h'])
   */
  getSupportedTimeframes(): string[]

  /**
   * Validates that the required environment variables are set
   * @throws Error if any required environment variables are missing
   */
  validateEnvVars(): void
}
