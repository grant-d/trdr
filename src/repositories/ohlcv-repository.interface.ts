import type { OhlcvDto } from '../models/ohlcv.dto'

/**
 * Query parameters for filtering OHLCV data
 */
export interface OhlcvQuery {
  /** Start timestamp (inclusive) in milliseconds */
  startTime?: number
  /** End timestamp (inclusive) in milliseconds */
  endTime?: number
  /** Filter by symbol (optional) */
  symbol?: string
  /** Filter by exchange (optional) */
  exchange?: string
  /** Limit number of results (optional) */
  limit?: number
  /** Offset for pagination (optional) */
  offset?: number
}

/**
 * Coefficient data for storing calculation results
 */
export interface CoefficientData {
  /** Unique identifier for the coefficient */
  name: string
  /** The coefficient value */
  value: number
  /** Optional metadata associated with the coefficient */
  metadata?: Record<string, unknown>
  /** Timestamp when the coefficient was calculated */
  timestamp: number
  /** Symbol this coefficient relates to (optional) */
  symbol?: string
  /** Exchange this coefficient relates to (optional) */
  exchange?: string
}

/**
 * Attached database configuration for SQLite schemas
 */
export interface AttachedDatabase {
  /** Schema name to use for this database */
  schema: string
  /** File path to the database */
  path: string
}

/**
 * Configuration options for repository initialization
 */
export interface RepositoryConfig {
  /** Connection string or file path */
  connectionString: string
  /** Additional configuration options specific to the repository type */
  options?: Record<string, unknown>
  /** Attached databases for SQLite schema support */
  attachedDatabases?: AttachedDatabase[]
  /** Default schema to use for tables (defaults to 'main') */
  defaultSchema?: string
}

/**
 * Repository interface for OHLCV data storage and retrieval
 * Supports multiple storage backends (SQLite, CSV, Parquet) with the same interface
 */
export interface OhlcvRepository {
  /**
   * Initialize the repository and set up storage backend
   * @param config Repository configuration
   */
  initialize(config: RepositoryConfig): Promise<void>

  /**
   * Save a single OHLCV record
   * @param data OHLCV data to save
   */
  save(data: OhlcvDto): Promise<void>

  /**
   * Save multiple OHLCV records in a batch operation
   * @param data Array of OHLCV data to save
   */
  saveMany(data: OhlcvDto[]): Promise<void>

  /**
   * Append a batch of OHLCV records (optimized for streaming writes)
   * @param data Array of OHLCV data to append
   */
  appendBatch(data: OhlcvDto[]): Promise<void>

  /**
   * Get OHLCV data within a specific date range
   * @param startTime Start timestamp in milliseconds (inclusive)
   * @param endTime End timestamp in milliseconds (inclusive)
   * @param symbol Optional symbol filter
   * @param exchange Optional exchange filter
   * @returns Array of OHLCV data
   */
  getBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<OhlcvDto[]>

  /**
   * Get OHLCV data for a specific symbol
   * @param symbol Symbol to filter by
   * @param exchange Optional exchange filter
   * @param limit Optional limit on number of results
   * @param offset Optional offset for pagination
   * @returns Array of OHLCV data
   */
  getBySymbol(
    symbol: string,
    exchange?: string,
    limit?: number,
    offset?: number
  ): Promise<OhlcvDto[]>

  /**
   * Get OHLCV data using flexible query parameters
   * @param query Query parameters for filtering
   * @returns Array of OHLCV data
   */
  query(query: OhlcvQuery): Promise<OhlcvDto[]>

  /**
   * Get the most recent timestamp for a specific symbol
   * Useful for resuming data collection from the last known point
   * @param symbol Symbol to check
   * @param exchange Optional exchange filter
   * @returns Latest timestamp in milliseconds, or null if no data exists
   */
  getLastTimestamp(symbol: string, exchange?: string): Promise<number | null>

  /**
   * Get the earliest timestamp for a specific symbol
   * @param symbol Symbol to check
   * @param exchange Optional exchange filter
   * @returns Earliest timestamp in milliseconds, or null if no data exists
   */
  getFirstTimestamp(symbol: string, exchange?: string): Promise<number | null>

  /**
   * Get count of records for a symbol
   * @param symbol Symbol to count
   * @param exchange Optional exchange filter
   * @returns Number of records
   */
  getCount(symbol: string, exchange?: string): Promise<number>

  /**
   * Save a coefficient value
   * @param coefficient Coefficient data to save
   */
  saveCoefficient(coefficient: CoefficientData): Promise<void>

  /**
   * Save multiple coefficient values in a batch
   * @param coefficients Array of coefficient data to save
   */
  saveCoefficients(coefficients: CoefficientData[]): Promise<void>

  /**
   * Get a coefficient value by name
   * @param name Name of the coefficient
   * @param symbol Optional symbol filter
   * @param exchange Optional exchange filter
   * @returns Coefficient data or null if not found
   */
  getCoefficient(
    name: string,
    symbol?: string,
    exchange?: string
  ): Promise<CoefficientData | null>

  /**
   * Get multiple coefficients by name pattern
   * @param namePattern Pattern to match coefficient names (supports wildcards)
   * @param symbol Optional symbol filter
   * @param exchange Optional exchange filter
   * @returns Array of coefficient data
   */
  getCoefficients(
    namePattern?: string,
    symbol?: string,
    exchange?: string
  ): Promise<CoefficientData[]>

  /**
   * Delete coefficients by name pattern
   * @param namePattern Pattern to match coefficient names
   * @param symbol Optional symbol filter
   * @param exchange Optional exchange filter
   * @returns Number of deleted coefficients
   */
  deleteCoefficients(
    namePattern: string,
    symbol?: string,
    exchange?: string
  ): Promise<number>

  /**
   * Get all unique symbols in the repository
   * @param exchange Optional exchange filter
   * @returns Array of unique symbols
   */
  getSymbols(exchange?: string): Promise<string[]>

  /**
   * Get all unique exchanges in the repository
   * @returns Array of unique exchanges
   */
  getExchanges(): Promise<string[]>

  /**
   * Delete OHLCV data within a specific date range
   * @param startTime Start timestamp in milliseconds (inclusive)
   * @param endTime End timestamp in milliseconds (inclusive)
   * @param symbol Optional symbol filter
   * @param exchange Optional exchange filter
   * @returns Number of deleted records
   */
  deleteBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<number>

  /**
   * Check if the repository is properly initialized and ready to use
   * @returns true if ready, false otherwise
   */
  isReady(): boolean

  /**
   * Get repository statistics and health information
   * @returns Repository statistics
   */
  getStats(): Promise<{
    totalRecords: number
    uniqueSymbols: number
    uniqueExchanges: number
    dataDateRange: {
      earliest: number | null
      latest: number | null
    }
    storageSize?: number
  }>

  /**
   * Flush any pending writes to ensure data persistence
   */
  flush(): Promise<void>

  /**
   * Close the repository and clean up resources
   */
  close(): Promise<void>
}

/**
 * Repository error types
 */
export class RepositoryError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly cause?: Error
  ) {
    super(message)
    this.name = 'RepositoryError'
  }
}

/**
 * Connection error when repository cannot connect to storage backend
 */
export class RepositoryConnectionError extends RepositoryError {
  constructor(message: string, cause?: Error) {
    super(message, 'CONNECTION_ERROR', cause)
    this.name = 'RepositoryConnectionError'
  }
}

/**
 * Validation error when data doesn't meet repository requirements
 */
export class RepositoryValidationError extends RepositoryError {
  constructor(message: string, cause?: Error) {
    super(message, 'VALIDATION_ERROR', cause)
    this.name = 'RepositoryValidationError'
  }
}

/**
 * Storage error when there are issues with the underlying storage
 */
export class RepositoryStorageError extends RepositoryError {
  constructor(message: string, cause?: Error) {
    super(message, 'STORAGE_ERROR', cause)
    this.name = 'RepositoryStorageError'
  }
}