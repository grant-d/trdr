import type { DataProviderConfig } from '../../interfaces/data-provider.interface'

/**
 * Column mapping configuration for file providers
 */
export interface ColumnMapping {
  timestamp: string
  open: string
  high: string
  low: string
  close: string
  volume: string
  symbol?: string
  exchange?: string
}

/**
 * Configuration for file-based data providers
 */
export interface FileProviderConfig extends DataProviderConfig {
  /** Path to the data file */
  path: string
  /** File format (auto-detected if not specified) */
  format?: 'csv' | 'parquet'
  /** Column mapping configuration */
  columnMapping?: ColumnMapping
  /** Chunk size for streaming (number of rows) */
  chunkSize?: number
  /** Default exchange name for data */
  exchange?: string
  /** Default symbol for data (if not in file) */
  symbol?: string
  /** CSV-specific: delimiter character */
  delimiter?: string
}