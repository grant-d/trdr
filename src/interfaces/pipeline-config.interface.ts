import type { FileProviderConfig } from '../providers'
import type { TransformConfig } from './transform.interface'

/**
 * Input configuration for file-based data sources
 */
export interface FileInputConfig extends FileProviderConfig {
  /** Input type */
  type: 'file'
}

/**
 * Input configuration for provider-based data sources
 */
export interface ProviderInputConfig {
  /** Input type */
  type: 'provider'
  /** Provider name (e.g., 'coinbase', 'alpaca') */
  provider: 'coinbase' | 'alpaca'
  /** Trading symbols to fetch */
  symbols: string[]
  /** Timeframe for the data (e.g., '1m', '5m', '1h') */
  timeframe: string
  /** Duration for data fetching */
  duration?: string // e.g., '1h', '1000bars', 'continuous'
}

/**
 * Union type for all input configurations
 */
export type InputConfig = FileInputConfig | ProviderInputConfig

/**
 * Output configuration for the pipeline
 */
export interface OutputConfig {
  /** Output file path */
  path: string
  /** Output format */
  format: 'csv' | 'jsonl'
  /** Whether to overwrite existing files */
  overwrite?: boolean
  /** Column mapping for output (if different from default) */
  columnMapping?: Record<string, string>
}

/**
 * Processing options for the pipeline
 */
export interface ProcessingOptions {
  /** Number of rows to process in each chunk */
  chunkSize?: number
  /** Whether to continue processing on errors */
  continueOnError?: boolean
  /** Maximum number of errors before stopping */
  maxErrors?: number
  /** Whether to show progress indicators */
  showProgress?: boolean
  /** Whether to validate data during processing */
  validateData?: boolean
  /** Input timezone for non-UTC data */
  inputTimezone?: string
  /** Whether to backfill gaps on startup */
  backfillOnStartup?: boolean
  /** Maximum backfill window (e.g., '24h', '7d') */
  maxBackfillWindow?: string
  /** Retry configuration for provider connections */
  retryConfig?: {
    maxRetries: number
    backoffMultiplier: number
    maxBackoffSeconds: number
  }
}

/**
 * Complete pipeline configuration
 */
export interface PipelineConfig {
  /** Input data source configuration */
  input: InputConfig

  /** Output configuration */
  output: OutputConfig

  /** Array of transformations to apply */
  transformations: TransformConfig[]

  /** Processing options */
  options?: ProcessingOptions

  /** Pipeline metadata */
  metadata?: {
    /** Pipeline name/description */
    name?: string
    /** Version of the pipeline */
    version?: string
    /** Pipeline description */
    description?: string
    /** Author information */
    author?: string
    /** Creation timestamp */
    created?: string
    /** Last modified timestamp */
    modified?: string
  }
}

/**
 * Type guard to check if value is a valid PipelineConfig
 */
export function isPipelineConfig(val: unknown): val is PipelineConfig {
  if (!val || typeof val !== 'object') {
    return false
  }

  const obj = val as Record<string, unknown>

  return (
    'input' in obj &&
    'output' in obj &&
    'transformations' in obj &&
    typeof obj.input === 'object' &&
    typeof obj.output === 'object' &&
    Array.isArray(obj.transformations)
  )
}
