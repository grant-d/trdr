import type { FileProviderConfig } from '../providers'
import type { TransformConfig } from './transform.interface'

/**
 * Output configuration for the pipeline
 */
export interface OutputConfig {
  /** Output file path */
  path: string
  /** Output format */
  format: 'csv' | 'jsonl' | 'sqlite'
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
}

/**
 * Complete pipeline configuration
 */
export interface PipelineConfig {
  /** Input data source configuration */
  input: FileProviderConfig

  /** Output configuration */
  output: OutputConfig

  /** Array of transformations to apply */
  transformations: TransformConfig[]

  /** Processing options */
  options: ProcessingOptions

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
