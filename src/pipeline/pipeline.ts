import type { Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import type { FileProvider } from '../providers'
import type { OhlcvRepository } from '../repositories'
import { TransformSerializer } from '../transforms'

/**
 * Pipeline options
 */
export interface PipelineOptions {
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
 * Pipeline metadata
 */
export interface PipelineMetadata {
  /** Pipeline name */
  name?: string
  /** Pipeline version */
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

/**
 * Pipeline configuration
 */
export interface PipelineConfig {
  /** Data provider */
  provider: FileProvider
  /** Optional transform or transform pipeline */
  transform?: Transform
  /** Output repository */
  repository: OhlcvRepository
  /** Processing options */
  options?: PipelineOptions
  /** Pipeline metadata */
  metadata?: PipelineMetadata
}

/**
 * Pipeline execution result
 */
export interface PipelineResult {
  /** Number of records processed */
  recordsProcessed: number
  /** Number of records written */
  recordsWritten: number
  /** Number of errors encountered */
  errors: number
  /** Execution time in milliseconds */
  executionTime: number
  /** Transform coefficients if available */
  coefficients?: any[]
}

/**
 * Progress callback function
 */
export type ProgressCallback = (progress: {
  current: number
  total?: number
  percentage?: number
  message?: string
}) => void

/**
 * Main Pipeline class that orchestrates data flow
 * from provider through transforms to repository
 */
export class Pipeline {
  private readonly config: PipelineConfig
  private readonly options: Required<PipelineOptions>
  private progressCallback?: ProgressCallback
  private errorCount = 0

  constructor(config: PipelineConfig) {
    this.config = config
    this.options = {
      chunkSize: config.options?.chunkSize ?? 1000,
      continueOnError: config.options?.continueOnError ?? false,
      maxErrors: config.options?.maxErrors ?? 100,
      showProgress: config.options?.showProgress ?? true,
    }
  }

  /**
   * Set progress callback
   */
  public onProgress(callback: ProgressCallback): Pipeline {
    this.progressCallback = callback
    return this
  }

  /**
   * Execute the pipeline
   */
  public async execute(): Promise<PipelineResult> {
    const startTime = Date.now()
    let recordsProcessed = 0
    let recordsWritten = 0
    const coefficients: any[] = []

    try {
      // Connect provider if not already connected
      if (!this.config.provider.isConnected()) {
        await this.config.provider.connect()
      }

      // Get data stream from provider
      const dataStream = this.config.provider.getHistoricalData({
        symbols: [], // Empty array means all symbols
        start: 0,
        end: Date.now(),
        timeframe: '1m',
      })

      // Apply transform if provided
      let processedStream: AsyncIterableIterator<OhlcvDto>
      if (this.config.transform) {
        const result = await this.config.transform.apply(dataStream)
        // Convert AsyncIterator to AsyncIterableIterator if needed
        processedStream = result.data as AsyncIterableIterator<OhlcvDto>

        // Collect coefficients if available
        if (result.coefficients) {
          coefficients.push(result.coefficients)
        }
      } else {
        processedStream = dataStream
      }

      // Process data in chunks
      const chunk: OhlcvDto[] = []

      for await (const record of processedStream) {
        recordsProcessed++
        chunk.push(record)

        // Write chunk when it reaches the configured size
        if (chunk.length >= this.options.chunkSize) {
          try {
            await this.config.repository.appendBatch(chunk)
            recordsWritten += chunk.length
            chunk.length = 0

            // Report progress
            this.reportProgress(recordsProcessed)
          } catch (error) {
            this.handleError(error)
          }
        }
      }

      // Write remaining records
      if (chunk.length > 0) {
        try {
          await this.config.repository.appendBatch(chunk)
          recordsWritten += chunk.length
        } catch (error) {
          this.handleError(error)
        }
      }

      // Flush repository to ensure all data is persisted
      await this.config.repository.flush()

      // Store coefficients if repository supports it
      if (coefficients.length > 0 && this.config.transform) {
        await this.storeCoefficients(coefficients)
      }

      const executionTime = Date.now() - startTime

      // Disconnect provider
      await this.config.provider.disconnect()

      return {
        recordsProcessed,
        recordsWritten,
        errors: this.errorCount,
        executionTime,
        coefficients,
      }
    } catch (error) {
      // Make sure to disconnect on error
      try {
        await this.config.provider.disconnect()
      } catch {
        // Ignore disconnect errors
      }
      
      throw new Error(
        `Pipeline execution failed: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`,
      )
    }
  }

  /**
   * Handle errors during processing
   */
  private handleError(error: unknown): void {
    this.errorCount++

    if (!this.options.continueOnError || this.errorCount >= this.options.maxErrors) {
      throw error
    }

    // Log error but continue processing
    console.error('Pipeline error:', error)
  }

  /**
   * Report progress
   */
  private reportProgress(current: number): void {
    if (!this.options.showProgress || !this.progressCallback) {
      return
    }

    this.progressCallback({
      current,
      message: `Processing record ${current}`,
    })
  }

  /**
   * Store transform coefficients in repository
   */
  private async storeCoefficients(coefficients: any[]): Promise<void> {
    if (!this.config.transform) return

    // Use imported TransformSerializer

    for (const coeff of coefficients) {
      if (!coeff) continue

      const coeffData = TransformSerializer.coefficientsToRepositoryFormat(
        coeff,
        this.config.transform.name || 'transform',
      )

      await this.config.repository.saveCoefficients(coeffData)
    }
  }

  /**
   * Get pipeline metadata
   */
  public getMetadata(): PipelineMetadata | undefined {
    return this.config.metadata
  }

  /**
   * Get pipeline configuration
   */
  public getConfig(): Readonly<PipelineConfig> {
    return this.config
  }

  /**
   * Check if pipeline has transforms
   */
  public hasTransforms(): boolean {
    return this.config.transform !== undefined
  }

  /**
   * Get transform information
   */
  public getTransformInfo(): { name: string; type: string } | null {
    if (!this.config.transform) return null

    return {
      name: this.config.transform.name,
      type: this.config.transform.type,
    }
  }
}
