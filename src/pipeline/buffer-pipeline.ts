import type { Transform } from '../interfaces'
import type { FileProvider } from '../providers'
import type { OhlcvRepository } from '../repositories'
import type { DataBuffer } from '../utils'
import logger from '../utils/logger'

/**
 * Buffer-based pipeline configuration
 */
export interface BufferPipelineConfig {
  /** Data provider that creates and manages buffers */
  provider: FileProvider;
  /** Transforms to apply in sequence */
  transforms: Transform[];
  /** Output repository */
  repository: OhlcvRepository;
  /** Number of rows to load per batch */
  batchSize?: number;
  /** The initial buffer created by the provider */
  initialBuffer: DataBuffer;
  /** The final buffer after all transforms (may be same as initial) */
  finalBuffer: DataBuffer;
}

/**
 * Pipeline result statistics
 */
export interface BufferPipelineResult {
  recordsProcessed: number;
  recordsWritten: number;
  errors: number;
  executionTime: number;
}

/**
 * Buffer-based pipeline that processes data in batches
 * Provider loads batches of data, pipeline runs transforms, repository writes results
 */
export class BufferPipeline {
  private readonly config: BufferPipelineConfig
  private recordsProcessed = 0
  private recordsWritten = 0
  private errors = 0

  constructor(config: BufferPipelineConfig) {
    this.config = config
  }

  /**
   * Call next() on each transform to process the batch
   */
  public next(from: number, to: number): void {
    // Tell each transform to process the window (from, to) on their internal buffer
    // The provider has already appended new data to the buffers
    for (const transform of this.config.transforms) {
      transform.next(from, to)
    }

    this.recordsProcessed += to - from
  }

  /**
   * Get the final buffer from the pipeline
   */
  private getBuffer(): DataBuffer {
    return this.config.finalBuffer
  }

  /**
   * Flush buffer to repository
   */
  public async flush(): Promise<void> {
    const buffer = this.getBuffer()
    if (!buffer || buffer.length() === 0) {
      return
    }

    try {
      // Pop all rows from buffer and convert to OhlcvDto
      const ohlcvBatch: any[] = []
      while (buffer.length() > 0) {
        const row = buffer.pop()
        if (row) {
          // Convert Row to OhlcvDto
          const ohlcv: any = {
            timestamp: row.timestamp!,
            open: row.open!,
            high: row.high!,
            low: row.low!,
            close: row.close!,
            volume: row.volume!
          }

          // Add any additional fields from transforms
          for (const [key, value] of Object.entries(row)) {
            if (
              ![
                'timestamp',
                'open',
                'high',
                'low',
                'close',
                'volume'
              ].includes(key)
            ) {
              ohlcv[key] = value
            }
          }

          ohlcvBatch.push(ohlcv)
        }
      }

      if (ohlcvBatch.length > 0) {
        await this.config.repository.appendBatch(ohlcvBatch)
        this.recordsWritten += ohlcvBatch.length
      }
    } catch (error) {
      this.errors++
      logger.error('Error flushing buffer', { error })
      throw error
    }
  }

  /**
   * Execute the pipeline by processing all data from provider
   */
  public async execute(): Promise<BufferPipelineResult> {
    const startTime = Date.now()

    try {
      // Connect provider if needed
      if (!this.config.provider.isConnected()) {
        await this.config.provider.connect()
      }

      // Initialize repository
      if ('setExpectedOutputFields' in this.config.repository) {
        // Set expected fields based on buffer columns after transforms
        const buffer = this.getBuffer()
        if (buffer) {
          (this.config.repository as any).setExpectedOutputFields(
            buffer.getColumns()
          )
        }
      }

      // Set the pipeline on the provider so it can call next()
      if ('setPipeline' in this.config.provider) {
        (this.config.provider as any).setPipeline(this)
      }

      // Process all data from provider
      if ('processHistoricalData' in this.config.provider) {
        // New buffer-based providers
        await (this.config.provider as any).processHistoricalData({
          symbols: [],
          start: 0,
          end: Date.now(),
          timeframe: '1m'
        })
      } else {
        // Legacy streaming providers
        const dataStream = this.config.provider.getHistoricalData({
          symbols: [],
          start: 0,
          end: Date.now(),
          timeframe: '1m'
        })

        for await (const _row of dataStream) {
          this.next(0, 1)
        }
      }

      // Final flush
      await this.flush()

      // Ensure repository writes everything
      await this.config.repository.flush()

      // Disconnect provider
      await this.config.provider.disconnect()

      return {
        recordsProcessed: this.recordsProcessed,
        recordsWritten: this.recordsWritten,
        errors: this.errors,
        executionTime: Date.now() - startTime
      }
    } catch (error) {
      logger.error('Pipeline execution failed', { error })
      throw error
    }
  }

  /**
   * Get current pipeline statistics
   */
  public getStats(): {
    recordsProcessed: number;
    recordsWritten: number;
    errors: number;
    bufferSize: number;
  } {
    const buffer = this.getBuffer()
    return {
      recordsProcessed: this.recordsProcessed,
      recordsWritten: this.recordsWritten,
      errors: this.errors,
      bufferSize: buffer?.length() ?? 0
    }
  }
}
