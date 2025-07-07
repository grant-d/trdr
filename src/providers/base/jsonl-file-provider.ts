import type { HistoricalParams } from '../../interfaces'
import logger from '../../utils/logger'
import { FileProvider } from './file-provider.base'
import type { FileProviderConfig } from './types'

/**
 * JSONL file provider implementation
 * Reads JSONL data and pushes directly to buffer
 */
export class JsonlFileProvider extends FileProvider {
  private pipeline?: { next(from: number, to: number): void }

  constructor(config: FileProviderConfig) {
    super(config)
  }

  /**
   * Set the pipeline to notify when rows are added
   */
  setPipeline(pipeline: { next(): void }): void {
    this.pipeline = pipeline
  }

  /**
   * Process historical data from JSONL file
   */
  async processHistoricalData(params: HistoricalParams): Promise<void> {
    if (!this.connected) {
      throw new Error('Provider not connected. Call connect() first.')
    }

    logger.info('Processing JSONL data', {
      path: this.filePath,
      symbols: params.symbols,
      start: new Date(params.start).toISOString(),
      end: new Date(params.end).toISOString()
    })

    await this.processFile(this.pipeline as any)
  }

  /**
   * Process a single JSONL line
   */
  protected processLine(line: string): boolean {
    try {
      const rawData = JSON.parse(line)

      // Extract values using column mapping
      const timestamp = this.parseTimestamp(rawData[this.columnMapping.timestamp])
      const open = this.parseNumber(rawData[this.columnMapping.open], 'open')
      const high = this.parseNumber(rawData[this.columnMapping.high], 'high')
      const low = this.parseNumber(rawData[this.columnMapping.low], 'low')
      const close = this.parseNumber(rawData[this.columnMapping.close], 'close')
      const volume = this.parseNumber(rawData[this.columnMapping.volume], 'volume')

      // Push to buffer
      this._buffer!.push({
        timestamp,
        open,
        high,
        low,
        close,
        volume
      })

      return true
    } catch (error) {
      logger.warn('Failed to parse JSON line', {
        error: error instanceof Error ? error.message : String(error),
        line: line.substring(0, 100)
      })
      return false
    }
  }

  /**
   * Parse timestamp helper
   */
  protected parseTimestamp(value: unknown): number {
    return super.parseTimestamp(value, 0)
  }

  /**
   * Parse number helper
   */
  protected parseNumber(value: unknown, field: string): number {
    return super.parseNumber(value, field, 0)
  }

  /**
   * Required by FileProvider interface - just processes the data
   */
  async* getHistoricalData(
    params: HistoricalParams
  ): AsyncIterableIterator<any> {
    // Process all data into buffer
    await this.processHistoricalData(params)

    // Don't yield anything - data is in buffer
    return
  }
}
