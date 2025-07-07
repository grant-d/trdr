import type { HistoricalParams } from '../../interfaces'
import logger from '../../utils/logger'
import { FileProvider } from './file-provider.base'
import type { FileProviderConfig } from './types'

/**
 * CSV file provider implementation
 * Reads CSV data and pushes directly to buffer
 */
export class CsvFileProvider extends FileProvider {
  private headers: string[] = []
  private headersParsed = false
  private readonly delimiter: string
  private pipeline?: { next(from: number, to: number): void }

  constructor(config: FileProviderConfig) {
    super(config)
    this.delimiter =
      'delimiter' in config && typeof config.delimiter === 'string'
        ? config.delimiter
        : ','
  }

  /**
   * Set the pipeline to notify when rows are added
   */
  setPipeline(pipeline: { next(from: number, to: number): void }): void {
    this.pipeline = pipeline
  }

  /**
   * Process historical data from CSV file
   */
  async processHistoricalData(params: HistoricalParams): Promise<void> {
    if (!this.connected) {
      throw new Error('Provider not connected. Call connect() first.')
    }

    logger.info('Processing CSV data', {
      path: this.filePath,
      symbols: params.symbols,
      start: new Date(params.start).toISOString(),
      end: new Date(params.end).toISOString()
    })

    await this.processFile(this.pipeline!)
  }

  /**
   * Process a single CSV line
   */
  protected processLine(line: string): boolean {
    const values = this.parseCsvLine(line)

    if (!this.headersParsed) {
      this.headers = values
      this.headersParsed = true
      logger.debug('CSV headers parsed', { headers: this.headers })
      return false // Headers don't count as data
    }

    if (values.length !== this.headers.length) {
      logger.warn('Skipping malformed row', {
        expected: this.headers.length,
        actual: values.length
      })
      return false
    }

    // Create object from headers and values
    const rawData: Record<string, string> = {}
    for (let i = 0; i < this.headers.length; i++) {
      rawData[this.headers[i]!] = values[i] || ''
    }

    // Transform and push to buffer
    try {
      const timestamp = this.parseTimestamp(rawData[this.columnMapping.timestamp])
      const open = this.parseNumber(rawData[this.columnMapping.open], 'open')
      const high = this.parseNumber(rawData[this.columnMapping.high], 'high')
      const low = this.parseNumber(rawData[this.columnMapping.low], 'low')
      const close = this.parseNumber(rawData[this.columnMapping.close], 'close')
      const volume = this.parseNumber(rawData[this.columnMapping.volume], 'volume')

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
      logger.error('Failed to parse CSV row', { error })
      return false
    }
  }

  /**
   * Parse a CSV line handling quoted values
   */
  private parseCsvLine(line: string): string[] {
    const values: string[] = []
    let current = ''
    let inQuotes = false
    let i = 0

    while (i < line.length) {
      const char = line[i]
      const nextChar = line[i + 1]

      if (char === '"') {
        if (inQuotes && nextChar === '"') {
          // Escaped quote
          current += '"'
          i += 2
          continue
        }
        inQuotes = !inQuotes
        i++
        continue
      }

      if (char === this.delimiter && !inQuotes) {
        values.push(current.trim())
        current = ''
        i++
        continue
      }

      current += char
      i++
    }

    // Add last value
    values.push(current.trim())

    return values
  }

  /**
   * Parse timestamp helper (delegates to base class)
   */
  protected parseTimestamp(value: unknown): number {
    return super.parseTimestamp(value, 0)
  }

  /**
   * Parse number helper (delegates to base class)
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

  /**
   * Override disconnect to reset CSV-specific state
   */
  async disconnect(): Promise<void> {
    this.headers = []
    this.headersParsed = false
    await super.disconnect()
  }
}
