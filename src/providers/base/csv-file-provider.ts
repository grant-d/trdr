import { createReadStream } from 'node:fs'
import { createInterface } from 'node:readline'
import type { HistoricalParams } from '../../interfaces'
import type { OhlcvDto } from '../../models'
import logger from '../../utils/logger'
import { FileProvider } from './file-provider.base'
import type { FileProviderConfig } from './types'

/**
 * CSV file provider implementation
 * Handles streaming large CSV files with configurable chunk processing
 */
export class CsvFileProvider extends FileProvider {
  private headers: string[] = []
  private readonly delimiter: string

  constructor(config: FileProviderConfig) {
    super(config)

    // Allow custom delimiter
    this.delimiter = ('delimiter' in config && typeof config.delimiter === 'string')
      ? config.delimiter
      : ','
  }

  /**
   * Streams historical data from CSV file
   */
  async* getHistoricalData(params: HistoricalParams): AsyncIterableIterator<OhlcvDto> {
    if (!this.connected) {
      throw new Error('Provider not connected. Call connect() first.')
    }

    logger.info('Starting CSV data stream', {
      path: this.filePath,
      symbols: params.symbols,
      start: new Date(params.start).toISOString(),
      end: new Date(params.end).toISOString()
    })

    const fileStream = createReadStream(this.filePath, { encoding: 'utf8' })
    const rl = createInterface({
      input: fileStream,
      crlfDelay: Infinity
    })

    try {
      let rowNumber = 0
      let headersParsed = false
      let yieldedCount = 0
      const chunk: OhlcvDto[] = []

      for await (const line of rl) {
        if (!line.trim()) continue

        rowNumber++

        // Parse headers from first row
        if (!headersParsed) {
          this.headers = this.parseCsvLine(line)
          headersParsed = true
          logger.debug('CSV headers parsed', { headers: this.headers })
          continue
        }

        // Parse data row
        const values = this.parseCsvLine(line)
        if (values.length !== this.headers.length) {
          logger.warn('Skipping malformed row', {
            row: rowNumber,
            expected: this.headers.length,
            actual: values.length
          })
          continue
        }

        // Create object from headers and values
        const rawData: Record<string, string> = {}
        for (let i = 0; i < this.headers.length; i++) {
          rawData[this.headers[i]!] = values[i] || ''
        }

        // Transform to OHLCV
        const ohlcv = this.validateAndTransform(rawData, rowNumber)
        if (!ohlcv) continue

        // Filter by params
        if (!this.matchesParams(ohlcv, params)) continue

        // Add to chunk
        chunk.push(ohlcv)

        // Yield chunk when it reaches configured size
        if (chunk.length >= this.chunkSize) {
          for (const item of chunk) {
            yield item
            yieldedCount++
          }
          chunk.length = 0
        }
      }

      // Yield remaining items in chunk
      for (const item of chunk) {
        yield item
        yieldedCount++
      }

      logger.info('CSV streaming completed', {
        path: this.filePath,
        rowsProcessed: rowNumber,
        rowsYielded: yieldedCount
      })
    } catch (error) {
      logger.error('Error reading CSV file', { error, path: this.filePath })
      throw error
    } finally {
      // Ensure readline interface is closed
      rl.close()
      // Destroy the underlying stream
      fileStream.destroy()
    }
  }

  /**
   * Parses a CSV line handling quoted values and escaped characters
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
        values.push(current)
        current = ''
        i++
        continue
      }

      current += char
      i++
    }

    // Add last value
    values.push(current)

    // Clean up values - remove quotes if present
    return values.map(val => {
      val = val.trim()
      if (val.startsWith('"') && val.endsWith('"')) {
        return val.slice(1, -1)
      }
      return val
    })
  }

  /**
   * Checks if OHLCV data matches the query parameters
   */
  private matchesParams(ohlcv: OhlcvDto, params: HistoricalParams): boolean {
    // Check timestamp range
    if (ohlcv.timestamp < params.start || ohlcv.timestamp > params.end) {
      return false
    }

    // Check symbol filter
    return !(params.symbols.length > 0 && !params.symbols.includes(ohlcv.symbol))
  }
}
