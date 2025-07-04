import { createReadStream } from 'node:fs'
import { Transform } from 'node:stream'
import type { HistoricalParams } from '../../interfaces/data-provider.interface'
import type { OhlcvDto } from '../../models/ohlcv.dto'
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
  async *getHistoricalData(params: HistoricalParams): AsyncIterableIterator<OhlcvDto> {
    if (!this.connected) {
      throw new Error('Provider not connected. Call connect() first.')
    }

    logger.info('Starting CSV data stream', {
      path: this.filePath,
      symbols: params.symbols,
      start: new Date(params.start).toISOString(),
      end: new Date(params.end).toISOString()
    })

    const stream = createReadStream(this.filePath, {
      encoding: 'utf8',
      highWaterMark: 64 * 1024 // 64KB buffer
    })

    let buffer = ''
    let rowNumber = 0
    let headersParsed = false
    let yieldedCount = 0
    const chunk: OhlcvDto[] = []

    try {
      for await (const data of stream) {
        buffer += data

        // Process complete lines
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer

        for (const line of lines) {
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
      }

      // Process remaining buffer
      if (buffer.trim() && headersParsed) {
        rowNumber++
        const values = this.parseCsvLine(buffer)
        if (values.length === this.headers.length) {
          const rawData: Record<string, string> = {}
          for (let i = 0; i < this.headers.length; i++) {
            rawData[this.headers[i]!] = values[i] || ''
          }
          const ohlcv = this.validateAndTransform(rawData, rowNumber)
          if (ohlcv && this.matchesParams(ohlcv, params)) {
            chunk.push(ohlcv)
          }
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
      logger.error('Error streaming CSV file', { error, path: this.filePath })
      throw error
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
    if (params.symbols.length > 0 && !params.symbols.includes(ohlcv.symbol)) {
      return false
    }

    return true
  }

  /**
   * Creates a transform stream for processing CSV data
   * Useful for advanced streaming scenarios
   */
  createTransformStream(params: HistoricalParams): Transform {
    let buffer = ''
    let rowNumber = 0
    let headersParsed = false
    let headers: string[] = []

    // Store references to avoid 'this' context issues
    const parseCsvLine = this.parseCsvLine.bind(this)
    const validateAndTransform = this.validateAndTransform.bind(this)
    const matchesParams = this.matchesParams.bind(this)

    const transformStream = new Transform({
      objectMode: true,
      transform(chunk: Buffer, _encoding, callback) {
        buffer += chunk.toString()
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.trim()) continue

          rowNumber++

          if (!headersParsed) {
            headers = parseCsvLine(line)
            headersParsed = true
            continue
          }

          const values = parseCsvLine(line)
          if (values.length !== headers.length) continue

          const rawData: Record<string, string> = {}
          for (let i = 0; i < headers.length; i++) {
            rawData[headers[i]!] = values[i] || ''
          }

          const ohlcv = validateAndTransform(rawData, rowNumber)
          if (ohlcv && matchesParams(ohlcv, params)) {
            this.push(ohlcv)
          }
        }

        callback()
      },
      flush(callback) {
        // Process remaining buffer
        if (buffer.trim() && headersParsed) {
          rowNumber++
          const values = parseCsvLine(buffer)
          if (values.length === headers.length) {
            const rawData: Record<string, string> = {}
            for (let i = 0; i < headers.length; i++) {
              rawData[headers[i]!] = values[i] || ''
            }
            const ohlcv = validateAndTransform(rawData, rowNumber)
            if (ohlcv && matchesParams(ohlcv, params)) {
              this.push(ohlcv)
            }
          }
        }
        callback()
      }
    })

    return transformStream
  }
}