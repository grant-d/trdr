import { createReadStream } from 'node:fs'
import { createInterface } from 'node:readline'
import type { HistoricalParams } from '../../interfaces'
import type { OhlcvDto } from '../../models'
import logger from '../../utils/logger'
import { FileProvider } from './file-provider.base'
import type { FileProviderConfig } from './types'

/**
 * JSONL (JSON Lines) file provider implementation
 * Handles reading and writing JSONL files with streaming support
 * Each line contains a complete JSON object representing one OHLCV record
 */
export class JsonlFileProvider extends FileProvider {
  constructor(config: FileProviderConfig) {
    super(config)
  }

  /**
   * Streams historical data from JSONL file
   */
  async* getHistoricalData(params: HistoricalParams): AsyncIterableIterator<OhlcvDto> {
    if (!this.connected) {
      throw new Error('Provider not connected. Call connect() first.')
    }

    logger.info('Starting JSONL data stream', {
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
      let yieldedCount = 0
      const chunk: OhlcvDto[] = []

      for await (const line of rl) {
        rowNumber++
        
        // Skip empty lines
        if (!line.trim()) continue

        try {
          // Parse JSON line
          const row = JSON.parse(line)
          
          // Transform row to OHLCV
          const ohlcv = this.validateAndTransform(row, rowNumber)
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
        } catch (parseError) {
          logger.warn('Failed to parse JSONL line', { 
            line: rowNumber, 
            error: parseError 
          })
        }
      }

      // Yield remaining items
      for (const item of chunk) {
        yield item
        yieldedCount++
      }

      logger.info('JSONL streaming completed', {
        path: this.filePath,
        rowsProcessed: rowNumber,
        rowsYielded: yieldedCount
      })
    } catch (error) {
      logger.error('Error reading JSONL file', { error, path: this.filePath })
      throw error
    } finally {
      // Ensure readline interface is closed
      rl.close()
      // Destroy the underlying stream
      fileStream.destroy()
    }
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
