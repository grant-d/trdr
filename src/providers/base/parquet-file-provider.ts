import type { HistoricalParams } from '../../interfaces/data-provider.interface'
import type { OhlcvDto } from '../../models/ohlcv.dto'
import logger from '../../utils/logger'
import { FileProvider } from './file-provider.base'
import type { FileProviderConfig } from './types'

/**
 * Parquet file provider implementation
 * Handles reading Parquet files with efficient columnar access
 */
export class ParquetFileProvider extends FileProvider {
  constructor(config: FileProviderConfig) {
    super(config)
  }

  /**
   * Streams historical data from Parquet file
   */
  async *getHistoricalData(params: HistoricalParams): AsyncIterableIterator<OhlcvDto> {
    if (!this.connected) {
      throw new Error('Provider not connected. Call connect() first.')
    }

    logger.info('Starting Parquet data stream', {
      path: this.filePath,
      symbols: params.symbols,
      start: new Date(params.start).toISOString(),
      end: new Date(params.end).toISOString()
    })

    try {
      // Dynamic import to handle ESM issues
      const { asyncBufferFromFile, parquetReadObjects } = await import('hyparquet')
      
      // Read the parquet file
      const file = await asyncBufferFromFile(this.filePath)
      
      // Read parquet data with column selection
      const rows = await parquetReadObjects({
        file,
        columns: this.getRequiredColumns(), // Only read needed columns
      })

      let rowNumber = 0
      let yieldedCount = 0
      const chunk: OhlcvDto[] = []

      // Process rows
      for (const row of rows) {
        rowNumber++

        // Transform row to OHLCV
        const ohlcv = this.validateAndTransform(row as Record<string, unknown>, rowNumber)
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

      // Yield remaining items
      for (const item of chunk) {
        yield item
        yieldedCount++
      }

      logger.info('Parquet streaming completed', {
        path: this.filePath,
        rowsProcessed: rowNumber,
        rowsYielded: yieldedCount
      })
    } catch (error) {
      logger.error('Error reading Parquet file', { error, path: this.filePath })
      throw error
    }
  }

  /**
   * Gets list of columns to read from Parquet file
   * This optimization reduces memory usage by only reading needed columns
   */
  private getRequiredColumns(): string[] {
    const columns: string[] = []
    
    // Add mapped columns
    columns.push(
      this.columnMapping.timestamp,
      this.columnMapping.open,
      this.columnMapping.high,
      this.columnMapping.low,
      this.columnMapping.close,
      this.columnMapping.volume
    )

    // Add optional columns if mapped
    if (this.columnMapping.symbol) {
      columns.push(this.columnMapping.symbol)
    }
    if (this.columnMapping.exchange) {
      columns.push(this.columnMapping.exchange)
    }

    // Remove duplicates
    return [...new Set(columns)]
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
   * Alternative method using streaming for very large files
   * Note: hyparquet currently requires the full file in memory,
   * but this method prepares for future streaming support
   */
  async *getHistoricalDataStreaming(params: HistoricalParams): AsyncIterableIterator<OhlcvDto> {
    if (!this.connected) {
      throw new Error('Provider not connected. Call connect() first.')
    }

    logger.info('Starting Parquet streaming (chunked)', {
      path: this.filePath,
      symbols: params.symbols
    })

    try {
      // Dynamic import to handle ESM issues
      const { asyncBufferFromFile, parquetReadObjects } = await import('hyparquet')
      
      // Read the parquet file
      const file = await asyncBufferFromFile(this.filePath)
      
      // Read parquet data
      const rows = await parquetReadObjects({
        file,
        columns: this.getRequiredColumns(),
        // Future: rowGroups option for chunked reading
      })

      let batchCount = 0
      const batch: OhlcvDto[] = []

      for (const row of rows) {
        const ohlcv = this.validateAndTransform(row as Record<string, unknown>, batchCount + 1)
        
        if (ohlcv && this.matchesParams(ohlcv, params)) {
          batch.push(ohlcv)
        }

        batchCount++

        // Yield batch when full
        if (batch.length >= this.chunkSize) {
          logger.debug(`Yielding batch of ${batch.length} rows`)
          for (const item of batch) {
            yield item
          }
          batch.length = 0
        }
      }

      // Yield remaining items
      if (batch.length > 0) {
        logger.debug(`Yielding final batch of ${batch.length} rows`)
        for (const item of batch) {
          yield item
        }
      }

      logger.info('Parquet streaming completed', {
        path: this.filePath,
        totalRows: batchCount
      })
    } catch (error) {
      logger.error('Error in Parquet streaming', { error, path: this.filePath })
      throw error
    }
  }

  /**
   * Gets metadata about the Parquet file
   * Useful for understanding schema and row groups
   */
  async getMetadata(): Promise<{
    numRows: number
    columns: string[]
    rowGroups: number
  }> {
    // Dynamic import to handle ESM issues
    const { asyncBufferFromFile, parquetReadObjects } = await import('hyparquet')
    
    // Read the parquet file
    const file = await asyncBufferFromFile(this.filePath)
    
    // For now, read a single row to get column names
    // Future: use parquet metadata APIs when available
    const sample = await parquetReadObjects({
      file,
      rowStart: 0,
      rowEnd: 1,
    })

    return {
      numRows: 0, // Would be extracted from metadata
      columns: sample.length > 0 && sample[0] ? Object.keys(sample[0]) : [],
      rowGroups: 0, // Would count row groups
    }
  }
}