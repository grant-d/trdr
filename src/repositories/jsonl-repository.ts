import { createReadStream, existsSync } from 'node:fs'
import { rename, rm, stat } from 'node:fs/promises'
import { createInterface } from 'node:readline'
import type { OhlcvDto } from '../models'
import logger from '../utils/logger'
import { FileBasedRepository } from './file-based-repository'
import type { OhlcvQuery, RepositoryConfig } from './ohlcv-repository.interface'
import { RepositoryStorageError } from './ohlcv-repository.interface'

/**
 * JSONL-based implementation of the OhlcvRepository interface
 * Extends FileBasedRepository to gain deduplication and shared functionality
 * Stores each OHLCV record as a JSON object on a separate line
 * Optimized for streaming and append operations
 */
export class JsonlRepository extends FileBasedRepository {

  /**
   * Get the repository type name for logging
   */
  protected getRepositoryType(): string {
    return 'JSONL'
  }

  /**
   * Perform additional initialization specific to JSONL format
   */
  protected async performAdditionalInitialization(_config: RepositoryConfig): Promise<void> {
    // No additional initialization needed for JSONL
  }

  /**
   * Perform additional cleanup specific to JSONL format
   */
  protected async performAdditionalCleanup(): Promise<void> {
    // No additional cleanup needed - streams are handled by base class
  }


  /**
   * Write a batch of OHLCV records to JSONL storage
   */
  protected async writeOhlcvBatch(data: OhlcvDto[]): Promise<void> {
    // Write to a single file
    const filePath = this.basePath
    
    // Prepare content with abbreviated property names
    let content = ''
    for (const record of data) {
      const abbreviated: any = {
        x: record.exchange,
        s: record.symbol,
        t: record.timestamp,
        o: record.open,
        h: record.high,
        l: record.low,
        c: record.close,
        v: record.volume
      }
      
      // Add any additional fields from transforms
      const standardKeys = new Set(['timestamp', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume'])
      for (const [key, value] of Object.entries(record)) {
        if (!standardKeys.has(key)) {
          abbreviated[key] = value
        }
      }
      
      content += JSON.stringify(abbreviated) + '\n'
    }
    
    // Append to file
    await this.appendToFile(filePath, content)
  }

  /**
   * Execute a query against the JSONL storage backend
   */
  async query(query: OhlcvQuery): Promise<OhlcvDto[]> {
    this.ensureReady()
    
    // Flush buffer before reading to ensure we have all data
    await this.flush()
    
    try {
      const results: OhlcvDto[] = []
      const files = await this.getRelevantFiles(query)
      
      for (const file of files) {
        const stream = createReadStream(file, { encoding: 'utf8' })
        const rl = createInterface({ input: stream })
        
        for await (const line of rl) {
          if (!line.trim()) continue
          
          try {
            const rawRecord = JSON.parse(line)
            const record = this.normalizeRecord(rawRecord)
            
            if (this.matchesQuery(record, query)) {
              results.push(record)
            }
          } catch (error) {
            logger.warn('Failed to parse JSONL record', { file, error })
          }
        }
      }
      
      // Sort by timestamp
      results.sort((a, b) => a.timestamp - b.timestamp)
      
      return this.applyPagination(results, query)
    } catch (error) {
      logger.error('Failed to execute query', { error, query })
      throw new RepositoryStorageError(
        `Failed to execute query: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get the most recent timestamp for a specific symbol
   */
  async getLastTimestamp(symbol: string, exchange?: string): Promise<number | null> {
    this.ensureReady()
    
    // Flush buffer before reading to ensure we have all data
    await this.flush()

    try {
      const results = await this.getBySymbol(symbol, exchange)
      if (results.length === 0) return null
      
      // Find latest
      let latest = results[0]!.timestamp
      for (const item of results) {
        if (item.timestamp > latest) {
          latest = item.timestamp
        }
      }
      return latest
    } catch (error) {
      logger.error('Failed to get last timestamp', { error, symbol })
      throw new RepositoryStorageError(
        `Failed to get last timestamp: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get the earliest timestamp for a specific symbol
   */
  async getFirstTimestamp(symbol: string, exchange?: string): Promise<number | null> {
    this.ensureReady()
    
    // Flush buffer before reading to ensure we have all data
    await this.flush()

    try {
      const results = await this.getBySymbol(symbol, exchange)
      if (results.length === 0) return null
      
      // Find earliest
      let earliest = results[0]!.timestamp
      for (const item of results) {
        if (item.timestamp < earliest) {
          earliest = item.timestamp
        }
      }
      return earliest
    } catch (error) {
      logger.error('Failed to get first timestamp', { error, symbol })
      throw new RepositoryStorageError(
        `Failed to get first timestamp: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get count of records for a symbol
   */
  async getCount(symbol: string, exchange?: string): Promise<number> {
    this.ensureReady()
    
    // Flush buffer before reading to ensure we have all data
    await this.flush()

    try {
      const results = await this.getBySymbol(symbol, exchange)
      return results.length
    } catch (error) {
      logger.error('Failed to get count', { error, symbol })
      throw new RepositoryStorageError(
        `Failed to get count: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Delete OHLCV data within a specific date range
   */
  async deleteBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<number> {
    this.ensureReady()
    
    // Flush buffer before deleting
    await this.flush()
    
    try {
      let deletedCount = 0
      const files = await this.getRelevantFiles({ symbol, exchange })
      
      for (const file of files) {
        // Read all records
        const keepRecords: OhlcvDto[] = []
        const stream = createReadStream(file, { encoding: 'utf8' })
        const rl = createInterface({ input: stream })
        
        for await (const line of rl) {
          if (!line.trim()) continue
          
          try {
            const rawRecord = JSON.parse(line)
            const record = this.normalizeRecord(rawRecord)
            
            if (record.timestamp >= startTime && record.timestamp <= endTime &&
                (!symbol || record.symbol === symbol) &&
                (!exchange || record.exchange === exchange)) {
              deletedCount++
            } else {
              keepRecords.push(record)
            }
          } catch (error) {
            // Keep unparseable records
            logger.warn('Failed to parse record during delete', { file, error: error instanceof Error ? error.message : String(error) })
          }
        }
        
        // Rewrite file with remaining records
        if (deletedCount > 0) {
          await this.rewriteFile(file, keepRecords)
        }
      }
      
      return deletedCount
    } catch (error) {
      logger.error('Failed to delete between dates', { error, startTime, endTime })
      throw new RepositoryStorageError(
        `Failed to delete between dates: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }


  /**
   * Get all unique symbols in the repository
   */
  async getSymbols(exchange?: string): Promise<string[]> {
    this.ensureReady()
    
    // Flush buffer before reading
    await this.flush()
    
    try {
      const symbols = new Set<string>()
      const files = await this.getRelevantFiles({})
      
      for (const file of files) {
        const stream = createReadStream(file, { encoding: 'utf8' })
        const rl = createInterface({ input: stream })
        
        for await (const line of rl) {
          if (!line.trim()) continue
          
          try {
            const rawRecord = JSON.parse(line)
            const record = this.normalizeRecord(rawRecord)
            if (!exchange || record.exchange === exchange) {
              symbols.add(record.symbol)
            }
          } catch (error) {
            // Skip invalid records
          }
        }
      }
      
      return Array.from(symbols).sort()
    } catch (error) {
      logger.error('Failed to get symbols', { error })
      throw new RepositoryStorageError(
        `Failed to get symbols: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get all unique exchanges in the repository
   */
  async getExchanges(): Promise<string[]> {
    this.ensureReady()
    
    // Flush buffer before reading
    await this.flush()
    
    try {
      const exchanges = new Set<string>()
      const files = await this.getRelevantFiles({})
      
      for (const file of files) {
        const stream = createReadStream(file, { encoding: 'utf8' })
        const rl = createInterface({ input: stream })
        
        for await (const line of rl) {
          if (!line.trim()) continue
          
          try {
            const rawRecord = JSON.parse(line)
            const record = this.normalizeRecord(rawRecord)
            exchanges.add(record.exchange)
          } catch (error) {
            // Skip invalid records
          }
        }
      }
      
      return Array.from(exchanges).sort()
    } catch (error) {
      logger.error('Failed to get exchanges', { error })
      throw new RepositoryStorageError(
        `Failed to get exchanges: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get repository statistics and health information
   */
  async getStats(): Promise<{
    totalRecords: number
    uniqueSymbols: number
    uniqueExchanges: number
    dataDateRange: {
      earliest: number | null
      latest: number | null
    }
    storageSize?: number
  }> {
    this.ensureReady()
    
    // Flush buffer before reading
    await this.flush()
    
    try {
      let totalRecords = 0
      let earliest: number | null = null
      let latest: number | null = null
      const symbols = new Set<string>()
      const exchanges = new Set<string>()
      let storageSize = 0
      
      const files = await this.getRelevantFiles({})
      
      for (const file of files) {
        const stats = await stat(file)
        storageSize += stats.size
        
        const stream = createReadStream(file, { encoding: 'utf8' })
        const rl = createInterface({ input: stream })
        
        for await (const line of rl) {
          if (!line.trim()) continue
          
          try {
            const rawRecord = JSON.parse(line)
            const record = this.normalizeRecord(rawRecord)
            totalRecords++
            symbols.add(record.symbol)
            exchanges.add(record.exchange)
            
            if (earliest === null || record.timestamp < earliest) {
              earliest = record.timestamp
            }
            if (latest === null || record.timestamp > latest) {
              latest = record.timestamp
            }
          } catch (error) {
            // Skip invalid records
          }
        }
      }
      
      return {
        totalRecords,
        uniqueSymbols: symbols.size,
        uniqueExchanges: exchanges.size,
        dataDateRange: { earliest, latest },
        storageSize
      }
    } catch (error) {
      logger.error('Failed to get stats', { error })
      throw new RepositoryStorageError(
        `Failed to get stats: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }


  /**
   * Private helper methods
   */

  private async getRelevantFiles(_query: OhlcvQuery): Promise<string[]> {
    // Simply return the single JSONL file if it exists
    if (existsSync(this.basePath)) {
      return [this.basePath]
    }
    return []
  }

  
  /**
   * Normalizes a record from abbreviated or full property names to standard OhlcvDto
   */
  private normalizeRecord(record: any): OhlcvDto {
    // Handle abbreviated format
    if ('t' in record && 'o' in record) {
      const normalized: any = {
        exchange: record.x || record.e || record.exchange,
        symbol: record.s || record.symbol,
        timestamp: record.t || record.timestamp,
        open: record.o || record.open,
        high: record.h || record.high,
        low: record.l || record.low,
        close: record.c || record.close,
        volume: record.v || record.volume
      }
      
      // Add any additional fields (transform outputs)
      const abbreviatedKeys = new Set(['x', 'e', 's', 't', 'o', 'h', 'l', 'c', 'v', 'exchange', 'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'])
      for (const [key, value] of Object.entries(record)) {
        if (!abbreviatedKeys.has(key)) {
          normalized[key] = value
        }
      }
      
      return normalized as OhlcvDto
    }
    
    // Already in full format
    return record as OhlcvDto
  }


  private async rewriteFile(filePath: string, records: OhlcvDto[]): Promise<void> {
    // Write to temp file first
    const tempPath = filePath + '.tmp'
    
    // Prepare content
    let content = ''
    for (const record of records) {
      // Write in abbreviated format to save space
      const abbreviated: any = {
        x: record.exchange,
        s: record.symbol,
        t: record.timestamp,
        o: record.open,
        h: record.high,
        l: record.low,
        c: record.close,
        v: record.volume
      }
      
      // Add any additional fields from transforms
      const standardKeys = new Set(['timestamp', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume'])
      for (const [key, value] of Object.entries(record)) {
        if (!standardKeys.has(key)) {
          abbreviated[key] = value
        }
      }
      
      content += JSON.stringify(abbreviated) + '\n'
    }
    
    // Write to temp file
    await this.writeToFile(tempPath, content)
    
    // Atomic rename
    await rm(filePath)
    await rename(tempPath, filePath)
  }

}