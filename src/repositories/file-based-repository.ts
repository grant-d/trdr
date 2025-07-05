import { mkdir, stat, unlink } from 'node:fs/promises'
import * as path from 'node:path'
import type { OhlcvDto } from '../models'
import { isValidOhlcv } from '../models'
import logger from '../utils/logger'
import type { OhlcvQuery, OhlcvRepository, RepositoryConfig } from './ohlcv-repository.interface'
import { RepositoryConnectionError, RepositoryStorageError, RepositoryValidationError } from './ohlcv-repository.interface'

/**
 * Abstract base class for file-based OHLCV repositories (CSV, JSONL)
 * Provides shared functionality for validation, deduplication, file management, and query operations
 */
export abstract class FileBasedRepository implements OhlcvRepository {
  protected ready = false
  protected basePath = ''
  protected batchSize = 1000
  
  // Single symbol/exchange enforcement
  protected singleSymbol: string | null = null
  protected singleExchange: string | null = null
  
  // Deduplication using JSONL's superior strategy
  protected lastPendingRecord: OhlcvDto | null = null
  protected buffer: OhlcvDto[] = []
  

  /**
   * Initialize the repository with common setup logic
   */
  async initialize(config: RepositoryConfig): Promise<void> {
    try {
      this.basePath = config.connectionString
      this.batchSize = (config.options?.batchSize as number | undefined) || 1000
      
      // Ensure the directory exists
      const dir = path.dirname(this.basePath)
      await mkdir(dir, { recursive: true })
      
      // Clear existing file if overwrite is requested
      if (config.options?.overwrite && await this.fileExists(this.basePath)) {
        await unlink(this.basePath)
        logger.debug('Removed existing file due to overwrite option')
      }
      
      // Allow subclasses to perform additional initialization
      await this.performAdditionalInitialization(config)
      
      this.ready = true
      logger.info(`${this.getRepositoryType()} repository initialized`, {
        basePath: this.basePath,
        options: config.options
      })
    } catch (error) {
      logger.error(`Failed to initialize ${this.getRepositoryType()} repository`, { error })
      throw new RepositoryConnectionError(
        `Failed to initialize ${this.getRepositoryType()} repository: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Save a single OHLCV record with deduplication
   */
  async save(data: OhlcvDto): Promise<void> {
    this.ensureReady()
    this.validateOhlcvData(data)
    this.enforceSingleSymbolExchange(data)
    
    try {
      // Apply JSONL's superior deduplication strategy
      if (this.lastPendingRecord) {
        const lastKey = this.generateRecordKey(this.lastPendingRecord)
        const newKey = this.generateRecordKey(data)
        
        if (lastKey === newKey) {
          // Same key - just overwrite the pending record (deduplication)
          this.lastPendingRecord = data
        } else {
          // Different key - write the pending record and store new one
          this.buffer.push(this.lastPendingRecord)
          this.lastPendingRecord = data
          
          if (this.buffer.length >= this.batchSize) {
            await this.flush()
          }
        }
      } else {
        // First record - just store it
        this.lastPendingRecord = data
      }
    } catch (error) {
      logger.error('Failed to save OHLCV data', { error, symbol: data.symbol })
      throw new RepositoryStorageError(
        `Failed to save OHLCV data: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Save multiple OHLCV records with batch processing and deduplication
   */
  async saveMany(data: OhlcvDto[]): Promise<void> {
    this.ensureReady()
    
    if (data.length === 0) return
    
    // Validate all data first
    for (const item of data) {
      this.validateOhlcvData(item)
    }
    
    // Validate all data is for the same symbol/exchange
    if (data.length > 0) {
      const firstItem = data[0]!
      if (this.singleSymbol === null) {
        this.singleSymbol = firstItem.symbol
        this.singleExchange = firstItem.exchange
      }
      
      for (const item of data) {
        this.enforceSingleSymbolExchange(item)
      }
    }
    
    try {
      // Process each record through deduplication logic to maintain streaming behavior
      // Bulk optimization commented out - pipeline emulates streaming/live data service
      /*
      if (data.length > 100) {
        // Sort by timestamp to improve deduplication
        const sortedData = [...data].sort((a, b) => a.timestamp - b.timestamp)
        
        // Deduplicate using a map
        const deduplicatedMap = new Map<string, OhlcvDto>()
        for (const item of sortedData) {
          const key = this.generateRecordKey(item)
          deduplicatedMap.set(key, item)
        }
        
        // Convert back to array
        const deduplicated = Array.from(deduplicatedMap.values())
        
        // Write in batches
        for (let i = 0; i < deduplicated.length; i += this.batchSize) {
          const batch = deduplicated.slice(i, i + this.batchSize)
          await this.writeOhlcvBatch(batch)
        }
      } else {
      */
      
      // Process each record individually to maintain streaming semantics
      for (const item of data) {
        await this.save(item)
      }
      // Ensure everything is flushed at the end
      await this.flush()
      
      logger.debug(`Saved OHLCV batch to ${this.getRepositoryType()}`, { count: data.length })
    } catch (error) {
      logger.error('Failed to save OHLCV batch', { error, count: data.length })
      throw new RepositoryStorageError(
        `Failed to save OHLCV batch: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Append a batch of OHLCV records (alias for saveMany)
   */
  async appendBatch(data: OhlcvDto[]): Promise<void> {
    return this.saveMany(data)
  }

  /**
   * Get OHLCV data within a specific date range
   */
  async getBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<OhlcvDto[]> {
    const query: OhlcvQuery = {
      startTime,
      endTime,
      symbol,
      exchange
    }
    return this.query(query)
  }

  /**
   * Get OHLCV data for a specific symbol
   */
  async getBySymbol(
    symbol: string,
    exchange?: string,
    limit?: number,
    offset?: number
  ): Promise<OhlcvDto[]> {
    const query: OhlcvQuery = {
      symbol,
      exchange,
      limit,
      offset
    }
    return this.query(query)
  }

  /**
   * Check if the repository is properly initialized and ready to use
   */
  isReady(): boolean {
    return this.ready
  }

  /**
   * Flush any pending writes including the last pending record
   */
  async flush(): Promise<void> {
    // Write any pending record first (JSONL's strategy)
    if (this.lastPendingRecord) {
      this.buffer.push(this.lastPendingRecord)
      this.lastPendingRecord = null
    }
    
    // Flush OHLCV data
    if (this.buffer.length > 0) {
      await this.writeOhlcvBatch(this.buffer)
      this.buffer = []
    }
  }

  /**
   * Close the repository and clean up resources
   */
  async close(): Promise<void> {
    if (this.ready) {
      try {
        await this.flush()
        await this.performAdditionalCleanup()
        this.ready = false
        logger.info(`${this.getRepositoryType()} repository closed`)
      } catch (error) {
        logger.error(`Error closing ${this.getRepositoryType()} repository`, { error })
        throw new RepositoryStorageError(
          `Error closing repository: ${String(error)}`,
          error instanceof Error ? error : undefined
        )
      }
    }
  }

  // --- Protected helper methods ---

  /**
   * Ensure the repository is ready for operations
   */
  protected ensureReady(): void {
    if (!this.ready) {
      throw new RepositoryConnectionError('Repository not initialized')
    }
  }

  /**
   * Validate OHLCV data
   */
  protected validateOhlcvData(data: OhlcvDto): void {
    if (!isValidOhlcv(data)) {
      throw new RepositoryValidationError('Invalid OHLCV data')
    }
  }

  /**
   * Enforce single symbol/exchange per repository instance
   */
  protected enforceSingleSymbolExchange(data: OhlcvDto): void {
    if (this.singleSymbol === null) {
      this.singleSymbol = data.symbol
      this.singleExchange = data.exchange
    } else if (this.singleSymbol !== data.symbol || this.singleExchange !== data.exchange) {
      throw new RepositoryValidationError(
        `${this.getRepositoryType()} file can only contain data for one symbol/exchange. Expected ${this.singleSymbol}/${this.singleExchange}, got ${data.symbol}/${data.exchange}`
      )
    }
  }

  /**
   * Generate a unique key for deduplication
   */
  protected generateRecordKey(data: OhlcvDto): string {
    return `${data.timestamp}:${data.symbol}:${data.exchange}`
  }

  /**
   * Check if a file exists
   */
  protected async fileExists(filePath: string): Promise<boolean> {
    try {
      await stat(filePath)
      return true
    } catch {
      return false
    }
  }

  /**
   * Simple pattern matching for names (supports * and ? wildcards)
   */
  protected matchesPattern(name: string, pattern: string): boolean {
    // Convert glob pattern to regex
    const regexPattern = pattern
      .replace(/\*/g, '.*')
      .replace(/\?/g, '.')
    
    const regex = new RegExp(`^${regexPattern}$`, 'i')
    return regex.test(name)
  }

  /**
   * Apply query filters to a record
   */
  protected matchesQuery(record: OhlcvDto, query: OhlcvQuery): boolean {
    if (query.startTime && record.timestamp < query.startTime) return false
    if (query.endTime && record.timestamp > query.endTime) return false
    if (query.symbol && record.symbol !== query.symbol) return false
    if (query.exchange && record.exchange !== query.exchange) return false
    
    return true
  }

  /**
   * Apply pagination to results
   */
  protected applyPagination<T>(results: T[], query: OhlcvQuery): T[] {
    let filtered = results
    
    // Apply offset
    if (query.offset) {
      filtered = filtered.slice(query.offset)
    }
    
    // Apply limit
    if (query.limit) {
      filtered = filtered.slice(0, query.limit)
    }
    
    return filtered
  }

  /**
   * Append data to a file
   */
  protected async appendToFile(filePath: string, data: string): Promise<void> {
    const { appendFile } = await import('node:fs/promises')
    await appendFile(filePath, data, 'utf8')
  }

  /**
   * Write data to a file (overwrites existing content)
   */
  protected async writeToFile(filePath: string, data: string): Promise<void> {
    const { writeFile } = await import('node:fs/promises')
    await writeFile(filePath, data, 'utf8')
  }


  // --- Abstract methods that subclasses must implement ---

  /**
   * Get the repository type name for logging
   */
  protected abstract getRepositoryType(): string

  /**
   * Perform additional initialization specific to the storage format
   */
  protected abstract performAdditionalInitialization(config: RepositoryConfig): Promise<void>

  /**
   * Perform additional cleanup specific to the storage format
   */
  protected abstract performAdditionalCleanup(): Promise<void>

  /**
   * Write a batch of OHLCV records to storage
   */
  protected abstract writeOhlcvBatch(data: OhlcvDto[]): Promise<void>

  /**
   * Execute a query against the storage backend
   */
  public abstract query(query: OhlcvQuery): Promise<OhlcvDto[]>

  /**
   * Get the most recent timestamp for a specific symbol
   */
  public abstract getLastTimestamp(symbol: string, exchange?: string): Promise<number | null>

  /**
   * Get the earliest timestamp for a specific symbol
   */
  public abstract getFirstTimestamp(symbol: string, exchange?: string): Promise<number | null>

  /**
   * Get count of records for a symbol
   */
  public abstract getCount(symbol: string, exchange?: string): Promise<number>

  /**
   * Get all unique symbols in the repository
   */
  public abstract getSymbols(exchange?: string): Promise<string[]>

  /**
   * Get all unique exchanges in the repository
   */
  public abstract getExchanges(): Promise<string[]>

  /**
   * Delete OHLCV data within a specific date range
   */
  public abstract deleteBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<number>

  /**
   * Get repository statistics and health information
   */
  public abstract getStats(): Promise<{
    totalRecords: number
    uniqueSymbols: number
    uniqueExchanges: number
    dataDateRange: {
      earliest: number | null
      latest: number | null
    }
    storageSize?: number
  }>

  /**
   * Force close all resources immediately without flushing
   * Use only in test cleanup scenarios
   */
  public forceClose(): void {
    this.ready = false
  }
}