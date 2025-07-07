import { mkdir, stat, unlink } from 'node:fs/promises'
import * as path from 'node:path'
import type { OhlcvDto } from '../models'
import type { ColumnDefinition } from '../utils'
import { DataBuffer } from '../utils'
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

  // DataBuffer for memory-efficient operations
  protected dataBuffer?: DataBuffer
  protected columnDefinitions?: Record<string, ColumnDefinition>

  // Expected output fields from transforms (if any)
  protected expectedOutputFields: string[] = []

  /**
   * Initialize the repository with common setup logic
   */
  async initialize(config: RepositoryConfig): Promise<void> {
    try {
      this.basePath = config.connectionString
      this.batchSize =
        (config.options?.batchSize as number | undefined) || 1000

      // Ensure the directory exists
      const dir = path.dirname(this.basePath)
      await mkdir(dir, { recursive: true })

      // Clear existing file if overwrite is requested
      if (config.options?.overwrite && (await this.fileExists(this.basePath))) {
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
      logger.error(
        `Failed to initialize ${this.getRepositoryType()} repository`,
        { error }
      )
      throw new RepositoryConnectionError(
        `Failed to initialize ${this.getRepositoryType()} repository: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Initialize the data buffer based on expected output fields
   */
  protected initializeDataBuffer(): void {
    // Create column definitions from expected fields or default OHLCV
    const columns: Record<string, ColumnDefinition> = {
      timestamp: { index: 0 },
      open: { index: 1 },
      high: { index: 2 },
      low: { index: 3 },
      close: { index: 4 },
      volume: { index: 5 }
    }

    // Add any additional expected fields
    if (this.expectedOutputFields.length > 0) {
      const standardFields = new Set([
        'timestamp',
        'open',
        'high',
        'low',
        'close',
        'volume'
      ])
      let nextIndex = 6

      for (const field of this.expectedOutputFields) {
        if (!standardFields.has(field)) {
          columns[field] = { index: nextIndex++ }
        }
      }
    }

    this.columnDefinitions = columns
    this.dataBuffer = new DataBuffer({ columns })
  }

  /**
   * Save a single OHLCV record using DataBuffer
   */
  async save(data: OhlcvDto): Promise<void> {
    this.ensureReady()
    this.validateOhlcvData(data)
    this.enforceSingleSymbolExchange(data)

    // Initialize buffer if not already done
    if (!this.dataBuffer) {
      this.initializeDataBuffer()
    }

    try {
      // Convert OhlcvDto to Row format for DataBuffer
      const row: Record<string, any> = {
        timestamp: data.timestamp,
        open: data.open,
        high: data.high,
        low: data.low,
        close: data.close,
        volume: data.volume
      }

      // Add any additional fields
      for (const key in data) {
        if (!(key in row)) {
          row[key] = (data as any)[key]
        }
      }

      // Push to DataBuffer
      this.dataBuffer!.push(row)

      // Check if we should flush
      if (this.dataBuffer!.length() >= this.batchSize) {
        await this.flush()
      }
    } catch (error) {
      logger.error('Failed to save OHLCV data', { error })
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
      if (this.singleSymbol === null) {
        this.singleSymbol = 'UNKNOWN'
        this.singleExchange = 'UNKNOWN'
      }

      for (const item of data) {
        this.enforceSingleSymbolExchange(item)
      }
    }

    try {
      // Process each record through the buffer
      for (const item of data) {
        await this.save(item)
      }
      // Ensure everything is flushed at the end
      await this.flush()

      logger.debug(`Saved OHLCV batch to ${this.getRepositoryType()}`, {
        count: data.length
      })
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
   * Set expected output fields from transform pipeline
   * This should be called before any data is written to ensure correct headers
   */
  public setExpectedOutputFields(fields: string[]): void {
    this.expectedOutputFields = [...fields]
    this.initializeDataBuffer()
  }

  /**
   * Check if the repository is properly initialized and ready to use
   */
  isReady(): boolean {
    return this.ready
  }

  /**
   * Flush the DataBuffer by popping items and writing them
   */
  async flush(): Promise<void> {
    if (!this.dataBuffer || this.dataBuffer.isEmpty()) {
      return
    }

    // Pop all items from buffer and convert back to OhlcvDto
    const items: OhlcvDto[] = []

    while (!this.dataBuffer.isEmpty()) {
      const row = this.dataBuffer.pop()
      if (row) {
        // Convert Row back to OhlcvDto
        const ohlcv: OhlcvDto = {
          timestamp: row.timestamp!,
          open: row.open!,
          high: row.high!,
          low: row.low!,
          close: row.close!,
          volume: row.volume!
        }

        // Add any additional fields
        for (const key in row) {
          if (!(key in ohlcv)) {
            (ohlcv as any)[key] = row[key]
          }
        }

        items.push(ohlcv)
      }
    }

    // Write the batch
    if (items.length > 0) {
      await this.writeOhlcvBatch(items)
      logger.debug(`Flushed ${items.length} items from buffer`)
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
        logger.error(`Error closing ${this.getRepositoryType()} repository`, {
          error
        })
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
    // Only validate the basic required fields, not the OHLCV relationships
    // since transforms may have modified the values
    if (typeof data.timestamp !== 'number') {
      throw new RepositoryValidationError(
        'Missing required field: timestamp'
      )
    }

    // Check that timestamp is valid
    if (isNaN(data.timestamp) || data.timestamp <= 0) {
      throw new RepositoryValidationError('Invalid timestamp')
    }
  }

  /**
   * Enforce single symbol/exchange per repository instance
   */
  protected enforceSingleSymbolExchange(_data: OhlcvDto): void {
    // Symbol and exchange are no longer part of OhlcvDto
    // This validation is now simplified since we track symbol/exchange at the repository level
    if (this.singleSymbol === null) {
      this.singleSymbol = 'UNKNOWN'
      this.singleExchange = 'UNKNOWN'
    }
    // No need to validate individual records since symbol/exchange are not in the data
  }

  /**
   * Generate a unique key for deduplication
   */
  protected generateRecordKey(data: OhlcvDto): string {
    return `${data.timestamp}`
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
    const regexPattern = pattern.replace(/\*/g, '.*').replace(/\?/g, '.')

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
  protected abstract getRepositoryType(): string;

  /**
   * Perform additional initialization specific to the storage format
   */
  protected abstract performAdditionalInitialization(
    config: RepositoryConfig
  ): Promise<void>;

  /**
   * Perform additional cleanup specific to the storage format
   */
  protected abstract performAdditionalCleanup(): Promise<void>;

  /**
   * Write a batch of OHLCV records to storage
   */
  protected abstract writeOhlcvBatch(data: OhlcvDto[]): Promise<void>;

  /**
   * Execute a query against the storage backend
   */
  public abstract query(query: OhlcvQuery): Promise<OhlcvDto[]>;

  /**
   * Get the most recent timestamp for a specific symbol
   */
  public abstract getLastTimestamp(
    symbol: string,
    exchange?: string
  ): Promise<number | null>;

  /**
   * Get the earliest timestamp for a specific symbol
   */
  public abstract getFirstTimestamp(
    symbol: string,
    exchange?: string
  ): Promise<number | null>;

  /**
   * Get count of records for a symbol
   */
  public abstract getCount(symbol: string, exchange?: string): Promise<number>;

  /**
   * Get all unique symbols in the repository
   */
  public abstract getSymbols(exchange?: string): Promise<string[]>;

  /**
   * Get all unique exchanges in the repository
   */
  public abstract getExchanges(): Promise<string[]>;

  /**
   * Delete OHLCV data within a specific date range
   */
  public abstract deleteBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<number>;

  /**
   * Get repository statistics and health information
   */
  public abstract getStats(): Promise<{
    totalRecords: number;
    uniqueSymbols: number;
    uniqueExchanges: number;
    dataDateRange: {
      earliest: number | null;
      latest: number | null;
    };
    storageSize?: number;
  }>;

  /**
   * Get buffer statistics
   */
  public getBufferStats(): {
    currentSize: number;
    columns: string[];
    isEmpty: boolean;
  } {
    if (!this.dataBuffer) {
      return {
        currentSize: 0,
        columns: [],
        isEmpty: true
      }
    }

    return {
      currentSize: this.dataBuffer.length(),
      columns: [...this.dataBuffer.getColumns()],
      isEmpty: this.dataBuffer.isEmpty()
    }
  }

  /**
   * Force close all resources immediately without flushing
   * Use only in test cleanup scenarios
   */
  public forceClose(): void {
    this.ready = false
  }
}
