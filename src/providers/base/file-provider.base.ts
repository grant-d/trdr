import { constants as fsConstants, promises as fsPromises } from 'node:fs'
import * as path from 'node:path'
import type { DataProvider, HistoricalParams, RealtimeParams } from '../../interfaces'
import type { OhlcvDto } from '../../models'
import { isValidOhlcv } from '../../models'
import type { ColumnDefinition } from '../../utils'
import { DataBuffer } from '../../utils'
import logger from '../../utils/logger'
import type { ColumnMapping, FileProviderConfig } from './types'

/**
 * Abstract base class for file-based data providers
 * Provides common functionality for CSV, Jsonl, and other file formats
 */
export abstract class FileProvider implements DataProvider {
  readonly name = 'file'
  protected readonly filePath: string
  protected readonly format: 'csv' | 'jsonl'
  protected readonly columnMapping: ColumnMapping
  protected readonly chunkSize: number
  protected connected = false
  protected fileHandle?: fsPromises.FileHandle
  protected _buffer?: DataBuffer

  // Chunked reading state
  private currentPosition = 0
  private lineBuffer = ''
  private pendingLines: string[] = []

  protected constructor(config: FileProviderConfig) {
    this.filePath = path.resolve(config.path)
    this.format = config.format || this.detectFormatFromExtension(config.path)
    this.columnMapping = config.columnMapping || this.getDefaultColumnMapping()
    this.chunkSize = config.chunkSize || 1000

    // Initialize buffer with standard OHLCV columns
    this.initializeBuffer()

    logger.info('FileProvider initialized', {
      filePath: this.filePath,
      format: this.format,
      chunkSize: this.chunkSize
    })
  }

  /**
   * Get the data buffer (read-only)
   */
  get buffer(): DataBuffer {
    if (!this._buffer) {
      throw new Error('Buffer not initialized')
    }
    return this._buffer
  }

  /**
   * Set the data buffer (for shared buffer scenarios)
   */
  setBuffer(buffer: DataBuffer): void {
    this._buffer = buffer
  }

  /**
   * Check if the provider is ready (has data)
   */
  get ready(): boolean {
    return this._buffer !== undefined && !this._buffer.isEmpty()
  }

  /**
   * Initialize the data buffer with column definitions
   */
  protected initializeBuffer(): void {
    // Standard OHLCV columns in specific order
    const columns: Record<string, ColumnDefinition> = {
      timestamp: { index: 0 },
      open: { index: 1 },
      high: { index: 2 },
      low: { index: 3 },
      close: { index: 4 },
      volume: { index: 5 }
    }

    this._buffer = new DataBuffer({ columns })
  }

  /**
   * Validates file exists and is readable
   */
  async connect(): Promise<void> {
    try {
      // Check if file exists
      await fsPromises.access(this.filePath, fsConstants.R_OK)

      // Get file stats
      const stats = await fsPromises.stat(this.filePath)
      if (!stats.isFile()) {
        throw new Error(`Path is not a file: ${this.filePath}`)
      }

      logger.info('Connected to file', {
        path: this.filePath,
        size: stats.size
      })

      this.connected = true
    } catch (error) {
      logger.error('Failed to connect to file', { error, path: this.filePath })
      throw new Error(`Cannot access file: ${this.filePath}`)
    }
  }

  /**
   * Get the data buffer
   */
  getBuffer(): DataBuffer {
    if (!this._buffer) {
      throw new Error('Buffer not initialized')
    }
    return this._buffer
  }

  /**
   * Closes any open file handles
   */
  async disconnect(): Promise<void> {
    if (this.fileHandle) {
      await this.fileHandle.close()
      this.fileHandle = undefined
    }
    this.connected = false
    this.currentPosition = 0
    this.lineBuffer = ''
    this.pendingLines = []
    logger.info('Disconnected from file', { path: this.filePath })
  }

  /**
   * File providers don't support real-time data
   */
  subscribeRealtime(_params: RealtimeParams): AsyncIterableIterator<OhlcvDto> {
    throw new Error(
      'FileProvider does not support real-time data subscriptions'
    )
  }

  /**
   * No environment variables required for file providers
   */
  getRequiredEnvVars(): string[] {
    return []
  }

  /**
   * Validates environment variables (none required for files)
   */
  validateEnvVars(): void {
    // No environment variables to validate
  }

  /**
   * Checks if provider is connected
   */
  isConnected(): boolean {
    return this.connected
  }

  /**
   * Gets supported timeframes (all timeframes for historical data)
   */
  getSupportedTimeframes(): string[] {
    return [
      '1m',
      '3m',
      '5m',
      '15m',
      '30m',
      '1h',
      '2h',
      '4h',
      '6h',
      '12h',
      '1d',
      '1w'
    ]
  }

  /**
   * Load next batch of rows from file
   * Returns number of rows loaded into buffer
   */
  async loadBatch(batchSize: number): Promise<number> {
    if (!this.fileHandle) {
      this.fileHandle = await fsPromises.open(this.filePath, 'r')
    }

    let loadedCount = 0
    const chunkSize = 8192 // 8KB chunks
    const readBuffer = Buffer.alloc(chunkSize)

    // Main processing loop
    while (loadedCount < batchSize) {
      // Process pending lines first
      while (this.pendingLines.length > 0 && loadedCount < batchSize) {
        const line = this.pendingLines.shift()!
        if (this.processLine(line)) {
          loadedCount++
        }
      }

      if (loadedCount >= batchSize) {
        return loadedCount
      }
      // Check if we have complete lines in buffer
      const newlineIndex = this.lineBuffer.indexOf('\n')
      if (newlineIndex !== -1) {
        // Split all complete lines
        const allLines = this.lineBuffer.split('\n')
        this.lineBuffer = allLines.pop() || '' // Keep last incomplete line

        // Add all complete lines to pending
        this.pendingLines.push(...allLines)
        continue // Process pending lines
      }

      // Read more from file
      const { bytesRead } = await this.fileHandle.read(
        readBuffer,
        0,
        chunkSize,
        this.currentPosition
      )

      if (bytesRead === 0) {
        // End of file - process remaining buffer
        if (this.lineBuffer.trim() && this.processLine(this.lineBuffer)) {
          loadedCount++
        }
        this.lineBuffer = ''
        break
      }

      this.currentPosition += bytesRead
      const chunk = readBuffer.subarray(0, bytesRead).toString('utf8')
      this.lineBuffer += chunk

    }

    return loadedCount
  }

  /**
   * Process all data from file in batches
   */
  async processFile(pipeline: { next(from: number, to: number): void }): Promise<void> {
    let totalProcessed = 0

    try {
      while (true) {
        const batchStart = this._buffer!.length()
        const rowsLoaded = await this.loadBatch(this.chunkSize)

        if (rowsLoaded === 0) break

        totalProcessed += rowsLoaded

        // Notify pipeline
        if (pipeline) {
          const batchEnd = this._buffer!.length()
          pipeline.next(batchStart, batchEnd)
        }
      }

      logger.info(`${this.format} processing complete`, { rowsProcessed: totalProcessed })
    } finally {
      // Clean up
      if (this.fileHandle) {
        await this.fileHandle.close()
        this.fileHandle = undefined
      }
      this.currentPosition = 0
      this.lineBuffer = ''
      this.pendingLines = []
    }
  }

  /**
   * Process a single line - must be implemented by subclasses
   * Returns true if line was successfully processed and added to buffer
   */
  protected abstract processLine(line: string): boolean

  /**
   * Abstract method that child classes must implement
   * Returns an async iterator that yields OHLCV data points
   */
  abstract getHistoricalData(
    params: HistoricalParams
  ): AsyncIterableIterator<OhlcvDto>

  /**
   * Detects file format from extension
   */
  protected detectFormatFromExtension(filePath: string): 'csv' | 'jsonl' {
    const ext = path.extname(filePath).toLowerCase()
    switch (ext) {
      case '.csv':
        return 'csv'
      case '.jsonl':
        return 'jsonl'
      default:
        throw new Error(`Unsupported file extension: ${ext}`)
    }
  }

  /**
   * Returns default column mapping
   */
  protected getDefaultColumnMapping(): ColumnMapping {
    return {
      timestamp: 'timestamp',
      open: 'open',
      high: 'high',
      low: 'low',
      close: 'close',
      volume: 'volume'
    }
  }

  /**
   * Validates and transforms raw data, pushing it into the buffer
   */
  protected validateAndTransform(
    rawData: Record<string, unknown>,
    rowNumber: number
  ): boolean {
    try {
      // Extract values using column mapping
      const timestamp = this.parseTimestamp(
        rawData[this.columnMapping.timestamp],
        rowNumber
      )
      const open = this.parseNumber(
        rawData[this.columnMapping.open],
        'open',
        rowNumber
      )
      const high = this.parseNumber(
        rawData[this.columnMapping.high],
        'high',
        rowNumber
      )
      const low = this.parseNumber(
        rawData[this.columnMapping.low],
        'low',
        rowNumber
      )
      const close = this.parseNumber(
        rawData[this.columnMapping.close],
        'close',
        rowNumber
      )
      const volume = this.parseNumber(
        rawData[this.columnMapping.volume],
        'volume',
        rowNumber
      )

      // Get symbol and exchange
      // const symbol =
      //   this.columnMapping.symbol && rawData[this.columnMapping.symbol]
      //     ? String(rawData[this.columnMapping.symbol])
      //     : this.symbol
      // const exchange =
      //   this.columnMapping.exchange && rawData[this.columnMapping.exchange]
      //     ? String(rawData[this.columnMapping.exchange])
      //     : this.exchange

      const ohlcv: OhlcvDto = {
        timestamp,
        open,
        high,
        low,
        close,
        volume
      }

      // Validate the data
      if (!isValidOhlcv(ohlcv)) {
        logger.warn('Invalid OHLCV data', { row: rowNumber, data: ohlcv })
        return false
      }

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
      logger.error('Failed to transform row', { row: rowNumber, error })
      return false
    }
  }

  /**
   * Parses timestamp from various formats to Unix milliseconds
   */
  protected parseTimestamp(value: unknown, rowNumber: number): number {
    if (!value) {
      throw new Error(`Missing timestamp at row ${rowNumber}`)
    }

    // Threshold to determine if timestamp is in seconds or milliseconds
    // Using year 2001 as threshold: timestamps before this in raw number form are likely seconds
    // 2001-01-01 00:00:00 UTC = 978,307,200 seconds = 978,307,200,000 milliseconds
    const SECONDS_THRESHOLD = 978307200000 // Year 2001 in milliseconds

    const convertIfSeconds = (num: number): number => {
      // If the number is less than our threshold, it's likely in seconds
      // Also check if it's greater than 0 to avoid negative timestamps
      return num > 0 && num < SECONDS_THRESHOLD ? num * 1000 : num
    }

    // Handle BigInt from Jsonl files
    if (typeof value === 'bigint') {
      const num = Number(value)
      return convertIfSeconds(num)
    }

    // If already a number, assume it's Unix timestamp
    if (typeof value === 'number') {
      return convertIfSeconds(value)
    }

    // Convert to string
    let valueStr: string
    if (typeof value === 'object' && value !== null) {
      valueStr = JSON.stringify(value)
    } else {
      // eslint-disable-next-line @typescript-eslint/no-base-to-string
      valueStr = String(value)
    }

    // Check if it's a numeric string
    const numericValue = Number(valueStr)
    if (!isNaN(numericValue)) {
      return convertIfSeconds(numericValue)
    }

    // Try to parse as date string
    const date = new Date(valueStr)
    if (isNaN(date.getTime())) {
      throw new Error(`Invalid timestamp at row ${rowNumber}: ${valueStr}`)
    }

    return date.getTime()
  }

  /**
   * Parses numeric value
   */
  protected parseNumber(
    value: unknown,
    field: string,
    rowNumber: number
  ): number {
    if (value === null || value === undefined || value === '') {
      throw new Error(`Missing ${field} at row ${rowNumber}`)
    }

    // Convert to string for processing
    let valueStr: string
    if (typeof value === 'object' && value !== null) {
      valueStr = JSON.stringify(value)
    } else {
      // eslint-disable-next-line @typescript-eslint/no-base-to-string
      valueStr = String(value)
    }

    // Remove commas from numbers (e.g., "1,000.50" -> "1000.50")
    valueStr = valueStr.replace(/,/g, '')

    const num = Number(valueStr)
    if (isNaN(num)) {
      throw new Error(`Invalid ${field} at row ${rowNumber}: ${valueStr}`)
    }

    return num
  }
}
