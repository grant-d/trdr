import { asyncBufferFromFile, parquetReadObjects } from 'hyparquet'
import { parquetWriteFile } from 'hyparquet-writer'
import { mkdir, readdir, stat } from 'node:fs/promises'
import * as path from 'node:path'
import type { OhlcvDto } from '../models/ohlcv.dto'
import { isValidOhlcv } from '../models/ohlcv.dto'
import logger from '../utils/logger'
import type {
  CoefficientData,
  OhlcvQuery,
  OhlcvRepository,
  RepositoryConfig
} from './ohlcv-repository.interface'
import {
  RepositoryConnectionError,
  RepositoryStorageError,
  RepositoryValidationError
} from './ohlcv-repository.interface'

// Parquet row type definitions
interface ParquetOhlcvRow {
  readonly timestamp: number | bigint
  readonly symbol: string
  readonly exchange: string
  readonly open: number
  readonly high: number
  readonly low: number
  readonly close: number
  readonly volume: number
}

interface ParquetCoefficientRow {
  readonly name: string
  readonly symbol: string
  readonly exchange: string
  readonly value: number
  readonly metadata: string
  readonly timestamp: number | bigint
}

/**
 * Parquet-based implementation of the OhlcvRepository interface
 * Optimized for columnar storage and analytical queries
 */
export class ParquetRepository implements OhlcvRepository {
  private ready = false
  private basePath = ''
  private batchSize = 10000

  // In-memory buffers for batch writing
  private ohlcvBuffer: OhlcvDto[] = []
  private coefficientBuffer: CoefficientData[] = []

  /**
   * Initialize the Parquet repository and set up directory structure
   */
  async initialize(config: RepositoryConfig): Promise<void> {
    try {
      this.basePath = config.connectionString
      
      // Configure options if provided
      if (config.options) {
        this.batchSize = (config.options.batchSize as number) || this.batchSize
      }

      // Create base directory if it doesn't exist
      await mkdir(this.basePath, { recursive: true })

      // Create subdirectories for organization
      await mkdir(path.join(this.basePath, 'ohlcv'), { recursive: true })
      await mkdir(path.join(this.basePath, 'coefficients'), { recursive: true })

      this.ready = true
      logger.info('Parquet repository initialized', {
        basePath: this.basePath,
        batchSize: this.batchSize,
        options: config.options
      })
    } catch (error) {
      logger.error('Failed to initialize Parquet repository', { error })
      throw new RepositoryConnectionError(
        `Failed to initialize Parquet repository: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Save a single OHLCV record (buffers for batch writing)
   */
  async save(data: OhlcvDto): Promise<void> {
    this.ensureReady()
    
    if (!isValidOhlcv(data)) {
      throw new RepositoryValidationError('Invalid OHLCV data')
    }

    try {
      this.ohlcvBuffer.push(data)
      
      // Flush buffer if it reaches batch size
      if (this.ohlcvBuffer.length >= this.batchSize) {
        await this.flushOhlcvBuffer()
      }
    } catch (error) {
      logger.error('Failed to save OHLCV data', { error, data })
      throw new RepositoryStorageError(
        `Failed to save OHLCV data: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Save multiple OHLCV records in a batch operation
   */
  async saveMany(data: OhlcvDto[]): Promise<void> {
    this.ensureReady()

    if (data.length === 0) return

    // Validate all data first
    for (const item of data) {
      if (!isValidOhlcv(item)) {
        throw new RepositoryValidationError(`Invalid OHLCV data for ${item.symbol}`)
      }
    }

    try {
      // Group data by symbol-exchange for optimal file organization
      const dataBySymbol = this.groupDataBySymbol(data)

      for (const [symbolKey, items] of dataBySymbol) {
        await this.writeParquetFile(symbolKey, items)
      }

      logger.debug('Saved OHLCV batch to Parquet', { count: data.length })
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
   * Group OHLCV data by symbol-exchange combination
   */
  private groupDataBySymbol(data: OhlcvDto[]): Map<string, OhlcvDto[]> {
    const grouped = new Map<string, OhlcvDto[]>()

    for (const item of data) {
      const key = this.getSymbolKey(item.symbol, item.exchange)
      if (!grouped.has(key)) {
        grouped.set(key, [])
      }
      grouped.get(key)!.push(item)
    }

    return grouped
  }

  /**
   * Write OHLCV data to a Parquet file
   */
  private async writeParquetFile(symbolKey: string, data: OhlcvDto[]): Promise<void> {
    const filePath = this.getOhlcvFilePath(symbolKey)
    
    // Sort data by timestamp for optimal columnar compression
    const sortedData = [...data].sort((a, b) => a.timestamp - b.timestamp)


    try {
      // Check if file exists to determine if we need to append
      const fileExists = await this.fileExists(filePath)
      
      if (fileExists) {
        // Read existing data and merge
        const existingData = await this.readParquetFile(filePath)
        const mergedData = [...existingData, ...sortedData]
        const uniqueData = this.deduplicateOhlcv(mergedData)
        
        // Rewrite with merged data
        await this.writeParquetData(filePath, uniqueData)
      } else {
        // Write new file
        await this.writeParquetData(filePath, sortedData)
      }
    } catch (error) {
      logger.error('Failed to write Parquet file', { error, filePath, count: sortedData.length })
      throw new RepositoryStorageError(
        `Failed to write Parquet file: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Write OHLCV data to Parquet format
   */
  private async writeParquetData(filePath: string, data: OhlcvDto[]): Promise<void> {
    // Ensure directory exists
    await mkdir(path.dirname(filePath), { recursive: true })

    // Sort data by timestamp for optimal columnar compression
    const sortedData = [...data].sort((a, b) => a.timestamp - b.timestamp)

    try {
      // Write Parquet file using hyparquet-writer
      parquetWriteFile({
        filename: filePath,
        columnData: [
          { name: 'timestamp', data: sortedData.map(d => BigInt(d.timestamp)), type: 'INT64' },
          { name: 'symbol', data: sortedData.map(d => d.symbol), type: 'STRING' },
          { name: 'exchange', data: sortedData.map(d => d.exchange), type: 'STRING' },
          { name: 'open', data: sortedData.map(d => d.open), type: 'DOUBLE' },
          { name: 'high', data: sortedData.map(d => d.high), type: 'DOUBLE' },
          { name: 'low', data: sortedData.map(d => d.low), type: 'DOUBLE' },
          { name: 'close', data: sortedData.map(d => d.close), type: 'DOUBLE' },
          { name: 'volume', data: sortedData.map(d => d.volume), type: 'DOUBLE' }
        ]
      })
    } catch (error) {
      logger.error('Failed to write Parquet file', { error, filePath, count: sortedData.length })
      throw new RepositoryStorageError(
        `Failed to write Parquet file: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Read OHLCV data from a Parquet file
   */
  private async readParquetFile(filePath: string): Promise<OhlcvDto[]> {
    try {
      const buffer = await asyncBufferFromFile(filePath)
      const rows = await parquetReadObjects({ file: buffer })
      const data: OhlcvDto[] = []
      
      for (const row of rows) {
        const rowData = row as Partial<ParquetOhlcvRow>
        const ohlcv: OhlcvDto = {
          timestamp: Number(rowData.timestamp),
          symbol: String(rowData.symbol),
          exchange: String(rowData.exchange),
          open: Number(rowData.open),
          high: Number(rowData.high),
          low: Number(rowData.low),
          close: Number(rowData.close),
          volume: Number(rowData.volume)
        }
        
        if (isValidOhlcv(ohlcv)) {
          data.push(ohlcv)
        }
      }

      return data
    } catch (error) {
      logger.error('Failed to read Parquet file', { error, filePath })
      throw new RepositoryStorageError(
        `Failed to read Parquet file: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Remove duplicate OHLCV records based on timestamp, symbol, exchange
   */
  private deduplicateOhlcv(data: OhlcvDto[]): OhlcvDto[] {
    const seen = new Set<string>()
    const unique: OhlcvDto[] = []

    for (const item of data) {
      const key = `${item.timestamp}-${item.symbol}-${item.exchange}`
      if (!seen.has(key)) {
        seen.add(key)
        unique.push(item)
      }
    }

    return unique.sort((a, b) => a.timestamp - b.timestamp)
  }

  /**
   * Flush the OHLCV buffer to disk
   */
  private async flushOhlcvBuffer(): Promise<void> {
    if (this.ohlcvBuffer.length === 0) return

    const buffer = [...this.ohlcvBuffer]
    this.ohlcvBuffer = []

    await this.saveMany(buffer)
  }

  /**
   * Generate a unique key for symbol-exchange combination
   */
  private getSymbolKey(symbol: string, exchange: string): string {
    return `${symbol}_${exchange}`.replace(/[^a-zA-Z0-9_-]/g, '_')
  }

  /**
   * Get the file path for OHLCV data for a specific symbol
   */
  private getOhlcvFilePath(symbolKey: string): string {
    return path.join(this.basePath, 'ohlcv', `${symbolKey}.parquet`)
  }

  /**
   * Get the file path for coefficient data
   */
  private getCoefficientFilePath(): string {
    return path.join(this.basePath, 'coefficients', 'coefficients.parquet')
  }

  /**
   * Check if a file exists
   */
  private async fileExists(filePath: string): Promise<boolean> {
    try {
      await stat(filePath)
      return true
    } catch {
      return false
    }
  }

  /**
   * Get relevant Parquet files based on symbol and exchange filters
   */
  private async getRelevantFiles(symbol?: string, exchange?: string): Promise<string[]> {
    const ohlcvDir = path.join(this.basePath, 'ohlcv')
    
    try {
      const files = await readdir(ohlcvDir)
      const parquetFiles = files.filter(file => file.endsWith('.parquet'))

      if (symbol && exchange) {
        const symbolKey = this.getSymbolKey(symbol, exchange)
        const targetFile = `${symbolKey}.parquet`
        return parquetFiles.includes(targetFile) 
          ? [path.join(ohlcvDir, targetFile)]
          : []
      }

      if (symbol) {
        const sanitizedSymbol = symbol.replace(/[^a-zA-Z0-9_-]/g, '_')
        const matchingFiles = parquetFiles.filter(file => 
          file.includes(sanitizedSymbol)
        )
        return matchingFiles.map(file => path.join(ohlcvDir, file))
      }

      // Return all files
      return parquetFiles.map(file => path.join(ohlcvDir, file))
    } catch {
      // Directory might not exist yet
      return []
    }
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
    this.ensureReady()

    try {
      const files = await this.getRelevantFiles(symbol, exchange)
      const allData: OhlcvDto[] = []

      for (const file of files) {
        const data = await this.readParquetFile(file)
        const filtered = data.filter(
          item => item.timestamp >= startTime && item.timestamp <= endTime
        )
        allData.push(...filtered)
      }

      // Sort by timestamp
      return allData.sort((a, b) => a.timestamp - b.timestamp)
    } catch (error) {
      logger.error('Failed to get OHLCV data by date range', { error, startTime, endTime })
      throw new RepositoryStorageError(
        `Failed to get OHLCV data by date range: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
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
    this.ensureReady()

    try {
      const files = await this.getRelevantFiles(symbol, exchange)
      const allData: OhlcvDto[] = []

      for (const file of files) {
        const data = await this.readParquetFile(file)
        allData.push(...data)
      }

      // Sort by timestamp (most recent first)
      const sorted = allData.sort((a, b) => b.timestamp - a.timestamp)

      // Apply pagination
      const start = offset || 0
      const end = limit ? start + limit : undefined
      return sorted.slice(start, end)
    } catch (error) {
      logger.error('Failed to get OHLCV data by symbol', { error, symbol })
      throw new RepositoryStorageError(
        `Failed to get OHLCV data by symbol: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get OHLCV data using flexible query parameters
   */
  async query(query: OhlcvQuery): Promise<OhlcvDto[]> {
    this.ensureReady()

    try {
      const files = await this.getRelevantFiles(query.symbol, query.exchange)
      const allData: OhlcvDto[] = []

      for (const file of files) {
        const data = await this.readParquetFile(file)
        let filtered = data

        // Apply filters
        if (query.startTime) {
          filtered = filtered.filter(item => item.timestamp >= query.startTime!)
        }
        if (query.endTime) {
          filtered = filtered.filter(item => item.timestamp <= query.endTime!)
        }

        allData.push(...filtered)
      }

      // Sort by timestamp
      let sorted = allData.sort((a, b) => a.timestamp - b.timestamp)

      // Apply pagination
      if (query.offset) {
        sorted = sorted.slice(query.offset)
      }
      if (query.limit) {
        sorted = sorted.slice(0, query.limit)
      }

      return sorted
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

    try {
      const data = await this.getBySymbol(symbol, exchange, 1)
      return data.length > 0 ? data[0]!.timestamp : null
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

    try {
      const data = await this.getBySymbol(symbol, exchange)
      if (data.length === 0) return null

      // Find the earliest timestamp
      return Math.min(...data.map(item => item.timestamp))
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

    try {
      const data = await this.getBySymbol(symbol, exchange)
      return data.length
    } catch (error) {
      logger.error('Failed to get count', { error, symbol })
      throw new RepositoryStorageError(
        `Failed to get count: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Save a coefficient value
   */
  async saveCoefficient(coefficient: CoefficientData): Promise<void> {
    this.ensureReady()

    try {
      this.coefficientBuffer.push(coefficient)
      
      // Flush buffer if it reaches batch size
      if (this.coefficientBuffer.length >= this.batchSize) {
        await this.flushCoefficientBuffer()
      }
    } catch (error) {
      logger.error('Failed to save coefficient', { error, coefficient })
      throw new RepositoryStorageError(
        `Failed to save coefficient: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Save multiple coefficient values in a batch
   */
  async saveCoefficients(coefficients: CoefficientData[]): Promise<void> {
    this.ensureReady()

    if (coefficients.length === 0) return

    try {
      await this.writeCoefficientFile(coefficients)
      logger.debug('Saved coefficients batch to Parquet', { count: coefficients.length })
    } catch (error) {
      logger.error('Failed to save coefficients batch', { error, count: coefficients.length })
      throw new RepositoryStorageError(
        `Failed to save coefficients batch: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Write coefficient data to Parquet file
   */
  private async writeCoefficientFile(coefficients: CoefficientData[]): Promise<void> {
    const filePath = this.getCoefficientFilePath()
    
    try {
      // Check if file exists to determine if we need to append
      const fileExists = await this.fileExists(filePath)
      
      if (fileExists) {
        // Read existing data and merge
        const existingData = await this.readCoefficientFile()
        const mergedData = [...existingData, ...coefficients]
        
        // Rewrite with merged data
        await this.writeCoefficientData(filePath, mergedData)
      } else {
        // Write new file
        await this.writeCoefficientData(filePath, coefficients)
      }
    } catch (error) {
      logger.error('Failed to write coefficient Parquet file', { error, filePath })
      throw new RepositoryStorageError(
        `Failed to write coefficient file: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Write coefficient data to Parquet format
   */
  private async writeCoefficientData(filePath: string, data: CoefficientData[]): Promise<void> {
    // Ensure directory exists
    await mkdir(path.dirname(filePath), { recursive: true })

    // Sort data by timestamp for optimal columnar compression
    const sortedData = [...data].sort((a, b) => a.timestamp - b.timestamp)

    try {
      // Write coefficient Parquet file
      parquetWriteFile({
        filename: filePath,
        columnData: [
          { name: 'name', data: sortedData.map(d => d.name), type: 'STRING' },
          { name: 'symbol', data: sortedData.map(d => d.symbol || ''), type: 'STRING' },
          { name: 'exchange', data: sortedData.map(d => d.exchange || ''), type: 'STRING' },
          { name: 'value', data: sortedData.map(d => d.value), type: 'DOUBLE' },
          { name: 'metadata', data: sortedData.map(d => d.metadata ? JSON.stringify(d.metadata) : ''), type: 'STRING' },
          { name: 'timestamp', data: sortedData.map(d => BigInt(d.timestamp)), type: 'INT64' }
        ]
      })
    } catch (error) {
      logger.error('Failed to write coefficient Parquet file', { error, filePath })
      throw new RepositoryStorageError(
        `Failed to write coefficient file: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Read coefficient data from Parquet file
   */
  private async readCoefficientFile(): Promise<CoefficientData[]> {
    const filePath = this.getCoefficientFilePath()
    
    if (!(await this.fileExists(filePath))) {
      return []
    }

    try {
      const buffer = await asyncBufferFromFile(filePath)
      const rows = await parquetReadObjects({ file: buffer })
      const data: CoefficientData[] = []
      
      for (const row of rows) {
        const rowData = row as Partial<ParquetCoefficientRow>
        const coefficient: CoefficientData = {
          name: String(rowData.name),
          symbol: String(rowData.symbol) || undefined,
          exchange: String(rowData.exchange) || undefined,
          value: Number(rowData.value),
          metadata: rowData.metadata ? JSON.parse(String(rowData.metadata)) as Record<string, unknown> : undefined,
          timestamp: Number(rowData.timestamp)
        }
        
        data.push(coefficient)
      }

      return data
    } catch (error) {
      logger.error('Failed to read coefficient Parquet file', { error, filePath })
      throw new RepositoryStorageError(
        `Failed to read coefficient file: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Flush the coefficient buffer to disk
   */
  private async flushCoefficientBuffer(): Promise<void> {
    if (this.coefficientBuffer.length === 0) return

    const buffer = [...this.coefficientBuffer]
    this.coefficientBuffer = []

    await this.saveCoefficients(buffer)
  }

  /**
   * Get a coefficient value by name (basic implementation)
   */
  async getCoefficient(
    name: string,
    symbol?: string,
    exchange?: string
  ): Promise<CoefficientData | null> {
    this.ensureReady()
    
    try {
      const coefficients = await this.getCoefficients(name, symbol, exchange)
      return coefficients.length > 0 ? coefficients[0]! : null
    } catch (error) {
      logger.error('Failed to get coefficient', { error, name })
      throw new RepositoryStorageError(
        `Failed to get coefficient: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get multiple coefficients by name pattern
   */
  async getCoefficients(
    namePattern?: string,
    symbol?: string,
    exchange?: string
  ): Promise<CoefficientData[]> {
    this.ensureReady()

    try {
      const allCoefficients = await this.readCoefficientFile()
      
      return allCoefficients.filter(coeff => {
        // Apply filters
        if (namePattern && !this.matchesPattern(coeff.name, namePattern)) {
          return false
        }
        if (symbol && coeff.symbol !== symbol) {
          return false
        }
        if (exchange && coeff.exchange !== exchange) {
          return false
        }
        return true
      })
    } catch (error) {
      logger.error('Failed to get coefficients', { error, namePattern })
      throw new RepositoryStorageError(
        `Failed to get coefficients: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Simple pattern matching for coefficient names
   */
  private matchesPattern(name: string, pattern: string): boolean {
    // Convert glob pattern to regex
    const regexPattern = pattern
      .replace(/\*/g, '.*')
      .replace(/\?/g, '.')
    
    const regex = new RegExp(`^${regexPattern}$`, 'i')
    return regex.test(name)
  }

  /**
   * Delete coefficients by name pattern (inefficient for Parquet)
   */
  async deleteCoefficients(
    namePattern: string,
    symbol?: string,
    exchange?: string
  ): Promise<number> {
    this.ensureReady()

    logger.warn('deleteCoefficients is inefficient with Parquet storage - consider using SQLite for deletions')

    try {
      const allCoefficients = await this.readCoefficientFile()
      const filtered = allCoefficients.filter(coeff => {
        if (this.matchesPattern(coeff.name, namePattern)) {
          if (symbol && coeff.symbol !== symbol) return true
          if (exchange && coeff.exchange !== exchange) return true
          return false // This one should be deleted
        }
        return true // Keep this one
      })

      const deletedCount = allCoefficients.length - filtered.length

      // Rewrite the file with remaining coefficients
      const filePath = this.getCoefficientFilePath()
      if (filtered.length === 0) {
        // Delete the file if no coefficients remain
        try {
          const fs = await import('node:fs/promises')
          await fs.unlink(filePath)
        } catch {
          // File might not exist
        }
      } else {
        await this.writeCoefficientData(filePath, filtered)
      }

      return deletedCount
    } catch (error) {
      logger.error('Failed to delete coefficients', { error, namePattern })
      throw new RepositoryStorageError(
        `Failed to delete coefficients: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Get all unique symbols in the repository
   */
  async getSymbols(exchange?: string): Promise<string[]> {
    this.ensureReady()

    try {
      const files = await this.getRelevantFiles(undefined, exchange)
      const symbols = new Set<string>()

      for (const file of files) {
        const data = await this.readParquetFile(file)
        for (const item of data) {
          if (!exchange || item.exchange === exchange) {
            symbols.add(item.symbol)
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

    try {
      const files = await this.getRelevantFiles()
      const exchanges = new Set<string>()

      for (const file of files) {
        const data = await this.readParquetFile(file)
        for (const item of data) {
          exchanges.add(item.exchange)
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
   * Delete OHLCV data within a specific date range (inefficient for Parquet)
   */
  async deleteBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<number> {
    this.ensureReady()

    logger.warn('deleteBetweenDates is inefficient with Parquet storage - consider using SQLite for deletions')

    try {
      const files = await this.getRelevantFiles(symbol, exchange)
      let totalDeleted = 0

      for (const file of files) {
        const data = await this.readParquetFile(file)
        const filtered = data.filter(
          item => item.timestamp < startTime || item.timestamp > endTime
        )

        const deletedCount = data.length - filtered.length
        totalDeleted += deletedCount

        if (deletedCount > 0) {
          // Rewrite the file with remaining data
          if (filtered.length === 0) {
            // Delete the file if no data remains
            const fs = await import('node:fs/promises')
            await fs.unlink(file)
          } else {
            await this.writeParquetData(file, filtered)
          }
        }
      }

      return totalDeleted
    } catch (error) {
      logger.error('Failed to delete between dates', { error, startTime, endTime })
      throw new RepositoryStorageError(
        `Failed to delete between dates: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Check if the repository is properly initialized and ready to use
   */
  isReady(): boolean {
    return this.ready
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

    try {
      const files = await this.getRelevantFiles()
      let totalRecords = 0
      let earliest: number | null = null
      let latest: number | null = null
      const symbols = new Set<string>()
      const exchanges = new Set<string>()
      let storageSize = 0

      for (const file of files) {
        const data = await this.readParquetFile(file)
        totalRecords += data.length

        for (const item of data) {
          symbols.add(item.symbol)
          exchanges.add(item.exchange)

          if (earliest === null || item.timestamp < earliest) {
            earliest = item.timestamp
          }
          if (latest === null || item.timestamp > latest) {
            latest = item.timestamp
          }
        }

        // Add file size
        try {
          const stats = await stat(file)
          storageSize += stats.size
        } catch {
          // File might not be accessible
        }
      }

      // Add coefficients file size
      const coeffPath = this.getCoefficientFilePath()
      if (await this.fileExists(coeffPath)) {
        try {
          const stats = await stat(coeffPath)
          storageSize += stats.size
        } catch {
          // File might not be accessible
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
   * Flush any pending writes to ensure data persistence
   */
  async flush(): Promise<void> {
    this.ensureReady()

    try {
      await this.flushOhlcvBuffer()
      await this.flushCoefficientBuffer()
      logger.debug('Flushed Parquet repository buffers')
    } catch (error) {
      logger.error('Failed to flush Parquet repository', { error })
      throw new RepositoryStorageError(
        `Failed to flush repository: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Close the repository and clean up resources
   */
  async close(): Promise<void> {
    if (this.ready) {
      try {
        await this.flush()
        this.ready = false
        logger.info('Parquet repository closed')
      } catch (error) {
        logger.error('Error closing Parquet repository', { error })
        throw new RepositoryStorageError(
          `Error closing repository: ${String(error)}`,
          error instanceof Error ? error : undefined
        )
      }
    }
  }

  /**
   * Ensure the repository is ready for operations
   */
  private ensureReady(): void {
    if (!this.ready) {
      throw new RepositoryConnectionError('Repository not initialized')
    }
  }
}