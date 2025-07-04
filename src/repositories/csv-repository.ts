import { createWriteStream, promises as fs } from 'node:fs'
import * as path from 'node:path'
import type { Writable } from 'node:stream'
import type { OhlcvDto } from '../models'
import { formatOhlcv, isValidOhlcv } from '../models'
import logger from '../utils/logger'
import type { CoefficientData, OhlcvQuery, OhlcvRepository, RepositoryConfig } from './ohlcv-repository.interface'
import { RepositoryConnectionError, RepositoryStorageError, RepositoryValidationError } from './ohlcv-repository.interface'

/**
 * CSV-based implementation of the OhlcvRepository interface
 * Organizes data by symbol in separate CSV files with streaming writes
 */
export class CsvRepository implements OhlcvRepository {
  private ready = false
  private basePath = ''
  private readonly writeStreams = new Map<string, Writable>()

  // CSV configuration
  private readonly csvDelimiter = ','
  private readonly csvHeaders = [
    'timestamp',
    'symbol',
    'exchange',
    'open',
    'high',
    'low',
    'close',
    'volume'
  ]

  /**
   * Initialize the CSV repository and set up directory structure
   */
  async initialize(config: RepositoryConfig): Promise<void> {
    try {
      this.basePath = config.connectionString

      // Create base directory if it doesn't exist
      await fs.mkdir(this.basePath, { recursive: true })

      // Create subdirectories for organization
      await fs.mkdir(path.join(this.basePath, 'ohlcv'), { recursive: true })
      await fs.mkdir(path.join(this.basePath, 'coefficients'), { recursive: true })

      this.ready = true
      logger.info('CSV repository initialized', {
        basePath: this.basePath,
        options: config.options
      })
    } catch (error) {
      logger.error('Failed to initialize CSV repository', { error })
      throw new RepositoryConnectionError(
        `Failed to initialize CSV repository: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Save a single OHLCV record
   */
  async save(data: OhlcvDto): Promise<void> {
    this.ensureReady()

    if (!isValidOhlcv(data)) {
      throw new RepositoryValidationError('Invalid OHLCV data')
    }

    try {
      await this.appendToFile(data)
    } catch (error) {
      logger.error('Failed to save OHLCV data', { error, data: formatOhlcv(data) })
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
      // Group data by symbol for efficient file operations
      const dataBySymbol = this.groupDataBySymbol(data)

      // Write each group to its respective file
      for (const [symbol, items] of dataBySymbol) {
        await this.appendManyToFile(symbol, items)
      }

      logger.debug('Saved OHLCV batch to CSV', { count: data.length })
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
   * Group OHLCV data by symbol for efficient batch operations
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
   * Append a single OHLCV record to the appropriate CSV file
   */
  private async appendToFile(data: OhlcvDto): Promise<void> {
    const symbolKey = this.getSymbolKey(data.symbol, data.exchange)
    const filePath = this.getOhlcvFilePath(symbolKey)

    // Check if file exists and needs headers
    const needsHeaders = !(await this.fileExists(filePath))

    // Get or create write stream for this symbol
    const stream = await this.getWriteStream(symbolKey, filePath)

    // Write headers if this is a new file
    if (needsHeaders) {
      await this.writeHeaders(stream)
    }

    // Convert data to CSV row and write
    const csvRow = this.formatOhlcvAsCsvRow(data)
    await this.writeToStream(stream, csvRow + '\n')
  }

  /**
   * Append multiple OHLCV records to a CSV file efficiently
   */
  private async appendManyToFile(symbolKey: string, data: OhlcvDto[]): Promise<void> {
    const filePath = this.getOhlcvFilePath(symbolKey)

    // Check if file exists and needs headers
    const needsHeaders = !(await this.fileExists(filePath))

    // Get or create write stream for this symbol
    const stream = await this.getWriteStream(symbolKey, filePath)

    // Write headers if this is a new file
    if (needsHeaders) {
      await this.writeHeaders(stream)
    }

    // Sort data by timestamp for consistent ordering
    const sortedData = [...data].sort((a, b) => a.timestamp - b.timestamp)

    // Write all rows
    for (const item of sortedData) {
      const csvRow = this.formatOhlcvAsCsvRow(item)
      await this.writeToStream(stream, csvRow + '\n')
    }
  }

  /**
   * Get or create a write stream for a symbol
   */
  private async getWriteStream(symbolKey: string, filePath: string): Promise<Writable> {
    if (this.writeStreams.has(symbolKey)) {
      return this.writeStreams.get(symbolKey)!
    }

    // Ensure directory exists
    await fs.mkdir(path.dirname(filePath), { recursive: true })

    // Create write stream in append mode
    const stream = createWriteStream(filePath, {
      flags: 'a',
      encoding: 'utf8',
      highWaterMark: 64 * 1024 // 64KB buffer
    })

    this.writeStreams.set(symbolKey, stream)
    return stream
  }

  /**
   * Write headers to a CSV file
   */
  private async writeHeaders(stream: Writable): Promise<void> {
    const headerRow = this.csvHeaders.join(this.csvDelimiter) + '\n'
    await this.writeToStream(stream, headerRow)
  }

  /**
   * Write data to a stream with proper error handling
   */
  private writeToStream(stream: Writable, data: string): Promise<void> {
    return new Promise((resolve, reject) => {
      stream.write(data, (error) => {
        if (error) {
          reject(error)
        } else {
          resolve()
        }
      })
    })
  }

  /**
   * Format OHLCV data as a CSV row
   */
  private formatOhlcvAsCsvRow(data: OhlcvDto): string {
    const values = [
      data.timestamp,
      this.escapeCsvValue(data.symbol),
      this.escapeCsvValue(data.exchange),
      data.open,
      data.high,
      data.low,
      data.close,
      data.volume
    ]

    return values.join(this.csvDelimiter)
  }

  /**
   * Escape CSV values that contain special characters
   */
  private escapeCsvValue(value: string): string {
    if (value.includes(this.csvDelimiter) || value.includes('"') || value.includes('\n')) {
      return `"${value.replace(/"/g, '""')}"`
    }
    return value
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
    return path.join(this.basePath, 'ohlcv', `${symbolKey}.csv`)
  }

  /**
   * Get the file path for coefficient data
   */
  private getCoefficientFilePath(): string {
    return path.join(this.basePath, 'coefficients', 'coefficients.csv')
  }

  /**
   * Check if a file exists
   */
  private async fileExists(filePath: string): Promise<boolean> {
    try {
      await fs.access(filePath)
      return true
    } catch {
      return false
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
        const data = await this.readCsvFile(file)
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
        const data = await this.readCsvFile(file)
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
        const data = await this.readCsvFile(file)
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
   * Get relevant CSV files based on symbol and exchange filters
   */
  private async getRelevantFiles(symbol?: string, exchange?: string): Promise<string[]> {
    const ohlcvDir = path.join(this.basePath, 'ohlcv')

    try {
      const files = await fs.readdir(ohlcvDir)
      const csvFiles = files.filter(file => file.endsWith('.csv'))

      if (symbol && exchange) {
        const symbolKey = this.getSymbolKey(symbol, exchange)
        const targetFile = `${symbolKey}.csv`
        return csvFiles.includes(targetFile)
          ? [path.join(ohlcvDir, targetFile)]
          : []
      }

      if (symbol) {
        const matchingFiles = csvFiles.filter(file =>
          file.includes(symbol.replace(/[^a-zA-Z0-9_-]/g, '_')),
        )
        return matchingFiles.map(file => path.join(ohlcvDir, file))
      }

      // Return all files
      return csvFiles.map(file => path.join(ohlcvDir, file))
    } catch (error) {
      // Directory might not exist yet
      logger.debug('Directory does not exist', { error })
      return []
    }
  }

  /**
   * Read and parse a CSV file
   */
  private async readCsvFile(filePath: string): Promise<OhlcvDto[]> {
    const data: OhlcvDto[] = []

    try {
      const content = await fs.readFile(filePath, 'utf8')
      const lines = content.split('\n').filter(line => line.trim())

      if (lines.length === 0) return []

      // Skip header row
      const dataLines = lines.slice(1)

      for (const line of dataLines) {
        const parsedRow = this.parseCsvRow(line)
        if (parsedRow) {
          data.push(parsedRow)
        }
      }

      return data
    } catch (error) {
      logger.error('Failed to read CSV file', { error, filePath })
      throw new RepositoryStorageError(
        `Failed to read CSV file: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Parse a single CSV row into OhlcvDto
   */
  private parseCsvRow(line: string): OhlcvDto | null {
    try {
      const values = this.parseCsvLine(line)

      if (values.length < this.csvHeaders.length) {
        return null
      }

      const ohlcv: OhlcvDto = {
        timestamp: parseInt(values[0]!, 10),
        symbol: values[1]!,
        exchange: values[2]!,
        open: parseFloat(values[3]!),
        high: parseFloat(values[4]!),
        low: parseFloat(values[5]!),
        close: parseFloat(values[6]!),
        volume: parseFloat(values[7]!)
      }

      return isValidOhlcv(ohlcv) ? ohlcv : null
    } catch {
      return null
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

      if (char === this.csvDelimiter && !inQuotes) {
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
      await this.appendCoefficientToFile(coefficient)
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
      for (const coefficient of coefficients) {
        await this.appendCoefficientToFile(coefficient)
      }
      logger.debug('Saved coefficients batch to CSV', { count: coefficients.length })
    } catch (error) {
      logger.error('Failed to save coefficients batch', { error, count: coefficients.length })
      throw new RepositoryStorageError(
        `Failed to save coefficients batch: ${String(error)}`,
        error instanceof Error ? error : undefined
      )
    }
  }

  /**
   * Append a coefficient to the coefficients CSV file
   */
  private async appendCoefficientToFile(coefficient: CoefficientData): Promise<void> {
    const filePath = this.getCoefficientFilePath()
    const needsHeaders = !(await this.fileExists(filePath))

    // Ensure directory exists
    await fs.mkdir(path.dirname(filePath), { recursive: true })

    // Format coefficient as CSV row
    const csvRow = [
      this.escapeCsvValue(coefficient.name),
      coefficient.symbol ? this.escapeCsvValue(coefficient.symbol) : '',
      coefficient.exchange ? this.escapeCsvValue(coefficient.exchange) : '',
      coefficient.value,
      coefficient.metadata ? this.escapeCsvValue(JSON.stringify(coefficient.metadata)) : '',
      coefficient.timestamp
    ].join(this.csvDelimiter)

    let content = csvRow + '\n'

    // Add headers if this is a new file
    if (needsHeaders) {
      const headers = ['name', 'symbol', 'exchange', 'value', 'metadata', 'timestamp']
      content = headers.join(this.csvDelimiter) + '\n' + content
    }

    // Append to file
    await fs.appendFile(filePath, content, 'utf8')
  }

  /**
   * Get a coefficient value by name (not fully implemented for CSV - would require reading entire file)
   */
  async getCoefficient(
    name: string,
    symbol?: string,
    exchange?: string
  ): Promise<CoefficientData | null> {
    this.ensureReady()

    // For CSV implementation, this is inefficient as we need to read the entire file
    // In a production system, you might want to use SQLite for coefficients even with CSV for OHLCV
    logger.warn('getCoefficient is inefficient with CSV storage - consider using SQLite for coefficients')

    try {
      const coefficients = await this.getCoefficients(name, symbol, exchange)
      return coefficients.length > 0 ? coefficients[0]! : null
    } catch (error) {
      logger.error('Failed to get coefficient', { error, name })
      throw new RepositoryStorageError(
        `Failed to get coefficient: ${String(error)}`,
        error instanceof Error ? error : undefined,
      )
    }
  }

  /**
   * Get multiple coefficients by name pattern (basic implementation)
   */
  async getCoefficients(
    namePattern?: string,
    symbol?: string,
    exchange?: string
  ): Promise<CoefficientData[]> {
    this.ensureReady()

    const filePath = this.getCoefficientFilePath()

    if (!(await this.fileExists(filePath))) {
      return []
    }

    try {
      const content = await fs.readFile(filePath, 'utf8')
      const lines = content.split('\n').filter(line => line.trim())

      if (lines.length <= 1) return [] // No data or just headers

      const coefficients: CoefficientData[] = []
      const dataLines = lines.slice(1) // Skip headers

      for (const line of dataLines) {
        const values = this.parseCsvLine(line)
        if (values.length >= 6) {
          const coefficient: CoefficientData = {
            name: values[0]!,
            symbol: values[1] || undefined,
            exchange: values[2] || undefined,
            value: parseFloat(values[3]!),
            metadata: values[4] && values[4] !== '' ? JSON.parse(values[4]) as Record<string, unknown> : undefined,
            timestamp: parseInt(values[5]!, 10)
          }

          // Apply filters
          if (namePattern && !this.matchesPattern(coefficient.name, namePattern)) {
            continue
          }
          if (symbol && coefficient.symbol !== symbol) {
            continue
          }
          if (exchange && coefficient.exchange !== exchange) {
            continue
          }

          coefficients.push(coefficient)
        }
      }

      return coefficients
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
   * Delete coefficients by name pattern (not efficiently implemented for CSV)
   */
  async deleteCoefficients(
    namePattern: string,
    symbol?: string,
    exchange?: string
  ): Promise<number> {
    this.ensureReady()

    logger.warn('deleteCoefficients is inefficient with CSV storage - consider using SQLite for coefficients')

    try {
      const allCoefficients = await this.getCoefficients()
      const filtered = allCoefficients.filter(coeff => {
        if (this.matchesPattern(coeff.name, namePattern)) {
          if (symbol && coeff.symbol !== symbol) return true
          return !!(exchange && coeff.exchange !== exchange)
           // This one should be deleted
        }
        return true // Keep this one
      })

      const deletedCount = allCoefficients.length - filtered.length

      // Rewrite the entire file with remaining coefficients
      if (filtered.length === 0) {
        // Delete the file
        const filePath = this.getCoefficientFilePath()
        if (await this.fileExists(filePath)) {
          await fs.unlink(filePath)
        }
      } else {
        // Rewrite with remaining data
        const filePath = this.getCoefficientFilePath()
        const headers = ['name', 'symbol', 'exchange', 'value', 'metadata', 'timestamp']
        let content = headers.join(this.csvDelimiter) + '\n'

        for (const coeff of filtered) {
          const row = [
            this.escapeCsvValue(coeff.name),
            coeff.symbol ? this.escapeCsvValue(coeff.symbol) : '',
            coeff.exchange ? this.escapeCsvValue(coeff.exchange) : '',
            coeff.value,
            coeff.metadata ? this.escapeCsvValue(JSON.stringify(coeff.metadata)) : '',
            coeff.timestamp
          ].join(this.csvDelimiter)
          content += row + '\n'
        }

        await fs.writeFile(filePath, content, 'utf8')
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
        const data = await this.readCsvFile(file)
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
        const data = await this.readCsvFile(file)
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
   * Delete OHLCV data within a specific date range (not efficiently implemented for CSV)
   */
  async deleteBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<number> {
    this.ensureReady()

    logger.warn('deleteBetweenDates is inefficient with CSV storage - consider using SQLite for deletions')

    try {
      const files = await this.getRelevantFiles(symbol, exchange)
      let totalDeleted = 0

      for (const file of files) {
        const data = await this.readCsvFile(file)
        const filtered = data.filter(
          item => item.timestamp < startTime || item.timestamp > endTime,
        )

        const deletedCount = data.length - filtered.length
        totalDeleted += deletedCount

        if (deletedCount > 0) {
          // Rewrite the file with remaining data
          await this.rewriteCsvFile(file, filtered)
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
   * Rewrite a CSV file with new data
   */
  private async rewriteCsvFile(filePath: string, data: OhlcvDto[]): Promise<void> {
    if (data.length === 0) {
      // Delete the file if no data remains
      await fs.unlink(filePath)
      return
    }

    // Create new content
    let content = this.csvHeaders.join(this.csvDelimiter) + '\n'
    for (const item of data) {
      content += this.formatOhlcvAsCsvRow(item) + '\n'
    }

    await fs.writeFile(filePath, content, 'utf8')
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
        const data = await this.readCsvFile(file)
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
          const stats = await fs.stat(file)
          storageSize += stats.size
        } catch {
          // File might not be accessible
        }
      }

      // Add coefficients file size
      const coeffPath = this.getCoefficientFilePath()
      if (await this.fileExists(coeffPath)) {
        try {
          const stats = await fs.stat(coeffPath)
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
      // Close and reopen all write streams to flush buffers
      for (const [, stream] of this.writeStreams) {
        await new Promise<void>((resolve, reject) => {
          stream.end((error?: Error) => {
            if (error) reject(error)
            else resolve()
          })
        })
      }

      this.writeStreams.clear()
      logger.debug('Flushed CSV repository write streams')
    } catch (error) {
      logger.error('Failed to flush CSV repository', { error })
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
        logger.info('CSV repository closed')
      } catch (error) {
        logger.error('Error closing CSV repository', { error })
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
