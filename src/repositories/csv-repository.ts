import { promises as fs } from 'node:fs'
import type { OhlcvDto } from '../models'
import { isValidOhlcv } from '../models'
import logger from '../utils/logger'
import { FileBasedRepository } from './file-based-repository'
import type { OhlcvQuery, RepositoryConfig } from './ohlcv-repository.interface'
import { RepositoryStorageError } from './ohlcv-repository.interface'

/**
 * CSV-based implementation of the OhlcvRepository interface
 * Extends FileBasedRepository to gain deduplication and shared functionality
 * Organizes data in CSV files with streaming writes
 */
export class CsvRepository extends FileBasedRepository {
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
   * Get the repository type name for logging
   */
  protected getRepositoryType(): string {
    return 'CSV'
  }


  /**
   * Perform additional initialization specific to CSV format
   */
  protected async performAdditionalInitialization(_config: RepositoryConfig): Promise<void> {
    // No additional initialization needed for CSV
  }

  /**
   * Perform additional cleanup specific to CSV format
   */
  protected async performAdditionalCleanup(): Promise<void> {
    // No additional cleanup needed - streams are handled by base class
  }

  /**
   * Write a batch of OHLCV records to CSV storage
   */
  protected async writeOhlcvBatch(data: OhlcvDto[]): Promise<void> {
    // Sort data by timestamp for consistent ordering
    const sortedData = [...data].sort((a, b) => a.timestamp - b.timestamp)
    
    // Check if file exists and needs headers
    const needsHeaders = !(await this.fileExists(this.basePath))
    
    // Prepare the content to write
    let content = ''
    
    // Add headers if this is a new file
    if (needsHeaders && sortedData.length > 0) {
      // Generate headers using expected fields if available, otherwise from first record
      const headers = this.expectedOutputFields.length > 0 
        ? this.generateHeadersFromExpected()
        : this.generateHeaders(sortedData[0]!)
      content += headers.join(this.csvDelimiter) + '\n'
    }
    
    // Add all rows
    for (const item of sortedData) {
      content += this.formatOhlcvAsCsvRow(item) + '\n'
    }
    
    // Write or append based on whether file exists
    if (needsHeaders) {
      await this.writeToFile(this.basePath, content)
    } else {
      await this.appendToFile(this.basePath, content)
    }
  }


  /**
   * Get OHLCV data using flexible query parameters
   */
  async query(query: OhlcvQuery): Promise<OhlcvDto[]> {
    this.ensureReady()
    
    // Flush buffer before reading to ensure we have all data
    await this.flush()

    try {
      const files = await this.getRelevantFiles(query.symbol, query.exchange)
      const allData: OhlcvDto[] = []

      for (const file of files) {
        const data = await this.readCsvFile(file)
        const filtered = data.filter(item => this.matchesQuery(item, query))
        allData.push(...filtered)
      }

      // Sort by timestamp
      const sorted = allData.sort((a, b) => a.timestamp - b.timestamp)
      
      return this.applyPagination(sorted, query)
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
    // Since we're using a single file approach, just return the base path if it exists
    if (await this.fileExists(this.basePath)) {
      // If filtering by symbol/exchange, check if it matches our single symbol/exchange
      if (symbol && this.singleSymbol && symbol !== this.singleSymbol) {
        return []
      }
      if (exchange && this.singleExchange && exchange !== this.singleExchange) {
        return []
      }
      return [this.basePath]
    }
    return []
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

      // Parse header row to know field names
      const headerLine = lines[0]
      const headers = headerLine ? this.parseCsvLine(headerLine) : undefined
      
      // Skip header row
      const dataLines = lines.slice(1)

      for (const line of dataLines) {
        const parsedRow = this.parseCsvRow(line, headers)
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
  private parseCsvRow(line: string, headers?: string[]): OhlcvDto | null {
    try {
      const values = this.parseCsvLine(line)

      if (values.length < this.csvHeaders.length) {
        return null
      }

      const ohlcv: Partial<OhlcvDto> = {
        timestamp: this.parseTimestamp(values[0]!),
        symbol: values[1]!,
        exchange: values[2]!,
        open: parseFloat(values[3]!),
        high: parseFloat(values[4]!),
        low: parseFloat(values[5]!),
        close: parseFloat(values[6]!),
        volume: parseFloat(values[7]!)
      }

      // Add any additional fields from transforms
      if (headers && values.length > this.csvHeaders.length) {
        for (let i = this.csvHeaders.length; i < headers.length && i < values.length; i++) {
          const fieldName = headers[i]
          if (fieldName) {
            const value = values[i]
            if (value !== undefined && value !== '') {
              // Try to parse as number first, fallback to string
              const numValue = parseFloat(value)
              ohlcv[fieldName] = isNaN(numValue) ? value : numValue
            }
          }
        }
      }

      return isValidOhlcv(ohlcv) ? ohlcv as OhlcvDto : null
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
    
    // Flush buffer before reading to ensure we have all data
    await this.flush()

    try {
      const files = await this.getRelevantFiles(symbol, exchange)
      let latest: number | null = null

      for (const file of files) {
        const data = await this.readCsvFile(file)
        for (const item of data) {
          if (latest === null || item.timestamp > latest) {
            latest = item.timestamp
          }
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
      const files = await this.getRelevantFiles(symbol, exchange)
      let earliest: number | null = null

      for (const file of files) {
        const data = await this.readCsvFile(file)
        for (const item of data) {
          if (earliest === null || item.timestamp < earliest) {
            earliest = item.timestamp
          }
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
      const files = await this.getRelevantFiles(symbol, exchange)
      let count = 0

      for (const file of files) {
        const data = await this.readCsvFile(file)
        count += data.length
      }

      return count
    } catch (error) {
      logger.error('Failed to get count', { error, symbol })
      throw new RepositoryStorageError(
        `Failed to get count: ${String(error)}`,
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
   * Delete OHLCV data within a specific date range
   */
  async deleteBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<number> {
    this.ensureReady()
    
    // Flush buffer before reading to ensure we have all data
    await this.flush()

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

    // Create new content with dynamic headers
    const headers = this.generateHeaders(data[0]!)
    let content = headers.join(this.csvDelimiter) + '\n'
    for (const item of data) {
      content += this.formatOhlcvAsCsvRow(item) + '\n'
    }

    await fs.writeFile(filePath, content, 'utf8')
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

  // --- Private helper methods specific to CSV ---

  /**
   * Generate CSV headers from expected output fields
   */
  private generateHeadersFromExpected(): string[] {
    // Standard headers in order
    const headers = [...this.csvHeaders]
    
    // Add expected additional fields
    const standardKeys = new Set(['timestamp', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume'])
    
    // Get additional fields that aren't standard OHLCV
    const additionalFields = this.expectedOutputFields.filter(field => !standardKeys.has(field))
    
    // Add additional fields to headers
    headers.push(...additionalFields)
    
    return headers
  }

  /**
   * Generate CSV headers dynamically based on data fields
   */
  private generateHeaders(data: OhlcvDto): string[] {
    // Standard headers in order
    const headers = [...this.csvHeaders]
    
    // Add any additional fields from transforms
    const standardKeys = new Set(['timestamp', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume'])
    
    // Get all additional keys
    const additionalKeys = Object.keys(data)
      .filter(key => !standardKeys.has(key))
    
    // Group by transform name (suffix after underscore)
    const transformGroups = new Map<string, string[]>()
    
    for (const key of additionalKeys) {
      // Extract transform name from keys like "o_t1", "h_t1" -> "t1"
      const match = /_(.+)$/.exec(key)
      const transformName = match?.[1] ? match[1] : 'other'
      
      if (!transformGroups.has(transformName)) {
        transformGroups.set(transformName, [])
      }
      transformGroups.get(transformName)!.push(key)
    }
    
    // Sort transform names and add columns in grouped order
    const sortedTransformNames = Array.from(transformGroups.keys()).sort()
    
    for (const transformName of sortedTransformNames) {
      const columns = transformGroups.get(transformName)!
      // Sort columns within each transform group (o_, h_, l_, c_, v_ order)
      const sortedColumns = columns.sort((a, b) => {
        const prefixOrder = ['o_', 'h_', 'l_', 'c_', 'v_']
        const aPrefix = a.substring(0, 2)
        const bPrefix = b.substring(0, 2)
        const aIndex = prefixOrder.indexOf(aPrefix)
        const bIndex = prefixOrder.indexOf(bPrefix)
        
        if (aIndex !== -1 && bIndex !== -1) {
          return aIndex - bIndex
        }
        return a.localeCompare(b)
      })
      headers.push(...sortedColumns)
    }
    
    return headers
  }

  /**
   * Format OHLCV data as a CSV row
   */
  private formatOhlcvAsCsvRow(data: OhlcvDto): string {
    // Always include standard fields first
    const standardValues = [
      new Date(data.timestamp).toISOString(),
      this.escapeCsvValue(data.symbol),
      this.escapeCsvValue(data.exchange),
      data.open,
      data.high,
      data.low,
      data.close,
      data.volume
    ]

    // Check for additional fields added by transforms
    const standardKeys = new Set(['timestamp', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume'])
    
    // Get all additional keys
    const additionalKeys = Object.keys(data)
      .filter(key => !standardKeys.has(key))
    
    // Group by transform name (suffix after underscore)
    const transformGroups = new Map<string, string[]>()
    
    for (const key of additionalKeys) {
      // Extract transform name from keys like "o_t1", "h_t1" -> "t1"
      const match = /_(.+)$/.exec(key)
      const transformName = match?.[1] ? match[1] : 'other'
      
      if (!transformGroups.has(transformName)) {
        transformGroups.set(transformName, [])
      }
      transformGroups.get(transformName)!.push(key)
    }
    
    // Sort transform names and add values in grouped order
    const additionalFields: string[] = []
    const sortedTransformNames = Array.from(transformGroups.keys()).sort()
    
    for (const transformName of sortedTransformNames) {
      const columns = transformGroups.get(transformName)!
      // Sort columns within each transform group (o_, h_, l_, c_, v_ order)
      const sortedColumns = columns.sort((a, b) => {
        const prefixOrder = ['o_', 'h_', 'l_', 'c_', 'v_']
        const aPrefix = a.substring(0, 2)
        const bPrefix = b.substring(0, 2)
        const aIndex = prefixOrder.indexOf(aPrefix)
        const bIndex = prefixOrder.indexOf(bPrefix)
        
        if (aIndex !== -1 && bIndex !== -1) {
          return aIndex - bIndex
        }
        return a.localeCompare(b)
      })
      
      // Add values for each column
      for (const key of sortedColumns) {
        const value = (data as any)[key]
        additionalFields.push(value !== null && value !== undefined ? String(value) : '')
      }
    }

    // Combine standard and additional values
    const allValues = [...standardValues, ...additionalFields]
    return allValues.join(this.csvDelimiter)
  }

  /**
   * Parse timestamp from various formats (number or ISO string) to Unix milliseconds
   */
  private parseTimestamp(value: string): number {
    // If it looks like a number, parse as epoch milliseconds
    const numericValue = Number(value)
    if (!isNaN(numericValue)) {
      // Handle both seconds and milliseconds
      // If the number is less than year 2001 in milliseconds, it's likely seconds
      const SECONDS_THRESHOLD = 978307200000 // Year 2001 in milliseconds
      return numericValue > 0 && numericValue < SECONDS_THRESHOLD ? numericValue * 1000 : numericValue
    }

    // Try to parse as ISO date string
    const date = new Date(value)
    if (isNaN(date.getTime())) {
      throw new Error(`Invalid timestamp format: ${value}`)
    }

    return date.getTime()
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
}
