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
    if (needsHeaders) {
      content += this.csvHeaders.join(this.csvDelimiter) + '\n'
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

    // Create new content
    let content = this.csvHeaders.join(this.csvDelimiter) + '\n'
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
}
