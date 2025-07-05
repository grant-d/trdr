import { createReadStream, createWriteStream, existsSync } from 'node:fs'
import { mkdir, readdir, rm, stat, unlink, rename } from 'node:fs/promises'
import * as path from 'node:path'
import { createInterface } from 'node:readline'
import type { OhlcvDto } from '../models'
import { isValidOhlcv } from '../models'
import logger from '../utils/logger'
import type { CoefficientData, OhlcvQuery, OhlcvRepository, RepositoryConfig } from './ohlcv-repository.interface'
import { RepositoryConnectionError, RepositoryValidationError } from './ohlcv-repository.interface'

/**
 * JSONL-based implementation of the OhlcvRepository interface
 * Stores each OHLCV record as a JSON object on a separate line
 * Optimized for streaming and append operations
 */
export class JsonlRepository implements OhlcvRepository {
  private ready = false
  private basePath = ''
  private batchSize = 1000
  
  // In-memory buffers for batch writing
  private ohlcvBuffer: OhlcvDto[] = []
  private coefficientBuffer: CoefficientData[] = []
  
  // Active write streams
  private readonly activeStreams = new Map<string, NodeJS.WritableStream>()

  /**
   * Initialize the JSONL repository and set up directory structure
   */
  async initialize(config: RepositoryConfig): Promise<void> {
    try {
      this.basePath = config.connectionString
      this.batchSize = (config.options?.batchSize as number | undefined) || 1000
      
      // Create base directories
      await mkdir(path.join(this.basePath, 'ohlcv'), { recursive: true })
      await mkdir(path.join(this.basePath, 'coefficients'), { recursive: true })
      
      this.ready = true
      logger.info('JSONL repository initialized', { basePath: this.basePath })
    } catch (error) {
      throw new RepositoryConnectionError(`Failed to initialize JSONL repository: ${error}`)
    }
  }

  async close(): Promise<void> {
    // Flush any remaining data
    await this.flush()
    
    // Close all active streams
    for (const [path, stream] of this.activeStreams) {
      stream.end()
      this.activeStreams.delete(path)
    }
    
    this.ready = false
    logger.info('JSONL repository closed')
  }

  async save(data: OhlcvDto): Promise<void> {
    this.assertReady()
    
    if (!isValidOhlcv(data)) {
      throw new RepositoryValidationError('Invalid OHLCV data')
    }
    
    this.ohlcvBuffer.push(data)
    
    if (this.ohlcvBuffer.length >= this.batchSize) {
      await this.flush()
    }
  }

  async saveMany(data: OhlcvDto[]): Promise<void> {
    this.assertReady()
    
    // Validate all data
    for (const item of data) {
      if (!isValidOhlcv(item)) {
        throw new RepositoryValidationError('Invalid OHLCV data in batch')
      }
    }
    
    this.ohlcvBuffer.push(...data)
    
    if (this.ohlcvBuffer.length >= this.batchSize) {
      await this.flush()
    }
  }

  async appendBatch(data: OhlcvDto[]): Promise<void> {
    return this.saveMany(data)
  }

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

  async query(query: OhlcvQuery): Promise<OhlcvDto[]> {
    this.assertReady()
    
    // Flush buffer before reading to ensure we have all data
    await this.flush()
    
    const results: OhlcvDto[] = []
    const files = await this.getRelevantFiles(query)
    
    for (const file of files) {
      const stream = createReadStream(file, { encoding: 'utf8' })
      const rl = createInterface({ input: stream })
      
      for await (const line of rl) {
        if (!line.trim()) continue
        
        try {
          const record = JSON.parse(line) as OhlcvDto
          
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
    
    // Apply offset
    let filtered = results
    if (query.offset) {
      filtered = filtered.slice(query.offset)
    }
    
    // Apply limit
    if (query.limit) {
      filtered = filtered.slice(0, query.limit)
    }
    
    return filtered
  }

  async getLastTimestamp(symbol: string, exchange?: string): Promise<number | null> {
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
  }

  async getFirstTimestamp(symbol: string, exchange?: string): Promise<number | null> {
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
  }

  async getCount(symbol: string, exchange?: string): Promise<number> {
    const results = await this.getBySymbol(symbol, exchange)
    return results.length
  }

  async deleteBetweenDates(
    startTime: number,
    endTime: number,
    symbol?: string,
    exchange?: string
  ): Promise<number> {
    this.assertReady()
    
    // Flush buffer before deleting
    await this.flush()
    
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
          const record = JSON.parse(line) as OhlcvDto
          
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
  }

  async saveCoefficient(data: CoefficientData): Promise<void> {
    this.assertReady()
    
    this.coefficientBuffer.push(data)
    
    if (this.coefficientBuffer.length >= this.batchSize) {
      await this.flush()
    }
  }

  async saveCoefficients(data: CoefficientData[]): Promise<void> {
    this.assertReady()
    
    this.coefficientBuffer.push(...data)
    
    if (this.coefficientBuffer.length >= this.batchSize) {
      await this.flush()
    }
  }

  async getCoefficient(name: string, symbol?: string, exchange?: string): Promise<CoefficientData | null> {
    const results = await this.getCoefficients(name, symbol, exchange)
    return results.length > 0 ? results[0]! : null
  }

  async getCoefficients(namePattern?: string, symbol?: string, exchange?: string): Promise<CoefficientData[]> {
    this.assertReady()
    
    // Flush buffer before reading
    await this.flush()
    
    const results: CoefficientData[] = []
    const coeffFile = path.join(this.basePath, 'coefficients', 'coefficients.jsonl')
    
    if (!existsSync(coeffFile)) {
      return results
    }
    
    const stream = createReadStream(coeffFile, { encoding: 'utf8' })
    const rl = createInterface({ input: stream })
    
    for await (const line of rl) {
      if (!line.trim()) continue
      
      try {
        const record = JSON.parse(line) as CoefficientData
        
        if ((!namePattern || this.matchesPattern(record.name, namePattern)) &&
            (!symbol || record.symbol === symbol) &&
            (!exchange || record.exchange === exchange)) {
          results.push(record)
        }
      } catch (error) {
        logger.warn('Failed to parse coefficient record', { error })
      }
    }
    
    return results
  }

  async deleteCoefficients(namePattern: string, symbol?: string, exchange?: string): Promise<number> {
    this.assertReady()
    
    // Flush buffer before deleting
    await this.flush()
    
    const coeffFile = path.join(this.basePath, 'coefficients', 'coefficients.jsonl')
    if (!existsSync(coeffFile)) {
      return 0
    }
    
    let deletedCount = 0
    const keepRecords: CoefficientData[] = []
    
    const stream = createReadStream(coeffFile, { encoding: 'utf8' })
    const rl = createInterface({ input: stream })
    
    for await (const line of rl) {
      if (!line.trim()) continue
      
      try {
        const record = JSON.parse(line) as CoefficientData
        
        if (this.matchesPattern(record.name, namePattern) &&
            (!symbol || record.symbol === symbol) &&
            (!exchange || record.exchange === exchange)) {
          deletedCount++
        } else {
          keepRecords.push(record)
        }
      } catch (error) {
        logger.warn('Failed to parse coefficient during delete', { error })
      }
    }
    
    // Rewrite file or delete if empty
    if (deletedCount > 0) {
      if (keepRecords.length > 0) {
        await this.rewriteCoefficientFile(coeffFile, keepRecords)
      } else {
        await unlink(coeffFile)
      }
    }
    
    return deletedCount
  }

  async getSymbols(exchange?: string): Promise<string[]> {
    this.assertReady()
    
    await this.flush()
    
    const symbols = new Set<string>()
    const files = await this.getRelevantFiles({})
    
    for (const file of files) {
      const stream = createReadStream(file, { encoding: 'utf8' })
      const rl = createInterface({ input: stream })
      
      for await (const line of rl) {
        if (!line.trim()) continue
        
        try {
          const record = JSON.parse(line) as OhlcvDto
          if (!exchange || record.exchange === exchange) {
            symbols.add(record.symbol)
          }
        } catch (error) {
          // Skip invalid records
        }
      }
    }
    
    return Array.from(symbols).sort()
  }

  async getExchanges(): Promise<string[]> {
    this.assertReady()
    
    await this.flush()
    
    const exchanges = new Set<string>()
    const files = await this.getRelevantFiles({})
    
    for (const file of files) {
      const stream = createReadStream(file, { encoding: 'utf8' })
      const rl = createInterface({ input: stream })
      
      for await (const line of rl) {
        if (!line.trim()) continue
        
        try {
          const record = JSON.parse(line) as OhlcvDto
          exchanges.add(record.exchange)
        } catch (error) {
          // Skip invalid records
        }
      }
    }
    
    return Array.from(exchanges).sort()
  }

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
    this.assertReady()
    
    await this.flush()
    
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
          const record = JSON.parse(line) as OhlcvDto
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
  }

  isReady(): boolean {
    return this.ready
  }

  async flush(): Promise<void> {
    // Flush OHLCV data
    if (this.ohlcvBuffer.length > 0) {
      await this.writeOhlcvBatch(this.ohlcvBuffer)
      this.ohlcvBuffer = []
    }
    
    // Flush coefficient data
    if (this.coefficientBuffer.length > 0) {
      await this.writeCoefficientBatch(this.coefficientBuffer)
      this.coefficientBuffer = []
    }
  }

  /**
   * Private helper methods
   */
  
  private assertReady(): void {
    if (!this.ready) {
      throw new RepositoryConnectionError('Repository not initialized')
    }
  }

  private async writeOhlcvBatch(data: OhlcvDto[]): Promise<void> {
    // Group by date for file organization
    const grouped = new Map<string, OhlcvDto[]>()
    
    for (const record of data) {
      const date = new Date(record.timestamp)
      const dateKey = `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, '0')}-${String(date.getUTCDate()).padStart(2, '0')}`
      
      if (!grouped.has(dateKey)) {
        grouped.set(dateKey, [])
      }
      grouped.get(dateKey)!.push(record)
    }
    
    // Write to files
    for (const [dateKey, records] of grouped) {
      const filePath = path.join(this.basePath, 'ohlcv', `${dateKey}.jsonl`)
      
      // Get or create write stream
      let stream = this.activeStreams.get(filePath)
      if (!stream) {
        stream = createWriteStream(filePath, { flags: 'a' })
        this.activeStreams.set(filePath, stream)
      }
      
      // Write records
      for (const record of records) {
        await new Promise<void>((resolve, reject) => {
          stream.write(JSON.stringify(record) + '\n', (err) => {
            if (err) reject(err)
            else resolve()
          })
        })
      }
    }
  }

  private async writeCoefficientBatch(data: CoefficientData[]): Promise<void> {
    const filePath = path.join(this.basePath, 'coefficients', 'coefficients.jsonl')
    
    // Get or create write stream
    let stream = this.activeStreams.get(filePath)
    if (!stream) {
      stream = createWriteStream(filePath, { flags: 'a' })
      this.activeStreams.set(filePath, stream)
    }
    
    // Write records
    for (const record of data) {
      await new Promise<void>((resolve, reject) => {
        stream.write(JSON.stringify(record) + '\n', (err) => {
          if (err) reject(err)
          else resolve()
        })
      })
    }
  }

  private async getRelevantFiles(query: OhlcvQuery): Promise<string[]> {
    const ohlcvDir = path.join(this.basePath, 'ohlcv')
    const files: string[] = []
    
    try {
      const entries = await readdir(ohlcvDir)
      
      for (const entry of entries) {
        if (entry.endsWith('.jsonl')) {
          // If query has date range, filter files
          if (query.startTime || query.endTime) {
            const dateStr = entry.replace('.jsonl', '')
            const fileDate = new Date(dateStr + 'T00:00:00Z')
            const fileTime = fileDate.getTime()
            
            if (query.startTime && fileTime < query.startTime - 86400000) continue // Subtract 1 day for safety
            if (query.endTime && fileTime > query.endTime + 86400000) continue // Add 1 day for safety
          }
          
          files.push(path.join(ohlcvDir, entry))
        }
      }
    } catch (error) {
      // Directory might not exist yet
      logger.debug('OHLCV directory does not exist yet', { error })
    }
    
    return files.sort()
  }

  private matchesQuery(record: OhlcvDto, query: OhlcvQuery): boolean {
    if (query.startTime && record.timestamp < query.startTime) return false
    if (query.endTime && record.timestamp > query.endTime) return false
    if (query.symbol && record.symbol !== query.symbol) return false
    if (query.exchange && record.exchange !== query.exchange) return false
    
    return true
  }

  private matchesPattern(name: string, pattern: string): boolean {
    // Convert glob pattern to regex
    const regexPattern = pattern
      .replace(/\*/g, '.*')
      .replace(/\?/g, '.')
    
    const regex = new RegExp(`^${regexPattern}$`)
    return regex.test(name)
  }

  private async rewriteFile(filePath: string, records: OhlcvDto[]): Promise<void> {
    // Write to temp file first
    const tempPath = filePath + '.tmp'
    const stream = createWriteStream(tempPath)
    
    for (const record of records) {
      await new Promise<void>((resolve, reject) => {
        stream.write(JSON.stringify(record) + '\n', (err) => {
          if (err) reject(err)
          else resolve()
        })
      })
    }
    
    await new Promise<void>((resolve, reject) => {
      stream.end((err: Error | null | undefined) => {
        if (err) reject(err)
        else resolve()
      })
    })
    
    // Atomic rename
    await rm(filePath)
    await rename(tempPath, filePath)
  }

  private async rewriteCoefficientFile(filePath: string, records: CoefficientData[]): Promise<void> {
    // Write to temp file first
    const tempPath = filePath + '.tmp'
    const stream = createWriteStream(tempPath)
    
    for (const record of records) {
      await new Promise<void>((resolve, reject) => {
        stream.write(JSON.stringify(record) + '\n', (err) => {
          if (err) reject(err)
          else resolve()
        })
      })
    }
    
    await new Promise<void>((resolve, reject) => {
      stream.end((err: Error | null | undefined) => {
        if (err) reject(err)
        else resolve()
      })
    })
    
    // Atomic rename
    await rm(filePath)
    await rename(tempPath, filePath)
  }
}