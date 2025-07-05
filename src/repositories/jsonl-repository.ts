import { createReadStream, createWriteStream, existsSync } from 'node:fs'
import { mkdir, rm, stat, unlink, rename } from 'node:fs/promises'
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
  
  // Last pending record for deduplication
  private lastPendingRecord: OhlcvDto | null = null
  
  // Active write streams
  private readonly activeStreams = new Map<string, NodeJS.WritableStream & { destroy?: () => void }>()
  
  // Track single symbol/exchange per file
  private singleSymbol: string | null = null
  private singleExchange: string | null = null

  /**
   * Initialize the JSONL repository and set up directory structure
   */
  async initialize(config: RepositoryConfig): Promise<void> {
    try {
      this.basePath = config.connectionString
      this.batchSize = (config.options?.batchSize as number | undefined) || 1000
      
      // Ensure the directory exists
      const dir = path.dirname(this.basePath)
      await mkdir(dir, { recursive: true })
      
      // Clear existing file if overwrite is requested
      if (config.options?.overwrite && existsSync(this.basePath)) {
        await unlink(this.basePath)
      }
      
      this.ready = true
      logger.info('JSONL repository initialized', { basePath: this.basePath })
    } catch (error) {
      throw new RepositoryConnectionError(`Failed to initialize JSONL repository: ${error}`)
    }
  }

  async close(): Promise<void> {
    // Flush any remaining data (including pending record)
    await this.flush()
    
    // Close all active streams and wait for them to complete
    const closePromises: Promise<void>[] = []
    for (const [path, stream] of this.activeStreams) {
      const closePromise = new Promise<void>((resolve, reject) => {
        stream.end((error?: Error) => {
          if (error) {
            reject(error)
          } else {
            resolve()
          }
        })
      })
      closePromises.push(closePromise)
      this.activeStreams.delete(path)
    }
    
    // Wait for all streams to close
    await Promise.all(closePromises)
    
    this.ready = false
    logger.info('JSONL repository closed')
  }

  /**
   * Force close all streams immediately without flushing
   * Use only in test cleanup scenarios
   */
  forceClose(): void {
    for (const [, stream] of this.activeStreams) {
      try {
        if (stream?.destroy) {
          stream.destroy()
        }
      } catch {
        // Ignore errors during force close
      }
    }
    this.activeStreams.clear()
    this.ready = false
  }

  async save(data: OhlcvDto): Promise<void> {
    this.assertReady()
    
    if (!isValidOhlcv(data)) {
      throw new RepositoryValidationError('Invalid OHLCV data')
    }
    
    // Enforce single symbol/exchange per file
    if (this.singleSymbol === null) {
      this.singleSymbol = data.symbol
      this.singleExchange = data.exchange
    } else if (this.singleSymbol !== data.symbol || this.singleExchange !== data.exchange) {
      throw new RepositoryValidationError(
        `JSONL file can only contain data for one symbol/exchange. Expected ${this.singleSymbol}/${this.singleExchange}, got ${data.symbol}/${data.exchange}`
      )
    }
    
    // If we have a pending record, check if new record has same key
    if (this.lastPendingRecord) {
      const lastKey = `${this.lastPendingRecord.timestamp}:${this.lastPendingRecord.symbol}:${this.lastPendingRecord.exchange}`
      const newKey = `${data.timestamp}:${data.symbol}:${data.exchange}`
      
      if (lastKey === newKey) {
        // Same key - just overwrite the pending record
        this.lastPendingRecord = data
      } else {
        // Different key - write the pending record and store new one
        this.ohlcvBuffer.push(this.lastPendingRecord)
        this.lastPendingRecord = data
        
        if (this.ohlcvBuffer.length >= this.batchSize) {
          await this.flush()
        }
      }
    } else {
      // First record - just store it
      this.lastPendingRecord = data
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
    
    // Validate all data is for the same symbol/exchange
    if (data.length > 0) {
      const firstItem = data[0]!
      if (this.singleSymbol === null) {
        this.singleSymbol = firstItem.symbol
        this.singleExchange = firstItem.exchange
      }
      
      for (const item of data) {
        if (item.symbol !== this.singleSymbol || item.exchange !== this.singleExchange) {
          throw new RepositoryValidationError(
            `JSONL file can only contain data for one symbol/exchange. Expected ${this.singleSymbol}/${this.singleExchange}, got ${item.symbol}/${item.exchange}`
          )
        }
      }
    }
    
    // Process each record through deduplication logic
    for (const item of data) {
      await this.save(item)
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
    const dir = path.dirname(this.basePath)
    const basename = path.basename(this.basePath, '.jsonl')
    const coeffFile = path.join(dir, `${basename}.coefficients.jsonl`)
    
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
    
    const dir = path.dirname(this.basePath)
    const basename = path.basename(this.basePath, '.jsonl')
    const coeffFile = path.join(dir, `${basename}.coefficients.jsonl`)
    
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
          const rawRecord = JSON.parse(line)
          const record = this.normalizeRecord(rawRecord)
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
  }

  isReady(): boolean {
    return this.ready
  }

  async flush(): Promise<void> {
    // Write any pending record first
    if (this.lastPendingRecord) {
      this.ohlcvBuffer.push(this.lastPendingRecord)
      this.lastPendingRecord = null
    }
    
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
    // Write to a single file
    const filePath = this.basePath
    
    // Get or create write stream
    let stream = this.activeStreams.get(filePath)
    if (!stream) {
      // Check if file exists to determine the correct flag
      const fileExists = existsSync(filePath)
      stream = createWriteStream(filePath, { flags: fileExists ? 'a' : 'w' })
      this.activeStreams.set(filePath, stream)
    }
    
    // Write records with abbreviated property names
    for (const record of data) {
      const abbreviated = {
        x: record.exchange,
        s: record.symbol,
        t: record.timestamp,
        o: record.open,
        h: record.high,
        l: record.low,
        c: record.close,
        v: record.volume
      }
      await new Promise<void>((resolve, reject) => {
        stream.write(JSON.stringify(abbreviated) + '\n', (err) => {
          if (err) reject(err)
          else resolve()
        })
      })
    }
  }

  private async writeCoefficientBatch(data: CoefficientData[]): Promise<void> {
    // Write coefficients to a separate file in the same directory
    const dir = path.dirname(this.basePath)
    const basename = path.basename(this.basePath, '.jsonl')
    const filePath = path.join(dir, `${basename}.coefficients.jsonl`)
    
    // Get or create write stream
    let stream = this.activeStreams.get(filePath)
    if (!stream) {
      // Check if file exists to determine the correct flag
      const fileExists = existsSync(filePath)
      stream = createWriteStream(filePath, { flags: fileExists ? 'a' : 'w' })
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

  private async getRelevantFiles(_query: OhlcvQuery): Promise<string[]> {
    // Simply return the single JSONL file if it exists
    if (existsSync(this.basePath)) {
      return [this.basePath]
    }
    return []
  }

  private matchesQuery(record: OhlcvDto, query: OhlcvQuery): boolean {
    if (query.startTime && record.timestamp < query.startTime) return false
    if (query.endTime && record.timestamp > query.endTime) return false
    if (query.symbol && record.symbol !== query.symbol) return false
    if (query.exchange && record.exchange !== query.exchange) return false
    
    return true
  }
  
  /**
   * Normalizes a record from abbreviated or full property names to standard OhlcvDto
   */
  private normalizeRecord(record: any): OhlcvDto {
    // Handle abbreviated format
    if ('t' in record && 'o' in record) {
      return {
        exchange: record.x || record.e || record.exchange,
        symbol: record.s || record.symbol,
        timestamp: record.t || record.timestamp,
        open: record.o || record.open,
        high: record.h || record.high,
        low: record.l || record.low,
        close: record.c || record.close,
        volume: record.v || record.volume
      }
    }
    
    // Already in full format
    return record as OhlcvDto
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
      // Write in abbreviated format to save space
      const abbreviated = {
        x: record.exchange,
        s: record.symbol,
        t: record.timestamp,
        o: record.open,
        h: record.high,
        l: record.low,
        c: record.close,
        v: record.volume
      }
      await new Promise<void>((resolve, reject) => {
        stream.write(JSON.stringify(abbreviated) + '\n', (err) => {
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