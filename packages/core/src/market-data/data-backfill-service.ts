import type { MarketDataRepository } from '@trdr/data'
import type { EpochDate, StockSymbol } from '@trdr/shared'
import { epochDateNow, toEpochDate, toStockSymbol } from '@trdr/shared'
import type { Candle, Logger } from '@trdr/types'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import type { CoinbaseDataFeed } from './coinbase-data-feed'
import type { DataGap } from './historical-data-manager'
export type { DataGap } from './historical-data-manager'

/**
 * Backfill progress tracking
 */
export interface BackfillProgress {
  readonly symbol: StockSymbol
  readonly interval: string
  readonly totalGaps: number
  readonly processedGaps: number
  readonly totalCandles: number
  readonly successfulCandles: number
  readonly failedCandles: number
  readonly startTime: EpochDate
  readonly estimatedCompletion?: EpochDate
  readonly currentGap?: DataGap
  readonly status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  readonly errors: readonly string[]
}

/**
 * Backfill operation handle for tracking and control
 */
export interface BackfillOperation {
  readonly id: string
  readonly symbol: StockSymbol
  readonly interval: string
  readonly startDate: EpochDate
  readonly endDate: EpochDate
  readonly mode: 'automatic' | 'manual'
  getProgress(): BackfillProgress
  cancel(): Promise<void>
  wait(): Promise<void>
}

/**
 * Configuration for backfill operations
 */
export interface BackfillConfig {
  /** Max concurrent backfill operations */
  readonly maxConcurrentOperations?: number
  /** Max candles per API request */
  readonly batchSize?: number
  /** Delay between batches (ms) */
  readonly batchDelayMs?: number
  /** Max retries per batch */
  readonly maxRetries?: number
  /** Retry delay multiplier */
  readonly retryDelayMultiplier?: number
  /** Enable progress tracking */
  readonly trackProgress?: boolean
  /** Min gap size to backfill (in intervals) */
  readonly minGapSize?: number
  /** Max gap size per operation (in intervals) */
  readonly maxGapSize?: number
}

/**
 * Advanced data backfilling service with gap detection,
 * progress tracking, and operation management
 */
export class DataBackfillService {
  private readonly config: Required<BackfillConfig>
  private readonly repository: MarketDataRepository
  private readonly dataFeed: CoinbaseDataFeed
  private readonly eventBus: EventBus
  private readonly logger?: Logger
  
  /** Active backfill operations */
  private readonly activeOperations = new Map<string, BackfillOperationImpl>()
  /** Operation queue for limiting concurrency */
  private readonly operationQueue: BackfillOperationImpl[] = []
  
  constructor(
    repository: MarketDataRepository,
    dataFeed: CoinbaseDataFeed,
    config: BackfillConfig = {},
    logger?: Logger
  ) {
    this.repository = repository
    this.dataFeed = dataFeed
    this.eventBus = EventBus.getInstance()
    this.logger = logger
    
    this.config = {
      maxConcurrentOperations: config.maxConcurrentOperations ?? 3,
      batchSize: config.batchSize ?? 300,
      batchDelayMs: config.batchDelayMs ?? 1000,
      maxRetries: config.maxRetries ?? 3,
      retryDelayMultiplier: config.retryDelayMultiplier ?? 2,
      trackProgress: config.trackProgress ?? true,
      minGapSize: config.minGapSize ?? 2,
      maxGapSize: config.maxGapSize ?? 10000
    }
  }

  /**
   * Detect gaps in historical data with advanced heuristics
   */
  async detectDataGaps(
    symbol: StockSymbol,
    interval: string,
    startDate: EpochDate | Date,
    endDate: EpochDate | Date
  ): Promise<DataGap[]> {
    const stockSymbol = typeof symbol === 'string' ? toStockSymbol(symbol) : symbol
    const start = startDate instanceof Date ? toEpochDate(startDate) : startDate
    const end = endDate instanceof Date ? toEpochDate(endDate) : endDate
    
    this.logger?.debug('Detecting data gaps', {
      symbol: stockSymbol,
      interval,
      startDate: new Date(start).toISOString(),
      endDate: new Date(end).toISOString()
    })
    
    // Fetch existing candles
    const candles = await this.repository.getCandles(stockSymbol, interval, start, end)
    
    // Calculate expected vs actual candles
    const intervalMs = this.intervalToMilliseconds(interval)
    const expectedCandleCount = Math.floor((end - start) / intervalMs)
    
    if (candles.length === 0) {
      // Entire range is missing
      return [{
        symbol: stockSymbol,
        interval,
        startTime: start,
        endTime: end,
        expectedCandles: expectedCandleCount,
        actualCandles: 0
      }]
    }
    
    // Sort candles by timestamp
    const sortedCandles = [...candles].sort((a, b) => a.timestamp - b.timestamp)
    
    // Detect gaps with advanced logic
    const gaps: DataGap[] = []
    
    // Check for gap at the beginning
    if (sortedCandles[0]!.timestamp > start + intervalMs) {
      const gapEnd = toEpochDate(sortedCandles[0]!.timestamp - intervalMs)
      const expectedCandles = Math.floor((gapEnd - start) / intervalMs) + 1
      
      if (expectedCandles >= this.config.minGapSize) {
        gaps.push({
          symbol: stockSymbol,
          interval,
          startTime: start,
          endTime: gapEnd,
          expectedCandles,
          actualCandles: 0
        })
      }
    }
    
    // Check for gaps between candles
    for (let i = 0; i < sortedCandles.length - 1; i++) {
      const current = sortedCandles[i]!
      const next = sortedCandles[i + 1]!
      const expectedNext = current.timestamp + intervalMs
      
      if (next.timestamp > expectedNext + intervalMs) {
        const gapStart = toEpochDate(current.timestamp + intervalMs)
        const gapEnd = toEpochDate(next.timestamp - intervalMs)
        const expectedCandles = Math.floor((gapEnd - gapStart) / intervalMs) + 1
        
        if (expectedCandles >= this.config.minGapSize) {
          gaps.push({
            symbol: stockSymbol,
            interval,
            startTime: gapStart,
            endTime: gapEnd,
            expectedCandles,
            actualCandles: 0
          })
        }
      }
    }
    
    // Check for gap at the end
    const lastCandle = sortedCandles[sortedCandles.length - 1]!
    if (lastCandle.timestamp < end - intervalMs) {
      const gapStart = toEpochDate(lastCandle.timestamp + intervalMs)
      const expectedCandles = Math.floor((end - gapStart) / intervalMs) + 1
      
      if (expectedCandles >= this.config.minGapSize) {
        gaps.push({
          symbol: stockSymbol,
          interval,
          startTime: gapStart,
          endTime: end,
          expectedCandles,
          actualCandles: 0
        })
      }
    }
    
    // Merge adjacent gaps if they're close enough
    const mergedGaps = this.mergeAdjacentGaps(gaps, intervalMs)
    
    this.logger?.info('Gap detection complete', {
      symbol: stockSymbol,
      interval,
      totalGaps: mergedGaps.length,
      totalMissingCandles: mergedGaps.reduce((sum, gap) => sum + gap.expectedCandles, 0)
    })
    
    return mergedGaps
  }

  /**
   * Request missing data for identified gaps
   */
  async requestMissingData(
    symbol: StockSymbol,
    interval: string,
    gaps: DataGap[],
    mode: 'automatic' | 'manual' = 'automatic'
  ): Promise<BackfillOperation> {
    const stockSymbol = typeof symbol === 'string' ? toStockSymbol(symbol) : symbol
    
    if (gaps.length === 0) {
      throw new Error('No gaps provided for backfilling')
    }
    
    // Create operation
    const operation = new BackfillOperationImpl(
      stockSymbol,
      interval,
      gaps,
      mode
    )
    
    // Add to queue or start immediately
    if (this.activeOperations.size < this.config.maxConcurrentOperations) {
      this.startOperation(operation)
    } else {
      this.operationQueue.push(operation)
      this.logger?.info('Backfill operation queued', {
        id: operation.id,
        symbol: stockSymbol,
        queueLength: this.operationQueue.length
      })
    }
    
    return operation
  }

  /**
   * Merge backfilled data with existing data
   */
  async mergeBackfilledData(
    existingData: Candle[],
    newData: Candle[]
  ): Promise<Candle[]> {
    if (newData.length === 0) return existingData
    if (existingData.length === 0) return newData
    
    // Create a map for efficient deduplication
    const candleMap = new Map<string, Candle>()
    
    // Add existing data
    for (const candle of existingData) {
      const key = `${candle.symbol}:${candle.interval}:${candle.timestamp}`
      candleMap.set(key, candle)
    }
    
    // Merge new data, preferring new over existing
    let updatedCount = 0
    let addedCount = 0
    
    for (const candle of newData) {
      const key = `${candle.symbol}:${candle.interval}:${candle.timestamp}`
      if (candleMap.has(key)) {
        updatedCount++
      } else {
        addedCount++
      }
      candleMap.set(key, candle)
    }
    
    this.logger?.debug('Data merge complete', {
      existingCount: existingData.length,
      newCount: newData.length,
      mergedCount: candleMap.size,
      updatedCount,
      addedCount
    })
    
    // Sort by timestamp and return
    return Array.from(candleMap.values()).sort((a, b) => a.timestamp - b.timestamp)
  }

  /**
   * Start a backfill operation
   */
  private startOperation(operation: BackfillOperationImpl): void {
    this.activeOperations.set(operation.id, operation)
    
    this.logger?.info('Starting backfill operation', {
      id: operation.id,
      symbol: operation.symbol,
      interval: operation.interval,
      totalGaps: operation.gaps.length
    })
    
    // Start the operation asynchronously
    void this.executeOperation(operation).catch(error => {
      this.logger?.error('Backfill operation failed', {
        id: operation.id,
        error: (error as Error).message
      })
    })
  }

  /**
   * Execute a backfill operation
   */
  private async executeOperation(operation: BackfillOperationImpl): Promise<void> {
    try {
      operation.setStatus('running')
      
      for (const gap of operation.gaps) {
        if (operation.isCancelled) {
          operation.setStatus('cancelled')
          break
        }
        
        operation.setCurrentGap(gap)
        
        try {
          await this.backfillGap(operation, gap)
          operation.incrementProcessedGaps()
        } catch (error) {
          operation.addError(`Failed to backfill gap: ${(error as Error).message}`)
          this.logger?.error('Gap backfill failed', {
            operationId: operation.id,
            gap,
            error: (error as Error).message
          })
        }
      }
      
      if (!operation.isCancelled) {
        operation.setStatus(operation.errors.length > 0 ? 'failed' : 'completed')
      }
      
      // Emit completion event
      this.eventBus.emit(EventTypes.BACKFILL_COMPLETED, {
        operationId: operation.id,
        symbol: operation.symbol,
        interval: operation.interval,
        progress: operation.getProgress(),
        timestamp: epochDateNow()
      })
      
    } finally {
      // Remove from active operations
      this.activeOperations.delete(operation.id)
      
      // Process next in queue if any
      if (this.operationQueue.length > 0) {
        const next = this.operationQueue.shift()!
        this.startOperation(next)
      }
    }
  }

  /**
   * Backfill a single gap
   */
  private async backfillGap(
    operation: BackfillOperationImpl,
    gap: DataGap
  ): Promise<void> {
    const { symbol, interval } = operation
    
    // Split large gaps into smaller chunks
    const chunks = this.splitGapIntoChunks(gap)
    
    for (const chunk of chunks) {
      if (operation.isCancelled) break
      
      let retries = 0
      let lastError: Error | null = null
      
      while (retries < this.config.maxRetries) {
        try {
          // Fetch candles for chunk
          const candles = await this.dataFeed.getHistorical({
            symbol,
            interval,
            start: chunk.startTime,
            end: chunk.endTime
          })
          
          // Transform and save candles
          if (candles.length > 0) {
            const transformedCandles: Candle[] = candles.map(candle => ({
              ...candle,
              symbol,
              interval,
              openTime: candle.timestamp,
              closeTime: toEpochDate(candle.timestamp + this.intervalToMilliseconds(interval))
            }))
            
            await this.repository.saveCandlesBatch(transformedCandles)
            operation.incrementSuccessfulCandles(transformedCandles.length)
          }
          
          // Success - break retry loop
          break
          
        } catch (error) {
          lastError = error as Error
          retries++
          
          if (retries < this.config.maxRetries) {
            const delay = this.config.batchDelayMs * Math.pow(this.config.retryDelayMultiplier, retries - 1)
            this.logger?.warn('Chunk backfill failed, retrying', {
              operationId: operation.id,
              chunk,
              attempt: retries,
              delay,
              error: lastError.message
            })
            await new Promise(resolve => setTimeout(resolve, delay))
          }
        }
      }
      
      if (lastError && retries >= this.config.maxRetries) {
        operation.incrementFailedCandles(chunk.expectedCandles)
        throw lastError
      }
      
      // Delay between chunks to avoid rate limiting
      if (chunks.indexOf(chunk) < chunks.length - 1) {
        await new Promise(resolve => setTimeout(resolve, this.config.batchDelayMs))
      }
    }
  }

  /**
   * Split a gap into smaller chunks for batching
   */
  private splitGapIntoChunks(gap: DataGap): DataGap[] {
    const chunks: DataGap[] = []
    const intervalMs = this.intervalToMilliseconds(gap.interval)
    const chunkSize = Math.min(this.config.batchSize, gap.expectedCandles)
    const chunkDurationMs = chunkSize * intervalMs
    
    let currentStart = gap.startTime
    
    while (currentStart < gap.endTime) {
      const currentEnd = Math.min(
        toEpochDate(currentStart + chunkDurationMs - intervalMs),
        gap.endTime
      ) as EpochDate
      
      const expectedCandles = Math.floor((currentEnd - currentStart) / intervalMs) + 1
      
      chunks.push({
        symbol: gap.symbol,
        interval: gap.interval,
        startTime: currentStart,
        endTime: currentEnd,
        expectedCandles,
        actualCandles: 0
      })
      
      currentStart = toEpochDate(currentEnd + intervalMs)
    }
    
    return chunks
  }

  /**
   * Merge adjacent gaps that are close together
   */
  private mergeAdjacentGaps(gaps: DataGap[], intervalMs: number): DataGap[] {
    if (gaps.length <= 1) return gaps
    
    const merged: DataGap[] = []
    let currentGap = { ...gaps[0]! }
    
    for (let i = 1; i < gaps.length; i++) {
      const nextGap = gaps[i]!
      
      // Check if gaps are adjacent (within 2 intervals)
      if (nextGap.startTime - currentGap.endTime <= 2 * intervalMs) {
        // Merge gaps
        currentGap = {
          ...currentGap,
          endTime: nextGap.endTime,
          expectedCandles: this.calculateExpectedCandles(
            currentGap.interval,
            currentGap.startTime,
            nextGap.endTime
          )
        }
      } else {
        // Gaps are not adjacent, save current and start new
        merged.push(currentGap)
        currentGap = { ...nextGap }
      }
    }
    
    // Don't forget the last gap
    merged.push(currentGap)
    
    return merged
  }

  /**
   * Calculate expected number of candles for a time range
   */
  private calculateExpectedCandles(interval: string, startTime: EpochDate, endTime: EpochDate): number {
    const intervalMs = this.intervalToMilliseconds(interval)
    return Math.floor((endTime - startTime) / intervalMs) + 1
  }

  /**
   * Convert interval string to milliseconds
   */
  private intervalToMilliseconds(interval: string): number {
    const intervalMap: Record<string, number> = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '6h': 6 * 60 * 60 * 1000,
      '1d': 24 * 60 * 60 * 1000
    }
    
    return intervalMap[interval] || 60 * 60 * 1000 // Default to 1 hour
  }

  /**
   * Get all active backfill operations
   */
  getActiveOperations(): BackfillOperation[] {
    return Array.from(this.activeOperations.values())
  }

  /**
   * Cancel all active operations
   */
  async cancelAllOperations(): Promise<void> {
    const operations = Array.from(this.activeOperations.values())
    await Promise.all(operations.map(op => op.cancel()))
  }
}

/**
 * Internal implementation of BackfillOperation
 */
class BackfillOperationImpl implements BackfillOperation {
  readonly id: string
  readonly symbol: StockSymbol
  readonly interval: string
  readonly startDate: EpochDate
  readonly endDate: EpochDate
  readonly mode: 'automatic' | 'manual'
  readonly gaps: DataGap[]
  
  private progress: BackfillProgress
  private cancelled = false
  private readonly completionPromise: Promise<void>
  private resolveCompletion!: () => void
  
  constructor(
    symbol: StockSymbol,
    interval: string,
    gaps: DataGap[],
    mode: 'automatic' | 'manual'
  ) {
    this.id = `backfill-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`
    this.symbol = symbol
    this.interval = interval
    this.gaps = gaps
    this.mode = mode
    
    // Calculate date range from gaps
    this.startDate = Math.min(...gaps.map(g => g.startTime)) as EpochDate
    this.endDate = Math.max(...gaps.map(g => g.endTime)) as EpochDate
    
    // Initialize progress
    this.progress = {
      symbol,
      interval,
      totalGaps: gaps.length,
      processedGaps: 0,
      totalCandles: gaps.reduce((sum, gap) => sum + gap.expectedCandles, 0),
      successfulCandles: 0,
      failedCandles: 0,
      startTime: epochDateNow(),
      status: 'pending',
      errors: []
    }
    
    // Create completion promise
    this.completionPromise = new Promise<void>(resolve => {
      this.resolveCompletion = resolve
    })
  }
  
  get isCancelled(): boolean {
    return this.cancelled
  }
  
  get errors(): readonly string[] {
    return this.progress.errors
  }
  
  getProgress(): BackfillProgress {
    // Calculate estimated completion if running
    if (this.progress.status === 'running' && this.progress.processedGaps > 0) {
      const elapsedTime = epochDateNow() - this.progress.startTime
      const avgTimePerGap = elapsedTime / this.progress.processedGaps
      const remainingGaps = this.progress.totalGaps - this.progress.processedGaps
      const estimatedRemainingTime = remainingGaps * avgTimePerGap
      
      this.progress = {
        ...this.progress,
        estimatedCompletion: toEpochDate(epochDateNow() + estimatedRemainingTime)
      }
    }
    
    return { ...this.progress }
  }
  
  async cancel(): Promise<void> {
    this.cancelled = true
    this.setStatus('cancelled')
  }
  
  async wait(): Promise<void> {
    return this.completionPromise
  }
  
  setStatus(status: BackfillProgress['status']): void {
    this.progress = { ...this.progress, status }
    
    if (status === 'completed' || status === 'failed' || status === 'cancelled') {
      this.resolveCompletion()
    }
  }
  
  setCurrentGap(gap: DataGap): void {
    this.progress = { ...this.progress, currentGap: gap }
  }
  
  incrementProcessedGaps(): void {
    this.progress = {
      ...this.progress,
      processedGaps: this.progress.processedGaps + 1
    }
  }
  
  incrementSuccessfulCandles(count: number): void {
    this.progress = {
      ...this.progress,
      successfulCandles: this.progress.successfulCandles + count
    }
  }
  
  incrementFailedCandles(count: number): void {
    this.progress = {
      ...this.progress,
      failedCandles: this.progress.failedCandles + count
    }
  }
  
  addError(error: string): void {
    this.progress = {
      ...this.progress,
      errors: [...this.progress.errors, error]
    }
  }
}