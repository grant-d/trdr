import type { MarketDataRepository } from '@trdr/data'
import type { EpochDate, StockSymbol } from '@trdr/shared'
import { epochDateNow, toEpochDate, toStockSymbol } from '@trdr/shared'
import type { Candle, Logger } from '@trdr/types'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import type { HistoricalDataRequest } from '../interfaces/market-data-pipeline'
import type { CoinbaseDataFeed } from './coinbase-data-feed'
import { CoinbaseWebSocketSource } from './coinbase-websocket-source'
import { DataBackfillService, type BackfillOperation } from './data-backfill-service'
import { DataValidator, type ValidationReport } from './data-validator'
import { LiveDataCollector } from './live-data-collector'

/**
 * Configuration for the Historical Data Manager
 */
export interface HistoricalDataManagerConfig {
  /** Maximum number of candles to fetch in a single request */
  readonly maxCandlesPerRequest?: number
  /** Maximum time range (in days) for a single request */
  readonly maxTimeRangePerRequest?: number
  /** Number of retry attempts for failed requests */
  readonly maxRetries?: number
  /** Delay between retry attempts (milliseconds) */
  readonly retryDelayMs?: number
  /** Enable automatic data validation */
  readonly enableValidation?: boolean
  /** Enable automatic backfilling of gaps */
  readonly enableAutoBackfill?: boolean
  /** Disable WebSocket initialization (set to true for tests) */
  readonly disableWebSocket?: boolean
  /** LiveDataCollector flush interval (set low for tests) */
  readonly liveDataFlushIntervalMs?: number
  /** Backfill batch delay (set low for tests) */
  readonly backfillBatchDelayMs?: number
}

/**
 * Represents a gap in historical data
 */
export interface DataGap {
  readonly symbol: StockSymbol
  readonly interval: string
  readonly startTime: EpochDate
  readonly endTime: EpochDate
  readonly expectedCandles: number
  readonly actualCandles: number
}

/**
 * Historical data summary
 */
export interface HistoricalDataSummary {
  readonly symbol: StockSymbol
  readonly interval: string
  readonly earliestTimestamp: EpochDate | null
  readonly latestTimestamp: EpochDate | null
  readonly totalCandles: number
  readonly gaps: readonly DataGap[]
  readonly lastUpdated: EpochDate
}

/**
 * HistoricalDataManager handles the collection, storage, and management of historical market data.
 * 
 * Features:
 * - Automatic gap detection and backfilling
 * - Live data collection integration
 * - Data validation and integrity checks
 * - Efficient storage with compression support
 * - Event-driven architecture
 */
export class HistoricalDataManager {
  private readonly config: Required<Omit<HistoricalDataManagerConfig, 'disableWebSocket' | 'liveDataFlushIntervalMs' | 'backfillBatchDelayMs'>>
  private readonly dataRepository: MarketDataRepository
  private readonly dataFeed: CoinbaseDataFeed
  private readonly eventBus: EventBus
  private readonly logger?: Logger
  private readonly backfillService: DataBackfillService
  private readonly dataValidator: DataValidator
  private readonly liveDataCollector: LiveDataCollector
  
  /** Map of active subscriptions for live data collection */
  private readonly activeSubscriptions = new Map<string, Set<string>>()
  /** Map to track last update time for each symbol/interval */
  private readonly lastUpdateTime = new Map<string, EpochDate>()
  /** Map of active backfill operations */
  private readonly activeBackfillOperations = new Map<string, BackfillOperation>()
  /** WebSocket data source */
  private websocketSource?: CoinbaseWebSocketSource
  
  constructor(
    dataRepository: MarketDataRepository,
    dataFeed: CoinbaseDataFeed,
    config: HistoricalDataManagerConfig = {},
    logger?: Logger
  ) {
    this.dataRepository = dataRepository
    this.dataFeed = dataFeed
    this.eventBus = EventBus.getInstance()
    this.logger = logger
    
    // Apply default configuration
    this.config = {
      maxCandlesPerRequest: config.maxCandlesPerRequest ?? 300,
      maxTimeRangePerRequest: config.maxTimeRangePerRequest ?? 30,
      maxRetries: config.maxRetries ?? 3,
      retryDelayMs: config.retryDelayMs ?? 1000,
      enableValidation: config.enableValidation ?? true,
      enableAutoBackfill: config.enableAutoBackfill ?? true
    }
    
    // Initialize backfill service
    this.backfillService = new DataBackfillService(
      dataRepository,
      dataFeed,
      {
        maxConcurrentOperations: 3,
        batchSize: this.config.maxCandlesPerRequest,
        maxRetries: this.config.maxRetries,
        retryDelayMultiplier: 2,
        trackProgress: true,
        batchDelayMs: config.backfillBatchDelayMs ?? 1000
      },
      logger
    )
    
    // Initialize data validator
    this.dataValidator = new DataValidator({
      repository: dataRepository,
      logger
    })
    
    // Initialize live data collector
    this.liveDataCollector = new LiveDataCollector(
      dataRepository,
      this,
      {
        batchSize: 100,
        flushIntervalMs: config.liveDataFlushIntervalMs ?? 5000,
        enableDeduplication: true,
        enableAutoReconnect: true,
        enableValidation: true
      },
      logger
    )
    
    this.setupEventHandlers()
    if (!config.disableWebSocket) {
      void this.initializeWebSocketSource()
    }
  }

  /**
   * Initialize data collection by checking existing data and backfilling as needed
   */
  async initializeDataCollection(
    symbols: string[],
    intervals: string[],
    startDate: EpochDate | Date,
    endDate?: EpochDate | Date
  ): Promise<void> {
    const start = startDate instanceof Date ? toEpochDate(startDate) : startDate
    const end = endDate ? (endDate instanceof Date ? toEpochDate(endDate) : endDate) : epochDateNow()
    
    this.logger?.info('Initializing historical data collection', {
      symbols,
      intervals,
      startDate: new Date(start).toISOString(),
      endDate: new Date(end).toISOString()
    })
    
    for (const symbol of symbols) {
      for (const interval of intervals) {
        try {
          // Check existing data
          await this.getDataSummary(symbol, interval)
          
          // Detect gaps using the backfill service
          const gaps = await this.backfillService.detectDataGaps(
            toStockSymbol(symbol),
            interval,
            start,
            end
          )
          
          if (gaps.length > 0) {
            this.logger?.info('Detected data gaps', {
              symbol,
              interval,
              gapCount: gaps.length,
              totalMissingCandles: gaps.reduce((sum, gap) => sum + gap.expectedCandles, 0)
            })
            
            // Backfill if enabled
            if (this.config.enableAutoBackfill) {
              const operation = await this.backfillService.requestMissingData(
                toStockSymbol(symbol),
                interval,
                gaps,
                'automatic'
              )
              
              const key = `${symbol}:${interval}`
              this.activeBackfillOperations.set(key, operation)
              
              // Don't wait for backfill to complete
              operation.wait().finally(() => {
                this.activeBackfillOperations.delete(key)
              })
            }
          } else {
            this.logger?.info('No gaps detected', { symbol, interval })
          }
          
          // Update last update time
          const key = `${symbol}:${interval}`
          this.lastUpdateTime.set(key, epochDateNow())
          
        } catch (error) {
          this.logger?.error('Failed to initialize data collection', {
            symbol,
            interval,
            error: (error as Error).message
          })
          
          // Emit error event
          this.eventBus.emit(EventTypes.SYSTEM_ERROR, {
            context: 'historical_data_initialization',
            error: error as Error,
            metadata: { symbol, interval },
            timestamp: epochDateNow()
          })
        }
      }
    }
  }

  /**
   * Start live data collection for specified symbols and intervals
   */
  async startLiveDataCollection(symbols: string[], intervals: string[]): Promise<void> {
    this.logger?.info('Starting live data collection', { symbols, intervals })
    
    for (const symbol of symbols) {
      if (!this.activeSubscriptions.has(symbol)) {
        this.activeSubscriptions.set(symbol, new Set())
      }
      
      const symbolIntervals = this.activeSubscriptions.get(symbol)!
      for (const interval of intervals) {
        symbolIntervals.add(interval)
      }
    }
    
    // Subscribe via live data collector (includes WebSocket)
    await this.liveDataCollector.subscribeToSymbols(symbols, intervals)
    
    // Also subscribe to REST API feed for redundancy
    await this.dataFeed.subscribe(symbols)
    
    this.logger?.info('Live data collection started')
  }

  /**
   * Stop live data collection for specified symbols and intervals
   */
  async stopLiveDataCollection(symbols?: string[], intervals?: string[]): Promise<void> {
    if (!symbols) {
      // Stop all subscriptions
      const allSymbols = Array.from(this.activeSubscriptions.keys())
      await this.liveDataCollector.unsubscribeFromSymbols(allSymbols)
      await this.dataFeed.unsubscribe(allSymbols)
      this.activeSubscriptions.clear()
      this.logger?.info('Stopped all live data collection')
      return
    }
    
    // Unsubscribe via live data collector
    await this.liveDataCollector.unsubscribeFromSymbols(symbols)
    
    for (const symbol of symbols) {
      if (!intervals) {
        // Stop all intervals for this symbol
        this.activeSubscriptions.delete(symbol)
        await this.dataFeed.unsubscribe([symbol])
      } else {
        // Stop specific intervals
        const symbolIntervals = this.activeSubscriptions.get(symbol)
        if (symbolIntervals) {
          for (const interval of intervals) {
            symbolIntervals.delete(interval)
          }
          
          if (symbolIntervals.size === 0) {
            this.activeSubscriptions.delete(symbol)
            await this.dataFeed.unsubscribe([symbol])
          }
        }
      }
    }
    
    this.logger?.info('Stopped live data collection', { symbols, intervals })
  }

  /**
   * Get historical data for a symbol
   */
  async getHistoricalData(
    symbol: string,
    interval: string,
    startTime: EpochDate | Date,
    endTime?: EpochDate | Date,
    limit?: number
  ): Promise<Candle[]> {
    const start = startTime instanceof Date ? toEpochDate(startTime) : startTime
    const end = endTime ? (endTime instanceof Date ? toEpochDate(endTime) : endTime) : epochDateNow()
    
    try {
      const candles = await this.dataRepository.getCandles(symbol, interval, start, end, limit)
      
      if (this.config.enableValidation) {
        const validatedCandles = await this.validateCandles(candles)
        return validatedCandles
      }
      
      return candles
    } catch (error) {
      this.logger?.error('Failed to get historical data', {
        symbol,
        interval,
        error: (error as Error).message
      })
      throw error
    }
  }

  /**
   * Save historical data
   */
  async saveHistoricalData(candles: Candle[]): Promise<void> {
    if (candles.length === 0) return
    
    try {
      await this.dataRepository.saveCandlesBatch(candles)
      
      this.logger?.debug('Saved historical data', {
        count: candles.length,
        symbol: candles[0]?.symbol,
        interval: candles[0]?.interval
      })
      
      // Emit data saved event
      this.eventBus.emit(EventTypes.HISTORICAL_DATA_SAVED, {
        count: candles.length,
        symbol: candles[0]?.symbol,
        interval: candles[0]?.interval,
        timestamp: epochDateNow()
      })
    } catch (error) {
      this.logger?.error('Failed to save historical data', {
        error: (error as Error).message,
        count: candles.length
      })
      throw error
    }
  }

  /**
   * Delete historical data for a time range
   */
  async deleteHistoricalData(
    symbol: string,
    interval: string,
    startTime: EpochDate | Date,
    endTime: EpochDate | Date
  ): Promise<number> {
    // This would require adding a delete method to the repository
    // For now, we'll use the cleanup method which deletes old data
    this.logger?.warn('Delete functionality not yet implemented', {
      symbol,
      interval,
      startTime,
      endTime
    })
    return 0
  }

  /**
   * List available data for all symbols
   */
  async listAvailableData(): Promise<HistoricalDataSummary[]> {
    // This would require querying distinct symbols and intervals from the database
    // For now, return data for active subscriptions
    const summaries: HistoricalDataSummary[] = []
    
    for (const [symbol, intervals] of this.activeSubscriptions.entries()) {
      for (const interval of intervals) {
        const summary = await this.getDataSummary(symbol, interval)
        summaries.push(summary)
      }
    }
    
    return summaries
  }

  /**
   * Backfill historical data for a specific time range
   * @returns BackfillOperation for tracking progress
   */
  async backfillHistoricalData(
    symbol: StockSymbol,
    interval: string,
    startTime: EpochDate | Date,
    endTime?: EpochDate | Date,
    mode: 'automatic' | 'manual' = 'manual'
  ): Promise<BackfillOperation> {
    const stockSymbol = typeof symbol === 'string' ? toStockSymbol(symbol) : symbol
    const start = startTime instanceof Date ? toEpochDate(startTime) : startTime
    const end = endTime ? (endTime instanceof Date ? toEpochDate(endTime) : endTime) : epochDateNow()
    
    this.logger?.info('Starting backfill', {
      symbol: stockSymbol,
      interval,
      startTime: new Date(start).toISOString(),
      endTime: new Date(end).toISOString(),
      mode
    })
    
    // Detect gaps for the specified range
    const gaps = await this.backfillService.detectDataGaps(
      stockSymbol,
      interval,
      start,
      end
    )
    
    if (gaps.length === 0) {
      this.logger?.info('No gaps to backfill', { symbol: stockSymbol, interval })
      // Create a dummy completed operation
      const dummyOp: BackfillOperation = {
        id: `no-gaps-${Date.now()}`,
        symbol: stockSymbol,
        interval,
        startDate: start,
        endDate: end,
        mode,
        getProgress: () => ({
          symbol: stockSymbol,
          interval,
          totalGaps: 0,
          processedGaps: 0,
          totalCandles: 0,
          successfulCandles: 0,
          failedCandles: 0,
          startTime: epochDateNow(),
          status: 'completed',
          errors: []
        }),
        cancel: async () => {},
        wait: async () => {}
      }
      return dummyOp
    }
    
    // Request backfill for detected gaps
    const operation = await this.backfillService.requestMissingData(
      stockSymbol,
      interval,
      gaps,
      mode
    )
    
    const key = `${stockSymbol}:${interval}`
    this.activeBackfillOperations.set(key, operation)
    
    // Clean up when done
    operation.wait().finally(() => {
      this.activeBackfillOperations.delete(key)
    })
    
    return operation
  }

  /**
   * Fetch historical candles from the API with retry logic
   */
  async fetchHistoricalCandles(request: HistoricalDataRequest): Promise<Candle[]> {
    let attempt = 0
    let lastError: Error | null = null
    
    while (attempt < this.config.maxRetries) {
      try {
        const rawCandles = await this.dataFeed.getHistorical(request)
        // Transform to our Candle type with symbol and interval
        const candles: Candle[] = rawCandles.map(candle => ({
          ...candle,
          symbol: request.symbol,
          interval: request.interval || '1h',
          openTime: candle.timestamp,
          closeTime: toEpochDate(candle.timestamp + this.intervalToMilliseconds(request.interval || '1h'))
        }))
        return candles
      } catch (error) {
        lastError = error as Error
        attempt++
        
        if (attempt < this.config.maxRetries) {
          this.logger?.warn('Fetch attempt failed, retrying', {
            attempt,
            maxRetries: this.config.maxRetries,
            error: lastError.message
          })
          
          // Wait before retrying
          await new Promise(resolve => setTimeout(resolve, this.config.retryDelayMs * attempt))
        }
      }
    }
    
    throw lastError || new Error('Failed to fetch historical candles')
  }

  /**
   * Get data summary for a symbol/interval pair
   */
  private async getDataSummary(symbol: string, interval: string): Promise<HistoricalDataSummary> {
    const latest = await this.dataRepository.getLatestCandle(symbol, interval)
    
    // For earliest, we'd need to add a method to the repository
    // For now, we'll fetch the oldest candles
    const oldestCandles = await this.dataRepository.getCandles(
      symbol,
      interval,
      toEpochDate(0), // Start from epoch
      epochDateNow(),
      1 // Limit to 1
    )
    
    const earliest = oldestCandles[0]
    
    // Detect gaps (simplified - would need more sophisticated logic)
    const gaps: DataGap[] = []
    
    return {
      symbol,
      interval,
      earliestTimestamp: earliest ? earliest.timestamp : null,
      latestTimestamp: latest ? latest.timestamp : null,
      totalCandles: 0, // Would need a count method
      gaps,
      lastUpdated: epochDateNow()
    }
  }

  /**
   * Get backfill operation status
   */
  getBackfillStatus(symbol: string, interval: string): BackfillOperation | undefined {
    const key = `${symbol}:${interval}`
    return this.activeBackfillOperations.get(key)
  }

  /**
   * Get all active backfill operations
   */
  getActiveBackfillOperations(): BackfillOperation[] {
    return Array.from(this.activeBackfillOperations.values())
  }

  /**
   * Cancel a backfill operation
   */
  async cancelBackfill(symbol: string, interval: string): Promise<void> {
    const key = `${symbol}:${interval}`
    const operation = this.activeBackfillOperations.get(key)
    if (operation) {
      await operation.cancel()
      this.activeBackfillOperations.delete(key)
    }
  }

  /**
   * Cancel all backfill operations
   */
  async cancelAllBackfills(): Promise<void> {
    await this.backfillService.cancelAllOperations()
    this.activeBackfillOperations.clear()
  }

  /**
   * Detect gaps in historical data
   * @deprecated Use backfillService.detectDataGaps instead
   */
  // @ts-ignore - deprecated method
  private async detectGaps(
    symbol: string,
    interval: string,
    startTime: EpochDate,
    endTime: EpochDate
  ): Promise<DataGap[]> {
    const gaps: DataGap[] = []
    
    // Fetch all candles in the time range
    const candles = await this.dataRepository.getCandles(symbol, interval, startTime, endTime)
    
    if (candles.length === 0) {
      // Entire range is a gap
      const expectedCandles = this.calculateExpectedCandles(interval, startTime, endTime)
      gaps.push({
        symbol: toStockSymbol(symbol),
        interval,
        startTime,
        endTime,
        expectedCandles,
        actualCandles: 0
      })
      return gaps
    }
    
    // Sort candles by timestamp
    candles.sort((a, b) => a.timestamp - b.timestamp)
    
    // Check for gap at the beginning
    if (candles[0]!.timestamp > startTime) {
      const expectedCandles = this.calculateExpectedCandles(interval, startTime, candles[0]!.timestamp)
      if (expectedCandles > 1) {
        gaps.push({
          symbol: toStockSymbol(symbol),
          interval,
          startTime,
          endTime: candles[0]!.timestamp,
          expectedCandles,
          actualCandles: 0
        })
      }
    }
    
    // Check for gaps between candles
    const intervalMs = this.intervalToMilliseconds(interval)
    for (let i = 0; i < candles.length - 1; i++) {
      const current = candles[i]!
      const next = candles[i + 1]!
      const expectedNext = current.timestamp + intervalMs
      
      if (next.timestamp > expectedNext + intervalMs) {
        const expectedCandles = this.calculateExpectedCandles(interval, current.timestamp, next.timestamp) - 1
        gaps.push({
          symbol: toStockSymbol(symbol),
          interval,
          startTime: toEpochDate(current.timestamp + intervalMs),
          endTime: toEpochDate(next.timestamp - intervalMs),
          expectedCandles,
          actualCandles: 0
        })
      }
    }
    
    // Check for gap at the end
    const lastCandle = candles[candles.length - 1]!
    if (lastCandle.timestamp < endTime - intervalMs) {
      const expectedCandles = this.calculateExpectedCandles(interval, lastCandle.timestamp, endTime) - 1
      if (expectedCandles > 0) {
        gaps.push({
          symbol: toStockSymbol(symbol),
          interval,
          startTime: toEpochDate(lastCandle.timestamp + intervalMs),
          endTime,
          expectedCandles,
          actualCandles: 0
        })
      }
    }
    
    return gaps
  }

  /**
   * Backfill detected gaps
   * @deprecated Use backfillService.requestMissingData instead
   */
  // @ts-ignore - deprecated method
  private async backfillGaps(gaps: DataGap[]): Promise<void> {
    if (gaps.length === 0) return
    
    const operation = await this.backfillService.requestMissingData(
      gaps[0]!.symbol,
      gaps[0]!.interval,
      gaps,
      'automatic'
    )
    
    // Wait for completion
    await operation.wait()
  }

  /**
   * Validate candles for data integrity
   */
  private async validateCandles(candles: Candle[]): Promise<Candle[]> {
    const validCandles: Candle[] = []
    
    for (const candle of candles) {
      // Basic validation
      if (candle.open <= 0 || candle.high <= 0 || candle.low <= 0 || candle.close <= 0) {
        this.logger?.warn('Invalid candle prices', { candle })
        continue
      }
      
      if (candle.high < candle.low) {
        this.logger?.warn('High price less than low price', { candle })
        continue
      }
      
      if (candle.high < candle.open || candle.high < candle.close) {
        this.logger?.warn('High price not the highest', { candle })
        continue
      }
      
      if (candle.low > candle.open || candle.low > candle.close) {
        this.logger?.warn('Low price not the lowest', { candle })
        continue
      }
      
      if (candle.volume < 0) {
        this.logger?.warn('Negative volume', { candle })
        continue
      }
      
      validCandles.push(candle)
    }
    
    if (validCandles.length < candles.length) {
      this.logger?.info('Filtered invalid candles', {
        original: candles.length,
        valid: validCandles.length,
        filtered: candles.length - validCandles.length
      })
    }
    
    return validCandles
  }

  /**
   * Calculate expected number of candles for a time range
   */
  private calculateExpectedCandles(interval: string, startTime: EpochDate, endTime: EpochDate): number {
    const intervalMs = this.intervalToMilliseconds(interval)
    return Math.floor((endTime - startTime) / intervalMs)
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
   * Initialize WebSocket data source
   */
  private async initializeWebSocketSource(): Promise<void> {
    try {
      // Create WebSocket source
      this.websocketSource = new CoinbaseWebSocketSource(
        {
          wsUrl: 'wss://ws-feed.exchange.coinbase.com',
          channels: ['ticker', 'matches'],
          enableHeartbeat: true,
          maxReconnectAttempts: 5
        },
        this.logger
      )
      
      // Register with live data collector
      await this.liveDataCollector.registerLiveDataSource(this.websocketSource)
      
      // Connect
      await this.websocketSource.connect()
      
      this.logger?.info('WebSocket source initialized and connected')
    } catch (error) {
      this.logger?.error('Failed to initialize WebSocket source', {
        error: (error as Error).message
      })
    }
  }

  /**
   * Setup event handlers for live data collection
   */
  private setupEventHandlers(): void {
    // Note: Candle events are now handled by the LiveDataCollector
    // which automatically saves to the repository and emits events.
    // We only need to handle specific historical data manager events here.
    
    // Handle backfill completion events
    this.eventBus.subscribe(EventTypes.BACKFILL_COMPLETED, (event: any) => {
      const key = `${event.symbol}:${event.interval}`
      this.lastUpdateTime.set(key, epochDateNow())
      
      this.logger?.info('Backfill completed', {
        symbol: event.symbol,
        interval: event.interval,
        candlesSaved: event.progress?.successfulCandles || 0
      })
    })
    
    // Handle system errors related to data collection
    this.eventBus.subscribe(EventTypes.SYSTEM_ERROR, (event: any) => {
      if (event.context === 'historical_data' || event.context === 'live_data') {
        this.logger?.error('Data collection error', {
          context: event.context,
          error: event.error?.message,
          metadata: event.metadata
        })
      }
    })
  }

  /**
   * Get statistics about stored historical data
   */
  async getDataStatistics(): Promise<{
    totalSymbols: number
    totalCandles: number
    dateRange: { earliest: Date | null; latest: Date | null }
    symbolStats: Array<{
      symbol: string
      intervals: string[]
      candleCount: number
    }>
  }> {
    // This would require additional repository methods
    // For now, return basic stats based on active subscriptions
    const symbolStats = []
    let totalCandles = 0
    
    for (const [symbol, intervals] of this.activeSubscriptions.entries()) {
      symbolStats.push({
        symbol,
        intervals: Array.from(intervals),
        candleCount: 0 // Would need to query from database
      })
      totalCandles += 1
    }
    
    return {
      totalSymbols: this.activeSubscriptions.size,
      totalCandles,
      dateRange: {
        earliest: null,
        latest: null
      },
      symbolStats
    }
  }

  /**
   * Validate data integrity for a symbol
   */
  async validateDataIntegrity(
    symbol: StockSymbol,
    interval: string,
    startTime?: EpochDate,
    endTime?: EpochDate
  ): Promise<ValidationReport> {
    const stockSymbol = typeof symbol === 'string' ? toStockSymbol(symbol) : symbol
    
    this.logger?.info('Starting data integrity validation', {
      symbol: stockSymbol,
      interval,
      startTime: startTime ? Number(startTime) : undefined,
      endTime: endTime ? Number(endTime) : undefined
    })
    
    const report = await this.dataValidator.verifyDataIntegrity(
      stockSymbol,
      interval,
      startTime,
      endTime,
      {
        autoRepair: this.config.enableAutoBackfill
      }
    )
    
    // Emit validation event
    this.eventBus.emit(EventTypes.HISTORICAL_DATA_VALIDATED, {
      symbol: stockSymbol,
      interval,
      reportId: `${stockSymbol}:${interval}:${Date.now()}`,
      overallHealth: report.overallHealth,
      issueCount: report.issues.length,
      timestamp: epochDateNow()
    })
    
    // Handle critical issues
    if (report.overallHealth === 'critical') {
      this.logger?.error('Critical data integrity issues detected', {
        symbol: stockSymbol,
        interval,
        issueCount: report.issues.length,
        criticalIssues: report.issues.filter(i => i.severity === 'critical').length
      })
    }
    
    // Auto-repair if enabled
    if (this.config.enableAutoBackfill && report.issues.length > 0) {
      const backfillableIssues = report.issues.filter(i => i.suggestedAction === 'backfill')
      if (backfillableIssues.length > 0) {
        this.logger?.info('Triggering auto-backfill for validation issues', {
          symbol: stockSymbol,
          interval,
          backfillableIssues: backfillableIssues.length
        })
        
        // Convert validation issues to gaps
        const gaps = backfillableIssues.map(issue => ({
          symbol: stockSymbol,
          interval,
          startTime: issue.startTime,
          endTime: issue.endTime,
          expectedCandles: issue.affectedCandles || 0,
          actualCandles: 0
        }))
        
        // Request backfill for each gap
        for (const gap of gaps) {
          await this.backfillHistoricalData(
            stockSymbol,
            interval,
            gap.startTime,
            gap.endTime,
            'automatic'
          )
        }
      }
    }
    
    return report
  }

  /**
   * Repair corrupted data based on validation results
   */
  async repairCorruptedData(
    symbol: StockSymbol,
    interval: string,
    report: ValidationReport
  ): Promise<{ repaired: number; failed: number }> {
    const stockSymbol = typeof symbol === 'string' ? toStockSymbol(symbol) : symbol
    
    if (report.issues.length === 0) {
      this.logger?.info('No issues to repair', {
        symbol: stockSymbol,
        interval
      })
      return { repaired: 0, failed: 0 }
    }
    
    const result = await this.dataValidator.repairCorruptedData(
      stockSymbol,
      interval,
      report.issues,
      {
        autoRepair: true,
        maxRepairAttempts: 3
      }
    )
    
    this.logger?.info('Data repair completed', {
      symbol: stockSymbol,
      interval,
      repaired: result.repaired,
      failed: result.failed,
      total: report.issues.length
    })
    
    return result
  }

  /**
   * Get comprehensive data quality metrics
   */
  async getDataQualityMetrics(
    symbol?: StockSymbol,
    interval?: string
  ): Promise<{
    overallQuality: 'excellent' | 'good' | 'fair' | 'poor'
    metrics: {
      completeness: number // 0-100%
      accuracy: number // 0-100%
      timeliness: number // 0-100%
      consistency: number // 0-100%
    }
    details: ValidationReport[]
  }> {
    const reports: ValidationReport[] = []
    
    // If specific symbol/interval provided, validate just that
    if (symbol && interval) {
      const stockSymbol = typeof symbol === 'string' ? toStockSymbol(symbol) : symbol
      const report = await this.validateDataIntegrity(stockSymbol, interval)
      reports.push(report)
    } else {
      // Validate all active subscriptions
      for (const [subscriptionKey] of this.activeSubscriptions) {
        const [sym, int] = subscriptionKey.split(':')
        if (sym && int) {
          const report = await this.validateDataIntegrity(toStockSymbol(sym), int)
          reports.push(report)
        }
      }
    }
    
    // Calculate overall metrics
    let totalCandles = 0
    let totalIssues = 0
    let gapIssues = 0
    let outlierIssues = 0
    let invalidIssues = 0
    
    for (const report of reports) {
      totalCandles += report.totalCandles
      totalIssues += report.issues.length
      gapIssues += report.issues.filter(i => i.type === 'gap').length
      outlierIssues += report.issues.filter(i => i.type === 'outlier').length
      invalidIssues += report.issues.filter(i => i.type === 'invalid' || i.type === 'corrupted').length
    }
    
    // Calculate quality scores
    const completeness = totalCandles > 0 ? Math.max(0, 100 - (gapIssues / totalCandles * 100)) : 0
    const accuracy = totalCandles > 0 ? Math.max(0, 100 - (outlierIssues / totalCandles * 100)) : 0
    const consistency = totalCandles > 0 ? Math.max(0, 100 - (invalidIssues / totalCandles * 100)) : 0
    const timeliness = 100 // Would calculate based on how recent the data is
    
    const avgQuality = (completeness + accuracy + consistency + timeliness) / 4
    
    let overallQuality: 'excellent' | 'good' | 'fair' | 'poor'
    if (avgQuality >= 95) overallQuality = 'excellent'
    else if (avgQuality >= 85) overallQuality = 'good'
    else if (avgQuality >= 70) overallQuality = 'fair'
    else overallQuality = 'poor'
    
    return {
      overallQuality,
      metrics: {
        completeness,
        accuracy,
        timeliness,
        consistency
      },
      details: reports
    }
  }
}