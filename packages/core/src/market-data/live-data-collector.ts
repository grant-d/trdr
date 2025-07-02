import type { MarketDataRepository } from '@trdr/data'
import type { EpochDate, StockSymbol } from '@trdr/shared'
import { epochDateNow, toStockSymbol } from '@trdr/shared'
import type { Candle, Logger } from '@trdr/types'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import type { HistoricalDataManager } from './historical-data-manager'

/**
 * Configuration for live data collection
 */
export interface LiveDataCollectorConfig {
  /** Buffer size for batching writes to database */
  readonly batchSize?: number
  /** Max time to wait before flushing buffer (ms) */
  readonly flushIntervalMs?: number
  /** Enable deduplication of incoming data */
  readonly enableDeduplication?: boolean
  /** Time window for deduplication (ms) */
  readonly deduplicationWindowMs?: number
  /** Enable automatic reconnection on disconnect */
  readonly enableAutoReconnect?: boolean
  /** Max reconnection attempts */
  readonly maxReconnectAttempts?: number
  /** Reconnect delay multiplier */
  readonly reconnectDelayMultiplier?: number
  /** Enable data validation */
  readonly enableValidation?: boolean
}

/**
 * Live data source interface
 */
export interface LiveDataSource {
  readonly name: string
  readonly type: 'websocket' | 'rest' | 'streaming'
  subscribe(symbols: string[]): Promise<void>
  unsubscribe(symbols: string[]): Promise<void>
  isConnected(): boolean
  reconnect?(): Promise<void>
}

/**
 * Connection state for monitoring
 */
export interface ConnectionState {
  readonly source: string
  readonly connected: boolean
  readonly lastConnected?: EpochDate
  readonly lastDisconnected?: EpochDate
  readonly reconnectAttempts: number
  readonly error?: string
}

/**
 * Live data statistics
 */
export interface LiveDataStats {
  readonly startTime: EpochDate
  readonly totalReceived: number
  readonly totalProcessed: number
  readonly totalErrors: number
  readonly totalDuplicates: number
  readonly bufferSize: number
  readonly connectionStates: ReadonlyMap<string, ConnectionState>
}

/**
 * Subscription info
 */
interface SubscriptionInfo {
  readonly symbol: StockSymbol
  readonly intervals: Set<string>
  readonly sources: Set<string>
  readonly lastUpdate?: EpochDate
}

/**
 * Candle buffer entry
 */
interface BufferedCandle {
  readonly candle: Candle
  readonly source: string
  readonly receivedAt: EpochDate
}

/**
 * LiveDataCollector manages real-time data collection from multiple sources
 * and seamlessly integrates with historical data storage.
 * 
 * Features:
 * - Multiple data source management
 * - Automatic reconnection handling
 * - Data deduplication
 * - Batch writing for efficiency
 * - Connection state monitoring
 * - Seamless historical data integration
 */
export class LiveDataCollector {
  private readonly config: Required<LiveDataCollectorConfig>
  private readonly repository: MarketDataRepository
  private readonly historicalManager: HistoricalDataManager
  private readonly eventBus: EventBus
  private readonly logger?: Logger
  
  /** Registered data sources */
  private readonly dataSources = new Map<string, LiveDataSource>()
  /** Active subscriptions */
  private readonly subscriptions = new Map<string, SubscriptionInfo>()
  /** Connection states */
  private readonly connectionStates = new Map<string, ConnectionState>()
  /** Candle buffer for batch writing */
  private readonly candleBuffer: BufferedCandle[] = []
  /** Deduplication cache */
  private readonly deduplicationCache = new Map<string, EpochDate>()
  /** Flush timer */
  private flushTimer?: NodeJS.Timeout
  /** Reconnect timers */
  private readonly reconnectTimers = new Map<string, NodeJS.Timeout>()
  /** Connection monitoring timers */
  private readonly monitorTimers = new Map<string, NodeJS.Timeout>()
  
  /** Statistics */
  private readonly stats = {
    startTime: epochDateNow(),
    totalReceived: 0,
    totalProcessed: 0,
    totalErrors: 0,
    totalDuplicates: 0
  }
  
  constructor(
    repository: MarketDataRepository,
    historicalManager: HistoricalDataManager,
    config: LiveDataCollectorConfig = {},
    logger?: Logger
  ) {
    this.repository = repository
    this.historicalManager = historicalManager
    this.eventBus = EventBus.getInstance()
    this.logger = logger
    
    this.config = {
      batchSize: config.batchSize ?? 100,
      flushIntervalMs: config.flushIntervalMs ?? 5000,
      enableDeduplication: config.enableDeduplication ?? true,
      deduplicationWindowMs: config.deduplicationWindowMs ?? 60000,
      enableAutoReconnect: config.enableAutoReconnect ?? true,
      maxReconnectAttempts: config.maxReconnectAttempts ?? 5,
      reconnectDelayMultiplier: config.reconnectDelayMultiplier ?? 2,
      enableValidation: config.enableValidation ?? true
    }
    
    this.setupEventHandlers()
    this.startFlushTimer()
  }

  /**
   * Register a live data source
   */
  async registerLiveDataSource(source: LiveDataSource): Promise<void> {
    if (this.dataSources.has(source.name)) {
      throw new Error(`Data source ${source.name} already registered`)
    }
    
    this.logger?.info('Registering live data source', { name: source.name, type: source.type })
    
    this.dataSources.set(source.name, source)
    this.connectionStates.set(source.name, {
      source: source.name,
      connected: source.isConnected(),
      reconnectAttempts: 0
    })
    
    // Setup connection monitoring
    if (source.type === 'websocket' || source.type === 'streaming') {
      this.monitorConnection(source)
    }
    
    this.eventBus.emit(EventTypes.SYSTEM_INFO, {
      message: `Registered live data source: ${source.name}`,
      context: 'live_data_collector',
      timestamp: epochDateNow()
    })
  }

  /**
   * Subscribe to symbols across all data sources
   */
  async subscribeToSymbols(
    symbols: Array<StockSymbol>,
    intervals: string[],
    sources?: string[]
  ): Promise<void> {
    const stockSymbols = symbols.map(s => typeof s === 'string' ? toStockSymbol(s) : s)
    const targetSources = sources || Array.from(this.dataSources.keys())
    
    this.logger?.info('Subscribing to symbols', {
      symbols: stockSymbols,
      intervals,
      sources: targetSources
    })
    
    // Update subscription info
    for (const symbol of stockSymbols) {
      const key = symbol
      if (!this.subscriptions.has(key)) {
        this.subscriptions.set(key, {
          symbol,
          intervals: new Set(),
          sources: new Set()
        })
      }
      
      const sub = this.subscriptions.get(key)!
      intervals.forEach(interval => sub.intervals.add(interval))
      targetSources.forEach(source => sub.sources.add(source))
    }
    
    // Subscribe on each source
    const subscribePromises = targetSources.map(async sourceName => {
      const source = this.dataSources.get(sourceName)
      if (!source) {
        this.logger?.warn('Data source not found', { source: sourceName })
        return
      }
      
      try {
        await source.subscribe(stockSymbols.map(s => s))
        
        // Also ensure historical data manager is collecting
        await this.historicalManager.startLiveDataCollection(
          stockSymbols.map(s => s),
          intervals
        )
      } catch (error) {
        this.logger?.error('Failed to subscribe', {
          source: sourceName,
          error: (error as Error).message
        })
        this.stats.totalErrors++
      }
    })
    
    await Promise.all(subscribePromises)
  }

  /**
   * Unsubscribe from symbols
   */
  async unsubscribeFromSymbols(
    symbols: Array<string>,
    sources?: string[]
  ): Promise<void> {
    const stockSymbols = symbols.map(s => typeof s === 'string' ? toStockSymbol(s) : s)
    const targetSources = sources || Array.from(this.dataSources.keys())
    
    this.logger?.info('Unsubscribing from symbols', {
      symbols: stockSymbols,
      sources: targetSources
    })
    
    // Update subscription info
    for (const symbol of stockSymbols) {
      const key = symbol
      const sub = this.subscriptions.get(key)
      if (sub) {
        if (sources) {
          sources.forEach(source => sub.sources.delete(source))
          if (sub.sources.size === 0) {
            this.subscriptions.delete(key)
          }
        } else {
          this.subscriptions.delete(key)
        }
      }
    }
    
    // Unsubscribe on each source
    const unsubscribePromises = targetSources.map(async sourceName => {
      const source = this.dataSources.get(sourceName)
      if (!source) return
      
      try {
        await source.unsubscribe(stockSymbols.map(s => s))
      } catch (error) {
        this.logger?.error('Failed to unsubscribe', {
          source: sourceName,
          error: (error as Error).message
        })
      }
    })
    
    await Promise.all(unsubscribePromises)
  }

  /**
   * Process incoming data from any source
   */
  async processIncomingData(data: unknown, source: string): Promise<void> {
    this.stats.totalReceived++
    
    try {
      // Convert to candle format
      const candle = this.convertToCandle(data, source)
      if (!candle) {
        this.logger?.warn('Failed to convert data to candle', { data, source })
        return
      }
      
      // Validate if enabled
      if (this.config.enableValidation && !this.validateCandle(candle)) {
        this.logger?.warn('Invalid candle data', { candle, source })
        this.stats.totalErrors++
        return
      }
      
      // Check for duplicates
      if (this.config.enableDeduplication && this.isDuplicate(candle)) {
        this.stats.totalDuplicates++
        return
      }
      
      // Add to buffer
      this.candleBuffer.push({
        candle,
        source,
        receivedAt: epochDateNow()
      })
      
      // Update subscription last update time
      const sub = this.subscriptions.get(candle.symbol)
      if (sub) {
        this.subscriptions.set(candle.symbol, {
          ...sub,
          lastUpdate: epochDateNow()
        })
      }
      
      // Emit event for real-time consumers
      this.eventBus.emit(EventTypes.CANDLE, {
        candle,
        source,
        timestamp: epochDateNow()
      })
      
      // Flush if buffer is full
      if (this.candleBuffer.length >= this.config.batchSize) {
        await this.flushBuffer()
      }
      
      this.stats.totalProcessed++
      
    } catch (error) {
      this.logger?.error('Error processing incoming data', {
        error: (error as Error).message,
        source
      })
      this.stats.totalErrors++
    }
  }

  /**
   * Append processed data to historical records
   */
  async appendToHistoricalRecords(candles: Candle[]): Promise<void> {
    if (candles.length === 0) return
    
    try {
      // Save to repository
      await this.repository.saveCandlesBatch(candles)
      
      // Emit event
      this.eventBus.emit(EventTypes.HISTORICAL_DATA_SAVED, {
        count: candles.length,
        symbol: candles[0]?.symbol,
        interval: candles[0]?.interval,
        timestamp: epochDateNow()
      })
      
      this.logger?.debug('Appended to historical records', {
        count: candles.length,
        firstTimestamp: candles[0]?.timestamp,
        lastTimestamp: candles[candles.length - 1]?.timestamp
      })
      
    } catch (error) {
      this.logger?.error('Failed to append to historical records', {
        error: (error as Error).message,
        candleCount: candles.length
      })
      throw error
    }
  }

  /**
   * Get live data statistics
   */
  getStatistics(): LiveDataStats {
    return {
      ...this.stats,
      bufferSize: this.candleBuffer.length,
      connectionStates: new Map(this.connectionStates)
    }
  }

  /**
   * Get active subscriptions
   */
  getActiveSubscriptions(): ReadonlyMap<string, SubscriptionInfo> {
    return new Map(this.subscriptions)
  }

  /**
   * Stop live data collection
   */
  async stop(): Promise<void> {
    this.logger?.info('Stopping live data collector')
    
    // Clear timers
    if (this.flushTimer) {
      clearInterval(this.flushTimer)
      this.flushTimer = undefined
    }
    
    for (const timer of this.reconnectTimers.values()) {
      clearTimeout(timer)
    }
    this.reconnectTimers.clear()
    
    // Clear connection monitoring timers
    for (const timer of this.monitorTimers.values()) {
      clearInterval(timer)
    }
    this.monitorTimers.clear()
    
    // Flush remaining buffer
    await this.flushBuffer()
    
    // Unsubscribe all
    const allSymbols = Array.from(this.subscriptions.keys())
    if (allSymbols.length > 0) {
      await this.unsubscribeFromSymbols(allSymbols)
    }
    
    this.logger?.info('Live data collector stopped')
  }

  /**
   * Setup event handlers
   */
  private setupEventHandlers(): void {
    // Handle candle events from data sources
    this.eventBus.subscribe(EventTypes.CANDLE, async (event: any) => {
      if (event.source && this.dataSources.has(event.source)) {
        await this.processIncomingData(event.candle, event.source)
      }
    })
    
    // Handle connection events
    this.eventBus.subscribe(EventTypes.CONNECTION_STATUS, (event: any) => {
      this.handleConnectionStatus(event)
    })
  }

  /**
   * Start flush timer
   */
  private startFlushTimer(): void {
    this.flushTimer = setInterval(async () => {
      if (this.candleBuffer.length > 0) {
        await this.flushBuffer()
      }
      
      // Clean old deduplication entries
      this.cleanDeduplicationCache()
    }, this.config.flushIntervalMs)
  }

  /**
   * Flush candle buffer to database
   */
  private async flushBuffer(): Promise<void> {
    if (this.candleBuffer.length === 0) return
    
    const candles = this.candleBuffer.map(b => b.candle)
    this.candleBuffer.length = 0 // Clear buffer
    
    try {
      await this.appendToHistoricalRecords(candles)
    } catch (error) {
      // Re-add to buffer on failure
      this.candleBuffer.push(...candles.map(candle => ({
        candle,
        source: 'retry',
        receivedAt: epochDateNow()
      })))
      
      this.logger?.error('Failed to flush buffer, will retry', {
        error: (error as Error).message,
        candleCount: candles.length
      })
    }
  }

  /**
   * Convert incoming data to candle format
   */
  private convertToCandle(data: unknown, _source: string): Candle | null {
    // This would need to handle different data formats from various sources
    // For now, assume data is already in candle format
    if (typeof data === 'object' && data !== null && 'timestamp' in data) {
      return data as Candle
    }
    
    return null
  }

  /**
   * Validate candle data
   */
  private validateCandle(candle: Candle): boolean {
    // Basic validation
    if (!candle.symbol || !candle.interval || !candle.timestamp) {
      return false
    }
    
    if (candle.open <= 0 || candle.high <= 0 || candle.low <= 0 || candle.close <= 0) {
      return false
    }
    
    if (candle.high < candle.low) {
      return false
    }
    
    if (candle.high < candle.open || candle.high < candle.close) {
      return false
    }
    
    if (candle.low > candle.open || candle.low > candle.close) {
      return false
    }
    
    if (candle.volume < 0) {
      return false
    }
    
    return true
  }

  /**
   * Check if candle is duplicate
   */
  private isDuplicate(candle: Candle): boolean {
    const key = `${candle.symbol}:${candle.interval}:${candle.timestamp}`
    const lastSeen = this.deduplicationCache.get(key)
    
    if (lastSeen) {
      return true
    }
    
    this.deduplicationCache.set(key, epochDateNow())
    return false
  }

  /**
   * Clean old entries from deduplication cache
   */
  private cleanDeduplicationCache(): void {
    const cutoff = epochDateNow() - this.config.deduplicationWindowMs
    
    for (const [key, timestamp] of this.deduplicationCache.entries()) {
      if (timestamp < cutoff) {
        this.deduplicationCache.delete(key)
      }
    }
  }

  /**
   * Monitor connection health
   */
  private monitorConnection(source: LiveDataSource): void {
    // Check connection periodically
    const checkInterval = setInterval(() => {
      const state = this.connectionStates.get(source.name)
      if (!state) {
        clearInterval(checkInterval)
        this.monitorTimers.delete(source.name)
        return
      }
      
      const wasConnected = state.connected
      const isConnected = source.isConnected()
      
      if (wasConnected && !isConnected) {
        // Connection lost
        this.handleDisconnection(source)
      } else if (!wasConnected && isConnected) {
        // Connection restored
        this.handleReconnection(source)
      }
    }, 5000) // Check every 5 seconds
    
    // Track the monitor timer for cleanup
    this.monitorTimers.set(source.name, checkInterval)
  }

  /**
   * Handle connection status events
   */
  private handleConnectionStatus(event: any): void {
    const { source: sourceName, status } = event
    const source = this.dataSources.get(sourceName)
    
    if (!source) return
    
    if (status === 'disconnected' || status === 'error') {
      this.handleDisconnection(source)
    } else if (status === 'connected') {
      this.handleReconnection(source)
    }
  }

  /**
   * Handle disconnection
   */
  private handleDisconnection(source: LiveDataSource): void {
    const state = this.connectionStates.get(source.name)
    if (!state) return
    
    this.connectionStates.set(source.name, {
      ...state,
      connected: false,
      lastDisconnected: epochDateNow()
    })
    
    this.logger?.warn('Data source disconnected', { source: source.name })
    
    if (this.config.enableAutoReconnect && source.reconnect) {
      this.scheduleReconnect(source)
    }
  }

  /**
   * Handle reconnection
   */
  private handleReconnection(source: LiveDataSource): void {
    const state = this.connectionStates.get(source.name)
    if (!state) return
    
    this.connectionStates.set(source.name, {
      ...state,
      connected: true,
      lastConnected: epochDateNow(),
      reconnectAttempts: 0
    })
    
    // Clear reconnect timer
    const timer = this.reconnectTimers.get(source.name)
    if (timer) {
      clearTimeout(timer)
      this.reconnectTimers.delete(source.name)
    }
    
    this.logger?.info('Data source reconnected', { source: source.name })
    
    // Re-subscribe to active symbols
    this.resubscribeToSource(source).catch(error => {
      this.logger?.error('Failed to resubscribe after reconnection', {
        source: source.name,
        error: (error as Error).message
      })
    })
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(source: LiveDataSource): void {
    const state = this.connectionStates.get(source.name)
    if (!state || state.reconnectAttempts >= this.config.maxReconnectAttempts) {
      this.logger?.error('Max reconnection attempts reached', {
        source: source.name,
        attempts: state?.reconnectAttempts
      })
      return
    }
    
    const delay = Math.min(
      1000 * Math.pow(this.config.reconnectDelayMultiplier, state.reconnectAttempts),
      30000 // Max 30 seconds
    )
    
    this.logger?.info('Scheduling reconnection attempt', {
      source: source.name,
      attempt: state.reconnectAttempts + 1,
      delay
    })
    
    const timer = setTimeout(async () => {
      try {
        if (source.reconnect) {
          await source.reconnect()
        }
        
        this.connectionStates.set(source.name, {
          ...state,
          reconnectAttempts: state.reconnectAttempts + 1
        })
      } catch (error) {
        this.logger?.error('Reconnection attempt failed', {
          source: source.name,
          error: (error as Error).message
        })
        
        // Schedule next attempt
        this.scheduleReconnect(source)
      }
    }, delay)
    
    this.reconnectTimers.set(source.name, timer)
  }

  /**
   * Resubscribe to active symbols on a source
   */
  private async resubscribeToSource(source: LiveDataSource): Promise<void> {
    const symbols: string[] = []
    
    for (const [symbol, info] of this.subscriptions.entries()) {
      if (info.sources.has(source.name)) {
        symbols.push(symbol)
      }
    }
    
    if (symbols.length > 0) {
      await source.subscribe(symbols)
    }
  }
}