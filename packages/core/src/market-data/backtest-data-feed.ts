import type { Candle, EpochDate } from '@trdr/shared'
import { epochDateNow, toEpochDate, toIsoDate } from '@trdr/shared'
import { EventTypes } from '../events/types'
import type {
  ConnectionStats,
  HistoricalDataRequest,
} from '../interfaces/market-data-pipeline'
import { EnhancedMarketDataFeed, type EnhancedDataFeedConfig } from './enhanced-market-data-feed'

/**
 * Configuration specific to backtest data feed with enhanced event capabilities
 */
export interface BacktestConfig extends EnhancedDataFeedConfig {
  /** Historical data source (file path or database connection) */
  readonly dataSource: string
  /** Replay speed multiplier (1 = real-time, 1000 = 1000x speed) */
  readonly speed?: number
  /** Start date for backtest */
  readonly startDate: EpochDate
  /** End date for backtest */
  readonly endDate: EpochDate
  /** Simulate network delays (in milliseconds) */
  readonly networkDelay?: number
  /** Simulate random failures (probability 0-1) */
  readonly failureRate?: number
  /** Batch size for data loading */
  readonly batchSize?: number
}

/**
 * Enhanced BacktestDataFeed implementation for historical data replay
 * Provides controlled time simulation with advanced event filtering and priority handling
 */
export class BacktestDataFeed extends EnhancedMarketDataFeed {
  private readonly data = new Map<string, Candle[]>()
  private readonly currentIndices = new Map<string, number>()
  private replayTimer: NodeJS.Timeout | null = null
  private isReplaying = false
  private speed: number
  private readonly networkDelay: number
  private readonly failureRate: number
  private currentTime: EpochDate
  private readonly startDate: EpochDate
  private readonly endDate: EpochDate
  private readonly dataSource: string

  constructor(config: BacktestConfig) {
    super(config)

    this.speed = config.speed || 1
    this.networkDelay = config.networkDelay || 0
    this.failureRate = config.failureRate || 0
    // TODO: Implement batch loading when SQLite integration is added
    this.startDate = config.startDate
    this.endDate = config.endDate
    this.currentTime = config.startDate
    this.dataSource = config.dataSource
  }

  /**
   * Start the backtest data feed
   */
  async start(): Promise<void> {
    this.debug('Starting backtest data feed')

    try {
      // Load historical data for all symbols
      await this.loadHistoricalData()

      this.connected = true
      this.startTime = epochDateNow()
      this.currentTime = toEpochDate(this.startDate)

      this.emitConnected()
      this.emitConnectionStatus('connected')
      this.debug(`Backtest data feed started. Speed: ${this.speed}x, Period: ${toIsoDate(this.startDate)} to ${toIsoDate(this.endDate)}`)
    } catch (error) {
      this.debug('Failed to start backtest data feed', error)
      this.emitError(error as Error)
      this.emitConnectionStatus('error')
      throw error
    }
  }

  /**
   * Stop the backtest data feed
   */
  async stop(): Promise<void> {
    this.debug('Stopping backtest data feed')

    this.isReplaying = false
    if (this.replayTimer) {
      clearTimeout(this.replayTimer)
      this.replayTimer = null
    }

    this.connected = false
    this.emitDisconnected('Manual stop')
    this.emitConnectionStatus('disconnected')
    await Promise.resolve()
  }

  /**
   * Subscribe to symbols for backtest replay
   */
  async subscribe(symbols: string[]): Promise<void> {
    this.debug('Subscribing to symbols for backtest', symbols)

    // Add symbols to subscription set
    symbols.forEach(symbol => {
      this.subscribedSymbols.add(symbol)
      this.currentIndices.set(symbol, 0)
    })

    // Start replay if we have data and not already replaying
    if (this.connected && !this.isReplaying && this.subscribedSymbols.size > 0) {
      await this.startReplay()
    }
  }

  /**
   * Unsubscribe from symbols
   */
  async unsubscribe(symbols: string[]): Promise<void> {
    this.debug('Unsubscribing from symbols', symbols)

    symbols.forEach(symbol => {
      this.subscribedSymbols.delete(symbol)
      this.currentIndices.delete(symbol)
    })

    // Stop replay if no symbols left
    if (this.subscribedSymbols.size === 0) {
      this.isReplaying = false
      if (this.replayTimer) {
        clearTimeout(this.replayTimer)
        this.replayTimer = null
      }
    }
    await Promise.resolve()
  }

  /**
   * Get historical data (returns pre-loaded data slice)
   */
  async getHistorical(request: HistoricalDataRequest): Promise<Candle[]> {
    this.debug('Fetching historical data', request)

    await this.simulateNetworkDelay()
    this.simulateFailure()

    const symbolData = this.data.get(request.symbol)
    if (!symbolData) {
      return []
    }

    // Filter data by date range
    const filtered = symbolData.filter(candle => {
      const startEpoch = request.start
      const endEpoch = request.end
      return candle.timestamp >= startEpoch && candle.timestamp <= endEpoch
    })

    // Apply limit if specified
    if (request.limit && filtered.length > request.limit) {
      return filtered.slice(-request.limit)
    }

    return filtered
  }

  /**
   * Get current price (latest price from current replay position)
   */
  async getCurrentPrice(symbol: string): Promise<number> {
    this.debug('Fetching current price for', symbol)

    await this.simulateNetworkDelay()
    this.simulateFailure()

    const symbolData = this.data.get(symbol)
    const currentIndex = this.currentIndices.get(symbol) || 0

    if (!symbolData || currentIndex >= symbolData.length) {
      return 0
    }

    const candle = symbolData[currentIndex]
    return candle ? candle.close : 0
  }

  /**
   * Get connection statistics
   */
  getStats(): ConnectionStats {
    return {
      connected: this.connected,
      uptime: this.startTime ? Date.now() - this.startTime : 0,
      reconnectAttempts: this.reconnectAttempts,
      lastError: this.lastError,
      lastMessageTime: this.lastMessageTime,
      messagesReceived: this.messagesReceived,
      subscribedSymbols: Array.from(this.subscribedSymbols),
    }
  }

  /**
   * Check if the data feed is healthy
   */
  isHealthy(): boolean {
    return this.connected
  }

  /**
   * Set replay speed
   */
  setSpeed(speed: number): void {
    this.speed = speed
    this.debug(`Replay speed changed to ${speed}x`)
  }

  /**
   * Get current backtest time
   */
  getCurrentTime(): Date {
    return new Date(this.currentTime)
  }

  /**
   * Seek to specific time in backtest
   */
  async seekToTime(time: EpochDate): Promise<void> {
    if (time < this.startDate || time > this.endDate) {
      throw new Error('Seek time outside of backtest range')
    }

    this.currentTime = time

    // Update all symbol indices to match the seek time
    this.subscribedSymbols.forEach(symbol => {
      const symbolData = this.data.get(symbol)
      if (symbolData) {
        const index = symbolData.findIndex(candle => candle.timestamp >= time)
        this.currentIndices.set(symbol, Math.max(0, index === -1 ? symbolData.length : index))
      }
    })

    this.debug(`Seeked to time: ${toIsoDate(time)}`)
    await Promise.resolve()
  }

  /**
   * Load historical data from source
   */
  private async loadHistoricalData(): Promise<void> {
    this.debug('Loading historical data from source', this.dataSource)

    // For now, we'll simulate loading data
    // In a real implementation, this would load from SQLite database or files
    // Based on the PRD requirement: "Backtesting: Sqlite cursor with speed control"

    // TODO: Implement actual SQLite data loading
    // This is a placeholder that generates sample data
    const symbols = ['BTC-USD', 'ETH-USD'] // Default symbols for testing

    // Only generate data if there's a meaningful time range and it's not far in the future
    const timeRange = this.endDate - this.startDate
    const now = epochDateNow()
    const isHistoricalData = this.startDate < now

    if (timeRange > 0 && isHistoricalData) {
      for (const symbol of symbols) {
        const candles = this.generateSampleData(symbol)
        this.data.set(symbol, candles)
        this.currentIndices.set(symbol, 0)
      }
    }

    this.debug(`Loaded data for ${symbols.length} symbols`)
    await Promise.resolve()
  }

  /**
   * Generate sample historical data for testing
   * TODO: Replace with actual SQLite data loading
   */
  private generateSampleData(_symbol: string): Candle[] {
    const candles: Candle[] = []
    const startTime = this.startDate
    const endTime = this.endDate
    const interval = 60000 // 1 minute intervals

    // Return empty array if invalid time range
    if (endTime <= startTime) {
      return []
    }

    let price = 50000 // Starting price

    for (let time = startTime as number; time <= endTime; time += interval) {
      // Simple random walk for price simulation
      const change = (Math.random() - 0.5) * 100
      price = Math.max(1000, price + change)

      const high = price + Math.random() * 50
      const low = price - Math.random() * 50
      const open = price + (Math.random() - 0.5) * 20
      const close = price + (Math.random() - 0.5) * 20
      const volume = Math.random() * 1000

      candles.push({
        timestamp: time as EpochDate,
        open,
        high: Math.max(open, close, high),
        low: Math.min(open, close, low),
        close,
        volume,
      })
    }

    return candles
  }

  /**
   * Start the data replay process
   */
  private async startReplay(): Promise<void> {
    if (this.isReplaying) {
      return
    }

    this.isReplaying = true
    this.debug('Starting data replay')

    await this.replayNextCandles()
  }

  /**
   * Replay next candles for all subscribed symbols
   */
  private async replayNextCandles(): Promise<void> {
    if (!this.isReplaying || this.subscribedSymbols.size === 0) {
      return
    }

    let hasMoreData = false

    // Process next candle for each subscribed symbol
    for (const symbol of this.subscribedSymbols) {
      const symbolData = this.data.get(symbol)
      const currentIndex = this.currentIndices.get(symbol) || 0

      if (symbolData && currentIndex < symbolData.length) {
        const candle = symbolData[currentIndex]
        if (!candle) continue

        // Check if this candle is at or before current time
        if (candle.timestamp <= this.currentTime) {
          // Emit enhanced candle event
          this.emitEnhancedCandle(candle, symbol, '1m')

          // Emit enhanced tick event
          this.emitEnhancedTick({
            symbol,
            price: candle.close,
            timestamp: candle.timestamp,
            volume: candle.volume,
          })

          // Update index
          this.currentIndices.set(symbol, currentIndex + 1)
          this.messagesReceived++
          this.lastMessageTime = epochDateNow()
        }

        // Check if there's more data
        if (currentIndex + 1 < symbolData.length) {
          hasMoreData = true
        }
      }
    }

    // Advance time
    this.currentTime = toEpochDate(this.currentTime + 60000) // 1 minute step

    // Schedule next replay if there's more data and we haven't reached end time
    if (hasMoreData && this.currentTime <= this.endDate) {
      const delay = Math.max(1, 60000 / this.speed) // Adjust delay based on speed
      this.replayTimer = setTimeout(() => {
        void this.replayNextCandles()
      }, delay)
    } else {
      // Backtest completed
      this.isReplaying = false
      this.debug('Backtest replay completed')
      this.eventBus.emit(EventTypes.SYSTEM_INFO, {
        message: 'Backtest completed',
        timestamp: epochDateNow(),
      })
    }
    await Promise.resolve()
  }

  /**
   * Simulate network delay
   */
  private async simulateNetworkDelay(): Promise<void> {
    if (this.networkDelay > 0) {
      const delay = this.networkDelay + (Math.random() * this.networkDelay * 0.5)
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }

  /**
   * Simulate random network failures
   */
  private simulateFailure(): void {
    if (this.failureRate > 0 && Math.random() < this.failureRate) {
      throw new Error('Simulated network failure')
    }
  }
}
