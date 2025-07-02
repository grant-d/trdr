import type { Candle, PriceTick } from '@trdr/shared'
import type {
  MarketDataPipeline,
  DataFeedConfig,
  HistoricalDataRequest,
  ConnectionStats,
} from '../interfaces/market-data-pipeline'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

/**
 * Base abstract class for market data feed implementations
 */
export abstract class BaseMarketDataFeed implements MarketDataPipeline {
  protected eventBus: EventBus
  protected config: DataFeedConfig
  protected connected = false
  protected uptime = 0
  protected reconnectAttempts = 0
  protected messagesReceived = 0
  protected lastError?: string
  protected lastMessageTime?: Date
  protected startTime: Date | null = null
  protected subscribedSymbols: Set<string> = new Set()

  constructor(config: DataFeedConfig) {
    this.config = config
    this.eventBus = EventBus.getInstance()
  }

  /**
   * Subscribe to real-time market data for specified symbols
   */
  abstract subscribe(symbols: string[]): Promise<void>

  /**
   * Unsubscribe from market data for specified symbols
   */
  abstract unsubscribe(symbols: string[]): Promise<void>

  /**
   * Get historical market data
   */
  abstract getHistorical(request: HistoricalDataRequest): Promise<Candle[]>

  /**
   * Get current price for a symbol
   */
  abstract getCurrentPrice(symbol: string): Promise<number>

  /**
   * Start the data feed connection
   */
  abstract start(): Promise<void>

  /**
   * Stop the data feed connection
   */
  abstract stop(): Promise<void>

  /**
   * Check if the data feed is connected and healthy
   */
  isHealthy(): boolean {
    return this.connected &&
      this.lastMessageTime !== undefined &&
      Date.now() - this.lastMessageTime.getTime() < 60000 // 1 minute
  }

  /**
   * Get connection statistics
   */
  getStats(): ConnectionStats {
    return {
      connected: this.connected,
      uptime: this.startTime ? Date.now() - this.startTime.getTime() : 0,
      reconnectAttempts: this.reconnectAttempts,
      messagesReceived: this.messagesReceived,
      lastError: this.lastError,
      lastMessageTime: this.lastMessageTime,
      subscribedSymbols: Array.from(this.subscribedSymbols),
    }
  }

  /**
   * Emit a candle event
   */
  protected emitCandle(candle: Candle, symbol: string, interval: string): void {
    this.updateMessageStats()
    this.eventBus.emit(EventTypes.MARKET_CANDLE, {
      timestamp: new Date(),
      symbol,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
      volume: candle.volume,
      interval,
    })
  }

  /**
   * Emit a tick event
   */
  protected emitTick(tick: PriceTick): void {
    this.updateMessageStats()
    this.eventBus.emit(EventTypes.MARKET_TICK, {
      timestamp: new Date(tick.timestamp),
      symbol: tick.symbol,
      price: tick.price,
      volume: tick.volume,
    })
  }

  /**
   * Emit a connection event
   */
  protected emitConnected(): void {
    this.connected = true
    this.startTime = new Date()
    this.eventBus.emit(EventTypes.SYSTEM_INFO, {
      timestamp: new Date(),
      message: `Market data feed connected: ${this.config.feedType}`,
      context: 'MarketDataFeed',
      details: { feedType: this.config.feedType, symbol: this.config.symbol },
    })
  }

  /**
   * Emit a disconnection event
   */
  protected emitDisconnected(reason: string): void {
    this.connected = false
    this.eventBus.emit(EventTypes.SYSTEM_WARNING, {
      timestamp: new Date(),
      message: `Market data feed disconnected: ${reason}`,
      context: 'MarketDataFeed',
      details: { feedType: this.config.feedType, reason },
    })
  }

  /**
   * Emit an error event
   */
  protected emitError(error: Error): void {
    this.lastError = error.message
    this.eventBus.emit(EventTypes.SYSTEM_ERROR, {
      timestamp: new Date(),
      error,
      context: 'MarketDataFeed',
      severity: 'medium',
    })
  }

  /**
   * Emit a reconnecting event
   */
  protected emitReconnecting(attempt: number): void {
    this.reconnectAttempts = attempt
    this.eventBus.emit(EventTypes.SYSTEM_INFO, {
      timestamp: new Date(),
      message: `Market data feed reconnecting: attempt ${attempt}`,
      context: 'MarketDataFeed',
      details: { feedType: this.config.feedType, attempt },
    })
  }

  /**
   * Update message statistics
   */
  private updateMessageStats(): void {
    this.messagesReceived++
    this.lastMessageTime = new Date()
  }

  /**
   * Log debug message if debug mode is enabled
   */
  protected debug(message: string, data?: any): void {
    if (this.config.debug) {
      console.debug(`[${this.config.feedType}] ${message}`, data)
    }
  }
}
