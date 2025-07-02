import type { Candle, EpochDate } from '@trdr/shared'
import type { HistoricalDataRequest } from '../interfaces/market-data-pipeline'
import { EnhancedMarketDataFeed, type EnhancedDataFeedConfig } from './enhanced-market-data-feed'
import { CoinbaseAdvTradeClient, ProductsService, CoinbaseAdvTradeCredentials } from '@coinbase-sample/advanced-trade-sdk-ts/dist/index'

/**
 * Coinbase Advanced Trade data feed implementation using the official SDK
 * Provides enhanced market data via REST API with future WebSocket support
 * Includes advanced event filtering, priority handling, and market statistics
 */
export class CoinbaseDataFeed extends EnhancedMarketDataFeed {
  private readonly client: CoinbaseAdvTradeClient
  private readonly productsService: ProductsService
  private wsConnected = false
  private reconnectTimer: NodeJS.Timeout | null = null

  constructor(config: EnhancedDataFeedConfig) {
    super(config)

    // Initialize Coinbase Advanced Trade client
    const credentials = new CoinbaseAdvTradeCredentials(
      config.apiKey || '',
      config.apiSecret || '',
    )
    this.client = new CoinbaseAdvTradeClient(credentials)

    // Initialize products service
    this.productsService = new ProductsService(this.client)
  }

  /**
   * Start the data feed connection
   */
  async start(): Promise<void> {
    this.debug('Starting Coinbase data feed')

    try {
      // For now, we'll focus on REST API functionality
      // WebSocket implementation can be added later
      this.emitConnected()
      this.emitConnectionStatus('connected')
    } catch (error) {
      this.debug('Failed to start data feed', error)
      this.emitError(error as Error)
      this.emitConnectionStatus('error')
      // Don't throw - we can still function with REST API only
    }
    await Promise.resolve()
  }

  /**
   * Stop the data feed connection
   */
  async stop(): Promise<void> {
    this.debug('Stopping Coinbase data feed')

    this.clearTimers()
    this.wsConnected = false

    this.emitDisconnected('Manual stop')
    // Method is async to satisfy the abstract interface requirement
    await Promise.resolve()
  }

  /**
   * Subscribe to real-time market data for specified symbols
   */
  async subscribe(symbols: string[]): Promise<void> {
    this.debug('Subscribing to symbols', symbols)

    // Add symbols to our set
    symbols.forEach(symbol => this.subscribedSymbols.add(symbol))

    // For now, we'll implement periodic price polling
    // Real WebSocket implementation can be added later
    this.debug('Symbols added to subscription list')
    // Method is async to satisfy the abstract interface requirement
    await Promise.resolve()
  }

  /**
   * Unsubscribe from market data for specified symbols
   */
  async unsubscribe(symbols: string[]): Promise<void> {
    this.debug('Unsubscribing from symbols', symbols)

    // Remove symbols from our set
    symbols.forEach(symbol => this.subscribedSymbols.delete(symbol))
    // Method is async to satisfy the abstract interface requirement
    await Promise.resolve()
  }

  /**
   * Get historical market data
   */
  async getHistorical(request: HistoricalDataRequest): Promise<Candle[]> {
    this.debug('Fetching historical data', request)

    try {
      const granularity = this.intervalToGranularity(request.interval || '1h')

      const response = await this.productsService.getProductCandles({
        productId: request.symbol,
        start: Math.floor(request.start / 1000).toString(),
        end: Math.floor(request.end / 1000).toString(),
        granularity: granularity,
      })

      // Check if response is successful (has candles property)
      if ('candles' in response) {
        return this.transformCandleData(response.candles || [])
      } else {
        // Handle error response
        throw new Error(`Failed to fetch candles: ${JSON.stringify(response)}`)
      }
    } catch (error) {
      this.debug('Failed to fetch historical data', error)
      throw error
    }
  }

  /**
   * Get current price for a symbol
   */
  async getCurrentPrice(symbol: string): Promise<number> {
    this.debug('Fetching current price for', symbol)

    try {
      const response = await this.productsService.getProduct({
        productId: symbol,
      })

      // Check if response is successful (has body property)
      if ('body' in response) {
        return parseFloat(response.body?.price || '0')
      } else {
        // Handle error response
        throw new Error(`Failed to fetch product: ${JSON.stringify(response)}`)
      }
    } catch (error) {
      this.debug('Failed to fetch current price', error)
      throw error
    }
  }

  /**
   * Schedule WebSocket reconnection
   */
  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      return // Already scheduled
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000) // Max 30s

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.emitReconnecting(this.reconnectAttempts + 1)

      void this.start().then(() => {
        this.debug('Reconnected successfully')
      }).catch((error) => {
        this.debug('Reconnection failed', error)
        this.scheduleReconnect()
      })
    }, delay)
  }

  /**
   * Clear all timers
   */
  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }

  /**
   * Convert interval string to Coinbase granularity (in seconds)
   */
  private intervalToGranularity(interval: string): string {
    const intervalMap: Record<string, string> = {
      '1m': 'ONE_MINUTE',
      '5m': 'FIVE_MINUTE',
      '15m': 'FIFTEEN_MINUTE',
      '1h': 'ONE_HOUR',
      '6h': 'SIX_HOUR',
      '1d': 'ONE_DAY',
    }

    return intervalMap[interval] || 'ONE_HOUR' // Default to 1 hour
  }

  /**
   * Transform Coinbase candle data to our format
   */
  private transformCandleData(candles: unknown[]): Candle[] {
    return candles.map(candle => {
      const candleData = candle as Record<string, unknown>
      return {
        timestamp: (parseInt(candleData.start as string) * 1000) as EpochDate, // Convert to milliseconds
        open: parseFloat(candleData.open as string),
        high: parseFloat(candleData.high as string),
        low: parseFloat(candleData.low as string),
        close: parseFloat(candleData.close as string),
        volume: parseFloat(candleData.volume as string),
      }
    })
  }

  /**
   * Check if the data feed is connected and healthy
   */
  isHealthy(): boolean {
    // For REST-only implementation, we're healthy if we can make API calls
    return this.connected || this.wsConnected
  }
}
