import { 
  CoinbaseAdvTradeClient,
  CoinbaseAdvTradeCredentials,
  ProductsService
} from '@coinbase-sample/advanced-trade-sdk-ts/dist/index.js'
import type { DataProvider, DataProviderConfig, HistoricalParams, RealtimeParams } from '../../interfaces'
import type { OhlcvDto } from '../../models'
import { isValidOhlcv } from '../../models'
import logger from '../../utils/logger'
import { RateLimiter } from './rate-limiter'

/**
 * Coinbase data provider implementation
 * Uses Coinbase Advanced Trade SDK for REST API access
 * WebSocket support to be implemented separately as SDK doesn't provide it
 */
export class CoinbaseProvider implements DataProvider {
  readonly name = 'coinbase'
  private client?: CoinbaseAdvTradeClient
  private productsService?: ProductsService
  private connected = false
  private readonly apiKey?: string
  private readonly apiSecret?: string
  private readonly rateLimiter: RateLimiter
  
  // Environment variable names
  private static readonly API_KEY_ENV = 'COINBASE_API_KEY'
  private static readonly API_SECRET_ENV = 'COINBASE_API_SECRET'
  
  // Map from standard timeframe to Coinbase granularity (in seconds)
  private static readonly TIMEFRAME_MAP: Record<string, string> = {
    '1m': 'ONE_MINUTE',
    '5m': 'FIVE_MINUTE',
    '15m': 'FIFTEEN_MINUTE',
    '30m': 'THIRTY_MINUTE',
    '1h': 'ONE_HOUR',
    '2h': 'TWO_HOUR',
    '6h': 'SIX_HOUR',
    '1d': 'ONE_DAY'
  }

  constructor(config: DataProviderConfig = {}) {
    this.apiKey = (config.apiKey as string) || process.env[CoinbaseProvider.API_KEY_ENV]
    this.apiSecret = (config.apiSecret as string) || process.env[CoinbaseProvider.API_SECRET_ENV]
    
    // Initialize rate limiter with config
    this.rateLimiter = new RateLimiter({
      maxRequestsPerSecond: (config.rateLimitPerSecond as number) || 10,
      maxRetries: (config.maxRetries as number) || 5,
      initialRetryDelayMs: (config.retryDelayMs as number) || 1000,
      maxRetryDelayMs: (config.maxRetryDelayMs as number) || 60000,
      backoffMultiplier: (config.backoffMultiplier as number) || 2
    })
    
    logger.info('CoinbaseProvider initialized')
  }

  async connect(): Promise<void> {
    if (this.connected) {
      logger.warn('CoinbaseProvider already connected')
      return
    }

    this.validateEnvVars()
    
    try {
      // Initialize the SDK client
      const credentials = new CoinbaseAdvTradeCredentials(
        this.apiKey,
        this.apiSecret
      )
      this.client = new CoinbaseAdvTradeClient(credentials)
      this.productsService = new ProductsService(this.client)
      
      // Test connection by fetching a product with rate limiting
      await this.rateLimiter.execute(
        () => this.productsService!.getProduct({ productId: 'BTC-USD' }),
        'connect'
      )
      
      this.connected = true
      logger.info('CoinbaseProvider connected successfully')
    } catch (error) {
      logger.error('Failed to connect to Coinbase', { error })
      throw new Error(`Failed to connect to Coinbase: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  async disconnect(): Promise<void> {
    if (!this.connected) {
      return
    }

    try {
      this.client = undefined
      this.productsService = undefined
      this.connected = false
      logger.info('CoinbaseProvider disconnected')
    } catch (error) {
      logger.error('Error during disconnect', { error })
      throw error
    }
  }

  async *getHistoricalData(params: HistoricalParams): AsyncIterableIterator<OhlcvDto> {
    if (!this.connected || !this.productsService) {
      throw new Error('Provider not connected')
    }

    const granularity = this.mapTimeframe(params.timeframe)
    
    for (const symbol of params.symbols) {
      logger.info('Fetching historical data', {
        symbol,
        start: new Date(params.start).toISOString(),
        end: new Date(params.end).toISOString(),
        timeframe: params.timeframe
      })

      try {
        // Convert timestamps to ISO strings for the API
        const startStr = new Date(params.start).toISOString()
        const endStr = new Date(params.end).toISOString()
        
        // Fetch candles from Coinbase with rate limiting
        const response = await this.rateLimiter.execute(
          () => this.productsService!.getProductCandles({
            productId: symbol,
            start: startStr,
            end: endStr,
            granularity,
            limit: 300 // Max allowed by Coinbase
          }),
          `getHistoricalData:${symbol}`
        )
        
        if ('candles' in response && response.candles) {
          // Sort candles by timestamp (oldest first)
          const sortedCandles = [...response.candles].sort((a, b) => {
            const timeA = parseInt(a.start || '0')
            const timeB = parseInt(b.start || '0')
            return timeA - timeB
          })
          
          for (const candle of sortedCandles) {
            const ohlcv = this.convertCandleToOhlcv(candle, symbol)
            if (ohlcv && isValidOhlcv(ohlcv)) {
              yield ohlcv
            }
          }
          
          logger.info('Historical data fetch completed', {
            symbol,
            count: response.candles.length
          })
        }
      } catch (error) {
        logger.error('Failed to fetch historical data', { symbol, error })
        throw error
      }
    }
  }

  async *subscribeRealtime(_params: RealtimeParams): AsyncIterableIterator<OhlcvDto> {
    // Note: Coinbase Advanced Trade currently doesn't provide candle/OHLCV data via WebSocket
    // Their WebSocket API only provides ticker and trade (match) data
    // This would need to be aggregated into candles on the client side
    // For now, we throw an error as proper candle WebSocket support is not available
    throw new Error('Coinbase WebSocket API does not provide candle/OHLCV data. Use ticker aggregation or polling REST API instead.')
  }

  getRequiredEnvVars(): string[] {
    return [CoinbaseProvider.API_KEY_ENV, CoinbaseProvider.API_SECRET_ENV]
  }

  validateEnvVars(): void {
    const missing: string[] = []
    
    if (!this.apiKey) {
      missing.push(CoinbaseProvider.API_KEY_ENV)
    }
    
    if (!this.apiSecret) {
      missing.push(CoinbaseProvider.API_SECRET_ENV)
    }

    if (missing.length > 0) {
      throw new Error(`Missing required environment variables: ${missing.join(', ')}`)
    }
  }

  isConnected(): boolean {
    return this.connected
  }

  getSupportedTimeframes(): string[] {
    return Object.keys(CoinbaseProvider.TIMEFRAME_MAP)
  }

  /**
   * Maps standard timeframe to Coinbase granularity
   */
  private mapTimeframe(timeframe: string): string {
    const granularity = CoinbaseProvider.TIMEFRAME_MAP[timeframe]
    if (!granularity) {
      throw new Error(`Unsupported timeframe: ${timeframe}. Supported: ${this.getSupportedTimeframes().join(', ')}`)
    }
    return granularity
  }

  /**
   * Converts Coinbase candle data to OhlcvDto
   */
  private convertCandleToOhlcv(candle: any, symbol: string): OhlcvDto | null {
    try {
      // Parse string values to numbers
      const timestamp = parseInt(candle.start || '0') * 1000 // Convert from seconds to milliseconds
      const open = parseFloat(candle.open || '0')
      const high = parseFloat(candle.high || '0')
      const low = parseFloat(candle.low || '0')
      const close = parseFloat(candle.close || '0')
      const volume = parseFloat(candle.volume || '0')
      
      return {
        exchange: 'coinbase',
        symbol,
        timestamp,
        open,
        high,
        low,
        close,
        volume
      }
    } catch (error) {
      logger.error('Failed to convert candle to OHLCV', { candle, error })
      return null
    }
  }
}