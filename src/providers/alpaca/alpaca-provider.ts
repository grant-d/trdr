import Alpaca from '@alpacahq/alpaca-trade-api'
import type { DataProvider, DataProviderConfig, HistoricalParams, RealtimeParams } from '../../interfaces'
import type { OhlcvDto } from '../../models'
import { isValidOhlcv } from '../../models'
import logger from '../../utils/logger'
import { RateLimiter } from '../coinbase'

/**
 * Alpaca data provider implementation
 * Supports both REST API for historical data and WebSocket for real-time data
 */
export class AlpacaProvider implements DataProvider {
  readonly name = 'alpaca'
  private client?: Alpaca
  private websocket?: any
  private cryptoWebsocket?: any
  private connected = false
  private readonly apiKey?: string
  private readonly apiSecret?: string
  private readonly isPaper: boolean
  private readonly rateLimiter: RateLimiter
  private readonly subscriptions = new Map<string, Set<string>>()

  // Environment variable names
  private static readonly PAPER_ENV = 'ALPACA_PAPER'
  private static readonly PAPER_API_KEY_ENV = 'ALPACA_PAPER_API_KEY'
  private static readonly PAPER_API_SECRET_ENV = 'ALPACA_PAPER_API_SECRET'
  private static readonly LIVE_API_KEY_ENV = 'ALPACA_LIVE_API_KEY'
  private static readonly LIVE_API_SECRET_ENV = 'ALPACA_LIVE_API_SECRET'

  constructor(config: DataProviderConfig = {}) {
    // Determine if using paper trading
    this.isPaper =
      config.paper !== undefined
        ? (config.paper as boolean)
        : process.env[AlpacaProvider.PAPER_ENV]?.toLowerCase() !== 'false'

    // Select appropriate API keys based on paper/live mode
    if (this.isPaper) {
      this.apiKey =
        (config.apiKey as string) ||
        process.env[AlpacaProvider.PAPER_API_KEY_ENV]
      this.apiSecret =
        (config.apiSecret as string) ||
        process.env[AlpacaProvider.PAPER_API_SECRET_ENV]
    } else {
      this.apiKey =
        (config.apiKey as string) ||
        process.env[AlpacaProvider.LIVE_API_KEY_ENV]
      this.apiSecret =
        (config.apiSecret as string) ||
        process.env[AlpacaProvider.LIVE_API_SECRET_ENV]
    }

    // Initialize rate limiter with config
    this.rateLimiter = new RateLimiter({
      maxRequestsPerSecond: (config.rateLimitPerSecond as number) || 200, // Alpaca has higher rate limits
      maxRetries: (config.maxRetries as number) || 5,
      initialRetryDelayMs: (config.retryDelayMs as number) || 1000,
      maxRetryDelayMs: (config.maxRetryDelayMs as number) || 60000,
      backoffMultiplier: (config.backoffMultiplier as number) || 2
    })

    logger.info('AlpacaProvider initialized', { paper: this.isPaper })
  }

  async connect(): Promise<void> {
    if (this.connected) {
      logger.warn('AlpacaProvider already connected')
      return
    }

    this.validateEnvVars()

    try {
      // Initialize the Alpaca client
      if (!this.apiKey || !this.apiSecret) {
        throw new Error('API credentials not available')
      }

      this.client = new Alpaca({
        keyId: this.apiKey,
        secretKey: this.apiSecret,
        paper: this.isPaper,
        feed: 'iex' // Use IEX feed which works with most subscription levels
      })

      // Test connection by fetching account info
      await this.rateLimiter.execute(async () => {
        if (!this.client) {
          throw new Error('Client not initialized')
        }
        return await this.client.getAccount()
      }, 'connect')

      this.connected = true
      logger.info('AlpacaProvider connected successfully')
    } catch (error) {
      logger.error('Failed to connect to Alpaca', { error })
      throw new Error(
        `Failed to connect to Alpaca: ${error instanceof Error ? error.message : 'Unknown error'}`
      )
    }
  }

  async disconnect(): Promise<void> {
    if (!this.connected) {
      return
    }

    try {
      // Disconnect WebSockets if active
      if (this.websocket) {
        this.websocket.disconnect()
        this.websocket = undefined
      }

      if (this.cryptoWebsocket) {
        this.cryptoWebsocket.disconnect()
        this.cryptoWebsocket = undefined
      }

      this.client = undefined
      this.connected = false
      this.subscriptions.clear()
      logger.info('AlpacaProvider disconnected')
    } catch (error) {
      logger.error('Error during disconnect', { error })
      throw error
    }
  }

  async* getHistoricalData(
    params: HistoricalParams
  ): AsyncIterableIterator<OhlcvDto> {
    if (!this.connected || !this.client) {
      throw new Error('Provider not connected')
    }

    for (const symbol of params.symbols) {
      logger.info('Fetching historical data', {
        symbol,
        start: new Date(params.start).toISOString(),
        end: new Date(params.end).toISOString(),
        timeframe: params.timeframe
      })

      try {
        // Create timeframe object using Alpaca's API
        const alpacaTimeframe = this.mapTimeframe(params.timeframe)

        // Determine if this is a crypto symbol
        const isCrypto = this.isCryptoSymbol(symbol)

        if (isCrypto) {
          // Normalize crypto symbol for Alpaca
          const normalizedSymbol = this.normalizeCryptoSymbol(symbol)

          // Use crypto-specific API
          try {
            // Note: TypeScript types say this returns Promise<Map>, but docs say AsyncGenerator
            // We'll handle it as a Promise<Map> based on the types
            const barsResult = await this.client.getCryptoBars(
              [normalizedSymbol],
              {
                start: new Date(params.start).toISOString(),
                end: new Date(params.end).toISOString(),
                timeframe: alpacaTimeframe
              }
            )

            // Process crypto bars (use original symbol for output)
            if (barsResult instanceof Map) {
              const symbolBars = barsResult.get(normalizedSymbol) || []
              for (const bar of symbolBars) {
                const ohlcv = this.convertBarToOhlcv(bar, symbol)
                if (ohlcv && isValidOhlcv(ohlcv)) {
                  yield ohlcv
                }
              }
            }
          } catch (error) {
            logger.error('Failed to fetch crypto bars', {
              symbol,
              normalizedSymbol,
              error
            })
            throw error
          }
        } else {
          // Use stock API with generator for streaming
          const barsGenerator = this.client.getBarsV2(symbol, {
            start: new Date(params.start).toISOString(),
            end: new Date(params.end).toISOString(),
            timeframe: alpacaTimeframe,
            feed: 'iex' // Explicitly use IEX feed for stocks
          })

          // Process bars as they arrive
          for await (const bar of barsGenerator) {
            const ohlcv = this.convertBarToOhlcv(bar, symbol)
            if (ohlcv && isValidOhlcv(ohlcv)) {
              yield ohlcv
            }
          }
        }

        logger.info('Historical data fetch completed', { symbol, isCrypto })
      } catch (error) {
        logger.error('Failed to fetch historical data', { symbol, error })
        throw error
      }
    }
  }

  async* subscribeRealtime(
    params: RealtimeParams
  ): AsyncIterableIterator<OhlcvDto> {
    if (!this.connected || !this.client) {
      throw new Error('Provider not connected')
    }

    // Separate crypto and stock symbols
    const cryptoSymbols = params.symbols.filter((s) => this.isCryptoSymbol(s))
    const stockSymbols = params.symbols.filter((s) => !this.isCryptoSymbol(s))

    // Initialize appropriate WebSocket connections
    if (stockSymbols.length > 0 && !this.websocket) {
      await this.initializeWebSocket()
    }
    if (cryptoSymbols.length > 0 && !this.cryptoWebsocket) {
      await this.initializeCryptoWebSocket()
    }

    // Subscribe to symbols
    for (const symbol of stockSymbols) {
      await this.subscribeToSymbol(symbol, params.timeframe)
    }
    for (const symbol of cryptoSymbols) {
      await this.subscribeToCryptoSymbol(symbol, params.timeframe)
    }

    // Create message queue for yielding data
    const messageQueue: OhlcvDto[] = []

    // Set up stock bar handler
    if (this.websocket) {
      this.websocket.onStockBar((bar: any) => {
        const ohlcv = this.convertBarToOhlcv(bar, bar.S || bar.Symbol)
        if (ohlcv && isValidOhlcv(ohlcv)) {
          messageQueue.push(ohlcv)
        }
      })
    }

    // Set up crypto bar handler
    if (this.cryptoWebsocket) {
      this.cryptoWebsocket.onCryptoBar((bar: any) => {
        const ohlcv = this.convertBarToOhlcv(bar, bar.S || bar.Symbol)
        if (ohlcv && isValidOhlcv(ohlcv)) {
          messageQueue.push(ohlcv)
        }
      })
    }

    // Yield data as it arrives
    while (this.connected && (this.websocket || this.cryptoWebsocket)) {
      while (messageQueue.length > 0) {
        const msg = messageQueue.shift()
        if (msg) {
          yield msg
        }
      }

      // Wait a bit before checking again with timeout to allow breaking from loop
      await new Promise((resolve) => setTimeout(resolve, 100))

      // Additional check to break loop when disconnected
      if (!this.connected) {
        break
      }
    }
  }

  getRequiredEnvVars(): string[] {
    if (this.isPaper) {
      return [
        AlpacaProvider.PAPER_API_KEY_ENV,
        AlpacaProvider.PAPER_API_SECRET_ENV
      ]
    } else {
      return [
        AlpacaProvider.LIVE_API_KEY_ENV,
        AlpacaProvider.LIVE_API_SECRET_ENV
      ]
    }
  }

  validateEnvVars(): void {
    const missing: string[] = []

    if (!this.apiKey) {
      missing.push(
        this.isPaper
          ? AlpacaProvider.PAPER_API_KEY_ENV
          : AlpacaProvider.LIVE_API_KEY_ENV
      )
    }

    if (!this.apiSecret) {
      missing.push(
        this.isPaper
          ? AlpacaProvider.PAPER_API_SECRET_ENV
          : AlpacaProvider.LIVE_API_SECRET_ENV
      )
    }

    if (missing.length > 0) {
      throw new Error(
        `Missing required environment variables: ${missing.join(', ')}`
      )
    }
  }

  isConnected(): boolean {
    return this.connected
  }

  getSupportedTimeframes(): string[] {
    // Return common timeframes as examples, but we support arbitrary timeframes
    return [
      '1m',
      '3m',
      '5m',
      '15m',
      '17m',
      '30m',
      '90s',
      '1h',
      '2h',
      '4h',
      '1d',
      '1w',
      '1M'
    ]
  }

  /**
   * Determines if a symbol is a cryptocurrency
   * Crypto symbols must contain a delimiter: '/', '-', '.', or '_'
   * Examples: BTC/USD, BTC-USD, BTC.USD, BTC_USD
   */
  private isCryptoSymbol(symbol: string): boolean {
    return /[\/\-\._]/.test(symbol)
  }

  /**
   * Normalizes crypto symbols to Alpaca's expected format
   * Alpaca expects crypto symbols with forward slash: BTC/USD
   */
  private normalizeCryptoSymbol(symbol: string): string {
    // Replace any delimiter with forward slash
    return symbol.replace(/[\-\._]/g, '/')
  }

  /**
   * Parses timeframe string to extract amount and unit
   * Supports arbitrary timeframes like '3m', '17m', '90s', etc.
   */
  private parseTimeframe(timeframe: string): {
    amount: number
    unitChar: string
  } {
    // Parse the timeframe string
    const match = /^(\d+)([smhdwM])$/.exec(timeframe)
    if (!match) {
      throw new Error(
        `Invalid timeframe format: ${timeframe}. Expected format: <number><unit> (e.g., '5m', '1h', '90s')`
      )
    }

    const amount = parseInt(match[1]!, 10)
    const unitChar = match[2]!

    // Handle seconds conversion to minutes
    if (unitChar === 's') {
      if (amount % 60 !== 0) {
        throw new Error(
          `Alpaca only supports timeframes in whole minutes. ${amount}s cannot be converted to minutes.`
        )
      }
      return { amount: amount / 60, unitChar: 'm' }
    }

    return { amount, unitChar }
  }

  /**
   * Maps standard timeframe to Alpaca timeframe object
   */
  private mapTimeframe(timeframe: string): any {
    if (!this.client) {
      throw new Error('Client not initialized')
    }

    const { amount, unitChar } = this.parseTimeframe(timeframe)

    // Map unit character to Alpaca TimeFrameUnit
    let unit: any
    switch (unitChar) {
      case 'm':
        unit = this.client.timeframeUnit.MIN
        break
      case 'h':
        unit = this.client.timeframeUnit.HOUR
        break
      case 'd':
        unit = this.client.timeframeUnit.DAY
        break
      case 'w':
        unit = this.client.timeframeUnit.WEEK
        break
      case 'M':
        unit = this.client.timeframeUnit.MONTH
        break
      default:
        throw new Error(`Unsupported timeframe unit: ${unitChar}`)
    }

    // Use the newTimeframe method to create the timeframe object
    return this.client.newTimeframe(amount, unit)
  }

  /**
   * Initialize WebSocket connection
   */
  private async initializeWebSocket(): Promise<void> {
    if (!this.client) {
      throw new Error('Client not initialized')
    }

    this.websocket = this.client.data_stream_v2

    // Set up event handlers
    this.websocket.onConnect(() => {
      logger.info('AlpacaProvider WebSocket connected')

      // Resubscribe to previous subscriptions if reconnecting
      for (const [symbol] of this.subscriptions) {
        this.websocket.subscribeForBars([symbol])
      }
    })

    this.websocket.onDisconnect(() => {
      logger.warn('AlpacaProvider WebSocket disconnected')
    })

    this.websocket.onStateChange((state: string) => {
      logger.info('AlpacaProvider WebSocket state changed', { state })
    })

    this.websocket.onError((error: any) => {
      logger.error('AlpacaProvider WebSocket error', { error })
    })

    // Connect to WebSocket
    await this.websocket.connect()
  }

  /**
   * Initialize crypto WebSocket connection
   */
  private async initializeCryptoWebSocket(): Promise<void> {
    if (!this.client) {
      throw new Error('Client not initialized')
    }

    this.cryptoWebsocket = this.client.crypto_stream_v1beta3

    // Set up event handlers
    this.cryptoWebsocket.onConnect(() => {
      logger.info('AlpacaProvider crypto WebSocket connected')

      // Resubscribe to previous subscriptions if reconnecting
      for (const [symbol] of this.subscriptions) {
        if (this.isCryptoSymbol(symbol)) {
          const normalizedSymbol = this.normalizeCryptoSymbol(symbol)
          this.cryptoWebsocket.subscribeForBars([normalizedSymbol])
        }
      }
    })

    this.cryptoWebsocket.onDisconnect(() => {
      logger.warn('AlpacaProvider crypto WebSocket disconnected')
    })

    this.cryptoWebsocket.onError((error: any) => {
      logger.error('AlpacaProvider crypto WebSocket error', { error })
    })

    // Connect to crypto WebSocket
    await this.cryptoWebsocket.connect()
  }

  /**
   * Subscribe to a symbol with specific timeframe
   */
  private async subscribeToSymbol(
    symbol: string,
    timeframe: string
  ): Promise<void> {
    if (!this.websocket) {
      throw new Error('WebSocket not initialized')
    }

    // Track subscription
    if (!this.subscriptions.has(symbol)) {
      this.subscriptions.set(symbol, new Set())
    }
    const subs = this.subscriptions.get(symbol)
    if (subs) {
      subs.add(timeframe)
    }

    // Subscribe to bars
    this.websocket.subscribeForBars([symbol])
    logger.info('Subscribed to symbol bars', { symbol, timeframe })
  }

  /**
   * Subscribe to a crypto symbol with specific timeframe
   */
  private async subscribeToCryptoSymbol(
    symbol: string,
    timeframe: string
  ): Promise<void> {
    if (!this.cryptoWebsocket) {
      throw new Error('Crypto WebSocket not initialized')
    }

    // Normalize crypto symbol for Alpaca
    const normalizedSymbol = this.normalizeCryptoSymbol(symbol)

    // Track subscription with original symbol
    if (!this.subscriptions.has(symbol)) {
      this.subscriptions.set(symbol, new Set())
    }
    const subs = this.subscriptions.get(symbol)
    if (subs) {
      subs.add(timeframe)
    }

    // Subscribe to crypto bars with normalized symbol
    this.cryptoWebsocket.subscribeForBars([normalizedSymbol])
    logger.info('Subscribed to crypto symbol bars', {
      symbol,
      normalizedSymbol,
      timeframe
    })
  }

  /**
   * Converts Alpaca bar data to OhlcvDto
   */
  private convertBarToOhlcv(bar: any, _symbol: string): OhlcvDto | null {
    try {
      // Check different bar formats
      if (bar.OpenPrice !== undefined) {
        // Stock v2 format with OpenPrice/HighPrice/etc
        return {
          timestamp: new Date(bar.Timestamp).getTime(),
          open: bar.OpenPrice,
          high: bar.HighPrice,
          low: bar.LowPrice,
          close: bar.ClosePrice,
          volume: bar.Volume
        }
      } else if (bar.Open !== undefined) {
        // Crypto format with Open/High/etc
        return {
          timestamp: new Date(bar.Timestamp).getTime(),
          open: bar.Open,
          high: bar.High,
          low: bar.Low,
          close: bar.Close,
          volume: bar.Volume
        }
      } else {
        // Raw format with lowercase names (o/h/l/c)
        return {
          timestamp: new Date(bar.t).getTime(),
          open: bar.o,
          high: bar.h,
          low: bar.l,
          close: bar.c,
          volume: bar.v
        }
      }
    } catch (error) {
      logger.error('Failed to convert bar to OHLCV', { bar, error })
      return null
    }
  }
}
