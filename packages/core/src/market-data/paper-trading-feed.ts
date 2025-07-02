import type { Candle } from '@trdr/shared'
import type {
  HistoricalDataRequest,
  ConnectionStats,
} from '../interfaces/market-data-pipeline'
import { EnhancedMarketDataFeed, type EnhancedDataFeedConfig } from './enhanced-market-data-feed'
import { CoinbaseDataFeed } from './coinbase-data-feed'
import { EventTypes } from '../events/types'

/**
 * Configuration specific to paper trading data feed with enhanced event capabilities
 */
export interface PaperTradingConfig extends EnhancedDataFeedConfig {
  /** Base market data source (coinbase, etc.) */
  readonly baseDataSource: 'coinbase'
  /** Base feed configuration for the underlying real data source */
  readonly baseFeedConfig: EnhancedDataFeedConfig
  /** Simulated slippage in basis points (100 = 1%) */
  readonly slippage?: number
  /** Execution delay in milliseconds */
  readonly executionDelay?: number
  /** Maximum price impact for large orders (basis points) */
  readonly maxPriceImpact?: number
  /** Simulated liquidity depth multiplier */
  readonly liquidityMultiplier?: number
  /** Enable accelerated time mode */
  readonly acceleratedTime?: boolean
  /** Time acceleration factor (2 = 2x speed) */
  readonly timeAcceleration?: number
  /** Enable custom market scenarios */
  readonly enableCustomScenarios?: boolean
}

/**
 * Market scenario injection for testing specific conditions
 */
export interface MarketScenario {
  /** Scenario identifier */
  readonly id: string
  /** Start time for scenario */
  readonly startTime: Date
  /** Duration in milliseconds */
  readonly duration: number
  /** Price volatility multiplier (1.0 = normal, 2.0 = double volatility) */
  readonly volatilityMultiplier: number
  /** Trend direction (-1 = bearish, 0 = sideways, 1 = bullish) */
  readonly trendDirection: number
  /** Liquidity impact (0.1 = 10% liquidity, 1.0 = normal liquidity) */
  readonly liquidityImpact: number
}

/**
 * Order execution simulation parameters
 */
export interface ExecutionSimulation {
  /** Order size */
  readonly size: number
  /** Order side */
  readonly side: 'buy' | 'sell'
  /** Limit price (null for market orders) */
  readonly limitPrice?: number
  /** Timestamp when order was placed */
  readonly timestamp: Date
}

/**
 * Simulated execution result
 */
export interface SimulatedExecution {
  /** Executed price */
  readonly executedPrice: number
  /** Executed size */
  readonly executedSize: number
  /** Slippage applied in basis points */
  readonly slippage: number
  /** Price impact in basis points */
  readonly priceImpact: number
  /** Execution delay in milliseconds */
  readonly executionDelay: number
  /** Execution timestamp */
  readonly executionTime: Date
}

/**
 * Enhanced PaperTradingFeed implementation for simulated trading
 * Combines real market data with realistic execution simulation and enhanced event features
 */
export class PaperTradingFeed extends EnhancedMarketDataFeed {
  private baseFeed: CoinbaseDataFeed
  private slippage: number
  private executionDelay: number
  private maxPriceImpact: number
  private liquidityMultiplier: number
  private acceleratedTime: boolean
  private timeAcceleration: number
  private enableCustomScenarios: boolean
  private activeScenarios: Map<string, MarketScenario> = new Map()
  private currentPrices: Map<string, number> = new Map()
  protected priceHistory: Map<string, Array<{ price: number; time: Date }>> = new Map()

  constructor(config: PaperTradingConfig) {
    super(config)

    this.slippage = config.slippage || 10 // 0.1% default slippage
    this.executionDelay = config.executionDelay || 100 // 100ms default delay
    this.maxPriceImpact = config.maxPriceImpact || 50 // 0.5% max impact
    this.liquidityMultiplier = config.liquidityMultiplier || 1.0
    this.acceleratedTime = config.acceleratedTime || false
    this.timeAcceleration = config.timeAcceleration || 1
    this.enableCustomScenarios = config.enableCustomScenarios || false

    // Create base feed for real market data
    this.baseFeed = new CoinbaseDataFeed(config.baseFeedConfig)
  }

  /**
   * Start the paper trading data feed
   */
  async start(): Promise<void> {
    this.debug('Starting paper trading data feed')

    try {
      await this.baseFeed.start()

      this.connected = true
      this.startTime = new Date()

      // Setup event subscriptions after base feed is started
      this.setupBaseFeedSubscriptions()

      this.emitConnected()
      this.emitConnectionStatus('connected')
      this.debug('Paper trading data feed started')
    } catch (error) {
      this.debug('Failed to start paper trading data feed', error)
      this.emitError(error as Error)
      this.emitConnectionStatus('error')
      throw error
    }
  }

  /**
   * Stop the paper trading data feed
   */
  async stop(): Promise<void> {
    this.debug('Stopping paper trading data feed')

    await this.baseFeed.stop()
    this.connected = false
    this.emitDisconnected('Manual stop')
    this.emitConnectionStatus('disconnected')
  }

  /**
   * Subscribe to symbols for paper trading
   */
  async subscribe(symbols: string[]): Promise<void> {
    this.debug('Subscribing to symbols for paper trading', symbols)

    // Subscribe to base feed
    await this.baseFeed.subscribe(symbols)

    // Track symbols in our set
    symbols.forEach(symbol => {
      this.subscribedSymbols.add(symbol)
      this.currentPrices.set(symbol, 0)
      this.priceHistory.set(symbol, [])
    })
  }

  /**
   * Unsubscribe from symbols
   */
  async unsubscribe(symbols: string[]): Promise<void> {
    this.debug('Unsubscribing from symbols', symbols)

    await this.baseFeed.unsubscribe(symbols)

    symbols.forEach(symbol => {
      this.subscribedSymbols.delete(symbol)
      this.currentPrices.delete(symbol)
      this.priceHistory.delete(symbol)
    })
  }

  /**
   * Get historical data (passthrough to base feed)
   */
  async getHistorical(request: HistoricalDataRequest): Promise<Candle[]> {
    return this.baseFeed.getHistorical(request)
  }

  /**
   * Get current price with potential market scenario adjustments
   */
  async getCurrentPrice(symbol: string): Promise<number> {
    const basePrice = await this.baseFeed.getCurrentPrice(symbol)
    return this.applyMarketScenarios(symbol, basePrice)
  }

  /**
   * Get connection statistics
   */
  getStats(): ConnectionStats {
    const baseStats = this.baseFeed.getStats()
    return {
      ...baseStats,
      connected: this.connected,
      subscribedSymbols: Array.from(this.subscribedSymbols),
    }
  }

  /**
   * Check if the data feed is healthy
   */
  isHealthy(): boolean {
    return this.connected && this.baseFeed.isHealthy()
  }

  /**
   * Simulate order execution with realistic slippage and delays
   */
  async simulateExecution(symbol: string, execution: ExecutionSimulation): Promise<SimulatedExecution> {
    const currentPrice = await this.getCurrentPrice(symbol)

    // Calculate execution delay
    const delay = this.calculateExecutionDelay(execution.size)

    // Wait for execution delay
    await new Promise(resolve => setTimeout(resolve, delay))

    // Calculate slippage
    const slippage = this.calculateSlippage(symbol, execution.size, execution.side)

    // Calculate price impact
    const priceImpact = this.calculatePriceImpact(symbol, execution.size)

    // Calculate executed price
    const executedPrice = this.calculateExecutedPrice(
      currentPrice,
      execution.side,
      execution.limitPrice,
      slippage,
      priceImpact,
    )

    // For simplicity, assume full execution (in reality, partial fills might occur)
    const executedSize = execution.size

    const result: SimulatedExecution = {
      executedPrice,
      executedSize,
      slippage,
      priceImpact,
      executionDelay: delay,
      executionTime: this.getVirtualTime(),
    }

    // Emit execution event
    this.emitExecution(symbol, execution, result)

    return result
  }

  /**
   * Add a custom market scenario
   */
  addMarketScenario(scenario: MarketScenario): void {
    if (!this.enableCustomScenarios) {
      throw new Error('Custom scenarios are not enabled')
    }

    this.activeScenarios.set(scenario.id, scenario)
    this.debug('Added market scenario', scenario)
  }

  /**
   * Remove a market scenario
   */
  removeMarketScenario(scenarioId: string): void {
    this.activeScenarios.delete(scenarioId)
    this.debug('Removed market scenario', scenarioId)
  }

  /**
   * Get current virtual time (for accelerated time mode)
   */
  getVirtualTime(): Date {
    if (this.acceleratedTime) {
      const realTimeElapsed = Date.now() - (this.startTime?.getTime() || Date.now())
      const virtualTimeElapsed = realTimeElapsed * this.timeAcceleration
      return new Date((this.startTime?.getTime() || Date.now()) + virtualTimeElapsed)
    }
    return new Date()
  }

  /**
   * Set time acceleration factor
   */
  setTimeAcceleration(factor: number): void {
    this.timeAcceleration = Math.max(0.1, Math.min(100, factor))
    this.debug(`Time acceleration set to ${this.timeAcceleration}x`)
  }

  /**
   * Setup event subscriptions from base feed
   */
  private setupBaseFeedSubscriptions(): void {
    // Subscribe to candle events from base feed
    this.eventBus.subscribe(EventTypes.MARKET_CANDLE, (data: any) => {
      this.handleBaseFeedCandle(data)
    })

    // Subscribe to tick events from base feed
    this.eventBus.subscribe(EventTypes.MARKET_TICK, (data: any) => {
      this.handleBaseFeedTick(data)
    })
  }

  /**
   * Handle candle events from base feed
   */
  private handleBaseFeedCandle(data: any): void {
    // Apply market scenarios and re-emit
    const adjustedCandle = this.applyScenarioToCandle(data)

    // Emit enhanced candle event
    this.emitEnhancedCandle(adjustedCandle, data.symbol, data.interval || '1m')
  }

  /**
   * Handle tick events from base feed
   */
  private handleBaseFeedTick(data: any): void {
    const symbol = data.symbol
    const adjustedPrice = this.applyMarketScenarios(symbol, data.price)

    // Update price tracking
    this.currentPrices.set(symbol, adjustedPrice)
    this.updateVirtualPriceHistory(symbol, adjustedPrice)

    // Emit enhanced tick event
    this.emitEnhancedTick({
      symbol,
      price: adjustedPrice,
      timestamp: data.timestamp,
      volume: data.volume,
    })
  }

  /**
   * Apply active market scenarios to a price
   */
  private applyMarketScenarios(_symbol: string, basePrice: number): number {
    if (!this.enableCustomScenarios || this.activeScenarios.size === 0) {
      return basePrice
    }

    let adjustedPrice = basePrice
    const currentTime = this.getVirtualTime()

    for (const scenario of this.activeScenarios.values()) {
      const scenarioEndTime = new Date(scenario.startTime.getTime() + scenario.duration)

      if (currentTime >= scenario.startTime && currentTime <= scenarioEndTime) {
        // Apply scenario effects
        const progress = (currentTime.getTime() - scenario.startTime.getTime()) / scenario.duration

        // Apply trend
        const trendAdjustment = scenario.trendDirection * progress * 0.01 // 1% max trend per scenario
        adjustedPrice *= (1 + trendAdjustment)

        // Apply volatility (this is simplified - in reality would need more complex modeling)
        const volatilityFactor = 1 + (scenario.volatilityMultiplier - 1) * 0.5
        adjustedPrice *= volatilityFactor
      }
    }

    return adjustedPrice
  }

  /**
   * Apply scenario effects to candle data
   */
  private applyScenarioToCandle(candle: any): any {
    if (!this.enableCustomScenarios) {
      return candle
    }

    return {
      ...candle,
      open: this.applyMarketScenarios(candle.symbol, candle.open),
      high: this.applyMarketScenarios(candle.symbol, candle.high),
      low: this.applyMarketScenarios(candle.symbol, candle.low),
      close: this.applyMarketScenarios(candle.symbol, candle.close),
    }
  }

  /**
   * Calculate execution delay based on order size
   */
  private calculateExecutionDelay(size: number): number {
    // Larger orders take longer to execute
    const baseDelay = this.executionDelay
    const sizeMultiplier = Math.log10(Math.max(1, size / 1000)) // Log scale for size impact
    return Math.max(baseDelay, baseDelay * (1 + sizeMultiplier))
  }

  /**
   * Calculate slippage based on order parameters
   */
  private calculateSlippage(_symbol: string, size: number, _side: 'buy' | 'sell'): number {
    let slippage = this.slippage

    // Increase slippage for larger orders
    const sizeMultiplier = Math.sqrt(size / 1000) // Square root scaling
    slippage *= sizeMultiplier

    // Apply liquidity constraints from active scenarios
    for (const scenario of this.activeScenarios.values()) {
      const currentTime = this.getVirtualTime()
      const scenarioEndTime = new Date(scenario.startTime.getTime() + scenario.duration)

      if (currentTime >= scenario.startTime && currentTime <= scenarioEndTime) {
        slippage *= (2 - scenario.liquidityImpact) // Lower liquidity = higher slippage
      }
    }

    return Math.min(slippage, 500) // Cap at 5%
  }

  /**
   * Calculate price impact for large orders
   */
  private calculatePriceImpact(_symbol: string, size: number): number {
    const baseImpact = (size / 10000) * this.maxPriceImpact // Linear scaling for simplicity
    return Math.min(baseImpact / this.liquidityMultiplier, this.maxPriceImpact)
  }

  /**
   * Calculate final executed price
   */
  private calculateExecutedPrice(
    currentPrice: number,
    side: 'buy' | 'sell',
    limitPrice: number | undefined,
    slippage: number,
    priceImpact: number,
  ): number {
    const totalImpact = slippage + priceImpact
    const impactMultiplier = totalImpact / 10000 // Convert basis points to decimal

    let executedPrice: number
    if (side === 'buy') {
      executedPrice = currentPrice * (1 + impactMultiplier)
    } else {
      executedPrice = currentPrice * (1 - impactMultiplier)
    }

    // Apply limit price constraints
    if (limitPrice !== undefined) {
      if (side === 'buy' && executedPrice > limitPrice) {
        executedPrice = limitPrice
      } else if (side === 'sell' && executedPrice < limitPrice) {
        executedPrice = limitPrice
      }
    }

    return executedPrice
  }

  /**
   * Update price history for a symbol using virtual time
   */
  private updateVirtualPriceHistory(symbol: string, price: number): void {
    if (!this.priceHistory.has(symbol)) {
      this.priceHistory.set(symbol, [])
    }

    const history = this.priceHistory.get(symbol)!
    history.push({ price, time: this.getVirtualTime() })

    // Keep only last 1000 price points
    if (history.length > 1000) {
      history.shift()
    }
  }

  /**
   * Emit execution event
   */
  private emitExecution(symbol: string, order: ExecutionSimulation, result: SimulatedExecution): void {
    this.eventBus.emit(EventTypes.SYSTEM_INFO, {
      message: 'Paper trade executed',
      timestamp: result.executionTime,
      context: 'PaperTradingFeed',
      details: {
        symbol,
        side: order.side,
        size: result.executedSize,
        price: result.executedPrice,
        slippage: result.slippage,
        priceImpact: result.priceImpact,
        executionDelay: result.executionDelay,
      },
    })
  }
}
