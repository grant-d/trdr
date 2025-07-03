import type {
  Candle,
  GridConfig,
  GridLevel,
  GridLevelTrailingState,
  GridState,
  GridTrailingOrderConfig,
  StockSymbol
} from '@trdr/shared'
import { epochDateNow } from '@trdr/shared'
import type { Logger } from '@trdr/types'
import { EventEmitter } from 'events'
import { v4 as uuidv4 } from 'uuid'
import type { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import type { TrailingOrderManager } from '../orders/trailing-order-manager'
import {
  GridStatePersistence,
  type GridManagerSnapshot,
  type GridPersistenceConfig,
  type GridStateRepository,
  type PerformanceHistoryRecord,
  type SerializableGridState,
  type StateRecoveryInfo
} from './grid-state-persistence'
import { SelfTuningGridSpacing, type SelfTuningConfig } from './self-tuning-grid-spacing'
import type { VolatilitySpacingConfig } from './volatility-grid-spacing'

/**
 * Parameters for initializing a grid
 */
export interface GridInitializationParams {
  readonly symbol: StockSymbol
  readonly allocatedCapital: number
  readonly baseAmount: number
  readonly quoteAmount: number
  readonly riskLevel: number
}

/**
 * Result of grid initialization
 */
export interface GridInitializationResult {
  readonly gridId: string
  readonly centerPrice: number
  readonly levels: readonly GridLevel[]
  readonly totalLevels: number
  readonly spacing: number
}

/**
 * Grid events that can be emitted
 */
export interface GridEvents {
  gridCreated: (gridId: string, state: GridState) => void
  gridUpdated: (gridId: string, state: GridState) => void
  levelActivated: (gridId: string, level: GridLevel) => void
  levelFilled: (gridId: string, level: GridLevel, fillPrice: number) => void
  gridCancelled: (gridId: string, reason?: string) => void
}

/**
 * GridManager manages the creation, monitoring, and updating of grid trading systems.
 * 
 * Features:
 * - Dynamic grid level generation based on market conditions
 * - Integration with trailing order system
 * - Self-tuning spacing algorithms
 * - Persistent state management
 * - Event-driven architecture for grid lifecycle
 */
export class GridManager extends EventEmitter {
  private readonly activeGrids = new Map<string, GridState>()
  private readonly eventBus: EventBus
  private readonly trailingOrderManager: TrailingOrderManager
  private readonly spacingCalculator: SelfTuningGridSpacing
  private readonly persistence: GridStatePersistence
  private readonly logger?: Logger
  // Note: priceMonitoringInterval reserved for future price monitoring features
  private readonly lastPriceUpdates = new Map<string, { price: number; timestamp: number }>()
  
  // Persistence metadata
  private totalGridsCreated = 0
  private totalGridsCancelled = 0
  private readonly systemStartTime = epochDateNow()

  constructor(
    eventBus: EventBus,
    trailingOrderManager: TrailingOrderManager,
    volatilitySpacingConfig?: Partial<VolatilitySpacingConfig>,
    selfTuningConfig?: Partial<SelfTuningConfig>,
    persistenceConfig?: Partial<GridPersistenceConfig>,
    repository?: GridStateRepository,
    logger?: Logger
  ) {
    super()
    this.eventBus = eventBus
    this.trailingOrderManager = trailingOrderManager
    this.spacingCalculator = new SelfTuningGridSpacing(volatilitySpacingConfig, selfTuningConfig, logger)
    this.persistence = new GridStatePersistence(persistenceConfig, repository, logger)
    this.logger = logger

    this.setupEventHandlers()
  }

  /**
   * Creates a new grid trading system
   */
  async createGrid(
    config: GridConfig,
    params: GridInitializationParams
  ): Promise<GridInitializationResult> {
    const gridId = uuidv4()
    const now = epochDateNow()

    this.logger?.info('Creating new grid', {
      gridId,
      symbol: params.symbol,
      capital: params.allocatedCapital,
      config
    })

    // For now, use the current price as center price
    // TODO: Integrate with market data feed to get actual current price
    const centerPrice = 50000 // Placeholder

    // Generate initial grid levels
    const levels = this.generateGridLevels(
      centerPrice,
      config.gridSpacing,
      config.gridLevels,
      params.allocatedCapital
    )

    // Create initial grid state
    const gridState: GridState = {
      config,
      symbol: params.symbol,
      levels,
      centerPrice,
      currentSpacing: config.gridSpacing,
      allocatedCapital: params.allocatedCapital,
      availableCapital: params.allocatedCapital,
      currentPosition: 0,
      realizedPnl: 0,
      unrealizedPnl: 0,
      initializedAt: now,
      lastUpdatedAt: now,
      isActive: true
    }

    // Store the grid
    this.activeGrids.set(gridId, gridState)
    this.totalGridsCreated++

    // Emit creation event
    this.eventBus.emit(EventTypes.SYSTEM_INFO, {
      message: `Grid ${gridId} created for ${params.symbol}`,
      context: 'grid_manager',
      timestamp: now
    })
    this.emit('gridCreated', gridId, gridState)

    this.logger?.info('Grid created successfully', {
      gridId,
      totalLevels: levels.length,
      centerPrice,
      spacing: config.gridSpacing
    })

    return {
      gridId,
      centerPrice,
      levels,
      totalLevels: levels.length,
      spacing: config.gridSpacing
    }
  }

  /**
   * Places trailing orders at grid levels near the current price
   */
  async activateNearbyGrids(gridId: string, currentPrice: number): Promise<void> {
    const gridState = this.activeGrids.get(gridId)
    if (!gridState) {
      throw new Error(`Grid ${gridId} not found`)
    }

    const trailingConfig = this.getTrailingConfig(gridState.config)
    
    // Determine activation range based on configuration
    const activationRange = trailingConfig.enableProximityActivation 
      ? trailingConfig.activationThreshold / 100
      : 0.05 // Default 5%

    // Find levels within activation range of current price
    const nearbyLevels = gridState.levels.filter(level => 
      Math.abs(level.price - currentPrice) / currentPrice < activationRange
    )

    this.logger?.info('Activating nearby grid levels', {
      gridId,
      currentPrice,
      totalLevels: nearbyLevels.length,
      activationRange: activationRange * 100,
      strategy: trailingConfig.activationStrategy
    })

    // Activate each nearby level with advanced trailing logic
    for (const level of nearbyLevels) {
      if (!level.isActive) {
        await this.activateGridLevelWithAdvancedTrailing(gridId, level.id, currentPrice)
      }
    }
  }

  /**
   * Updates grid based on new market price with enhanced trailing monitoring
   */
  async processMarketUpdate(gridId: string, marketPrice: number, volume?: number): Promise<void> {
    const gridState = this.activeGrids.get(gridId)
    if (!gridState?.isActive) {
      return
    }

    // Update price tracking
    this.lastPriceUpdates.set(gridState.symbol, {
      price: marketPrice,
      timestamp: Date.now()
    })

    // Process regular grid updates
    await this.updateGrid(gridId, marketPrice)

    // Process advanced trailing order monitoring
    await this.monitorTrailingOrders(gridId, marketPrice, volume)
  }

  /**
   * Monitors and updates trailing orders based on market conditions
   */
  async monitorTrailingOrders(gridId: string, marketPrice: number, volume?: number): Promise<void> {
    const gridState = this.activeGrids.get(gridId)
    if (!gridState) {
      return
    }

    const trailingConfig = this.getTrailingConfig(gridState.config)
    const now = epochDateNow()

    // Check each level for trailing updates
    for (const level of gridState.levels) {
      if (!level.trailingState) continue

      const shouldUpdate = await this.shouldUpdateTrailingOrder(
        level,
        marketPrice,
        volume,
        trailingConfig
      )

      if (shouldUpdate) {
        await this.updateTrailingOrderForLevel(gridId, level.id, marketPrice)
      }

      // Check for activation timeout and fallback
      if (level.trailingState.status === 'approaching' || level.trailingState.status === 'active') {
        const timeoutExceeded = level.trailingState.activatedAt && 
          (now - level.trailingState.activatedAt) > trailingConfig.activationTimeoutMs

        if (timeoutExceeded && trailingConfig.enableDirectFallback && !level.trailingState.fallbackAttempted) {
          await this.executeFallbackOrder(gridId, level.id, marketPrice)
        }
      }
    }
  }

  /**
   * Updates the grid based on new market conditions
   */
  async updateGrid(gridId: string, marketPrice: number): Promise<void> {
    const gridState = this.activeGrids.get(gridId)
    if (!gridState?.isActive) {
      return
    }

    // Check if we need to rebalance based on price movement
    const priceMovement = Math.abs(marketPrice - gridState.centerPrice) / gridState.centerPrice

    if (priceMovement > gridState.config.rebalanceThreshold) {
      this.logger?.info('Grid rebalancing triggered', {
        gridId,
        currentPrice: marketPrice,
        centerPrice: gridState.centerPrice,
        movement: priceMovement,
        threshold: gridState.config.rebalanceThreshold
      })

      // Regenerate grid levels around new center price
      await this.rebalanceGrid(gridId, marketPrice)
    } else {
      // Just activate nearby levels if needed
      await this.activateNearbyGrids(gridId, marketPrice)

      // Update last updated timestamp since rebalanceGrid already updates it
      const currentState = this.activeGrids.get(gridId)
      if (currentState) {
        const updatedState: GridState = {
          ...currentState,
          lastUpdatedAt: epochDateNow()
        }
        this.activeGrids.set(gridId, updatedState)
      }
    }
  }

  /**
   * Cancels a grid and all its active orders
   */
  async cancelGrid(gridId: string, reason?: string): Promise<void> {
    const gridState = this.activeGrids.get(gridId)
    if (!gridState) {
      this.logger?.warn('Attempted to cancel non-existent grid', { gridId })
      return
    }

    this.logger?.info('Cancelling grid', { gridId, reason })

    // Record performance before cancelling if this was an active session
    if (gridState.isActive) {
      await this.recordGridPerformance(gridId, gridState)
    }

    // Cancel all active orders
    for (const level of gridState.levels) {
      if (level.isActive && level.orderId) {
        await this.trailingOrderManager.removeOrder(level.orderId, reason)
      }
    }

    // Update grid state
    const updatedState: GridState = {
      ...gridState,
      isActive: false,
      lastUpdatedAt: epochDateNow()
    }
    this.activeGrids.set(gridId, updatedState)
    this.totalGridsCancelled++

    // Emit cancellation event
    this.eventBus.emit(EventTypes.SYSTEM_INFO, {
      message: `Grid ${gridId} cancelled: ${reason || 'no reason provided'}`,
      context: 'grid_manager',
      timestamp: epochDateNow()
    })
    this.emit('gridCancelled', gridId, reason)
  }

  /**
   * Gets the current state of a grid
   */
  getGridState(gridId: string): GridState | undefined {
    return this.activeGrids.get(gridId)
  }

  /**
   * Gets all active grids
   */
  getActiveGrids(): readonly [string, GridState][] {
    return Array.from(this.activeGrids.entries())
      .filter(([, state]) => state.isActive)
  }

  /**
   * Calculates optimal grid spacing based on market volatility and performance feedback
   */
  async calculateOptimalSpacing(
    historicalCandles: readonly Candle[],
    currentPrice: number
  ): Promise<{ spacing: number; confidence: number; reasoning: string }> {
    // Get all active grid states for performance analysis
    const activeGridStates = Array.from(this.activeGrids.values()).filter(state => state.isActive)
    
    const result = await this.spacingCalculator.calculateOptimalSpacing(
      historicalCandles,
      currentPrice,
      activeGridStates
    )

    this.logger?.info('Optimal spacing calculated', {
      spacing: result.optimalSpacing,
      confidence: result.confidence,
      volatility: result.volatilityMetrics.currentVolatility
    })

    return {
      spacing: result.optimalSpacing,
      confidence: result.confidence,
      reasoning: result.reasoning
    }
  }

  /**
   * Records grid performance for self-tuning analysis
   */
  private async recordGridPerformance(gridId: string, gridState: GridState): Promise<void> {
    try {
      // Calculate market conditions for this session
      const marketConditions = await this.assessMarketConditions(gridState.symbol)
      
      // Record performance with the spacing calculator
      this.spacingCalculator.recordGridPerformance(
        gridState.currentSpacing,
        gridState,
        marketConditions
      )

      this.logger?.debug('Grid performance recorded for self-tuning', {
        gridId,
        spacing: gridState.currentSpacing,
        pnl: gridState.realizedPnl + gridState.unrealizedPnl,
        duration: epochDateNow() - gridState.initializedAt
      })
    } catch (error) {
      this.logger?.error('Failed to record grid performance', {
        gridId,
        error: (error as Error).message
      })
    }
  }

  /**
   * Assesses current market conditions for performance tracking
   */
  private async assessMarketConditions(symbol: string): Promise<{
    volatility: number
    trend: 'bullish' | 'bearish' | 'sideways'
    volume: number
  }> {
    // Simple market condition assessment
    // In a real implementation, this would use market data
    const priceUpdate = this.lastPriceUpdates.get(symbol)
    
    return {
      volatility: 0.02, // Default 2% volatility
      trend: 'sideways', // Default sideways trend
      volume: priceUpdate ? 1000 : 0 // Default volume
    }
  }

  /**
   * Gets performance statistics for self-tuning analysis
   */
  getPerformanceStatistics(): {
    historyCount: number
    averageSpacing: number
    bestPerformingSpacing: number
    currentMetrics: any
  } {
    return this.spacingCalculator.getPerformanceStatistics()
  }

  /**
   * Resets performance history (useful for testing)
   */
  resetPerformanceHistory(): void {
    this.spacingCalculator.resetPerformanceHistory()
  }

  /**
   * Generates grid levels around a center price with intelligent capital distribution
   */
  private generateGridLevels(
    centerPrice: number,
    spacingPercent: number,
    levelCount: number,
    totalCapital: number
  ): GridLevel[] {
    // Validate input parameters first
    this.validateGridParameters(centerPrice, spacingPercent, levelCount, totalCapital)

    const levels: GridLevel[] = []
    const now = epochDateNow()

    // Calculate optimal level distribution
    const { buyLevels, sellLevels, capitalDistribution } = this.calculateLevelDistribution(
      centerPrice,
      spacingPercent,
      levelCount,
      totalCapital
    )

    // Generate buy levels (below center price)
    for (let i = 0; i < buyLevels; i++) {
      const levelIndex = i + 1
      const price = centerPrice * (1 - levelIndex * (spacingPercent / 100))
      const allocatedCapital = capitalDistribution.buyCapital[i] || 0
      const size = allocatedCapital / price // Size in base currency

      levels.push({
        id: uuidv4(),
        price,
        side: 'buy',
        size,
        isActive: false,
        createdAt: now,
        updatedAt: now,
        fillCount: 0,
        pnl: 0
      })
    }

    // Generate sell levels (above center price)
    for (let i = 0; i < sellLevels; i++) {
      const levelIndex = i + 1
      const price = centerPrice * (1 + levelIndex * (spacingPercent / 100))
      const allocatedCapital = capitalDistribution.sellCapital[i] || 0
      const size = allocatedCapital / centerPrice // Size in base currency

      levels.push({
        id: uuidv4(),
        price,
        side: 'sell',
        size,
        isActive: false,
        createdAt: now,
        updatedAt: now,
        fillCount: 0,
        pnl: 0
      })
    }

    // Sort levels by price
    levels.sort((a, b) => a.price - b.price)

    this.logger?.debug('Generated grid levels', {
      totalLevels: levels.length,
      buyLevels,
      sellLevels,
      centerPrice,
      spacing: spacingPercent,
      totalCapital
    })

    return levels
  }

  /**
   * Calculates optimal distribution of levels and capital across buy/sell sides
   */
  private calculateLevelDistribution(
    centerPrice: number,
    spacingPercent: number,
    levelCount: number,
    totalCapital: number
  ): {
    buyLevels: number
    sellLevels: number
    capitalDistribution: {
      buyCapital: number[]
      sellCapital: number[]
    }
  } {
    // Default to equal distribution
    const buyLevels = Math.floor(levelCount / 2)
    const sellLevels = levelCount - buyLevels

    // Calculate capital allocation with gradual increase for levels further from center
    // This ensures better fill probability for levels closer to current price
    const buyCapital: number[] = []
    const sellCapital: number[] = []

    const totalBuyCapital = totalCapital * 0.5
    const totalSellCapital = totalCapital * 0.5

    // Calculate weights for progressive capital allocation
    const buyWeights = this.calculateProgressiveWeights(buyLevels)
    const sellWeights = this.calculateProgressiveWeights(sellLevels)

    // Distribute buy capital
    for (let i = 0; i < buyLevels; i++) {
      buyCapital.push(totalBuyCapital * buyWeights[i]!)
    }

    // Distribute sell capital
    for (let i = 0; i < sellLevels; i++) {
      sellCapital.push(totalSellCapital * sellWeights[i]!)
    }

    this.logger?.debug('Calculated level distribution', {
      centerPrice,
      spacingPercent,
      levelCount,
      buyLevels,
      sellLevels,
      totalBuyCapital,
      totalSellCapital
    })

    return {
      buyLevels,
      sellLevels,
      capitalDistribution: {
        buyCapital,
        sellCapital
      }
    }
  }

  /**
   * Calculates progressive weights for capital allocation
   * Closer levels get slightly more capital for better fill probability
   */
  private calculateProgressiveWeights(levelCount: number): number[] {
    if (levelCount === 0) return []
    if (levelCount === 1) return [1.0]

    const weights: number[] = []
    let totalWeight = 0

    // Progressive weighting: closer levels get more weight
    for (let i = 0; i < levelCount; i++) {
      // Weight decreases as we move further from center
      // First level gets 1.2x weight, gradually decreasing to 0.8x for furthest level
      const weight = 1.2 - (i / (levelCount - 1)) * 0.4
      weights.push(weight)
      totalWeight += weight
    }

    // Normalize weights to sum to 1
    return weights.map(w => w / totalWeight)
  }

  /**
   * Validates grid generation parameters to prevent invalid configurations
   */
  private validateGridParameters(
    centerPrice: number,
    spacingPercent: number,
    levelCount: number,
    totalCapital: number
  ): void {
    if (centerPrice <= 0) {
      throw new Error('Center price must be positive')
    }

    if (spacingPercent <= 0) {
      throw new Error('Grid spacing must be positive')
    }

    if (spacingPercent < 0.1) {
      throw new Error('Grid spacing too tight: minimum spacing is 0.1%')
    }

    if (spacingPercent > 50) {
      throw new Error('Grid spacing too wide: maximum spacing is 50%')
    }

    if (levelCount < 2) {
      throw new Error('Grid must have at least 2 levels')
    }

    if (levelCount > 200) {
      throw new Error('Grid cannot have more than 200 levels')
    }

    if (totalCapital <= 0) {
      throw new Error('Total capital must be positive')
    }

    // Validate minimum capital per level
    const minCapitalPerLevel = centerPrice * 0.00001 // Minimum $0.50 for $50k price
    const capitalPerLevel = totalCapital / levelCount

    if (capitalPerLevel < minCapitalPerLevel) {
      throw new Error(`Insufficient capital: need at least $${(minCapitalPerLevel * levelCount).toFixed(2)} for ${levelCount} levels`)
    }

    // Validate that spacing allows for reasonable price ranges
    const maxLevelsPerSide = Math.floor(levelCount / 2)
    const maxPriceDeviation = spacingPercent * maxLevelsPerSide / 100

    if (maxPriceDeviation > 0.9) {
      throw new Error(`Grid configuration would create extreme price ranges (${(maxPriceDeviation * 100).toFixed(1)}% from center)`)
    }

    this.logger?.debug('Grid parameters validated successfully', {
      centerPrice,
      spacingPercent,
      levelCount,
      totalCapital,
      capitalPerLevel,
      maxPriceDeviation
    })
  }


  /**
   * Rebalances the grid around a new center price
   */
  private async rebalanceGrid(gridId: string, newCenterPrice: number): Promise<void> {
    const gridState = this.activeGrids.get(gridId)
    if (!gridState) {
      throw new Error(`Grid ${gridId} not found`)
    }

    this.logger?.info('Rebalancing grid', {
      gridId,
      oldCenter: gridState.centerPrice,
      newCenter: newCenterPrice
    })

    // Cancel all existing active orders
    for (const level of gridState.levels) {
      if (level.isActive && level.orderId) {
        await this.trailingOrderManager.removeOrder(level.orderId, 'Grid rebalancing')
      }
    }

    // Generate new grid levels
    const newLevels = this.generateGridLevels(
      newCenterPrice,
      gridState.config.gridSpacing,
      gridState.config.gridLevels,
      gridState.availableCapital
    )

    // Update grid state
    const updatedState: GridState = {
      ...gridState,
      levels: newLevels,
      centerPrice: newCenterPrice,
      lastUpdatedAt: epochDateNow()
    }
    this.activeGrids.set(gridId, updatedState)

    // Activate nearby levels
    await this.activateNearbyGrids(gridId, newCenterPrice)

    this.emit('gridUpdated', gridId, updatedState)
  }

  /**
   * Sets up event handlers for order lifecycle events
   */
  private setupEventHandlers(): void {
    // Handle order fills to update grid state
    this.eventBus.subscribe(EventTypes.ORDER_FILLED, (event: any) => {
      if (event.type === 'filled' || event.type === 'partial_fill') {
        this.handleOrderFill(event.order, event.fill)
      }
    })

    // Handle order cancellations
    this.eventBus.subscribe(EventTypes.ORDER_CANCELLED, (event: any) => {
      if (event.type === 'cancelled') {
        this.handleOrderCancellation(event.order)
      }
    })
  }

  /**
   * Handles order fill events to update grid state
   */
  private handleOrderFill(order: any, fill: any): void {
    // Find which grid this order belongs to
    for (const [gridId, gridState] of this.activeGrids.entries()) {
      const level = gridState.levels.find(l => l.orderId === order.id)
      if (level) {
        this.updateLevelOnFill(gridId, level, fill)
        break
      }
    }
  }

  /**
   * Updates a grid level when its order is filled
   */
  private updateLevelOnFill(gridId: string, level: GridLevel, fill: any): void {
    const gridState = this.activeGrids.get(gridId)
    if (!gridState) return

    const now = epochDateNow()

    // Calculate PnL for this fill
    let pnlDelta = 0
    if (level.side === 'buy') {
      // For buy orders, PnL will be realized when we sell
      pnlDelta = 0
    } else {
      // For sell orders, calculate realized PnL
      pnlDelta = (fill.price - gridState.centerPrice) * fill.size
    }

    // Update level
    const updatedLevels = gridState.levels.map(l =>
      l.id === level.id
        ? {
            ...l,
            fillCount: l.fillCount + 1,
            pnl: l.pnl + pnlDelta,
            isActive: false, // Level is filled, no longer active
            orderId: undefined,
            updatedAt: now
          }
        : l
    )

    // Update grid state
    const updatedState: GridState = {
      ...gridState,
      levels: updatedLevels,
      realizedPnl: gridState.realizedPnl + pnlDelta,
      currentPosition: gridState.currentPosition + (level.side === 'buy' ? fill.size : -fill.size),
      lastUpdatedAt: now
    }
    this.activeGrids.set(gridId, updatedState)

    this.emit('levelFilled', gridId, level, fill.price)

    this.logger?.info('Grid level filled', {
      gridId,
      levelId: level.id,
      side: level.side,
      price: fill.price,
      size: fill.size,
      pnl: pnlDelta
    })
  }

  /**
   * Handles order cancellation events
   */
  private handleOrderCancellation(order: any): void {
    // Find which grid this order belongs to
    for (const [gridId, gridState] of this.activeGrids.entries()) {
      const level = gridState.levels.find(l => l.orderId === order.id)
      if (level) {
        // Update level to mark as inactive
        const updatedLevels = gridState.levels.map(l =>
          l.id === level.id
            ? { ...l, isActive: false, orderId: undefined, updatedAt: epochDateNow() }
            : l
        )

        const updatedState: GridState = {
          ...gridState,
          levels: updatedLevels,
          lastUpdatedAt: epochDateNow()
        }
        this.activeGrids.set(gridId, updatedState)

        this.logger?.debug('Grid level order cancelled', {
          gridId,
          levelId: level.id,
          orderId: order.id
        })
        break
      }
    }
  }

  /**
   * Gets trailing order configuration with defaults
   */
  private getTrailingConfig(gridConfig: GridConfig): Required<GridTrailingOrderConfig> {
    const defaults: Required<GridTrailingOrderConfig> = {
      activationThreshold: 3.0, // 3%
      enableProximityActivation: true,
      enableDirectFallback: true,
      activationTimeoutMs: 30000, // 30 seconds
      trailUpdateThrottleMs: 1000, // 1 second
      activationStrategy: 'proximity'
    }

    return {
      ...defaults,
      ...gridConfig.trailingOrderConfig
    }
  }

  /**
   * Activates a grid level with advanced trailing order logic
   */
  private async activateGridLevelWithAdvancedTrailing(
    gridId: string,
    levelId: string,
    currentPrice: number
  ): Promise<void> {
    const gridState = this.activeGrids.get(gridId)
    if (!gridState) {
      throw new Error(`Grid ${gridId} not found`)
    }

    const level = gridState.levels.find(l => l.id === levelId)
    if (!level) {
      throw new Error(`Level ${levelId} not found in grid ${gridId}`)
    }

    if (level.isActive) {
      this.logger?.debug('Level already active', { gridId, levelId })
      return
    }

    const trailingConfig = this.getTrailingConfig(gridState.config)
    const now = epochDateNow()

    try {
      // Determine if price is approaching this level
      const distanceToLevel = Math.abs(level.price - currentPrice) / currentPrice
      const isApproaching = distanceToLevel <= (trailingConfig.activationThreshold / 100)

      // Initialize trailing state
      const trailingState: GridLevelTrailingState = {
        status: isApproaching ? 'approaching' : 'pending',
        activationPrice: isApproaching ? currentPrice : undefined,
        activatedAt: isApproaching ? now : undefined,
        lastUpdatePrice: currentPrice,
        lastUpdatedAt: now,
        adjustmentCount: 0,
        fallbackAttempted: false
      }

      // Create trailing order with appropriate parameters
      const activationPrice = this.calculateActivationPrice(level, currentPrice, trailingConfig)
      
      const trailingOrder = await this.trailingOrderManager.createTrailingOrder({
        symbol: gridState.symbol,
        side: level.side,
        size: level.size,
        trailPercent: gridState.config.trailPercent,
        currentPrice: level.price,
        limitPrice: level.price,
        activationPrice: activationPrice
      })

      // Update level state with trailing information
      const updatedLevels = gridState.levels.map(l =>
        l.id === levelId
          ? { 
              ...l, 
              isActive: true, 
              orderId: trailingOrder.id, 
              updatedAt: now,
              trailingState
            }
          : l
      )

      const updatedState: GridState = {
        ...gridState,
        levels: updatedLevels,
        lastUpdatedAt: now
      }
      this.activeGrids.set(gridId, updatedState)

      this.emit('levelActivated', gridId, { 
        ...level, 
        isActive: true, 
        orderId: trailingOrder.id,
        trailingState 
      })

      this.logger?.info('Grid level activated with advanced trailing', {
        gridId,
        levelId,
        price: level.price,
        side: level.side,
        orderId: trailingOrder.id,
        trailingStatus: trailingState.status,
        activationPrice,
        strategy: trailingConfig.activationStrategy
      })

    } catch (error) {
      await this.handleTrailingActivationFailure(gridId, levelId, error as Error)
    }
  }

  /**
   * Determines if a trailing order should be updated
   */
  private async shouldUpdateTrailingOrder(
    level: GridLevel,
    marketPrice: number,
    volume: number | undefined,
    config: Required<GridTrailingOrderConfig>
  ): Promise<boolean> {
    if (!level.trailingState || !level.trailingState.lastUpdatedAt) {
      return false
    }

    // Check throttling
    const timeSinceLastUpdate = epochDateNow() - level.trailingState.lastUpdatedAt
    if (timeSinceLastUpdate < config.trailUpdateThrottleMs) {
      return false
    }

    // Check if price has moved significantly
    const priceChange = level.trailingState.lastUpdatePrice 
      ? Math.abs(marketPrice - level.trailingState.lastUpdatePrice) / level.trailingState.lastUpdatePrice
      : 1

    // Strategy-based update logic
    switch (config.activationStrategy) {
      case 'proximity':
        return priceChange > 0.001 // 0.1% price change

      case 'price_approach':
        const distanceToLevel = Math.abs(level.price - marketPrice) / level.price
        return distanceToLevel <= (config.activationThreshold / 100)

      case 'volume_spike':
        // TODO: Implement volume-based logic when volume data is consistently available
        return volume ? volume > 1000 : priceChange > 0.002

      case 'combined':
        const proximityCondition = priceChange > 0.001
        const approachCondition = Math.abs(level.price - marketPrice) / level.price <= (config.activationThreshold / 100)
        return proximityCondition || approachCondition

      default:
        return priceChange > 0.001
    }
  }

  /**
   * Updates trailing order for a specific level
   */
  private async updateTrailingOrderForLevel(
    gridId: string,
    levelId: string,
    marketPrice: number
  ): Promise<void> {
    const gridState = this.activeGrids.get(gridId)
    if (!gridState) return

    const level = gridState.levels.find(l => l.id === levelId)
    if (!level?.trailingState || !level.orderId) return

    const now = epochDateNow()

    try {
      // Process market update to trailing order manager
      await this.trailingOrderManager.processMarketUpdate(gridState.symbol, marketPrice)

      // Update trailing state
      const updatedTrailingState: GridLevelTrailingState = {
        ...level.trailingState,
        lastUpdatePrice: marketPrice,
        lastUpdatedAt: now,
        adjustmentCount: level.trailingState.adjustmentCount + 1,
        status: this.determineTrailingStatus(level, marketPrice, gridState.config)
      }

      // Update level in grid state
      const updatedLevels = gridState.levels.map(l =>
        l.id === levelId
          ? { ...l, trailingState: updatedTrailingState, updatedAt: now }
          : l
      )

      const updatedState: GridState = {
        ...gridState,
        levels: updatedLevels,
        lastUpdatedAt: now
      }
      this.activeGrids.set(gridId, updatedState)

      this.logger?.debug('Trailing order updated', {
        gridId,
        levelId,
        marketPrice,
        status: updatedTrailingState.status,
        adjustments: updatedTrailingState.adjustmentCount
      })

    } catch (error) {
      this.logger?.error('Failed to update trailing order', {
        gridId,
        levelId,
        error: (error as Error).message
      })
    }
  }

  /**
   * Calculates activation price for trailing order
   */
  private calculateActivationPrice(
    level: GridLevel,
    currentPrice: number,
    config: Required<GridTrailingOrderConfig>
  ): number | undefined {
    if (!config.enableProximityActivation) {
      return undefined
    }

    const threshold = config.activationThreshold / 100

    // Use current price for dynamic threshold calculation if needed
    const basePrice = currentPrice > 0 ? level.price : level.price

    if (level.side === 'buy') {
      // For buy orders, activate when price approaches from above
      return basePrice * (1 + threshold)
    } else {
      // For sell orders, activate when price approaches from below
      return basePrice * (1 - threshold)
    }
  }

  /**
   * Determines trailing status based on market conditions
   */
  private determineTrailingStatus(
    level: GridLevel,
    marketPrice: number,
    gridConfig: GridConfig
  ): GridLevelTrailingState['status'] {
    const config = this.getTrailingConfig(gridConfig)
    const distanceToLevel = Math.abs(level.price - marketPrice) / level.price
    const threshold = config.activationThreshold / 100

    if (distanceToLevel <= threshold * 0.5) {
      return 'active'
    } else if (distanceToLevel <= threshold) {
      return 'approaching'
    } else {
      return 'pending'
    }
  }

  /**
   * Executes fallback order when trailing fails
   */
  private async executeFallbackOrder(
    gridId: string,
    levelId: string,
    marketPrice: number
  ): Promise<void> {
    const gridState = this.activeGrids.get(gridId)
    if (!gridState) return

    const level = gridState.levels.find(l => l.id === levelId)
    if (!level?.trailingState) return

    this.logger?.warn('Executing fallback order for grid level', {
      gridId,
      levelId,
      levelPrice: level.price,
      marketPrice,
      reason: 'Trailing order timeout'
    })

    try {
      // Cancel existing trailing order if still active
      if (level.orderId) {
        await this.trailingOrderManager.removeOrder(level.orderId, 'Fallback execution')
      }

      // Create direct order at market price
      const directOrder = await this.trailingOrderManager.createTrailingOrder({
        symbol: gridState.symbol,
        side: level.side,
        size: level.size,
        trailPercent: 0.1, // Minimal trail for immediate execution
        currentPrice: marketPrice,
        limitPrice: marketPrice
      })

      // Update level state
      const updatedTrailingState: GridLevelTrailingState = {
        ...level.trailingState,
        status: 'triggered',
        fallbackAttempted: true,
        lastUpdatedAt: epochDateNow()
      }

      const updatedLevels = gridState.levels.map(l =>
        l.id === levelId
          ? { 
              ...l, 
              orderId: directOrder.id, 
              trailingState: updatedTrailingState,
              updatedAt: epochDateNow()
            }
          : l
      )

      const updatedState: GridState = {
        ...gridState,
        levels: updatedLevels,
        lastUpdatedAt: epochDateNow()
      }
      this.activeGrids.set(gridId, updatedState)

      this.emit('levelFallbackExecuted', gridId, level, marketPrice)

    } catch (error) {
      await this.handleTrailingActivationFailure(gridId, levelId, error as Error)
    }
  }

  /**
   * Handles trailing order activation failures
   */
  private async handleTrailingActivationFailure(
    gridId: string,
    levelId: string,
    error: Error
  ): Promise<void> {
    const gridState = this.activeGrids.get(gridId)
    if (!gridState) return

    const level = gridState.levels.find(l => l.id === levelId)
    if (!level) return

    this.logger?.error('Trailing order activation failed', {
      gridId,
      levelId,
      error: error.message
    })

    // Update level state to reflect failure
    const failedTrailingState: GridLevelTrailingState = {
      status: 'failed',
      lastUpdatedAt: epochDateNow(),
      adjustmentCount: 0,
      fallbackAttempted: false,
      failureReason: error.message
    }

    const updatedLevels = gridState.levels.map(l =>
      l.id === levelId
        ? { ...l, trailingState: failedTrailingState, updatedAt: epochDateNow() }
        : l
    )

    const updatedState: GridState = {
      ...gridState,
      levels: updatedLevels,
      lastUpdatedAt: epochDateNow()
    }
    this.activeGrids.set(gridId, updatedState)

    this.emit('levelActivationFailed', gridId, level, error)
  }

  // === PERSISTENCE METHODS ===

  /**
   * Initialize persistence system and attempt to restore state
   */
  async initializePersistence(): Promise<StateRecoveryInfo> {
    await this.persistence.initialize()
    
    const { snapshot, recoveryInfo } = await this.persistence.loadSnapshot()
    
    if (recoveryInfo.success && recoveryInfo.recoveredGrids > 0) {
      await this.restoreFromSnapshot(snapshot)
      
      // Start auto-save after successful restoration
      this.persistence.startAutoSave(() => this.createSnapshot())
      
      this.logger?.info('Grid state restored from persistence', {
        recoveredGrids: recoveryInfo.recoveredGrids,
        failedGrids: recoveryInfo.failedGrids
      })
    } else {
      // Start auto-save even with empty state
      this.persistence.startAutoSave(() => this.createSnapshot())
      
      this.logger?.info('Starting with empty grid state', {
        errors: recoveryInfo.errors
      })
    }
    
    return recoveryInfo
  }

  /**
   * Create a complete snapshot of current grid manager state
   */
  async createSnapshot(): Promise<GridManagerSnapshot> {
    const activeGrids: Record<string, SerializableGridState> = {}
    
    // Convert all active grids to serializable format
    for (const [gridId, gridState] of this.activeGrids.entries()) {
      activeGrids[gridId] = this.persistence.serializeGridState(gridState)
    }
    
    // Get performance history from self-tuning calculator
    const performanceHistory: PerformanceHistoryRecord[] = []
    
    // Convert last price updates to plain object
    const lastPriceUpdates: Record<string, { price: number; timestamp: number }> = {}
    for (const [symbol, update] of this.lastPriceUpdates.entries()) {
      lastPriceUpdates[symbol] = update
    }
    
    return {
      version: '1.0.0',
      timestamp: epochDateNow(),
      activeGrids,
      performanceHistory,
      lastPriceUpdates,
      metadata: {
        totalGridsCreated: this.totalGridsCreated,
        totalGridsCancelled: this.totalGridsCancelled,
        systemUptime: epochDateNow() - this.systemStartTime
      }
    }
  }

  /**
   * Restore grid manager state from a snapshot
   */
  private async restoreFromSnapshot(snapshot: GridManagerSnapshot): Promise<void> {
    try {
      // Clear current state
      this.activeGrids.clear()
      this.lastPriceUpdates.clear()
      
      // Restore active grids
      for (const [gridId, serializedState] of Object.entries(snapshot.activeGrids)) {
        const gridState = this.persistence.deserializeGridState(serializedState)
        this.activeGrids.set(gridId, gridState)
      }
      
      // Restore last price updates
      for (const [symbol, update] of Object.entries(snapshot.lastPriceUpdates)) {
        this.lastPriceUpdates.set(symbol, update)
      }
      
      // Restore metadata counters
      if (snapshot.metadata) {
        this.totalGridsCreated = snapshot.metadata.totalGridsCreated || 0
        this.totalGridsCancelled = snapshot.metadata.totalGridsCancelled || 0
      }
      
      // Restore performance history to self-tuning calculator
      if (snapshot.performanceHistory && snapshot.performanceHistory.length > 0) {
        this.spacingCalculator.resetPerformanceHistory()
        // TODO: Add method to bulk restore performance history
      }
      
      this.logger?.info('State restoration completed', {
        activeGrids: this.activeGrids.size,
        totalCreated: this.totalGridsCreated,
        totalCancelled: this.totalGridsCancelled
      })
      
    } catch (error) {
      this.logger?.error('Failed to restore state from snapshot', { error })
      throw new Error(`State restoration failed: ${error}`)
    }
  }

  /**
   * Manually save current state
   */
  async saveState(): Promise<void> {
    const snapshot = await this.createSnapshot()
    await this.persistence.saveSnapshot(snapshot)
    
    this.logger?.info('Grid state saved manually', {
      activeGrids: this.activeGrids.size,
      timestamp: snapshot.timestamp
    })
  }

  /**
   * Clean shutdown with state persistence
   */
  async shutdown(): Promise<void> {
    try {
      // Create final snapshot
      const finalSnapshot = await this.createSnapshot()
      
      // Shutdown persistence system
      await this.persistence.shutdown(finalSnapshot)
      
      // Cancel all active grids
      const activeGridIds = Array.from(this.activeGrids.keys())
      for (const gridId of activeGridIds) {
        await this.cancelGrid(gridId, 'System shutdown')
      }
      
      this.logger?.info('Grid manager shutdown completed', {
        finalActiveGrids: activeGridIds.length,
        totalCreated: this.totalGridsCreated,
        totalCancelled: this.totalGridsCancelled
      })
      
    } catch (error) {
      this.logger?.error('Error during grid manager shutdown', { error })
      throw error
    }
  }

  /**
   * Get current persistence statistics
   */
  getPersistenceStats(): {
    activeGrids: number
    totalCreated: number
    totalCancelled: number
    systemUptime: number
    lastSaveTime?: number
  } {
    // Count only grids where isActive is true
    const activeCount = Array.from(this.activeGrids.values())
      .filter(grid => grid.isActive).length
    
    return {
      activeGrids: activeCount,
      totalCreated: this.totalGridsCreated,
      totalCancelled: this.totalGridsCancelled,
      systemUptime: epochDateNow() - this.systemStartTime
    }
  }
}