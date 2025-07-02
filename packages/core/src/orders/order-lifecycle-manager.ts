import {
  EnhancedOrderState
} from '@trdr/shared'
import type {
  OrderAgentConsensus,
  ManagedOrder,
  OrderLifecycleConfig,
  OrderModification,
  OrderValidationResult,
  TrailingOrder,
  Mutable,
  OrderFill
} from '@trdr/shared'
import { OrderStateMachine } from './order-state-machine'
import { OrderExecutionMonitor } from './order-execution-monitor'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import { PositionSizingManager, type PositionSizingInput } from '../position-sizing'
import { ConsensusManager, type SignalRequest } from '../consensus'

/**
 * Manages the complete lifecycle of orders from agent consensus to execution.
 * 
 * Based on PRD section 3.3.1 - Order Lifecycle Management.
 * Implements agent signal processing, dynamic position sizing, order monitoring,
 * and order improvement strategies.
 */
export class OrderLifecycleManager {
  /** Map of active orders by order ID */
  private readonly activeOrders = new Map<string, ManagedOrder>()
  /** State machine for managing order state transitions */
  private readonly orderStateMachine: OrderStateMachine
  /** Event bus for publishing order lifecycle events */
  private readonly eventBus: EventBus
  /** Configuration for order lifecycle management */
  private readonly config: OrderLifecycleConfig
  /** Execution monitor for tracking order performance */
  private readonly executionMonitor: OrderExecutionMonitor
  /** Position sizing manager for dynamic risk management */
  private readonly positionSizingManager: PositionSizingManager
  /** Consensus manager for gathering agent signals */
  private readonly consensusManager: ConsensusManager

  /**
   * Creates a new OrderLifecycleManager instance.
   * 
   * @param config - Configuration parameters for order management
   * @param eventBus - Optional event bus instance (uses singleton if not provided)
   */
  constructor(
    config: OrderLifecycleConfig,
    eventBus?: EventBus
  ) {
    this.config = config
    this.eventBus = eventBus || EventBus.getInstance()
    this.orderStateMachine = new OrderStateMachine(this.eventBus)
    this.executionMonitor = new OrderExecutionMonitor(config.circuitBreaker, this.eventBus)
    
    // Initialize position sizing manager
    this.positionSizingManager = new PositionSizingManager({
      defaultStrategy: 'kelly', // Use Kelly Criterion by default
      strategies: ['kelly', 'fixed', 'volatility'],
      enableAdaptive: true,
      enableMarketAdjustments: true,
      minPositionSize: config.minOrderSize / 50000, // Convert USD to BTC (approx)
      maxPositionSize: config.maxOrderSize / 50000,
      enableBacktesting: false
    }, this.eventBus)
    
    // Initialize consensus manager
    this.consensusManager = new ConsensusManager({
      minConfidenceThreshold: config.minConfidenceThreshold,
      minAgentsRequired: 3, // Require at least 3 agents
      consensusTimeoutMs: 5000, // 5 second timeout
      fallbackStrategy: 'use-majority',
      useWeightedVoting: true,
      minAgreementThreshold: 0.6, // 60% agreement required
      enableAdaptiveWeights: true,
      defaultAgentWeight: 1.0,
      maxAgentWeight: 3.0,
      symbol: config.symbol
    }, undefined, this.eventBus)
    
    this.registerEventTypes()
    this.setupEventHandlers()
  }

  /**
   * Register additional event types used by lifecycle manager
   */
  private registerEventTypes(): void {
    this.eventBus.registerEvent(EventTypes.ORDER_CREATED)
    this.eventBus.registerEvent(EventTypes.ORDER_PARTIAL_FILL)
    this.eventBus.registerEvent(EventTypes.ORDER_CONSENSUS_REJECTED)
    this.eventBus.registerEvent(EventTypes.ORDER_SIZE_TOO_SMALL)
    this.eventBus.registerEvent(EventTypes.ORDER_VALIDATION_FAILED)
    this.eventBus.registerEvent(EventTypes.ORDER_CANCEL_FAILED)
    this.eventBus.registerEvent(EventTypes.ORDER_MODIFIED)
    this.eventBus.registerEvent(EventTypes.ORDER_MODIFICATION_FAILED)
    this.eventBus.registerEvent(EventTypes.ORDER_IMPROVED)
    this.eventBus.registerEvent(EventTypes.ORDER_IMPROVEMENT_FAILED)
    this.eventBus.registerEvent(EventTypes.ORDER_TIME_CONSTRAINT_ERROR)
    this.eventBus.registerEvent(EventTypes.ORDER_FILL)
    this.eventBus.registerEvent(EventTypes.MARKET_TICK)
  }

  /**
   * Process agent consensus and create order if conditions are met.
   * 
   * Implements consensus validation, position sizing, and order creation
   * as specified in PRD section 3.3.1.
   * 
   * @param consensus - Agent consensus containing trading decision and parameters
   * @returns Created trailing order if all conditions are met, null otherwise
   * 
   * @example
   * ```typescript
   * const consensus = {
   *   action: 'buy',
   *   confidence: 0.75,
   *   expectedWinRate: 0.65,
   *   expectedRiskReward: 2.0,
   *   trailDistance: 2.5,
   *   leadAgentId: 'agent-trend-1',
   *   agentSignals: [...]
   * }
   * const order = await manager.processAgentConsensus(consensus)
   * ```
   */
  processAgentConsensus(consensus: OrderAgentConsensus): TrailingOrder | null {
    // Check circuit breaker before processing
    if (this.executionMonitor.isCircuitBreakerActive()) {
      this.eventBus.emit(EventTypes.ORDER_CONSENSUS_REJECTED, {
        consensus,
        reason: 'Circuit breaker is active - trading suspended',
        timestamp: new Date()
      })
      return null
    }

    // Minimum confidence threshold check
    if (consensus.confidence < this.config.minConfidenceThreshold) {
      this.eventBus.emit(EventTypes.ORDER_CONSENSUS_REJECTED, {
        consensus,
        reason: `Confidence ${consensus.confidence} below threshold ${this.config.minConfidenceThreshold}`,
        timestamp: new Date()
      })
      return null
    }

    // Check for conflicting orders
    const existingOrder = this.findConflictingOrder(consensus)
    if (existingOrder) {
      this.modifyOrder(existingOrder, consensus)
      return null
    }

    // Calculate order size using dynamic position sizing
    const size = this.calculateOrderSize(consensus)
    if (size < this.config.minOrderSize) {
      this.eventBus.emit(EventTypes.ORDER_SIZE_TOO_SMALL, {
        consensus,
        calculatedSize: size,
        minSize: this.config.minOrderSize,
        timestamp: new Date()
      })
      return null
    }

    // Create order with enhanced metadata
    const order = this.createOrderFromConsensus(consensus, size)
    
    // Validate order
    const validation = this.validateOrder(order)
    if (!validation.valid) {
      this.eventBus.emit(EventTypes.ORDER_VALIDATION_FAILED, {
        order,
        errors: validation.errors,
        warnings: validation.warnings,
        timestamp: new Date()
      })
      return null
    }

    // Emit order creation event
    this.eventBus.emit(EventTypes.ORDER_CREATED, {
      order,
      consensus,
      timestamp: new Date()
    })

    return order
  }

  /**
   * Calculate dynamic order size based on consensus and risk parameters.
   * 
   * Uses pluggable position sizing strategies (Kelly, Fixed Fractional, Volatility Adjusted).
   * 
   * @param consensus - Agent consensus with expected win rate and risk/reward
   * @returns Calculated order size in base currency units (e.g., BTC)
   * 
   * @remarks
   * Delegates to PositionSizingManager which implements:
   * - Multiple position sizing strategies
   * - Market condition adjustments
   * - Risk limit enforcement
   * - Adaptive sizing based on performance
   */
  calculateOrderSize(consensus: OrderAgentConsensus): number {
    // Get current market conditions
    const marketConditions = this.getCurrentMarketConditions()
    
    // Get risk parameters
    const accountBalance = 10000 // TODO: Get from portfolio manager
    const currentExposure = this.getCurrentExposure()
    const openPositions = this.activeOrders.size
    
    const riskParams = PositionSizingManager.createRiskParameters(
      accountBalance,
      currentExposure,
      openPositions
    )
    
    // Get symbol from consensus or config
    const symbol = consensus.symbol || this.config.symbol || 'BTC-USD'
    
    // Calculate stop loss based on trail distance
    const currentPrice = this.getCurrentPrice(symbol)
    const stopLoss = consensus.action === 'buy'
      ? currentPrice * (1 - consensus.trailDistance / 100)
      : currentPrice * (1 + consensus.trailDistance / 100)
    
    // Create position sizing input
    const sizingInput: PositionSizingInput = {
      side: consensus.action,
      entryPrice: currentPrice,
      stopLoss,
      winRate: consensus.expectedWinRate,
      riskRewardRatio: consensus.expectedRiskReward,
      confidence: consensus.confidence,
      riskParams,
      marketConditions,
      historicalMetrics: this.getHistoricalMetrics()
    }
    
    // Calculate position size
    const sizingOutput = this.positionSizingManager.calculatePositionSize(sizingInput)
    
    // Log warnings if any
    if (sizingOutput.warnings.length > 0) {
      this.eventBus.emit(EventTypes.SYSTEM_WARNING, {
        context: 'position_sizing',
        warnings: sizingOutput.warnings,
        timestamp: new Date()
      })
    }
    
    // Convert from BTC to USD for order size
    const sizeInUSD = sizingOutput.positionSize * currentPrice
    
    return this.roundToValidSize(sizeInUSD)
  }

  /**
   * Submit order to exchange and begin monitoring.
   * 
   * @param order - Trailing order to submit
   * @returns Managed order with state tracking
   * @throws Error if order submission fails
   * 
   * @remarks
   * - Transitions order to SUBMITTED state
   * - Stores in active orders map
   * - Starts real-time monitoring for price updates
   * - Handles rejection with proper state transition
   */
  submitOrder(order: TrailingOrder): ManagedOrder {
    const managedOrder = this.createManagedOrder(order)
    
    try {
      // First transition to PENDING (validated)
      this.orderStateMachine.transition(
        managedOrder,
        EnhancedOrderState.PENDING
      )
      
      // Then transition to SUBMITTED
      this.orderStateMachine.transition(
        managedOrder,
        EnhancedOrderState.SUBMITTED
      )

      // Store in active orders
      this.activeOrders.set(managedOrder.id, managedOrder)

      // Start execution monitoring
      this.executionMonitor.startMonitoring(managedOrder)

      // Start order monitoring
      this.monitorOrder(managedOrder)

      return managedOrder
    } catch (error) {
      // Transition to rejected state on submission failure
      this.orderStateMachine.transition(
        managedOrder,
        EnhancedOrderState.REJECTED,
        (error as Error).message
      )
      throw error
    }
  }

  /**
   * Cancel an active order.
   * 
   * @param orderId - ID of the order to cancel
   * @param reason - Optional cancellation reason
   * @throws Error if order not found or cannot be cancelled in current state
   * 
   * @example
   * ```typescript
   * manager.cancelOrder('order_123', 'User requested cancellation')
   * ```
   */
  cancelOrder(orderId: string, reason?: string): void {
    const order = this.activeOrders.get(orderId)
    if (!order) {
      throw new Error(`Order ${orderId} not found`)
    }

    if (!this.orderStateMachine.canTransition(order.state, EnhancedOrderState.CANCELLED)) {
      throw new Error(`Cannot cancel order ${orderId} in state ${order.state}`)
    }

    try {
      // Cancel with exchange would go here
      // await this.exchange.cancelOrder(order.exchangeOrderId)

      this.orderStateMachine.transition(
        order,
        EnhancedOrderState.CANCELLED,
        reason || 'Manual cancellation'
      )

      this.activeOrders.delete(orderId)
    } catch (error) {
      this.eventBus.emit(EventTypes.ORDER_CANCEL_FAILED, {
        orderId,
        error: (error as Error).message,
        timestamp: new Date()
      })
      throw error
    }
  }

  /**
   * Modify an existing order based on new consensus.
   * 
   * @param order - Existing managed order to modify
   * @param consensus - New agent consensus with updated parameters
   * @returns Modified order if successful, null if order cannot be modified
   * 
   * @remarks
   * Only modifies orders in active states (PENDING, SUBMITTED, PARTIAL).
   * Updates order size and trail percentage based on new consensus.
   */
  modifyOrder(order: ManagedOrder, consensus: OrderAgentConsensus): ManagedOrder | null {
    if (!this.orderStateMachine.isActiveState(order.state)) {
      return null
    }

    const newSize = this.calculateOrderSize(consensus)
    const modification: OrderModification = {
      size: newSize,
      trailPercent: consensus.trailDistance
    }

    try {
      // Apply modification
      this.applyOrderModification(order, modification)
      
      this.eventBus.emit(EventTypes.ORDER_MODIFIED, {
        orderId: order.id,
        modification,
        consensus,
        timestamp: new Date()
      })

      return order
    } catch (error) {
      this.eventBus.emit(EventTypes.ORDER_MODIFICATION_FAILED, {
        orderId: order.id,
        error: (error as Error).message,
        timestamp: new Date()
      })
      return null
    }
  }

  /**
   * Monitor order for price updates and time constraints.
   * 
   * @param order - Order to monitor
   * 
   * @remarks
   * Sets up monitoring for:
   * - Real-time price updates via market tick events
   * - Time constraint violations (expiration, max duration)
   * - Automatic cleanup when order reaches terminal state
   * 
   * @privateRemarks
   * Called automatically by submitOrder() after successful submission
   */
  private monitorOrder(order: ManagedOrder): void {
    // Set up price monitoring
    const priceSubscription = this.eventBus.subscribe(EventTypes.MARKET_TICK, (data: unknown) => {
      const marketData = data as { symbol?: string, price?: number }
      if (marketData.symbol === order.symbol && typeof marketData.price === 'number') {
        this.handlePriceUpdate(order, marketData.price)
      }
    })

    // Set up time constraint monitoring
    if (order.metadata?.timeConstraints) {
      this.scheduleTimeConstraintChecks(order)
    }

    // Clean up subscriptions when order completes
    const cleanup = (): void => {
      priceSubscription.unsubscribe()
    }

    this.eventBus.subscribe(EventTypes.ORDER_STATE_CHANGED, (data: unknown) => {
      const stateChange = data as { orderId?: string, newState?: EnhancedOrderState }
      if (stateChange.orderId === order.id && stateChange.newState && this.orderStateMachine.isTerminalState(stateChange.newState)) {
        cleanup()
      }
    })
  }

  /**
   * Handle price updates for order improvement.
   * 
   * @param order - Order being monitored
   * @param currentPrice - Current market price
   * 
   * @remarks
   * Checks if order improvement is enabled and if current price
   * presents an opportunity to improve the order price.
   */
  private handlePriceUpdate(order: ManagedOrder, currentPrice: number): void {
    if (!this.config.enableOrderImprovement) {
      return
    }

    if (this.shouldImproveOrder(order, currentPrice)) {
      this.improveOrder(order, currentPrice)
    }
  }

  /**
   * Check if order should be improved based on price movement.
   * 
   * @param order - Order to check
   * @param currentPrice - Current market price
   * @returns true if order price should be improved
   * 
   * @remarks
   * Order improvement logic:
   * - Buy orders: Improve if current price is lower than order price
   * - Sell orders: Improve if current price is higher than order price
   * - Only improve if difference exceeds improvement threshold (0.1% default)
   */
  private shouldImproveOrder(order: ManagedOrder, currentPrice: number): boolean {
    if (!('price' in order) || !order.price) {
      return false
    }

    const improvement = order.side === 'buy'
      ? order.price - currentPrice
      : currentPrice - order.price

    return improvement / order.price > this.config.improvementThreshold
  }

  /**
   * Improve order price based on favorable market conditions.
   * 
   * @param order - Order to improve
   * @param newPrice - New improved price
   * 
   * @remarks
   * Modifies the order price to take advantage of favorable market movements.
   * Emits 'order.improved' event on success or 'order.improvement.failed' on error.
   * 
   * @example
   * - Buy order at $50,000, market drops to $49,500 → improve to $49,500
   * - Sell order at $50,000, market rises to $50,500 → improve to $50,500
   */
  private improveOrder(order: ManagedOrder, newPrice: number): void {
    try {
      const modification: OrderModification = { price: newPrice }
      this.applyOrderModification(order, modification)

      this.eventBus.emit(EventTypes.ORDER_IMPROVED, {
        orderId: order.id,
        oldPrice: ('price' in order) ? order.price : undefined,
        newPrice,
        timestamp: new Date()
      })
    } catch (error) {
      this.eventBus.emit(EventTypes.ORDER_IMPROVEMENT_FAILED, {
        orderId: order.id,
        error: (error as Error).message,
        timestamp: new Date()
      })
    }
  }

  /**
   * Apply modifications to an order
   */
  private applyOrderModification(
    order: ManagedOrder,
    modification: OrderModification
  ): void {
    const mut = order as Mutable<ManagedOrder>
    
    // Update order properties
    if (modification.price !== undefined && 'price' in order) {
      mut.price = modification.price
    }
    if (modification.size !== undefined) {
      mut.size = modification.size
    }
    if (modification.trailPercent !== undefined && order.type === 'trailing') {
      mut.trailPercent = modification.trailPercent
    }

    mut.lastModified = new Date()

    // Exchange modification would go here
    // await this.exchange.modifyOrder(order.exchangeOrderId, modification)
  }

  /**
   * Find conflicting orders that would compete with new consensus
   */
  private findConflictingOrder(consensus: OrderAgentConsensus): ManagedOrder | null {
    for (const order of this.activeOrders.values()) {
      if (order.side === consensus.action && 
          this.orderStateMachine.isActiveState(order.state)) {
        return order
      }
    }
    return null
  }

  /**
   * Create order from agent consensus
   */
  private createOrderFromConsensus(consensus: OrderAgentConsensus, size: number): TrailingOrder {
    return {
      id: this.generateOrderId(),
      symbol: consensus.symbol || this.config.symbol || 'BTC-USD',
      side: consensus.action,
      type: 'trailing',
      size,
      trailPercent: consensus.trailDistance,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      status: 'pending'
    }
  }

  /**
   * Create managed order from base order
   */
  private createManagedOrder(order: TrailingOrder): ManagedOrder {
    return {
      ...order,
      state: EnhancedOrderState.CREATED,
      filledSize: 0,
      averageFillPrice: 0,
      fees: 0,
      lastModified: new Date(),
      fills: []
    }
  }

  /**
   * Get current market conditions for position sizing
   */
  private getCurrentMarketConditions(): ReturnType<typeof PositionSizingManager.assessMarketConditions> {
    // TODO: Get actual market data from data provider
    const currentPrice = 50000 // BTC price
    const recentPrices = Array.from({ length: 50 }, (_, i) => 
      currentPrice + (Math.random() - 0.5) * 1000 - i * 10
    ).reverse()
    const volume = 1000000
    const avgVolume = 1200000
    
    return PositionSizingManager.assessMarketConditions(
      currentPrice,
      recentPrices,
      volume,
      avgVolume
    )
  }

  /**
   * Get current price for a symbol
   */
  private getCurrentPrice(_symbol: string): number {
    // TODO: Get actual price from market data provider
    // Will use _symbol parameter when market data provider is integrated
    return 50000 // BTC-USD price placeholder
  }

  /**
   * Get current exposure across all positions
   */
  private getCurrentExposure(): number {
    let totalExposure = 0
    for (const order of this.activeOrders.values()) {
      if (this.orderStateMachine.isActiveState(order.state)) {
        totalExposure += order.size - order.filledSize
      }
    }
    return totalExposure
  }

  /**
   * Get historical performance metrics
   */
  private getHistoricalMetrics(): {
    avgWinRate: number
    avgRiskReward: number
    maxConsecutiveLosses: number
    currentConsecutiveLosses: number
    sharpeRatio: number
    maxDrawdown: number
    profitFactor: number
  } {
    // TODO: Get actual metrics from performance tracker
    return {
      avgWinRate: 0.6,
      avgRiskReward: 2.0,
      maxConsecutiveLosses: 3,
      currentConsecutiveLosses: 0,
      sharpeRatio: 1.5,
      maxDrawdown: 0.15,
      profitFactor: 1.8
    }
  }

  /**
   * Round size to valid exchange increment
   */
  private roundToValidSize(size: number): number {
    // TODO: Get actual size increment from exchange info
    const increment = 0.00001 // BTC size increment
    return Math.round(size / increment) * increment
  }

  /**
   * Validate order before submission
   */
  private validateOrder(order: TrailingOrder): OrderValidationResult {
    const errors: string[] = []
    const warnings: string[] = []

    if (order.size < this.config.minOrderSize) {
      errors.push(`Order size ${order.size} below minimum ${this.config.minOrderSize}`)
    }

    if (order.size > this.config.maxOrderSize) {
      errors.push(`Order size ${order.size} exceeds maximum ${this.config.maxOrderSize}`)
    }

    if (!order.symbol) {
      errors.push('Order symbol is required')
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    }
  }

  /**
   * Schedule time constraint checks for order.
   * 
   * @param order - Order with time constraints to monitor
   * 
   * @remarks
   * Schedules automatic cancellation/expiration for:
   * - maxDuration: Cancel after specified milliseconds
   * - expiresAt: Cancel at specific date/time
   * 
   * @privateRemarks
   * Called by monitorOrder() if order has time constraints
   */
  private scheduleTimeConstraintChecks(order: ManagedOrder): void {
    const constraints = order.metadata?.timeConstraints
    if (!constraints) return

    if (constraints.maxDuration) {
      setTimeout(() => {
        this.handleTimeConstraintViolation(order, 'maxDuration')
      }, constraints.maxDuration)
    }

    if (constraints.expiresAt) {
      const timeToExpiry = constraints.expiresAt.getTime() - Date.now() // milliseconds
      if (timeToExpiry > 0) {
        setTimeout(() => {
          this.handleTimeConstraintViolation(order, 'expiration')
        }, timeToExpiry)
      }
    }
  }

  /**
   * Handle time constraint violations.
   * 
   * @param order - Order that violated time constraint
   * @param violationType - Type of violation ('maxDuration' or 'expiration')
   * 
   * @remarks
   * Transitions order to EXPIRED state if still active.
   * Removes order from active orders map.
   * Emits error event if transition fails.
   */
  private handleTimeConstraintViolation(
    order: ManagedOrder,
    violationType: string
  ): void {
    if (!this.orderStateMachine.isActiveState(order.state)) {
      return
    }

    try {
      this.orderStateMachine.transition(
        order,
        EnhancedOrderState.EXPIRED,
        `Time constraint violation: ${violationType}`
      )
      
      this.activeOrders.delete(order.id)
    } catch (error) {
      this.eventBus.emit(EventTypes.ORDER_TIME_CONSTRAINT_ERROR, {
        orderId: order.id,
        violationType,
        error: (error as Error).message,
        timestamp: new Date()
      })
    }
  }

  /**
   * Generate unique order ID
   */
  private generateOrderId(): string {
    return `order_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`
  }

  /**
   * Setup event handlers for order lifecycle
   */
  private setupEventHandlers(): void {
    // Handle order fills
    this.eventBus.subscribe(EventTypes.ORDER_FILL, (data: unknown) => {
      const fillData = data as { orderId?: string, fill?: OrderFill }
      if (typeof fillData.orderId === 'string' && fillData.fill) {
        this.handleOrderFill(fillData.orderId, fillData.fill)
      }
    })

    // Handle partial fills
    this.eventBus.subscribe(EventTypes.ORDER_PARTIAL_FILL, (data: unknown) => {
      const fillData = data as { orderId?: string, fill?: OrderFill }
      if (typeof fillData.orderId === 'string' && fillData.fill) {
        this.handlePartialFill(fillData.orderId, fillData.fill)
      }
    })
  }

  /**
   * Handle order fill events
   */
  private handleOrderFill(orderId: string, fill: OrderFill): void {
    const order = this.activeOrders.get(orderId)
    if (!order) return

    // Update order with fill information
    order.fills = [...order.fills, fill]
    order.filledSize += fill.size
    order.fees += fill.fee
    
    // Recalculate average fill price
    const totalValue = order.fills.reduce((sum, f) => sum + f.size * f.price, 0)
    order.averageFillPrice = totalValue / order.filledSize

    // Update execution metrics
    this.executionMonitor.updateFillMetrics(order, fill)

    // Check if completely filled
    if (order.filledSize >= order.size) {
      this.orderStateMachine.transition(order, EnhancedOrderState.FILLED)
      
      // Check circuit breaker on completion
      const isSuccess = order.state === EnhancedOrderState.FILLED
      const pnl = this.calculateOrderPnL(order)
      this.executionMonitor.checkCircuitBreaker(order, isSuccess, pnl < 0 ? Math.abs(pnl) : undefined)
      
      this.activeOrders.delete(orderId)
    } else {
      this.orderStateMachine.transition(order, EnhancedOrderState.PARTIALLY_FILLED)
    }
  }

  /**
   * Handle partial fill events
   */
  private handlePartialFill(orderId: string, fill: OrderFill): void {
    this.handleOrderFill(orderId, fill)
  }

  /**
   * Get all active orders.
   * 
   * @returns Read-only map of active orders by order ID
   * 
   * @example
   * ```typescript
   * const activeOrders = manager.getActiveOrders()
   * console.log(`Active orders: ${activeOrders.size}`)
   * ```
   */
  getActiveOrders(): ReadonlyMap<string, ManagedOrder> {
    return this.activeOrders
  }

  /**
   * Get order by ID.
   * 
   * @param orderId - ID of the order to retrieve
   * @returns Managed order if found, undefined otherwise
   * 
   * @example
   * ```typescript
   * const order = manager.getOrder('order_123')
   * if (order) {
   *   console.log(`Order state: ${order.state}`)
   * }
   * ```
   */
  getOrder(orderId: string): ManagedOrder | undefined {
    return this.activeOrders.get(orderId)
  }

  /**
   * Calculate P&L for a completed order.
   *
   * @returns P&L in USD (positive = profit, negative = loss)
   *
   * @privateRemarks
   * Simple P&L calculation based on fill price vs current market price.
   * In a real implementation, this would consider actual exit price.
   * @param _order
   */
  private calculateOrderPnL(_order: ManagedOrder): number {
    // TODO: Get actual market price for proper P&L calculation
    // For now, return 0 to indicate break-even
    return 0
  }

  /**
   * Get execution monitor statistics.
   * 
   * @param timeWindowMs - Time window for statistics (default: 24 hours)
   * @returns Execution statistics from the monitor
   * 
   * @example
   * ```typescript
   * const stats = manager.getExecutionStatistics()
   * console.log(`Success rate: ${stats.successRate.toFixed(1)}%`)
   * console.log(`Average slippage: ${stats.avgSlippage.toFixed(3)}%`)
   * ```
   */
  getExecutionStatistics(timeWindowMs?: number): ReturnType<typeof this.executionMonitor.getExecutionStatistics> {
    return this.executionMonitor.getExecutionStatistics(timeWindowMs)
  }

  /**
   * Get adaptive order type recommendation.
   * 
   * @param orderSize - Size of the order
   * @param side - Buy or sell
   * @returns Recommendation from the execution monitor
   * 
   * @example
   * ```typescript
   * const recommendation = manager.getOrderTypeRecommendation(1000, 'buy')
   * console.log(`Recommended: ${recommendation.orderType} (${recommendation.confidence} confidence)`)
   * console.log(`Reasoning: ${recommendation.reasoning}`)
   * ```
   */
  getOrderTypeRecommendation(orderSize: number, side: 'buy' | 'sell'): ReturnType<typeof this.executionMonitor.getAdaptiveOrderTypeRecommendation> {
    return this.executionMonitor.getAdaptiveOrderTypeRecommendation(orderSize, side)
  }

  /**
   * Check if circuit breaker is active.
   * 
   * @returns true if circuit breaker has tripped and trading is suspended
   */
  isCircuitBreakerActive(): boolean {
    return this.executionMonitor.isCircuitBreakerActive()
  }

  /**
   * Gather consensus from trading agents.
   * 
   * @param symbol - Trading symbol
   * @param expectedAgents - List of agent IDs expected to respond
   * @returns OrderAgentConsensus if action should be taken, null for hold
   * 
   * @example
   * ```typescript
   * const agents = ['volatility-agent', 'momentum-agent', 'volume-agent']
   * const consensus = await manager.gatherAgentConsensus('BTC-USD', agents)
   * if (consensus) {
   *   const order = await manager.processAgentConsensus(consensus)
   * }
   * ```
   */
  async gatherAgentConsensus(
    symbol: string,
    expectedAgents: string[]
  ): Promise<OrderAgentConsensus | null> {
    // Create signal request
    const request: SignalRequest = {
      requestId: `req_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`,
      symbol,
      currentPrice: this.getCurrentPrice(symbol),
      timestamp: new Date()
    }

    // Gather consensus from agents
    const result = await this.consensusManager.gatherConsensus(request, expectedAgents)

    // Update agent performance if we have historical data
    // This would be called later when we know if the prediction was correct
    // For now, just convert to OrderAgentConsensus format
    
    return ConsensusManager.toOrderAgentConsensus(result, symbol)
  }

  /**
   * Update agent performance after order completion.
   * 
   * @param orderId - Completed order ID
   * @param profitable - Whether the order was profitable
   * 
   * @remarks
   * Updates agent weights based on prediction accuracy and P&L contribution.
   * Should be called after order is closed to improve future consensus quality.
   */
  updateAgentPerformance(orderId: string, profitable: boolean): void {
    const order = this.activeOrders.get(orderId)
    if (!order?.metadata?.consensus) return

    const consensus = order.metadata.consensus
    const pnl = this.calculateOrderPnL(order)

    // Update performance for all agents that contributed
    for (const signal of consensus.agentSignals) {
      this.consensusManager.updateAgentPerformance(
        signal.agentId,
        profitable,
        signal.confidence,
        pnl * (signal.weight / consensus.agentSignals.reduce((sum, s) => sum + s.weight, 0))
      )
    }
  }

  /**
   * Get consensus manager configuration.
   */
  getConsensusConfig(): ReturnType<typeof this.consensusManager.getConfig> {
    return this.consensusManager.getConfig()
  }

  /**
   * Get agent performance metrics.
   */
  getAgentPerformanceMetrics(): ReturnType<typeof this.consensusManager.getAgentPerformanceMetrics> {
    return this.consensusManager.getAgentPerformanceMetrics()
  }
}