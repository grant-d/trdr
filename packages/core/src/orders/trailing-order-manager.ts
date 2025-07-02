import type {
  ManagedOrder,
  OrderEvent,
  OrderExecutionMetrics,
  OrderSide,
  StockSymbol,
  TrailingOrder
} from '@trdr/shared'
import { EnhancedOrderState, epochDateNow, type EpochDate } from '@trdr/shared'
import type { Logger } from '@trdr/types'
import { EventEmitter } from 'events'
import { v4 as uuidv4 } from 'uuid'
import type { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

/**
 * Parameters for creating a trailing order
 */
export interface TrailingOrderParams {
  readonly symbol: StockSymbol
  readonly side: OrderSide
  readonly size: number
  readonly trailPercent: number
  readonly currentPrice: number
  readonly limitPrice?: number
  readonly activationPrice?: number
}

/**
 * Internal representation of a trailing order with additional tracking
 */
interface InternalTrailingOrder extends ManagedOrder {
  readonly type: 'trailing'
  bestPrice: number
  triggerPrice: number
  lastUpdateTime: EpochDate
  isTriggered: boolean
  trailPercent: number
}

/**
 * Configuration for the trailing order manager
 */
export interface TrailingOrderManagerConfig {
  readonly minTrailPercent?: number
  readonly maxTrailPercent?: number
  readonly updateThrottleMs?: number
  readonly persistenceEnabled?: boolean
}

/**
 * TrailingOrderManager handles the creation, tracking, and execution of trailing orders.
 * 
 * Features:
 * - Dynamic trigger price adjustment based on market movements
 * - Separate trailing logic for buy and sell orders
 * - Price update throttling to prevent excessive recalculations
 * - Persistent storage of order state
 * - Event-driven architecture for order lifecycle
 */
export class TrailingOrderManager extends EventEmitter {
  private readonly activeOrders = new Map<string, InternalTrailingOrder>()
  private readonly eventBus: EventBus
  private readonly config: Required<TrailingOrderManagerConfig>
  private readonly lastPriceUpdate = new Map<string, EpochDate>()
  private readonly logger?: Logger

  constructor(
    eventBus: EventBus,
    config: TrailingOrderManagerConfig = {},
    logger?: Logger
  ) {
    super()
    this.eventBus = eventBus
    this.logger = logger
    
    // Apply default configuration
    this.config = {
      minTrailPercent: config.minTrailPercent ?? 0.1,
      maxTrailPercent: config.maxTrailPercent ?? 10,
      updateThrottleMs: config.updateThrottleMs ?? 100,
      persistenceEnabled: config.persistenceEnabled ?? true
    }

    this.setupEventHandlers()
  }

  /**
   * Creates a new trailing order
   */
  async createTrailingOrder(params: TrailingOrderParams): Promise<TrailingOrder> {
    // Validate parameters
    this.validateTrailingOrderParams(params)

    const orderId = uuidv4()
    const now = epochDateNow()

    const order: InternalTrailingOrder = {
      id: orderId,
      symbol: params.symbol,
      side: params.side,
      type: 'trailing',
      size: params.size,
      createdAt: now,
      updatedAt: now,
      status: 'pending',
      state: EnhancedOrderState.CREATED,
      
      // Trailing-specific properties
      trailPercent: params.trailPercent,
      limitPrice: params.limitPrice,
      activationPrice: params.activationPrice,
      bestPrice: params.currentPrice,
      triggerPrice: this.calculateInitialTrigger(params),
      highWaterMark: params.side === 'sell' ? params.currentPrice : undefined,
      lowWaterMark: params.side === 'buy' ? params.currentPrice : undefined,
      
      // Execution tracking
      filledSize: 0,
      averageFillPrice: 0,
      fees: 0,
      lastModified: now,
      fills: [],
      lastUpdateTime: now,
      isTriggered: false
    }

    // Store the order
    this.activeOrders.set(orderId, order)

    // Emit creation event
    const orderEvent: OrderEvent = { type: 'created', order }
    this.eventBus.emit(EventTypes.ORDER_CREATED, { ...orderEvent, timestamp: now })
    this.emit('orderCreated', order)

    // Log creation
    this.logger?.info('Trailing order created', {
      orderId,
      symbol: params.symbol,
      side: params.side,
      size: params.size,
      trailPercent: params.trailPercent,
      triggerPrice: order.triggerPrice
    })

    // Persist if enabled
    if (this.config.persistenceEnabled) {
      await this.persistOrder(order)
    }

    return order
  }

  /**
   * Updates trailing orders based on new market price
   */
  async processMarketUpdate(symbol: string, price: number): Promise<void> {
    // Throttle updates per symbol
    const lastUpdate = this.lastPriceUpdate.get(symbol)
    const now = epochDateNow()
    
    if (lastUpdate && (now - lastUpdate) < this.config.updateThrottleMs) {
      return
    }
    
    this.lastPriceUpdate.set(symbol, now)

    // Process all active orders for this symbol
    const ordersToProcess = Array.from(this.activeOrders.values())
      .filter(order => order.symbol === symbol && order.status === 'pending' && !order.isTriggered)

    for (const order of ordersToProcess) {
      await this.updateTrailingOrder(order, price)
    }
  }

  /**
   * Removes a trailing order
   */
  async removeOrder(orderId: string, reason?: string): Promise<void> {
    const order = this.activeOrders.get(orderId)
    if (!order) {
      this.logger?.warn('Attempted to remove non-existent order', { orderId })
      return
    }

    // Update order state
    order.status = 'cancelled'
    order.state = EnhancedOrderState.CANCELLED
    order.cancellationReason = reason
    order.updatedAt = epochDateNow()

    // Remove from active orders
    this.activeOrders.delete(orderId)

    // Emit cancellation event
    const orderEvent: OrderEvent = { type: 'cancelled', order, reason }
    this.eventBus.emit(EventTypes.ORDER_CANCELLED, { ...orderEvent, timestamp: epochDateNow() })
    this.emit('orderCancelled', order)

    this.logger?.info('Trailing order removed', { orderId, reason })

    // Persist final state if enabled
    if (this.config.persistenceEnabled) {
      await this.persistOrder(order)
    }
  }

  /**
   * Gets all active trailing orders
   */
  getActiveOrders(): readonly TrailingOrder[] {
    return Array.from(this.activeOrders.values())
      .filter(order => order.status === 'pending')
  }

  /**
   * Gets a specific order by ID
   */
  getOrder(orderId: string): TrailingOrder | undefined {
    return this.activeOrders.get(orderId)
  }

  /**
   * Gets execution metrics for an order
   */
  getOrderMetrics(orderId: string): OrderExecutionMetrics | undefined {
    const order = this.activeOrders.get(orderId)
    return order?.executionMetrics
  }

  /**
   * Calculates the initial trigger price for a trailing order
   */
  private calculateInitialTrigger(params: TrailingOrderParams): number {
    const { currentPrice, side, trailPercent } = params

    if (side === 'buy') {
      // For buy orders: trigger when price rises X% from the lowest point
      return currentPrice * (1 + trailPercent / 100)
    } else {
      // For sell orders: trigger when price falls X% from the highest point
      return currentPrice * (1 - trailPercent / 100)
    }
  }

  /**
   * Updates a trailing order based on new price
   */
  private async updateTrailingOrder(order: InternalTrailingOrder, price: number): Promise<void> {
    const previousBestPrice = order.bestPrice
    const previousTriggerPrice = order.triggerPrice

    // Update best price and trigger based on order side
    if (order.side === 'sell' && price > order.bestPrice) {
      // Sell order: track highest price
      order.bestPrice = price
      order.highWaterMark = price
      order.triggerPrice = price * (1 - order.trailPercent / 100)
    } else if (order.side === 'buy' && price < order.bestPrice) {
      // Buy order: track lowest price
      order.bestPrice = price
      order.lowWaterMark = price
      order.triggerPrice = price * (1 + order.trailPercent / 100)
    }

    // Update timestamp if price changed
    if (order.bestPrice !== previousBestPrice) {
      order.lastUpdateTime = epochDateNow()
      order.updatedAt = order.lastUpdateTime

      this.logger?.debug('Trailing order updated', {
        orderId: order.id,
        side: order.side,
        previousBest: previousBestPrice,
        newBest: order.bestPrice,
        previousTrigger: previousTriggerPrice,
        newTrigger: order.triggerPrice
      })
    }

    // Check if order should be triggered
    if (this.shouldTrigger(order, price)) {
      await this.triggerOrder(order, price)
    }
  }

  /**
   * Determines if a trailing order should be triggered
   */
  private shouldTrigger(order: InternalTrailingOrder, price: number): boolean {
    // Check activation price if set
    if (order.activationPrice) {
      if (order.side === 'buy' && price > order.activationPrice) {
        return false
      }
      if (order.side === 'sell' && price < order.activationPrice) {
        return false
      }
    }

    // Check trigger conditions
    if (order.side === 'sell') {
      // Sell order triggers when price falls below trigger price
      return price <= order.triggerPrice
    } else {
      // Buy order triggers when price rises above trigger price
      return price >= order.triggerPrice
    }
  }

  /**
   * Executes a triggered trailing order
   */
  private async triggerOrder(order: InternalTrailingOrder, triggerPrice: number): Promise<void> {
    order.isTriggered = true
    order.status = 'open'
    order.state = EnhancedOrderState.SUBMITTED
    order.updatedAt = epochDateNow()

    // Record execution start time
    if (!order.executionMetrics) {
      order.executionMetrics = {
        fillRate: 0,
        fillCount: 0,
        submittedAt: epochDateNow()
      }
    }

    this.logger?.info('Trailing order triggered', {
      orderId: order.id,
      symbol: order.symbol,
      side: order.side,
      size: order.size,
      triggerPrice,
      bestPrice: order.bestPrice,
      trailPercent: order.trailPercent
    })

    // Emit submission event
    const orderEvent: OrderEvent = { type: 'submitted', order }
    this.eventBus.emit(EventTypes.ORDER_SUBMITTED, { ...orderEvent, timestamp: epochDateNow() })
    this.emit('orderTriggered', order)

    // Persist state if enabled
    if (this.config.persistenceEnabled) {
      await this.persistOrder(order)
    }
  }

  /**
   * Validates trailing order parameters
   */
  private validateTrailingOrderParams(params: TrailingOrderParams): void {
    if (params.size <= 0) {
      throw new Error('Order size must be positive')
    }

    if (params.trailPercent < this.config.minTrailPercent) {
      throw new Error(`Trail percent must be at least ${this.config.minTrailPercent}%`)
    }

    if (params.trailPercent > this.config.maxTrailPercent) {
      throw new Error(`Trail percent must not exceed ${this.config.maxTrailPercent}%`)
    }

    if (params.currentPrice <= 0) {
      throw new Error('Current price must be positive')
    }

    if (params.limitPrice && params.limitPrice <= 0) {
      throw new Error('Limit price must be positive if specified')
    }

    if (params.activationPrice && params.activationPrice <= 0) {
      throw new Error('Activation price must be positive if specified')
    }
  }

  /**
   * Sets up event handlers for order lifecycle
   */
  private setupEventHandlers(): void {
    // Handle order fills
    this.eventBus.subscribe(EventTypes.ORDER_FILLED, (event: any) => {
      if (event.type === 'filled' || event.type === 'partial_fill') {
        const order = this.activeOrders.get(event.order.id)
        if (order && event.fill) {
          this.handleOrderFill(order, event.fill)
        }
      }
    })

    // Handle order rejections
    this.eventBus.subscribe(EventTypes.ORDER_REJECTED, (event: any) => {
      if (event.type === 'rejected') {
        const order = this.activeOrders.get(event.order.id)
        if (order) {
          order.status = 'rejected'
          order.state = EnhancedOrderState.REJECTED
          order.rejectionReason = event.reason
          order.updatedAt = epochDateNow()
          this.activeOrders.delete(order.id)
        }
      }
    })
  }

  /**
   * Handles order fill updates
   */
  private handleOrderFill(order: InternalTrailingOrder, fill: any): void {
    const previousFilledSize = order.filledSize
    order.filledSize += fill.size
    order.fills.push(fill)
    
    // Update average fill price
    order.averageFillPrice = previousFilledSize > 0
      ? (order.averageFillPrice * previousFilledSize + fill.price * fill.size) / order.filledSize
      : fill.price
    
    // Update fees
    order.fees += fill.fee || 0
    
    // Update execution metrics
    if (order.executionMetrics) {
      order.executionMetrics.fillCount += 1
      order.executionMetrics.fillRate = order.filledSize / order.size
      
      if (!order.executionMetrics.timeToFirstFill && previousFilledSize === 0) {
        order.executionMetrics.timeToFirstFill = epochDateNow() - (order.executionMetrics.submittedAt || order.createdAt)
      }
      
      if (order.filledSize >= order.size) {
        order.executionMetrics.timeToComplete = epochDateNow() - (order.executionMetrics.submittedAt || order.createdAt)
        order.executionMetrics.completedAt = epochDateNow()
      }
    }
    
    // Update order status
    if (order.filledSize >= order.size) {
      order.status = 'filled'
      order.state = EnhancedOrderState.FILLED
      this.activeOrders.delete(order.id)
    } else {
      order.status = 'partial'
      order.state = EnhancedOrderState.PARTIALLY_FILLED
    }
    
    order.updatedAt = epochDateNow()
  }

  /**
   * Persists order state (placeholder for actual implementation)
   */
  private async persistOrder(order: InternalTrailingOrder): Promise<void> {
    // TODO: Implement actual persistence logic
    // This would typically save to a database or file system
    this.logger?.debug('Persisting order state', { orderId: order.id })
  }

  /**
   * Restores orders from persistent storage
   */
  async restoreOrders(): Promise<void> {
    if (!this.config.persistenceEnabled) {
      return
    }

    // TODO: Implement actual restoration logic
    this.logger?.info('Restoring trailing orders from storage')
  }

  /**
   * Cleans up inactive orders
   */
  async cleanup(): Promise<void> {
    const ordersToRemove: string[] = []

    for (const [orderId, order] of this.activeOrders.entries()) {
      // Remove completed, cancelled, or rejected orders
      if (['filled', 'cancelled', 'rejected'].includes(order.status)) {
        ordersToRemove.push(orderId)
      }
    }

    for (const orderId of ordersToRemove) {
      this.activeOrders.delete(orderId)
    }

    if (ordersToRemove.length > 0) {
      this.logger?.debug('Cleaned up inactive orders', { count: ordersToRemove.length })
    }
  }
}