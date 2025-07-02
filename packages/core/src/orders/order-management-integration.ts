import type { ManagedOrder, OrderAgentConsensus, TrailingOrder } from '@trdr/shared'
import { epochDateNow } from '@trdr/shared'
import type { Logger } from '@trdr/types'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import { OrderLifecycleManager } from './order-lifecycle-manager'
import { TrailingOrderManager, type TrailingOrderParams } from './trailing-order-manager'

/**
 * Integration configuration for order management system
 */
export interface OrderManagementIntegrationConfig {
  readonly lifecycleConfig: ConstructorParameters<typeof OrderLifecycleManager>[0]
  readonly trailingOrderConfig?: ConstructorParameters<typeof TrailingOrderManager>[1]
  readonly enableAutoSubmission?: boolean
  readonly logger?: Logger
}

/**
 * OrderManagementIntegration provides seamless integration between
 * TrailingOrderManager and OrderLifecycleManager.
 * 
 * This class acts as a facade that:
 * - Routes agent consensus to create trailing orders
 * - Manages order submission and lifecycle
 * - Handles market data updates for trailing orders
 * - Synchronizes order state between managers
 */
export class OrderManagementIntegration {
  private readonly lifecycleManager: OrderLifecycleManager
  private readonly trailingOrderManager: TrailingOrderManager
  private readonly eventBus: EventBus
  private readonly logger?: Logger

  constructor(config: OrderManagementIntegrationConfig) {
    this.logger = config.logger
    this.eventBus = EventBus.getInstance()
    
    // Initialize managers
    this.lifecycleManager = new OrderLifecycleManager(config.lifecycleConfig, this.eventBus)
    this.trailingOrderManager = new TrailingOrderManager(
      this.eventBus,
      config.trailingOrderConfig,
      config.logger
    )
    
    this.setupEventHandlers()
    this.setupMarketDataRouting()
  }

  /**
   * Process agent consensus to create and manage orders
   */
  async processAgentConsensus(consensus: OrderAgentConsensus): Promise<ManagedOrder | null> {
    // First, let lifecycle manager validate consensus and calculate size
    const validatedOrder = this.lifecycleManager.processAgentConsensus(consensus)
    if (!validatedOrder) {
      return null
    }

    // Create trailing order parameters from validated order
    const currentPrice = this.getCurrentPrice(validatedOrder.symbol)
    const trailingParams: TrailingOrderParams = {
      symbol: validatedOrder.symbol,
      side: validatedOrder.side,
      size: validatedOrder.size,
      trailPercent: validatedOrder.trailPercent || consensus.trailDistance,
      currentPrice,
      limitPrice: validatedOrder.limitPrice,
      activationPrice: validatedOrder.activationPrice
    }

    try {
      // Create trailing order
      const trailingOrder = await this.trailingOrderManager.createTrailingOrder(trailingParams)
      
      // Submit to lifecycle manager for execution management
      const managedOrder = this.lifecycleManager.submitOrder(trailingOrder)
      
      this.logger?.info('Created and submitted trailing order', {
        orderId: managedOrder.id,
        symbol: managedOrder.symbol,
        side: managedOrder.side,
        size: managedOrder.size,
        trailPercent: trailingParams.trailPercent
      })
      
      return managedOrder
    } catch (error) {
      this.logger?.error('Failed to create trailing order', {
        error: (error as Error).message,
        consensus
      })
      return null
    }
  }

  /**
   * Create trailing order directly without consensus
   */
  async createTrailingOrder(params: TrailingOrderParams): Promise<ManagedOrder> {
    const trailingOrder = await this.trailingOrderManager.createTrailingOrder(params)
    return this.lifecycleManager.submitOrder(trailingOrder)
  }

  /**
   * Process market data update
   */
  async processMarketUpdate(symbol: string, price: number): Promise<void> {
    // Update trailing orders
    await this.trailingOrderManager.processMarketUpdate(symbol, price)
    
    // Also emit market tick for lifecycle manager monitoring
    this.eventBus.emit(EventTypes.MARKET_TICK, {
      symbol,
      price,
      timestamp: epochDateNow()
    })
  }

  /**
   * Cancel an order
   */
  async cancelOrder(orderId: string, reason?: string): Promise<void> {
    // Cancel in both managers
    await this.trailingOrderManager.removeOrder(orderId, reason)
    this.lifecycleManager.cancelOrder(orderId, reason)
  }

  /**
   * Get order by ID
   */
  getOrder(orderId: string): ManagedOrder | undefined {
    // Check lifecycle manager first (source of truth for managed orders)
    const managedOrder = this.lifecycleManager.getOrder(orderId)
    if (managedOrder) {
      return managedOrder
    }
    
    // Fall back to trailing order manager
    const trailingOrder = this.trailingOrderManager.getOrder(orderId)
    if (trailingOrder) {
      // Convert to managed order format
      return {
        ...trailingOrder,
        state: 'pending' as any, // Will be properly managed by lifecycle
        filledSize: 0,
        averageFillPrice: 0,
        fees: 0,
        lastModified: epochDateNow(),
        fills: []
      }
    }
    
    return undefined
  }

  /**
   * Get all active orders
   */
  getActiveOrders(): readonly ManagedOrder[] {
    const lifecycleOrders = Array.from(this.lifecycleManager.getActiveOrders().values())
    return lifecycleOrders
  }

  /**
   * Get execution statistics
   */
  getExecutionStatistics(timeWindowMs?: number) {
    return this.lifecycleManager.getExecutionStatistics(timeWindowMs)
  }

  /**
   * Check if circuit breaker is active
   */
  isCircuitBreakerActive(): boolean {
    return this.lifecycleManager.isCircuitBreakerActive()
  }

  /**
   * Gather agent consensus
   */
  async gatherAgentConsensus(symbol: string, expectedAgents: string[]): Promise<OrderAgentConsensus | null> {
    return this.lifecycleManager.gatherAgentConsensus(symbol, expectedAgents)
  }

  /**
   * Setup event handlers for order synchronization
   */
  private setupEventHandlers(): void {
    // Handle trailing order triggers
    this.trailingOrderManager.on('orderTriggered', (order: TrailingOrder) => {
      this.handleTrailingOrderTriggered(order)
    })
    
    // Handle order state changes from lifecycle manager
    this.eventBus.subscribe(EventTypes.ORDER_STATE_CHANGED, (data: any) => {
      this.handleOrderStateChange(data)
    })
    
    // Handle order fills
    this.eventBus.subscribe(EventTypes.ORDER_FILLED, (data: any) => {
      this.handleOrderFilled(data)
    })
    
    this.eventBus.subscribe(EventTypes.ORDER_PARTIAL_FILL, (data: any) => {
      this.handleOrderFilled(data)
    })
    
    // Handle order rejections
    this.eventBus.subscribe(EventTypes.ORDER_REJECTED, (data: any) => {
      this.handleOrderRejected(data)
    })
  }

  /**
   * Setup market data routing to trailing order manager
   */
  private setupMarketDataRouting(): void {
    // Subscribe to market data events and route to trailing order manager
    this.eventBus.subscribe(EventTypes.MARKET_TICK, async (data: any) => {
      if (data.symbol && typeof data.price === 'number') {
        await this.trailingOrderManager.processMarketUpdate(data.symbol, data.price)
      }
    })
  }

  /**
   * Handle when a trailing order is triggered
   */
  private handleTrailingOrderTriggered(order: TrailingOrder): void {
    this.logger?.info('Trailing order triggered', {
      orderId: order.id,
      symbol: order.symbol,
      side: order.side,
      triggerPrice: (order as any).triggerPrice
    })
    
    // The order state will be updated by the lifecycle manager
    // through its own event handlers
  }

  /**
   * Handle order state changes
   */
  private handleOrderStateChange(data: any): void {
    if (!data.orderId || !data.newState) return
    
    // If order reaches terminal state, clean up from trailing manager
    const terminalStates = ['filled', 'cancelled', 'rejected', 'expired']
    if (terminalStates.includes(data.newState)) {
      // Let trailing manager clean up its internal state
      this.trailingOrderManager.cleanup().catch(error => {
        this.logger?.error('Failed to cleanup trailing orders', { error })
      })
    }
  }

  /**
   * Handle order fills
   */
  private handleOrderFilled(data: any): void {
    if (!data.orderId) return
    
    const order = this.trailingOrderManager.getOrder(data.orderId)
    if (order) {
      this.logger?.debug('Trailing order fill processed', {
        orderId: data.orderId,
        fillSize: data.fill?.size,
        fillPrice: data.fill?.price
      })
    }
  }

  /**
   * Handle order rejections
   */
  private handleOrderRejected(data: any): void {
    if (!data.orderId) return
    
    // Remove from trailing manager if it exists there
    this.trailingOrderManager.removeOrder(data.orderId, data.reason).catch(error => {
      this.logger?.error('Failed to remove rejected order from trailing manager', {
        orderId: data.orderId,
        error
      })
    })
  }

  /**
   * Get current price for a symbol
   */
  private getCurrentPrice(symbol: string): number {
    // TODO: Get from market data provider
    // For now, return a placeholder
    return symbol === 'BTC-USD' ? 50000 : 3000
  }

  /**
   * Restore orders from persistence
   */
  async restoreOrders(): Promise<void> {
    await this.trailingOrderManager.restoreOrders()
  }

  /**
   * Cleanup inactive orders
   */
  async cleanup(): Promise<void> {
    await this.trailingOrderManager.cleanup()
  }
}