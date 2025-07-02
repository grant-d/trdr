import type {
  ManagedOrder,
  OrderExecutionMetrics,
  CircuitBreakerConfig,
  OrderFill
} from '@trdr/shared'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

/**
 * Monitors order execution quality and manages circuit breakers.
 * 
 * Tracks execution metrics, detects adverse conditions, and provides
 * reporting capabilities for order performance analysis.
 */
export class OrderExecutionMonitor {
  private readonly eventBus: EventBus
  private readonly circuitBreakerConfig?: CircuitBreakerConfig
  
  // Circuit breaker state
  private circuitBreakerTripped = false
  private circuitBreakerTripTime?: Date
  private consecutiveFailures = 0
  private recentLosses: Array<{ amount: number; timestamp: Date }> = []
  
  // Execution history for adaptive algorithms
  private executionHistory: Map<string, OrderExecutionMetrics> = new Map()
  
  /**
   * Creates a new OrderExecutionMonitor instance.
   * 
   * @param circuitBreakerConfig - Optional circuit breaker configuration
   * @param eventBus - Optional event bus instance
   */
  constructor(
    circuitBreakerConfig?: CircuitBreakerConfig,
    eventBus?: EventBus
  ) {
    this.circuitBreakerConfig = circuitBreakerConfig
    this.eventBus = eventBus || EventBus.getInstance()
    this.registerEventTypes()
  }
  
  /**
   * Register monitoring-related event types.
   */
  private registerEventTypes(): void {
    this.eventBus.registerEvent(EventTypes.ORDER_EXECUTION_METRICS)
    this.eventBus.registerEvent(EventTypes.CIRCUIT_BREAKER_TRIPPED)
    this.eventBus.registerEvent(EventTypes.CIRCUIT_BREAKER_RESET)
    this.eventBus.registerEvent(EventTypes.ORDER_EXECUTION_POOR)
  }
  
  /**
   * Start monitoring an order's execution.
   * 
   * @param order - Order to monitor
   */
  startMonitoring(order: ManagedOrder): void {
    // Initialize execution metrics
    const metrics: OrderExecutionMetrics = {
      fillRate: 0,
      fillCount: 0,
      submittedAt: new Date()
    }
    
    // Store initial metrics
    this.executionHistory.set(order.id, metrics)
    
    // Update order with metrics
    order.executionMetrics = metrics
  }
  
  /**
   * Update execution metrics when order is filled.
   *
   * @param order - Order being filled
   * @param _fill
   */
  updateFillMetrics(order: ManagedOrder, _fill: OrderFill): void {
    const metrics = this.executionHistory.get(order.id)
    if (!metrics) return
    
    // Update fill count and rate
    metrics.fillCount++
    metrics.fillRate = order.filledSize / order.size
    
    // Calculate time to first fill
    if (!metrics.timeToFirstFill && metrics.submittedAt) {
      metrics.timeToFirstFill = Date.now() - metrics.submittedAt.getTime()
    }
    
    // Calculate slippage for limit orders
    if ('price' in order && order.price) {
      const expectedPrice = order.price
      const actualPrice = order.averageFillPrice
      
      // For buy orders, positive slippage means paying more
      // For sell orders, positive slippage means receiving less
      const slippageMultiplier = order.side === 'buy' ? 1 : -1
      metrics.slippageAmount = (actualPrice - expectedPrice) * order.filledSize * slippageMultiplier
      metrics.slippagePercent = ((actualPrice - expectedPrice) / expectedPrice) * 100 * slippageMultiplier
    }
    
    // Check if order is completely filled
    if (order.filledSize >= order.size && metrics.submittedAt) {
      metrics.timeToComplete = Date.now() - metrics.submittedAt.getTime()
      metrics.completedAt = new Date()
      
      // Emit execution metrics event
      this.eventBus.emit(EventTypes.ORDER_EXECUTION_METRICS, {
        orderId: order.id,
        metrics,
        timestamp: new Date()
      })
      
      // Check execution quality
      this.checkExecutionQuality(order, metrics)
    }
    
    // Update order metrics
    order.executionMetrics = metrics
  }
  
  /**
   * Check if circuit breaker should trip based on order result.
   * 
   * @param order - Completed order
   * @param success - Whether order was successful
   * @param loss - Loss amount if applicable
   */
  checkCircuitBreaker(order: ManagedOrder, success: boolean, loss?: number): void {
    if (!this.circuitBreakerConfig || this.circuitBreakerTripped) return
    
    const config = this.circuitBreakerConfig
    
    // Check consecutive failures
    if (!success) {
      this.consecutiveFailures++
      if (this.consecutiveFailures >= config.maxConsecutiveFailures) {
        this.tripCircuitBreaker('Max consecutive failures reached')
        return
      }
    } else {
      this.consecutiveFailures = 0
    }
    
    // Track losses
    if (loss && loss > 0) {
      this.recentLosses.push({ amount: loss, timestamp: new Date() })
      
      // Remove old losses outside window
      const cutoffTime = Date.now() - config.lossWindowMs
      this.recentLosses = this.recentLosses.filter(
        l => l.timestamp.getTime() > cutoffTime
      )
      
      // Calculate total recent loss
      const totalLoss = this.recentLosses.reduce((sum, l) => sum + l.amount, 0)
      if (totalLoss >= config.maxLossThreshold) {
        this.tripCircuitBreaker(`Loss threshold exceeded: $${totalLoss}`)
        return
      }
    }
    
    // Check slippage
    const metrics = order.executionMetrics
    if (metrics?.slippagePercent && metrics.slippagePercent > config.maxSlippagePercent) {
      this.tripCircuitBreaker(`Excessive slippage: ${metrics.slippagePercent.toFixed(2)}%`)
      return
    }
    
    // Check fill rate
    if (metrics?.fillRate && metrics.fillRate < config.minFillRate) {
      this.tripCircuitBreaker(`Poor fill rate: ${(metrics.fillRate * 100).toFixed(1)}%`)
    }
  }
  
  /**
   * Trip the circuit breaker.
   * 
   * @param reason - Reason for tripping
   */
  private tripCircuitBreaker(reason: string): void {
    this.circuitBreakerTripped = true
    this.circuitBreakerTripTime = new Date()
    
    this.eventBus.emit(EventTypes.CIRCUIT_BREAKER_TRIPPED, {
      reason,
      timestamp: new Date()
    })
  }
  
  /**
   * Check if circuit breaker should be reset.
   * 
   * @returns true if circuit breaker is active (tripped)
   */
  isCircuitBreakerActive(): boolean {
    if (!this.circuitBreakerTripped || !this.circuitBreakerTripTime) {
      return false
    }
    
    // Check if cooldown period has passed
    const cooldownExpired = 
      Date.now() - this.circuitBreakerTripTime.getTime() > 
      (this.circuitBreakerConfig?.cooldownPeriodMs || 0)
    
    if (cooldownExpired) {
      this.resetCircuitBreaker()
      return false
    }
    
    return true
  }
  
  /**
   * Reset the circuit breaker.
   */
  private resetCircuitBreaker(): void {
    this.circuitBreakerTripped = false
    this.circuitBreakerTripTime = undefined
    this.consecutiveFailures = 0
    
    this.eventBus.emit(EventTypes.CIRCUIT_BREAKER_RESET, {
      timestamp: new Date()
    })
  }
  
  /**
   * Check execution quality and emit warnings.
   * 
   * @param order - Completed order
   * @param metrics - Execution metrics
   */
  private checkExecutionQuality(order: ManagedOrder, metrics: OrderExecutionMetrics): void {
    const warnings: string[] = []
    
    // Check for poor execution time
    if (metrics.timeToComplete && metrics.timeToComplete > 60000) { // 1 minute
      warnings.push(`Slow execution: ${(metrics.timeToComplete / 1000).toFixed(1)}s`)
    }
    
    // Check for excessive slippage
    if (metrics.slippagePercent && Math.abs(metrics.slippagePercent) > 0.5) {
      warnings.push(`High slippage: ${metrics.slippagePercent.toFixed(2)}%`)
    }
    
    // Check for many partial fills
    if (metrics.fillCount > 5) {
      warnings.push(`Fragmented execution: ${metrics.fillCount} fills`)
    }
    
    if (warnings.length > 0) {
      this.eventBus.emit(EventTypes.ORDER_EXECUTION_POOR, {
        orderId: order.id,
        warnings,
        metrics,
        timestamp: new Date()
      })
    }
  }
  
  /**
   * Get execution statistics for reporting.
   * 
   * @param timeWindowMs - Time window for statistics (default: 24 hours)
   * @returns Execution statistics
   */
  getExecutionStatistics(timeWindowMs = 86400000): {
    totalOrders: number
    avgTimeToFill: number
    avgSlippage: number
    fillRateDistribution: Record<string, number>
    successRate: number
  } {
    const cutoffTime = Date.now() - timeWindowMs
    const recentMetrics = Array.from(this.executionHistory.values())
      .filter(m => m.submittedAt && m.submittedAt.getTime() > cutoffTime)
    
    if (recentMetrics.length === 0) {
      return {
        totalOrders: 0,
        avgTimeToFill: 0,
        avgSlippage: 0,
        fillRateDistribution: {},
        successRate: 0
      }
    }
    
    // Calculate averages
    const completedOrders = recentMetrics.filter(m => m.timeToComplete)
    const avgTimeToFill = completedOrders.length > 0
      ? completedOrders.reduce((sum, m) => sum + (m.timeToComplete || 0), 0) / completedOrders.length
      : 0
    
    const ordersWithSlippage = recentMetrics.filter(m => m.slippagePercent !== undefined)
    const avgSlippage = ordersWithSlippage.length > 0
      ? ordersWithSlippage.reduce((sum, m) => sum + (m.slippagePercent || 0), 0) / ordersWithSlippage.length
      : 0
    
    // Fill rate distribution
    const fillRateDistribution: Record<string, number> = {
      '0-25%': 0,
      '25-50%': 0,
      '50-75%': 0,
      '75-99%': 0,
      '100%': 0
    }
    
    recentMetrics.forEach(m => {
      const rate = m.fillRate * 100
      if (rate === 100) {
        fillRateDistribution['100%'] = (fillRateDistribution['100%'] || 0) + 1
      } else if (rate >= 75) {
        fillRateDistribution['75-99%'] = (fillRateDistribution['75-99%'] || 0) + 1
      } else if (rate >= 50) {
        fillRateDistribution['50-75%'] = (fillRateDistribution['50-75%'] || 0) + 1
      } else if (rate >= 25) {
        fillRateDistribution['25-50%'] = (fillRateDistribution['25-50%'] || 0) + 1
      } else {
        fillRateDistribution['0-25%'] = (fillRateDistribution['0-25%'] || 0) + 1
      }
    })
    
    const successRate = (completedOrders.length / recentMetrics.length) * 100
    
    return {
      totalOrders: recentMetrics.length,
      avgTimeToFill,
      avgSlippage,
      fillRateDistribution,
      successRate
    }
  }
  
  /**
   * Get recommendations for order type based on historical performance.
   * 
   * @param orderSize - Size of the order
   * @param _side - Buy or sell (reserved for future use)
   * @returns Recommended order type and parameters
   */
  getAdaptiveOrderTypeRecommendation(
    orderSize: number,
    _side: 'buy' | 'sell'
  ): {
    orderType: 'market' | 'limit' | 'trailing'
    confidence: number
    reasoning: string
  } {
    const stats = this.getExecutionStatistics()
    
    // Default to trailing orders
    let orderType: 'market' | 'limit' | 'trailing' = 'trailing'
    let confidence = 0.7
    let reasoning = 'Default trailing order strategy'
    
    // If high slippage, prefer limit orders
    if (Math.abs(stats.avgSlippage) > 0.3) {
      orderType = 'limit'
      confidence = 0.8
      reasoning = `High average slippage (${stats.avgSlippage.toFixed(2)}%), limit order recommended`
    }
    
    // If slow fills, consider market orders for small sizes
    if (stats.avgTimeToFill > 30000 && orderSize < 1000) { // 30 seconds
      orderType = 'market'
      confidence = 0.75
      reasoning = 'Slow fill times for small orders, market order recommended'
    }
    
    // If poor fill rates, stick with trailing
    const poorFillRate = (stats.fillRateDistribution['0-25%'] || 0) + 
                        (stats.fillRateDistribution['25-50%'] || 0)
    if (poorFillRate > stats.totalOrders * 0.3) {
      orderType = 'trailing'
      confidence = 0.85
      reasoning = 'Poor fill rates detected, trailing order provides flexibility'
    }
    
    return { orderType, confidence, reasoning }
  }
  
  /**
   * Clear old execution history.
   * 
   * @param olderThanMs - Clear entries older than this (default: 7 days)
   */
  clearOldHistory(olderThanMs = 604800000): void {
    const cutoffTime = Date.now() - olderThanMs
    
    for (const [orderId, metrics] of this.executionHistory.entries()) {
      if (metrics.submittedAt && metrics.submittedAt.getTime() < cutoffTime) {
        this.executionHistory.delete(orderId)
      }
    }
  }
}