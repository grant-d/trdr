import { describe, it, beforeEach, mock } from 'node:test'
import assert from 'node:assert/strict'
import { OrderManagementIntegration, type OrderManagementIntegrationConfig } from './order-management-integration'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import type { OrderAgentConsensus } from '@trdr/shared'
import { epochDateNow } from '@trdr/shared'

describe('OrderManagementIntegration', () => {
  let integration: OrderManagementIntegration
  let eventBus: EventBus
  let config: OrderManagementIntegrationConfig

  beforeEach(() => {
    eventBus = EventBus.getInstance()
    
    // Register all required events
    Object.values(EventTypes).forEach(eventType => {
      eventBus.registerEvent(eventType)
    })
    
    config = {
      lifecycleConfig: {
        symbol: 'BTC-USD',
        minOrderSize: 10,
        maxOrderSize: 10000,
        baseSizePercent: 2,
        maxPositionSize: 50000,
        defaultTimeInForce: 'GTC',
        enableOrderImprovement: true,
        improvementThreshold: 0.001,
        maxOrderDuration: 86400000, // 24 hours
        minConfidenceThreshold: 0.6,
        circuitBreaker: {
          maxConsecutiveFailures: 3,
          maxLossThreshold: 1000,
          lossWindowMs: 86400000, // 24 hours
          cooldownPeriodMs: 300000,
          maxSlippagePercent: 5,
          minFillRate: 0.95
        }
      },
      trailingOrderConfig: {
        minTrailPercent: 0.5,
        maxTrailPercent: 5,
        updateThrottleMs: 50,
        persistenceEnabled: false
      },
      enableAutoSubmission: true
    }
    
    integration = new OrderManagementIntegration(config)
  })

  describe('processAgentConsensus', () => {
    it('should create trailing order from consensus', async () => {
      const consensus: OrderAgentConsensus = {
        action: 'buy',
        confidence: 0.75,
        expectedWinRate: 0.65,
        expectedRiskReward: 2.0,
        trailDistance: 2,
        symbol: 'BTC-USD',
        leadAgentId: 'test-agent',
        agentSignals: [{
          agentId: 'test-agent',
          signal: 'buy',
          confidence: 0.75,
          reason: 'Test signal',
          weight: 1,
          timestamp: epochDateNow()
        }]
      }

      const orderCreatedHandler = mock.fn()
      eventBus.subscribe(EventTypes.ORDER_CREATED, orderCreatedHandler)

      const managedOrder = await integration.processAgentConsensus(consensus)

      assert.ok(managedOrder)
      assert.equal(managedOrder.symbol, 'BTC-USD')
      assert.equal(managedOrder.side, 'buy')
      assert.equal(managedOrder.type, 'trailing')
      assert.ok(managedOrder.size > 0)
      
      // Verify events were emitted
      assert.ok(orderCreatedHandler.mock.calls.length >= 1)
    })

    it('should reject consensus below confidence threshold', async () => {
      const consensus: OrderAgentConsensus = {
        action: 'buy',
        confidence: 0.5, // Below threshold of 0.6
        expectedWinRate: 0.65,
        expectedRiskReward: 2.0,
        trailDistance: 2,
        symbol: 'BTC-USD',
        leadAgentId: 'test-agent',
        agentSignals: [{
          agentId: 'test-agent',
          signal: 'buy',
          confidence: 0.5,
          reason: 'Test signal',
          weight: 1,
          timestamp: epochDateNow()
        }]
      }

      const consensusRejectedHandler = mock.fn()
      eventBus.subscribe(EventTypes.ORDER_CONSENSUS_REJECTED, consensusRejectedHandler)

      const managedOrder = await integration.processAgentConsensus(consensus)

      assert.equal(managedOrder, null)
      assert.equal(consensusRejectedHandler.mock.calls.length, 1)
    })
  })

  describe('createTrailingOrder', () => {
    it('should create trailing order directly', async () => {
      const params = {
        symbol: 'BTC-USD',
        side: 'sell' as const,
        size: 0.1,
        trailPercent: 1.5,
        currentPrice: 50000
      }

      const managedOrder = await integration.createTrailingOrder(params)

      assert.ok(managedOrder)
      assert.equal(managedOrder.symbol, 'BTC-USD')
      assert.equal(managedOrder.side, 'sell')
      assert.equal(managedOrder.type, 'trailing')
      assert.equal(managedOrder.size, 0.1)
    })
  })

  describe('processMarketUpdate', () => {
    it('should route market updates to trailing orders', async () => {
      // Create a trailing order first
      const params = {
        symbol: 'BTC-USD',
        side: 'buy' as const,
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      await integration.createTrailingOrder(params)
      
      // Wait for throttle
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Send market update
      const marketTickHandler = mock.fn()
      eventBus.subscribe(EventTypes.MARKET_TICK, marketTickHandler)
      
      await integration.processMarketUpdate('BTC-USD', 49000)
      
      // Verify market tick was emitted
      assert.equal(marketTickHandler.mock.calls.length, 1)
      const tickData = marketTickHandler.mock.calls[0]?.arguments[0]
      assert.equal(tickData.symbol, 'BTC-USD')
      assert.equal(tickData.price, 49000)
    })
  })

  describe('cancelOrder', () => {
    it('should cancel order in both managers', async () => {
      const params = {
        symbol: 'BTC-USD',
        side: 'buy' as const,
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      const order = await integration.createTrailingOrder(params)
      
      const orderCancelledHandler = mock.fn()
      eventBus.subscribe(EventTypes.ORDER_CANCELLED, orderCancelledHandler)
      
      await integration.cancelOrder(order.id, 'User requested')
      
      // Verify order is cancelled
      assert.equal(integration.getOrder(order.id), undefined)
      assert.ok(orderCancelledHandler.mock.calls.length >= 1)
    })
  })

  describe('getActiveOrders', () => {
    it('should return all active orders', async () => {
      const params1 = {
        symbol: 'BTC-USD',
        side: 'buy' as const,
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }
      
      const params2 = {
        symbol: 'ETH-USD',
        side: 'sell' as const,
        size: 1,
        trailPercent: 1.5,
        currentPrice: 3000
      }

      await integration.createTrailingOrder(params1)
      await integration.createTrailingOrder(params2)
      
      const activeOrders = integration.getActiveOrders()
      assert.equal(activeOrders.length, 2)
    })
  })

  describe('market data integration', () => {
    it('should trigger orders on market movements', async () => {
      const orderSubmittedHandler = mock.fn()
      eventBus.subscribe(EventTypes.ORDER_SUBMITTED, orderSubmittedHandler)

      // Create a buy trailing order
      const params = {
        symbol: 'BTC-USD',
        side: 'buy' as const,
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      await integration.createTrailingOrder(params)
      
      // Wait for throttle
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Simulate price drop (updates trailing price)
      await integration.processMarketUpdate('BTC-USD', 49000)
      
      // Wait for throttle
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Simulate price rise above trigger
      await integration.processMarketUpdate('BTC-USD', 50000)
      
      // Order should be triggered
      assert.ok(orderSubmittedHandler.mock.calls.length >= 1)
    })
  })

  describe('circuit breaker integration', () => {
    it('should respect circuit breaker status', async () => {
      // TODO: Add test for circuit breaker integration
      // This would require simulating multiple losing trades
      assert.ok(!integration.isCircuitBreakerActive())
    })
  })
})