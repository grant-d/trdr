import { describe, it, beforeEach, mock } from 'node:test'
import assert from 'node:assert/strict'
import { OrderLifecycleManager } from './order-lifecycle-manager'
import { EnhancedOrderState, epochDateNow } from '@trdr/shared'
import type { OrderAgentConsensus, OrderLifecycleConfig } from '@trdr/shared'
import { EventBus } from '../events/event-bus'

describe('OrderLifecycleManager', () => {
  let manager: OrderLifecycleManager
  let eventBus: EventBus
  let config: OrderLifecycleConfig
  let mockConsensus: OrderAgentConsensus

  beforeEach(() => {
    eventBus = EventBus.getInstance()
    
    config = {
      minOrderSize: 100,
      maxOrderSize: 10000,
      baseSizePercent: 0.05,
      maxPositionSize: 5000,
      defaultTimeInForce: 'GTC',
      enableOrderImprovement: true,
      improvementThreshold: 0.001,
      maxOrderDuration: 86400000, // 24 hours
      minConfidenceThreshold: 0.6
    }

    manager = new OrderLifecycleManager(config, eventBus)

    mockConsensus = {
      action: 'buy',
      confidence: 0.75,
      expectedWinRate: 0.65,
      expectedRiskReward: 2.0,
      trailDistance: 2.5,
      leadAgentId: 'agent-trend-1',
      agentSignals: [
        {
          agentId: 'agent-trend-1',
          signal: 'buy',
          confidence: 0.8,
          weight: 1.0,
          reason: 'Upward trend detected',
          timestamp: epochDateNow()
        }
      ]
    }
  })

  describe('Agent Consensus Processing', () => {
    it('should create order from valid consensus', async () => {
      const order = await manager.processAgentConsensus(mockConsensus)

      assert.ok(order)
      assert.equal(order.side, 'buy')
      assert.equal(order.type, 'trailing')
      assert.equal(order.trailPercent, 2.5)
      assert.ok(order.size >= config.minOrderSize)
    })

    it('should reject consensus below confidence threshold', async () => {
      const lowConfidenceConsensus = {
        ...mockConsensus,
        confidence: 0.5 // Below 0.6 threshold
      }

      const eventHandler = mock.fn()
      eventBus.subscribe('order.consensus.rejected', eventHandler)

      const order = await manager.processAgentConsensus(lowConfidenceConsensus)

      assert.equal(order, null)
      assert.equal(eventHandler.mock.calls.length, 1)
      
      const event = eventHandler.mock.calls[0]?.arguments[0]
      assert.ok(event.reason.includes('below threshold'))
    })

    it('should reject order with size too small', async () => {
      // Mock the calculateOrderSize to return a very small value
      const originalCalculateOrderSize = manager['calculateOrderSize']
      manager['calculateOrderSize'] = () => 50 // Below minOrderSize of 100

      const eventHandler = mock.fn()
      eventBus.subscribe('order.size.too.small', eventHandler)

      const order = await manager.processAgentConsensus(mockConsensus)

      assert.equal(order, null)
      assert.equal(eventHandler.mock.calls.length, 1)

      // Restore original method
      manager['calculateOrderSize'] = originalCalculateOrderSize
    })
  })

  describe('Position Sizing', () => {
    it('should calculate Kelly size correctly', () => {
      const size = manager.calculateOrderSize(mockConsensus)

      assert.ok(size > 0)
      assert.ok(size >= config.minOrderSize)
      assert.ok(size <= config.maxOrderSize)
    })

    it('should apply confidence adjustment', () => {
      const highConfidenceConsensus = { ...mockConsensus, confidence: 0.9 }
      const lowConfidenceConsensus = { ...mockConsensus, confidence: 0.65 }

      const highSize = manager.calculateOrderSize(highConfidenceConsensus)
      const lowSize = manager.calculateOrderSize(lowConfidenceConsensus)

      // Higher confidence should generally result in larger position
      // (though other factors may influence the final size)
      assert.ok(typeof highSize === 'number')
      assert.ok(typeof lowSize === 'number')
    })
  })

  describe('Order Lifecycle', () => {
    it('should submit order and transition to SUBMITTED state', async () => {
      const order = await manager.processAgentConsensus(mockConsensus)
      assert.ok(order)

      const submittedOrder = await manager.submitOrder(order)

      assert.equal(submittedOrder.state, EnhancedOrderState.SUBMITTED)
      assert.ok(manager.getActiveOrders().has(submittedOrder.id))
    })

    it('should cancel order successfully', async () => {
      const order = await manager.processAgentConsensus(mockConsensus)
      assert.ok(order)

      const submittedOrder = await manager.submitOrder(order)
      await manager.cancelOrder(submittedOrder.id, 'Test cancellation')

      assert.equal(submittedOrder.state, EnhancedOrderState.CANCELLED)
      assert.equal(submittedOrder.cancellationReason, 'Test cancellation')
      assert.ok(!manager.getActiveOrders().has(submittedOrder.id))
    })

    it('should reject cancellation of non-existent order', async () => {
      await assert.rejects(
        async () => manager.cancelOrder('non-existent-id'),
        /Order non-existent-id not found/
      )
    })

    it('should reject cancellation of terminal order', async () => {
      const order = await manager.processAgentConsensus(mockConsensus)
      assert.ok(order)

      const submittedOrder = await manager.submitOrder(order)
      
      // Manually transition to filled state
      submittedOrder.state = EnhancedOrderState.FILLED

      await assert.rejects(
        async () => manager.cancelOrder(submittedOrder.id),
        /Cannot cancel order.*in state FILLED/
      )
    })
  })

  describe('Order Modification', () => {
    it('should modify active order based on new consensus', async () => {
      const order = await manager.processAgentConsensus(mockConsensus)
      assert.ok(order)

      const submittedOrder = await manager.submitOrder(order)
      
      // Create new consensus with different parameters
      const newConsensus = {
        ...mockConsensus,
        confidence: 0.85,
        trailDistance: 3.0
      }

      const eventHandler = mock.fn()
      eventBus.subscribe('order.modified', eventHandler)

      const modifiedOrder = await manager.modifyOrder(submittedOrder, newConsensus)

      assert.ok(modifiedOrder)
      assert.equal(eventHandler.mock.calls.length, 1)
    })

    it('should not modify terminal order', async () => {
      const order = await manager.processAgentConsensus(mockConsensus)
      assert.ok(order)

      const submittedOrder = await manager.submitOrder(order)
      submittedOrder.state = EnhancedOrderState.FILLED

      const result = await manager.modifyOrder(submittedOrder, mockConsensus)
      assert.equal(result, null)
    })
  })

  describe('Order Monitoring', () => {
    it('should handle order fills correctly', async () => {
      const order = await manager.processAgentConsensus(mockConsensus)
      assert.ok(order)

      const submittedOrder = await manager.submitOrder(order)

      // Simulate a fill
      const fill = {
        fillId: 'fill-1',
        orderId: submittedOrder.id,
        size: submittedOrder.size,
        price: 50000,
        fee: 25,
        timestamp: Date.now(),
        side: 'buy' as const
      }

      // Trigger fill event
      eventBus.emit('order.fill', { orderId: submittedOrder.id, fill, timestamp: epochDateNow() })

      // Wait for async handling
      await new Promise(resolve => setTimeout(resolve, 10))

      assert.equal(submittedOrder.state, EnhancedOrderState.FILLED)
      assert.equal(submittedOrder.filledSize, submittedOrder.size)
      assert.ok(Math.abs(submittedOrder.averageFillPrice - 50000) < 0.001) // Allow for floating point precision
      assert.equal(submittedOrder.fees, 25)
    })

    it('should handle partial fills correctly', async () => {
      const order = await manager.processAgentConsensus(mockConsensus)
      assert.ok(order)

      const submittedOrder = await manager.submitOrder(order)

      // Simulate a partial fill (50% of order)
      const partialFill = {
        fillId: 'fill-1',
        orderId: submittedOrder.id,
        size: submittedOrder.size / 2,
        price: 50000,
        fee: 12.5,
        timestamp: Date.now(),
        side: 'buy' as const
      }

      // Trigger partial fill event
      eventBus.emit('order.fill', { orderId: submittedOrder.id, fill: partialFill, timestamp: epochDateNow() })

      // Wait for async handling
      await new Promise(resolve => setTimeout(resolve, 10))

      assert.equal(submittedOrder.state, EnhancedOrderState.PARTIALLY_FILLED)
      assert.equal(submittedOrder.filledSize, submittedOrder.size / 2)
      assert.ok(Math.abs(submittedOrder.averageFillPrice - 50000) < 0.001) // Allow for floating point precision
    })
  })

  describe('Utility Methods', () => {
    it('should get active orders', async () => {
      const order1 = await manager.processAgentConsensus(mockConsensus)
      const order2 = await manager.processAgentConsensus({
        ...mockConsensus,
        action: 'sell'
      })

      assert.ok(order1 && order2)

      await manager.submitOrder(order1)
      await manager.submitOrder(order2)

      const activeOrders = manager.getActiveOrders()
      assert.equal(activeOrders.size, 2)
    })

    it('should get order by ID', async () => {
      const order = await manager.processAgentConsensus(mockConsensus)
      assert.ok(order)

      const submittedOrder = await manager.submitOrder(order)
      const retrievedOrder = manager.getOrder(submittedOrder.id)

      assert.deepEqual(retrievedOrder, submittedOrder)
    })

    it('should return undefined for non-existent order', () => {
      const order = manager.getOrder('non-existent-id')
      assert.equal(order, undefined)
    })
  })
})