import { describe, it, beforeEach, mock } from 'node:test'
import assert from 'node:assert/strict'
import { TrailingOrderManager, type TrailingOrderParams } from './trailing-order-manager'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import type { OrderEvent } from '@trdr/shared'

describe('TrailingOrderManager', () => {
  let eventBus: EventBus
  let manager: TrailingOrderManager

  beforeEach(() => {
    eventBus = EventBus.getInstance()
    eventBus.registerEvent(EventTypes.ORDER_CREATED)
    eventBus.registerEvent(EventTypes.ORDER_SUBMITTED)
    eventBus.registerEvent(EventTypes.ORDER_CANCELLED)
    eventBus.registerEvent(EventTypes.ORDER_FILLED)
    eventBus.registerEvent(EventTypes.ORDER_REJECTED)
    
    manager = new TrailingOrderManager(eventBus, {
      minTrailPercent: 0.5,
      maxTrailPercent: 5,
      updateThrottleMs: 50,
      persistenceEnabled: false
    })
  })

  describe('createTrailingOrder', () => {
    it('should create a buy trailing order', async () => {
      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'buy',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      const orderCreatedHandler = mock.fn()
      eventBus.subscribe(EventTypes.ORDER_CREATED, orderCreatedHandler)

      const order = await manager.createTrailingOrder(params)

      assert.equal(order.symbol, 'BTC-USD')
      assert.equal(order.side, 'buy')
      assert.equal(order.type, 'trailing')
      assert.equal(order.size, 0.1)
      assert.equal(order.status, 'pending')
      
      // Check trailing-specific properties
      const trailingOrder = order as any
      assert.equal(trailingOrder.trailPercent, 2)
      assert.equal(trailingOrder.bestPrice, 50000)
      assert.equal(trailingOrder.lowWaterMark, 50000)
      assert.equal(trailingOrder.triggerPrice, 51000) // 50000 * 1.02
      
      // Verify event was emitted
      assert.equal(orderCreatedHandler.mock.calls.length, 1)
      const event = orderCreatedHandler.mock.calls[0]?.arguments[0] as OrderEvent
      assert.equal(event.type, 'created')
      assert.equal(event.order.id, order.id)
    })

    it('should create a sell trailing order', async () => {
      const params: TrailingOrderParams = {
        symbol: 'ETH-USD',
        side: 'sell',
        size: 1,
        trailPercent: 1.5,
        currentPrice: 3000
      }

      const order = await manager.createTrailingOrder(params)

      assert.equal(order.symbol, 'ETH-USD')
      assert.equal(order.side, 'sell')
      assert.equal(order.type, 'trailing')
      
      const trailingOrder = order as any
      assert.equal(trailingOrder.trailPercent, 1.5)
      assert.equal(trailingOrder.bestPrice, 3000)
      assert.equal(trailingOrder.highWaterMark, 3000)
      assert.equal(trailingOrder.triggerPrice, 2955) // 3000 * 0.985
    })

    it('should create order with optional parameters', async () => {
      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'sell',
        size: 0.5,
        trailPercent: 3,
        currentPrice: 60000,
        limitPrice: 59000,
        activationPrice: 61000
      }

      const order = await manager.createTrailingOrder(params)
      const trailingOrder = order as any

      assert.equal(trailingOrder.limitPrice, 59000)
      assert.equal(trailingOrder.activationPrice, 61000)
    })

    it('should reject invalid size', async () => {
      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'buy',
        size: -1,
        trailPercent: 2,
        currentPrice: 50000
      }

      await assert.rejects(
        () => manager.createTrailingOrder(params),
        /Order size must be positive/
      )
    })

    it('should reject trail percent below minimum', async () => {
      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'buy',
        size: 0.1,
        trailPercent: 0.1, // Below min of 0.5
        currentPrice: 50000
      }

      await assert.rejects(
        () => manager.createTrailingOrder(params),
        /Trail percent must be at least 0.5%/
      )
    })

    it('should reject trail percent above maximum', async () => {
      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'buy',
        size: 0.1,
        trailPercent: 10, // Above max of 5
        currentPrice: 50000
      }

      await assert.rejects(
        () => manager.createTrailingOrder(params),
        /Trail percent must not exceed 5%/
      )
    })
  })

  describe('processMarketUpdate', () => {
    it('should update buy order trigger on price decrease', async () => {
      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'buy',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      const order = await manager.createTrailingOrder(params)
      
      // Wait for throttle period
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Price drops - should update buy order
      await manager.processMarketUpdate('BTC-USD', 49000)

      const updatedOrder = manager.getOrder(order.id) as any
      assert.ok(updatedOrder)
      assert.equal(updatedOrder.bestPrice, 49000)
      assert.equal(updatedOrder.lowWaterMark, 49000)
      assert.equal(updatedOrder.triggerPrice, 49980) // 49000 * 1.02
    })

    it('should update sell order trigger on price increase', async () => {
      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'sell',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      const order = await manager.createTrailingOrder(params)
      
      // Wait for throttle period
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Price rises - should update sell order
      await manager.processMarketUpdate('BTC-USD', 51000)

      const updatedOrder = manager.getOrder(order.id) as any
      assert.ok(updatedOrder)
      assert.equal(updatedOrder.bestPrice, 51000)
      assert.equal(updatedOrder.highWaterMark, 51000)
      assert.equal(updatedOrder.triggerPrice, 49980) // 51000 * 0.98
    })

    it('should not update buy order on price increase', async () => {
      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'buy',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      const order = await manager.createTrailingOrder(params)
      
      // Wait for throttle period
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Price rises - should NOT update buy order
      await manager.processMarketUpdate('BTC-USD', 51000)

      const updatedOrder = manager.getOrder(order.id) as any
      assert.ok(updatedOrder)
      assert.equal(updatedOrder.bestPrice, 50000) // Unchanged
      assert.equal(updatedOrder.triggerPrice, 51000) // Unchanged
    })

    it('should not update sell order on price decrease', async () => {
      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'sell',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      const order = await manager.createTrailingOrder(params)
      
      // Wait for throttle period
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Price falls - should NOT update sell order
      await manager.processMarketUpdate('BTC-USD', 49000)

      const updatedOrder = manager.getOrder(order.id) as any
      assert.ok(updatedOrder)
      assert.equal(updatedOrder.bestPrice, 50000) // Unchanged
      assert.equal(updatedOrder.triggerPrice, 49000) // Unchanged
    })

    it('should throttle updates', async () => {
      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'buy',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      const order = await manager.createTrailingOrder(params)
      
      // Wait for initial throttle
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // First update
      await manager.processMarketUpdate('BTC-USD', 49000)
      
      // Immediate second update - should be throttled
      await manager.processMarketUpdate('BTC-USD', 48000)

      const updatedOrder = manager.getOrder(order.id) as any
      assert.ok(updatedOrder)
      assert.equal(updatedOrder.bestPrice, 49000) // Only first update applied
    })

    it('should trigger buy order when price rises above trigger', async () => {
      const orderSubmittedHandler = mock.fn()
      eventBus.subscribe(EventTypes.ORDER_SUBMITTED, orderSubmittedHandler)

      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'buy',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      await manager.createTrailingOrder(params)
      
      // Wait for throttle period
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Price drops to establish lower best price
      await manager.processMarketUpdate('BTC-USD', 49000)
      
      // Wait for throttle period
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Price rises above trigger (49000 * 1.02 = 49980)
      await manager.processMarketUpdate('BTC-USD', 50000)

      assert.equal(orderSubmittedHandler.mock.calls.length, 1)
      const event = orderSubmittedHandler.mock.calls[0]?.arguments[0] as OrderEvent
      assert.equal(event.type, 'submitted')
    })

    it('should trigger sell order when price falls below trigger', async () => {
      const orderSubmittedHandler = mock.fn()
      eventBus.subscribe(EventTypes.ORDER_SUBMITTED, orderSubmittedHandler)

      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'sell',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      await manager.createTrailingOrder(params)
      
      // Wait for throttle period
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Price rises to establish higher best price
      await manager.processMarketUpdate('BTC-USD', 51000)
      
      // Wait for throttle period
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Price falls below trigger (51000 * 0.98 = 49980)
      await manager.processMarketUpdate('BTC-USD', 49900)

      assert.equal(orderSubmittedHandler.mock.calls.length, 1)
      const event = orderSubmittedHandler.mock.calls[0]?.arguments[0] as OrderEvent
      assert.equal(event.type, 'submitted')
    })

    it('should respect activation price for sell orders', async () => {
      const orderSubmittedHandler = mock.fn()
      eventBus.subscribe(EventTypes.ORDER_SUBMITTED, orderSubmittedHandler)

      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'sell',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000,
        activationPrice: 52000 // Must reach 52k before activating
      }

      await manager.createTrailingOrder(params)
      
      // Wait for throttle period
      await new Promise(resolve => setTimeout(resolve, 60))
      
      // Price rises but not to activation level
      await manager.processMarketUpdate('BTC-USD', 51000)
      
      // Wait and try to trigger - should not work
      await new Promise(resolve => setTimeout(resolve, 60))
      await manager.processMarketUpdate('BTC-USD', 49000)

      // Should not trigger because activation price not reached
      assert.equal(orderSubmittedHandler.mock.calls.length, 0)
    })
  })

  describe('removeOrder', () => {
    it('should cancel an active order', async () => {
      const orderCancelledHandler = mock.fn()
      eventBus.subscribe(EventTypes.ORDER_CANCELLED, orderCancelledHandler)

      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'buy',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      const order = await manager.createTrailingOrder(params)
      await manager.removeOrder(order.id, 'User requested')

      // Check order is removed
      assert.equal(manager.getOrder(order.id), undefined)
      
      // Check event was emitted
      assert.equal(orderCancelledHandler.mock.calls.length, 1)
      const event = orderCancelledHandler.mock.calls[0]?.arguments[0] as OrderEvent
      assert.equal(event.type, 'cancelled')
      assert.equal(event.reason, 'User requested')
    })

    it('should handle removing non-existent order', async () => {
      await assert.doesNotReject(() => manager.removeOrder('non-existent-id'))
    })
  })

  describe('getActiveOrders', () => {
    it('should return only pending orders', async () => {
      const params1: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'buy',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      const params2: TrailingOrderParams = {
        symbol: 'ETH-USD',
        side: 'sell',
        size: 1,
        trailPercent: 1.5,
        currentPrice: 3000
      }

      const order1 = await manager.createTrailingOrder(params1)
      await manager.createTrailingOrder(params2)

      let activeOrders = manager.getActiveOrders()
      assert.equal(activeOrders.length, 2)

      // Cancel one order
      await manager.removeOrder(order1.id)

      activeOrders = manager.getActiveOrders()
      assert.equal(activeOrders.length, 1)
      assert.equal(activeOrders[0]?.symbol, 'ETH-USD')
    })
  })

  describe('cleanup', () => {
    it('should remove inactive orders', async () => {
      const params: TrailingOrderParams = {
        symbol: 'BTC-USD',
        side: 'buy',
        size: 0.1,
        trailPercent: 2,
        currentPrice: 50000
      }

      const order = await manager.createTrailingOrder(params)
      
      // Cancel the order
      await manager.removeOrder(order.id)
      
      // Run cleanup
      await manager.cleanup()
      
      // Order should be completely removed
      assert.equal(manager.getOrder(order.id), undefined)
    })
  })
})