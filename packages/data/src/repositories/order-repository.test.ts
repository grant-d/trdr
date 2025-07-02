import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { epochDateNow, toEpochDate } from '@trdr/shared'
import { OrderRepository } from './order-repository'
import { createConnectionManager } from '../db/connection-manager'
import type { ConnectionManager } from '../db/connection-manager'
import type { Order } from '../types/orders'

describe('OrderRepository', () => {
  let repository: OrderRepository
  let connectionManager: ConnectionManager

  beforeEach(async () => {
    connectionManager = createConnectionManager({ databasePath: ':memory:' })
    await connectionManager.initialize()

    // Create orders table
    await connectionManager.execute(`
      CREATE TABLE orders (
        id VARCHAR PRIMARY KEY,
        symbol VARCHAR NOT NULL,
        side VARCHAR NOT NULL CHECK (side IN ('buy', 'sell')),
        type VARCHAR NOT NULL CHECK (type IN ('market', 'limit', 'stop', 'trailing')),
        status VARCHAR NOT NULL CHECK (status IN ('pending', 'open', 'partial', 'filled', 'cancelled', 'rejected')),
        price DECIMAL(20, 8),
        size DECIMAL(20, 8) NOT NULL,
        filled_size DECIMAL(20, 8) DEFAULT 0,
        average_fill_price DECIMAL(20, 8),
        stop_price DECIMAL(20, 8),
        trail_distance DECIMAL(20, 8),
        agent_id VARCHAR,
        metadata JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        submitted_at TIMESTAMP,
        filled_at TIMESTAMP,
        cancelled_at TIMESTAMP
      )
    `)

    repository = new OrderRepository(connectionManager)
  })

  afterEach(async () => {
    await connectionManager.close()
  })

  describe('Order CRUD Operations', () => {
    const testOrder: Order = {
      id: 'order-123',
      symbol: 'BTC-USD',
      side: 'buy',
      type: 'limit',
      status: 'pending',
      price: 50000,
      size: 0.1,
      createdAt: epochDateNow(),
      updatedAt: epochDateNow(),
    }

    it('should create an order', async () => {
      await repository.createOrder(testOrder)

      const retrieved = await repository.getOrder(testOrder.id)

      assert.ok(retrieved)
      assert.equal(retrieved.id, testOrder.id)
      assert.equal(retrieved.symbol, testOrder.symbol)
      assert.equal(retrieved.side, testOrder.side)
      assert.equal(retrieved.price, testOrder.price)
    })

    it('should update an order', async () => {
      await repository.createOrder(testOrder)

      await repository.updateOrder(testOrder.id, {
        status: 'filled',
        filledSize: 0.1,
        averageFillPrice: 50000,
        filledAt: epochDateNow(),
      })

      const updated = await repository.getOrder(testOrder.id)

      assert.ok(updated)
      assert.equal(updated.status, 'filled')
      assert.equal(updated.filledSize, 0.1)
      assert.equal(updated.averageFillPrice, 50000)
      assert.ok(updated.filledAt)
    })

    it('should cancel an order', async () => {
      await repository.createOrder(testOrder)

      await repository.cancelOrder(testOrder.id, 'User requested')

      const cancelled = await repository.getOrder(testOrder.id)

      assert.ok(cancelled)
      assert.equal(cancelled.status, 'cancelled')
      assert.ok(cancelled.cancelledAt)
      assert.equal(cancelled.metadata?.cancellationReason, 'User requested')
    })

    it('should return null for non-existent order', async () => {
      const order = await repository.getOrder('non-existent')
      assert.equal(order, null)
    })
  })

  describe('Order Queries', () => {
    beforeEach(async () => {
      const orders: Order[] = [
        {
          id: 'order-1',
          symbol: 'BTC-USD',
          side: 'buy',
          type: 'limit',
          status: 'pending',
          price: 50000,
          size: 0.1,
          createdAt: toEpochDate(Date.now() - 3600000),
          updatedAt: toEpochDate(Date.now() - 3600000),
        },
        {
          id: 'order-2',
          symbol: 'BTC-USD',
          side: 'sell',
          type: 'limit',
          status: 'filled',
          price: 51000,
          size: 0.1,
          filledSize: 0.1,
          createdAt: toEpochDate(Date.now() - 1800000),
          updatedAt: toEpochDate(Date.now() - 1800000),
        },
        {
          id: 'order-3',
          symbol: 'ETH-USD',
          side: 'buy',
          type: 'market',
          status: 'filled',
          size: 1,
          filledSize: 1,
          createdAt: toEpochDate(Date.now() - 900000),
          updatedAt: toEpochDate(Date.now() - 900000),
        },
      ]

      for (const order of orders) {
        await repository.createOrder(order)
      }
    })

    it('should get orders by status', async () => {
      const pendingOrders = await repository.getOrdersByStatus('pending')
      assert.equal(pendingOrders.length, 1)
      assert.ok(pendingOrders[0])
      assert.equal(pendingOrders[0].id, 'order-1')

      const filledOrders = await repository.getOrdersByStatus('filled')
      assert.equal(filledOrders.length, 2)
    })

    it('should get active orders', async () => {
      const activeOrders = await repository.getActiveOrders()
      assert.equal(activeOrders.length, 1)
      assert.ok(activeOrders[0])
      assert.equal(activeOrders[0].status, 'pending')
    })

    it('should get active orders for symbol', async () => {
      const btcOrders = await repository.getActiveOrders('BTC-USD')
      assert.equal(btcOrders.length, 1)
      assert.ok(btcOrders[0])
      assert.equal(btcOrders[0].symbol, 'BTC-USD')
    })

    it('should get order history', async () => {
      const history = await repository.getOrderHistory(
        'BTC-USD',
        toEpochDate(Date.now() - 7200000),
        epochDateNow(),
      )

      assert.equal(history.length, 2)
      assert.ok(history.every(order => order.symbol === 'BTC-USD'))
    })

    it('should filter order history by status', async () => {
      const filledHistory = await repository.getOrderHistory(
        'BTC-USD',
        toEpochDate(Date.now() - 7200000),
        epochDateNow(),
        ['filled'],
      )

      assert.equal(filledHistory.length, 1)
      assert.ok(filledHistory[0])
      assert.equal(filledHistory[0].status, 'filled')
    })
  })

  describe('Order Statistics', () => {
    beforeEach(async () => {
      const orders: Order[] = [
        {
          id: 'order-stat-1',
          symbol: 'BTC-USD',
          side: 'buy',
          type: 'limit',
          status: 'filled',
          size: 0.1,
          createdAt: toEpochDate(Date.now() - 86400000),
          updatedAt: epochDateNow(),
        },
        {
          id: 'order-stat-2',
          symbol: 'BTC-USD',
          side: 'sell',
          type: 'market',
          status: 'filled',
          size: 0.1,
          createdAt: toEpochDate(Date.now() - 43200000),
          updatedAt: epochDateNow(),
        },
        {
          id: 'order-stat-3',
          symbol: 'BTC-USD',
          side: 'buy',
          type: 'limit',
          status: 'cancelled',
          size: 0.2,
          createdAt: toEpochDate(Date.now() - 3600000),
          updatedAt: epochDateNow(),
        },
      ]

      for (const order of orders) {
        await repository.createOrder(order)
      }
    })

    it('should calculate order statistics', async () => {
      const stats = await repository.getOrderStats('BTC-USD', 30)

      assert.equal(stats.totalOrders, 3)
      assert.equal(stats.filledOrders, 2)
      assert.equal(stats.cancelledOrders, 1)
      assert.equal(stats.averageFillRate, 2 / 3)
      assert.equal(stats.ordersByType['limit'], 2)
      assert.equal(stats.ordersByType['market'], 1)
      assert.equal(stats.ordersBySide['buy'], 2)
      assert.equal(stats.ordersBySide['sell'], 1)
    })

    it('should calculate stats for all symbols', async () => {
      const stats = await repository.getOrderStats(undefined, 30)
      assert.equal(stats.totalOrders, 3)
    })
  })

  describe('Cleanup Operations', () => {
    it('should cleanup old filled orders', async () => {
      const oldOrder: Order = {
        id: 'old-order',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'limit',
        status: 'filled',
        size: 0.1,
        createdAt: toEpochDate(Date.now() - 100 * 86400000),
        updatedAt: toEpochDate(Date.now() - 100 * 86400000),
      }

      const recentOrder: Order = {
        id: 'recent-order',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'limit',
        status: 'pending',
        size: 0.1,
        createdAt: epochDateNow(),
        updatedAt: epochDateNow(),
      }

      await repository.createOrder(oldOrder)
      await repository.createOrder(recentOrder)

      const deleted = await repository.cleanup(90)

      assert.equal(deleted, 1)

      const oldRetrieved = await repository.getOrder('old-order')
      assert.equal(oldRetrieved, null)

      const recentRetrieved = await repository.getOrder('recent-order')
      assert.ok(recentRetrieved)
    })

    it('should not cleanup active orders', async () => {
      const oldActiveOrder: Order = {
        id: 'old-active',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'limit',
        status: 'pending',
        size: 0.1,
        createdAt: toEpochDate(Date.now() - 100 * 86400000),
        updatedAt: toEpochDate(Date.now() - 100 * 86400000),
      }

      await repository.createOrder(oldActiveOrder)

      const deleted = await repository.cleanup(90)

      assert.equal(deleted, 0)

      const retrieved = await repository.getOrder('old-active')
      assert.ok(retrieved)
    })
  })

  describe('Agent Orders', () => {
    it('should get orders by agent', async () => {
      const agentOrder: Order = {
        id: 'agent-order-1',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'limit',
        status: 'pending',
        size: 0.1,
        agentId: 'agent-123',
        createdAt: epochDateNow(),
        updatedAt: epochDateNow(),
      }

      await repository.createOrder(agentOrder)

      const agentOrders = await repository.getOrdersByAgent('agent-123')
      assert.equal(agentOrders.length, 1)
      assert.ok(agentOrders[0])
      assert.equal(agentOrders[0].agentId, 'agent-123')
    })
  })
})
