import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { Database } from './index'
import { Candle } from '../types/market-data'
import { Order } from '../types/orders'
import { AgentSignal } from '../types/agents'
import { Trade } from '../repositories/trade-repository'
import fs from 'node:fs/promises'
import path from 'node:path'

describe('Database Integration', () => {
  let db: Database
  const testDbPath = path.join(__dirname, '../../test-data/test.db')

  beforeEach(async () => {
    // Ensure test directory exists
    await fs.mkdir(path.dirname(testDbPath), { recursive: true })
    
    // Create database with test configuration
    db = new Database({
      databasePath: testDbPath,
      enableLogging: false
    })
    
    await db.initialize()
  })

  afterEach(async () => {
    // Close database
    await db.close()
    
    // Clean up test database
    try {
      await fs.unlink(testDbPath)
    } catch (error) {
      // Ignore if file doesn't exist
    }
  })

  describe('Database Initialization', () => {
    it('should initialize database and run migrations', async () => {
      const stats = await db.getStats()
      
      assert.equal(stats.migration.currentVersion, 1)
      assert.equal(stats.migration.needsMigration, false)
      assert.ok(stats.connection.tables.includes('candles'))
      assert.ok(stats.connection.tables.includes('orders'))
      assert.ok(stats.connection.tables.includes('trades'))
    })
  })

  describe('Market Data Repository', () => {
    it('should save and retrieve candles', async () => {
      const candle: Candle = {
        symbol: 'BTC-USD',
        interval: '1h',
        openTime: new Date(Date.now() - 3600000),
        closeTime: new Date(),
        timestamp: new Date(),
        open: 50000,
        high: 51000,
        low: 49500,
        close: 50500,
        volume: 1000
      }
      
      await db.marketData.saveCandle(candle)
      
      const retrieved = await db.marketData.getLatestCandle('BTC-USD', '1h')
      assert.ok(retrieved)
      assert.equal(retrieved.symbol, candle.symbol)
      assert.equal(retrieved.close, candle.close)
    })

    it('should save candles in batch', async () => {
      const candles: Candle[] = Array.from({ length: 10 }, (_, i) => ({
        symbol: 'BTC-USD',
        interval: '1h',
        openTime: new Date(Date.now() - (10 - i) * 3600000),
        closeTime: new Date(Date.now() - (9 - i) * 3600000),
        timestamp: new Date(Date.now() - (9 - i) * 3600000),
        open: 50000 + i * 100,
        high: 50100 + i * 100,
        low: 49900 + i * 100,
        close: 50050 + i * 100,
        volume: 1000 + i * 10
      }))
      
      await db.marketData.saveCandlesBatch(candles)
      
      const retrieved = await db.marketData.getCandles(
        'BTC-USD',
        '1h',
        new Date(Date.now() - 11 * 3600000),
        new Date()
      )
      
      assert.equal(retrieved.length, 10)
    })
  })

  describe('Order Repository', () => {
    it('should create and update orders', async () => {
      const order: Order = {
        id: 'order-123',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'limit',
        status: 'pending',
        price: 50000,
        size: 0.1,
        createdAt: new Date(),
        updatedAt: new Date()
      }
      
      await db.orders.createOrder(order)
      
      // Update order
      await db.orders.updateOrder(order.id, {
        status: 'filled',
        filledSize: 0.1,
        averageFillPrice: 50000,
        filledAt: new Date()
      })
      
      const retrieved = await db.orders.getOrder(order.id)
      assert.ok(retrieved)
      assert.equal(retrieved.status, 'filled')
      assert.equal(retrieved.filledSize, 0.1)
    })

    it('should get active orders', async () => {
      const orders: Order[] = [
        {
          id: 'order-1',
          symbol: 'BTC-USD',
          side: 'buy',
          type: 'limit',
          status: 'pending',
          price: 50000,
          size: 0.1,
          createdAt: new Date(),
          updatedAt: new Date()
        },
        {
          id: 'order-2',
          symbol: 'BTC-USD',
          side: 'sell',
          type: 'limit',
          status: 'filled',
          price: 51000,
          size: 0.1,
          createdAt: new Date(),
          updatedAt: new Date()
        }
      ]
      
      for (const order of orders) {
        await db.orders.createOrder(order)
      }
      
      const activeOrders = await db.orders.getActiveOrders('BTC-USD')
      assert.equal(activeOrders.length, 1)
      assert.ok(activeOrders[0])
      assert.equal(activeOrders[0].id, 'order-1')
    })
  })

  describe('Trade Repository', () => {
    it('should record trades', async () => {
      // First create the order
      await db.orders.createOrder({
        id: 'order-123',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'limit',
        status: 'filled',
        price: 50000,
        size: 0.1,
        createdAt: new Date(),
        updatedAt: new Date()
      })
      
      const trade: Trade = {
        id: 'trade-123',
        orderId: 'order-123',
        symbol: 'BTC-USD',
        side: 'buy',
        price: 50000,
        size: 0.1,
        fee: 0.001,
        feeCurrency: 'BTC',
        pnl: 100,
        executedAt: new Date()
      }
      
      await db.trades.recordTrade(trade)
      
      const retrieved = await db.trades.getTrade(trade.id)
      assert.ok(retrieved)
      assert.equal(retrieved.price, trade.price)
      assert.equal(retrieved.pnl, trade.pnl)
    })

    it('should calculate P&L', async () => {
      // First create the orders
      await db.orders.createOrder({
        id: 'order-pnl-1',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'limit',
        status: 'filled',
        price: 50000,
        size: 0.1,
        createdAt: new Date(),
        updatedAt: new Date()
      })
      
      await db.orders.createOrder({
        id: 'order-pnl-2',
        symbol: 'BTC-USD',
        side: 'sell',
        type: 'limit',
        status: 'filled',
        price: 51000,
        size: 0.1,
        createdAt: new Date(),
        updatedAt: new Date()
      })
      
      const trades: Trade[] = [
        {
          id: 'trade-pnl-1',
          orderId: 'order-pnl-1',
          symbol: 'BTC-USD',
          side: 'buy',
          price: 50000,
          size: 0.1,
          fee: 0.001,
          pnl: 100,
          executedAt: new Date()
        },
        {
          id: 'trade-pnl-2',
          orderId: 'order-pnl-2',
          symbol: 'BTC-USD',
          side: 'sell',
          price: 51000,
          size: 0.1,
          fee: 0.001,
          pnl: 200,
          executedAt: new Date()
        }
      ]
      
      await db.trades.recordTradesBatch(trades)
      
      const pnl = await db.trades.calculatePnL('BTC-USD')
      assert.equal(pnl.totalPnL, 300)
      assert.equal(pnl.totalFees, 0.002)
      assert.equal(pnl.tradeCount, 2)
      assert.equal(pnl.winRate, 1) // Both trades profitable
    })
  })

  describe('Agent Repository', () => {
    it('should record agent decisions', async () => {
      const decision: AgentSignal & { agentType: 'momentum' } = {
        agentId: 'agent-1', 
        agentType: 'momentum',
        symbol: 'BTC-USD',
        action: 'TRAIL_BUY',
        confidence: 0.85,
        trailDistance: 100,
        reasoning: { momentum: 'strong', trend: 'up' },
        timestamp: new Date()
      }
      
      await db.agents.recordDecision(decision)
      
      const retrieved = await db.agents.getDecisions('agent-1')
      assert.equal(retrieved.length, 1)
      assert.ok(retrieved[0])
      assert.equal(retrieved[0].action, 'TRAIL_BUY')
      assert.equal(retrieved[0].confidence, 0.85)
    })

    it('should save and load checkpoints', async () => {
      const checkpoint = {
        id: 'checkpoint-123',
        type: 'agent-state',
        version: 1,
        state: {
          agentId: 'agent-1',
          position: 'long',
          entryPrice: 50000
        }
      }
      
      await db.agents.saveCheckpoint(checkpoint)
      
      const retrieved = await db.agents.loadLatestCheckpoint('agent-state')
      assert.ok(retrieved)
      assert.equal(retrieved.id, checkpoint.id)
      assert.deepEqual(retrieved.state, checkpoint.state)
    })
  })

  describe('Database Cleanup', () => {
    it('should clean up old data', async () => {
      // Add some test data
      const oldCandle: Candle = {
        symbol: 'BTC-USD',
        interval: '1h',
        openTime: new Date(Date.now() - 100 * 24 * 3600000), // 100 days ago
        closeTime: new Date(Date.now() - 100 * 24 * 3600000 + 3600000),
        timestamp: new Date(Date.now() - 100 * 24 * 3600000 + 3600000),
        open: 40000,
        high: 41000,
        low: 39000,
        close: 40500,
        volume: 1000
      }
      
      await db.marketData.saveCandle(oldCandle)
      
      // Run cleanup
      const cleanup = await db.cleanup(90)
      
      assert.ok(cleanup.marketData.candlesDeleted > 0)
    })
  })
})