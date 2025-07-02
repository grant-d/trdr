import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { Database, createDatabase } from './index'
import type { Candle, Order, AgentSignal } from '../types'
import type { Trade } from '../repositories/trade-repository'
import fs from 'node:fs/promises'
import path from 'node:path'

describe('Database Integration Tests', () => {
  let db: Database
  const testDbPath = path.join(__dirname, '../../test-data/integration.db')

  beforeEach(async () => {
    // Ensure test directory exists
    await fs.mkdir(path.dirname(testDbPath), { recursive: true })
    
    // Create and initialize database
    db = await createDatabase({
      databasePath: testDbPath,
      enableLogging: false
    })
  })

  afterEach(async () => {
    await db.close()
    
    // Clean up test database
    try {
      await fs.unlink(testDbPath)
    } catch (error) {
      // Ignore if file doesn't exist
    }
  })

  describe('Full Trading Workflow', () => {
    it('should handle complete order lifecycle', async () => {
      // 1. Save market data
      const candle: Candle = {
        symbol: 'BTC-USD',
        interval: '1h',
        timestamp: new Date(),
        openTime: new Date(Date.now() - 3600000),
        closeTime: new Date(),
        open: 50000,
        high: 51000,
        low: 49500,
        close: 50500,
        volume: 1000
      }
      
      await db.marketData.saveCandle(candle)
      
      // 2. Agent makes a decision
      const agentDecision: AgentSignal & { agentType: 'momentum'; symbol: string } = {
        agentId: 'momentum-agent-1',
        agentType: 'momentum',
        symbol: 'BTC-USD',
        action: 'TRAIL_BUY',
        confidence: 0.85,
        trailDistance: 0.02,
        reasoning: { momentum: 'strong', trend: 'up' },
        marketContext: { volume: 'high', volatility: 'medium' },
        timestamp: new Date()
      }
      
      await db.agents.recordDecision(agentDecision)
      
      // 3. Create order based on agent decision
      const order: Order = {
        id: 'order-workflow-1',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'trailing',
        status: 'pending',
        size: 0.1,
        trailPercent: 2,
        agentId: agentDecision.agentId,
        createdAt: new Date(),
        updatedAt: new Date()
      }
      
      await db.orders.createOrder(order)
      
      // 4. Update order status
      await db.orders.updateOrder(order.id, {
        status: 'filled',
        filledSize: 0.1,
        averageFillPrice: 50000,
        filledAt: new Date()
      })
      
      // 5. Record trade
      const trade: Trade = {
        id: 'trade-workflow-1',
        orderId: order.id,
        symbol: 'BTC-USD',
        side: 'buy',
        price: 50000,
        size: 0.1,
        fee: 0.0001,
        feeCurrency: 'BTC',
        pnl: 0, // Initial trade
        executedAt: new Date()
      }
      
      await db.trades.recordTrade(trade)
      
      // Verify complete workflow
      const latestCandle = await db.marketData.getLatestCandle('BTC-USD', '1h')
      assert.ok(latestCandle)
      
      const agentDecisions = await db.agents.getDecisions(agentDecision.agentId)
      assert.equal(agentDecisions.length, 1)
      
      const filledOrder = await db.orders.getOrder(order.id)
      assert.ok(filledOrder)
      assert.equal(filledOrder.status, 'filled')
      
      const orderTrades = await db.trades.getTradesByOrder(order.id)
      assert.equal(orderTrades.length, 1)
    })
  })

  describe('Cross-Repository Queries', () => {
    it('should track agent performance through orders and trades', async () => {
      const agentId = 'performance-agent-1'
      
      // Create multiple orders from agent
      const orderIds = ['perf-order-1', 'perf-order-2', 'perf-order-3']
      
      for (let i = 0; i < orderIds.length; i++) {
        // Record agent decision
        await db.agents.recordDecision({
          agentId,
          agentType: 'momentum',
          symbol: 'BTC-USD',
          action: i % 2 === 0 ? 'TRAIL_BUY' : 'TRAIL_SELL',
          confidence: 0.7 + i * 0.1,
          trailDistance: 0.02,
          reasoning: { index: i },
          marketContext: { volume: 'medium', volatility: 'low' },
          timestamp: new Date(Date.now() - (3 - i) * 3600000)
        })
        
        // Create order
        await db.orders.createOrder({
          id: orderIds[i]!,
          symbol: 'BTC-USD',
          side: i % 2 === 0 ? 'buy' as const : 'sell' as const,
          type: 'limit' as const,
          status: 'filled' as const,
          size: 0.1,
          agentId: agentId,
          createdAt: new Date(Date.now() - (3 - i) * 3600000),
          updatedAt: new Date()
        })
        
        // Record trade
        await db.trades.recordTrade({
          id: `trade-${i}`,
          orderId: orderIds[i]!,
          symbol: 'BTC-USD',
          side: i % 2 === 0 ? 'buy' as const : 'sell' as const,
          price: 50000 + i * 100,
          size: 0.1,
          fee: 0.0001,
          pnl: i === 0 ? 0 : (i % 2 === 0 ? -50 : 100),
          executedAt: new Date(Date.now() - (3 - i) * 3600000)
        })
      }
      
      // Analyze agent performance
      const agentOrders = await db.orders.getOrdersByAgent(agentId)
      assert.equal(agentOrders.length, 3)
      
      const decisions = await db.agents.getDecisions(agentId)
      assert.equal(decisions.length, 3)
      
      const pnl = await db.trades.calculatePnL('BTC-USD')
      assert.equal(pnl.totalPnL, 50) // 0 + 100 - 50
      assert.equal(pnl.tradeCount, 3)
    })
  })

  describe('Database Statistics', () => {
    it('should provide comprehensive stats', async () => {
      // Add some data
      await db.marketData.saveCandle({
        symbol: 'BTC-USD',
        interval: '1h',
        timestamp: new Date(),
        openTime: new Date(Date.now() - 3600000),
        closeTime: new Date(),
        open: 50000,
        high: 51000,
        low: 49500,
        close: 50500,
        volume: 1000
      })
      
      await db.orders.createOrder({
        id: 'stats-order-1',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'limit',
        status: 'pending',
        size: 0.1,
        createdAt: new Date(),
        updatedAt: new Date()
      })
      
      const stats = await db.getStats()
      
      assert.ok(stats.connection.tables.includes('candles'))
      assert.ok(stats.connection.tables.includes('orders'))
      assert.equal(stats.migration.needsMigration, false)
      assert.equal(stats.repositories.candles, 1)
      assert.equal(stats.repositories.orders, 1)
    })
  })

  describe('Checkpoint and Recovery', () => {
    it('should save and restore agent state', async () => {
      // Clean up any existing checkpoints from previous tests
      await db.connectionManager.execute('DELETE FROM checkpoints')
      const checkpoint = {
        id: 'checkpoint-1',
        type: 'agent-state',
        version: 1,
        state: {
          agentId: 'recovery-agent-1',
          position: 'long',
          entryPrice: 50000,
          stopLoss: 49000,
          metrics: {
            winRate: 0.65,
            totalTrades: 100
          }
        },
        metadata: {
          timestamp: new Date()
        }
      }
      
      await db.agents.saveCheckpoint(checkpoint)
      
      // Simulate restart - get latest checkpoint
      const restored = await db.agents.loadLatestCheckpoint('agent-state')
      
      assert.ok(restored)
      assert.equal(restored.id, checkpoint.id)
      assert.deepEqual(restored.state, checkpoint.state)
      assert.equal(restored.version, checkpoint.version)
    })

    it('should list checkpoints', async () => {
      // Clean up any existing checkpoints from previous tests
      await db.connectionManager.execute('DELETE FROM checkpoints')
      
      // Save multiple checkpoints with slight delay to ensure different timestamps
      const savedIds: string[] = []
      for (let i = 0; i < 3; i++) {
        const id = `checkpoint-list-${i}`
        await db.agents.saveCheckpoint({
          id,
          type: 'system-state',
          version: i + 1,
          state: { iteration: i }
        })
        savedIds.push(id)
        // Larger delay to ensure different created_at timestamps
        await new Promise(resolve => setTimeout(resolve, 50))
      }
      
      const checkpoints = await db.agents.listCheckpoints('system-state')
      
      // Verify we saved all 3 checkpoints
      for (const id of savedIds) {
        const checkpoint = await db.agents.loadCheckpoint(id)
        assert.ok(checkpoint, `Checkpoint ${id} should exist`)
      }
      
      assert.equal(checkpoints.length, 3, `Expected 3 checkpoints but got ${checkpoints.length}`)
      assert.ok(checkpoints[0])
      assert.equal(checkpoints[0].version, 3) // Most recent first
    })
  })

  describe('Time-based Queries', () => {
    it('should handle complex time-range queries', async () => {
      const now = Date.now()
      const symbol = 'ETH-USD'
      
      // Add candles over time
      const candles: Candle[] = Array.from({ length: 24 }, (_, i) => ({
        symbol,
        interval: '1h',
        timestamp: new Date(now - (24 - i) * 3600000),
        openTime: new Date(now - (24 - i) * 3600000),
        closeTime: new Date(now - (23 - i) * 3600000),
        open: 3000 + i * 10,
        high: 3010 + i * 10,
        low: 2990 + i * 10,
        close: 3005 + i * 10,
        volume: 100 + i
      }))
      
      await db.marketData.saveCandlesBatch(candles)
      
      // Query different time ranges
      const last6Hours = await db.marketData.getCandles(
        symbol,
        '1h',
        new Date(now - 6 * 3600000),
        new Date(now)
      )
      
      assert.equal(last6Hours.length, 6)
      assert.ok(last6Hours[0])
      assert.ok(last6Hours[0].openTime.getTime() >= now - 6 * 3600000)
    })
  })

  describe('Cleanup Integration', () => {
    it('should cleanup across all repositories', async () => {
      const oldDate = Date.now() - 100 * 86400000
      const recentDate = Date.now() - 3600000
      
      // Add old data
      await db.marketData.saveCandle({
        symbol: 'BTC-USD',
        interval: '1h',
        timestamp: new Date(oldDate),
        openTime: new Date(oldDate),
        closeTime: new Date(oldDate + 3600000),
        open: 40000,
        high: 41000,
        low: 39000,
        close: 40500,
        volume: 1000
      })
      
      await db.orders.createOrder({
        id: 'old-cleanup-order',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'limit',
        status: 'filled',
        size: 0.1,
        createdAt: new Date(oldDate),
        updatedAt: new Date(oldDate)
      })
      
      await db.trades.recordTrade({
        id: 'old-cleanup-trade',
        orderId: 'old-cleanup-order',
        symbol: 'BTC-USD',
        side: 'buy',
        price: 40000,
        size: 0.1,
        fee: 0.0001,
        executedAt: new Date(oldDate)
      })
      
      await db.agents.recordDecision({
        agentId: 'cleanup-agent',
        agentType: 'momentum',
        symbol: 'BTC-USD',
        action: 'HOLD',
        confidence: 0.5,
        trailDistance: 0,
        reasoning: {},
        marketContext: {},
        timestamp: new Date(oldDate)
      })
      
      // Add recent data
      await db.marketData.saveCandle({
        symbol: 'BTC-USD',
        interval: '1h',
        timestamp: new Date(recentDate),
        openTime: new Date(recentDate),
        closeTime: new Date(recentDate + 3600000),
        open: 50000,
        high: 51000,
        low: 49000,
        close: 50500,
        volume: 2000
      })
      
      // Run cleanup
      const cleanup = await db.cleanup(90)
      
      assert.ok(cleanup.marketData.candlesDeleted > 0)
      assert.ok(cleanup.ordersDeleted > 0)
      assert.ok(cleanup.tradesDeleted > 0)
      assert.ok(cleanup.agentData.decisionsDeleted > 0)
      
      // Verify recent data remains
      const recentCandles = await db.marketData.getLatestCandle('BTC-USD', '1h')
      assert.ok(recentCandles)
      assert.equal(recentCandles.timestamp.getTime(), recentDate)
    })
  })

  describe('Error Handling', () => {
    it('should handle constraint violations', async () => {
      const order: Order = {
        id: 'constraint-test',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'limit',
        status: 'pending',
        size: 0.1,
        createdAt: new Date(),
        updatedAt: new Date()
      }
      
      await db.orders.createOrder(order)
      
      // Try to create duplicate
      await assert.rejects(
        () => db.orders.createOrder(order),
        /UNIQUE constraint|duplicate key/i
      )
    })

    it('should handle invalid enum values', async () => {
      const invalidOrder = {
        id: 'invalid-enum',
        symbol: 'BTC-USD',
        side: 'invalid-side' as any,
        type: 'limit' as const,
        status: 'pending' as const,
        size: 0.1,
        createdAt: new Date(),
        updatedAt: new Date()
      }
      
      await assert.rejects(
        () => db.orders.createOrder(invalidOrder),
        /CHECK constraint/i
      )
    })
  })
})