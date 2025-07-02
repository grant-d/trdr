import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { TradeRepository } from './trade-repository'
import { createConnectionManager } from '../db/connection-manager'
import type { ConnectionManager } from '../db/connection-manager'
import type { Trade } from './trade-repository'

describe('TradeRepository', () => {
  let repository: TradeRepository
  let connectionManager: ConnectionManager

  beforeEach(async () => {
    connectionManager = createConnectionManager({ databasePath: ':memory:' })
    await connectionManager.initialize()
    
    // Create trades table
    await connectionManager.execute(`
      CREATE TABLE trades (
        id VARCHAR PRIMARY KEY,
        order_id VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        side VARCHAR NOT NULL CHECK (side IN ('buy', 'sell')),
        price DECIMAL(20, 8) NOT NULL,
        size DECIMAL(20, 8) NOT NULL,
        fee DECIMAL(20, 8) DEFAULT 0,
        fee_currency VARCHAR,
        pnl DECIMAL(20, 8),
        metadata JSON,
        executed_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `)
    
    repository = new TradeRepository(connectionManager)
  })

  afterEach(async () => {
    await connectionManager.close()
  })

  describe('Trade Operations', () => {
    const testTrade: Trade = {
      id: 'trade-123',
      orderId: 'order-123',
      symbol: 'BTC-USD',
      side: 'buy',
      price: 50000,
      size: 0.1,
      fee: 0.0001,
      feeCurrency: 'BTC',
      pnl: 0,
      executedAt: new Date()
    }

    it('should record a trade', async () => {
      await repository.recordTrade(testTrade)
      
      const retrieved = await repository.getTrade(testTrade.id)
      
      assert.ok(retrieved)
      assert.equal(retrieved.id, testTrade.id)
      assert.equal(retrieved.price, testTrade.price)
      assert.equal(retrieved.pnl, testTrade.pnl)
    })

    it('should record trades in batch', async () => {
      const trades: Trade[] = Array.from({ length: 5 }, (_, i) => ({
        id: `batch-trade-${i}`,
        orderId: `batch-order-${i}`,
        symbol: 'BTC-USD',
        side: i % 2 === 0 ? 'buy' : 'sell',
        price: 50000 + i * 100,
        size: 0.1,
        fee: 0.0001,
        executedAt: new Date(Date.now() - i * 1000)
      }))
      
      await repository.recordTradesBatch(trades)
      
      const firstTrade = await repository.getTrade('batch-trade-0')
      assert.ok(firstTrade)
      assert.equal(firstTrade.price, 50000)
    })

    it('should get trades by order', async () => {
      const orderId = 'multi-trade-order'
      const trades: Trade[] = [
        {
          id: 'partial-1',
          orderId,
          symbol: 'BTC-USD',
          side: 'buy',
          price: 50000,
          size: 0.05,
          fee: 0.00005,
          executedAt: new Date(Date.now() - 2000)
        },
        {
          id: 'partial-2',
          orderId,
          symbol: 'BTC-USD',
          side: 'buy',
          price: 50100,
          size: 0.05,
          fee: 0.00005,
          executedAt: new Date(Date.now() - 1000)
        }
      ]
      
      await repository.recordTradesBatch(trades)
      
      const orderTrades = await repository.getTradesByOrder(orderId)
      assert.equal(orderTrades.length, 2)
      assert.ok(orderTrades[0])
      assert.ok(orderTrades[1])
      assert.equal(orderTrades[0].executedAt < orderTrades[1].executedAt, true)
    })
  })

  describe('Trade History', () => {
    beforeEach(async () => {
      const trades: Trade[] = [
        {
          id: 'history-1',
          orderId: 'order-h1',
          symbol: 'BTC-USD',
          side: 'buy',
          price: 49000,
          size: 0.1,
          fee: 0.0001,
          pnl: 0,
          executedAt: new Date(Date.now() - 7200000)
        },
        {
          id: 'history-2',
          orderId: 'order-h2',
          symbol: 'BTC-USD',
          side: 'sell',
          price: 50000,
          size: 0.1,
          fee: 0.0001,
          pnl: 100,
          executedAt: new Date(Date.now() - 3600000)
        },
        {
          id: 'history-3',
          orderId: 'order-h3',
          symbol: 'ETH-USD',
          side: 'buy',
          price: 3000,
          size: 1,
          fee: 0.001,
          pnl: 0,
          executedAt: new Date(Date.now() - 1800000)
        }
      ]
      
      for (const trade of trades) {
        await repository.recordTrade(trade)
      }
    })

    it('should get trade history by time range', async () => {
      const history = await repository.getTradeHistory(
        'BTC-USD',
        new Date(Date.now() - 4000000),
        new Date()
      )
      
      assert.equal(history.length, 1)
      assert.ok(history[0])
      assert.equal(history[0].id, 'history-2')
    })

    it('should filter trade history by side', async () => {
      const buys = await repository.getTradeHistory(
        'BTC-USD',
        new Date(Date.now() - 8000000),
        new Date(),
        'buy'
      )
      
      assert.equal(buys.length, 1)
      assert.ok(buys[0])
      assert.equal(buys[0].side, 'buy')
    })
  })

  describe('P&L Calculations', () => {
    beforeEach(async () => {
      const trades: Trade[] = [
        {
          id: 'pnl-1',
          orderId: 'order-pnl1',
          symbol: 'BTC-USD',
          side: 'buy',
          price: 50000,
          size: 0.1,
          fee: 50,
          pnl: 0,
          executedAt: new Date(Date.now() - 3600000)
        },
        {
          id: 'pnl-2',
          orderId: 'order-pnl2',
          symbol: 'BTC-USD',
          side: 'sell',
          price: 51000,
          size: 0.1,
          fee: 51,
          pnl: 100,
          executedAt: new Date(Date.now() - 1800000)
        },
        {
          id: 'pnl-3',
          orderId: 'order-pnl3',
          symbol: 'BTC-USD',
          side: 'sell',
          price: 49000,
          size: 0.1,
          fee: 49,
          pnl: -100,
          executedAt: new Date()
        }
      ]
      
      await repository.recordTradesBatch(trades)
    })

    it('should calculate total P&L', async () => {
      const pnl = await repository.calculatePnL('BTC-USD')
      
      assert.equal(pnl.totalPnL, 0) // 0 + 100 - 100
      assert.equal(pnl.totalFees, 150) // 50 + 51 + 49
      assert.equal(pnl.realizedPnL, -150) // 0 - 150
      assert.equal(pnl.tradeCount, 3)
      assert.equal(pnl.winRate, 1 / 3) // 1 winning trade out of 3
    })

    it('should calculate P&L for time range', async () => {
      const now = new Date()
      const startTime = new Date(now.getTime() - 2000000) // 33 minutes ago
      const endTime = new Date(now)
      
      // Debug: check what trades we expect
      // pnl-2: executedAt = now - 1800000 (30 minutes ago) - should be included
      // pnl-3: executedAt = now - should be included
      // Total P&L should be 100 + (-100) = 0
      
      const pnl = await repository.calculatePnL(
        'BTC-USD',
        startTime,
        endTime
      )
      
      assert.equal(pnl.totalPnL, 0) // 100 + (-100) = 0
      assert.equal(pnl.tradeCount, 2)
    })
  })

  describe('Trade Statistics', () => {
    beforeEach(async () => {
      // Add trades over multiple days
      const trades: Trade[] = []
      for (let day = 0; day < 7; day++) {
        for (let hour = 0; hour < 4; hour++) {
          trades.push({
            id: `stat-${day}-${hour}`,
            orderId: `order-stat-${day}-${hour}`,
            symbol: 'BTC-USD',
            side: hour % 2 === 0 ? 'buy' : 'sell',
            price: 50000 + day * 100 + hour * 10,
            size: 0.1 + hour * 0.01,
            fee: 1,
            pnl: hour % 2 === 0 ? -10 : 20,
            executedAt: new Date(Date.now() - day * 86400000 - hour * 3600000)
          })
        }
      }
      
      await repository.recordTradesBatch(trades)
    })

    it('should get trade stats by day', async () => {
      const stats = await repository.getTradeStatsByPeriod('BTC-USD', 'day', 7)
      
      assert.ok(stats.length > 0)
      assert.ok(stats[0])
      assert.ok(stats[0].tradeCount >= 4)
      assert.ok(stats[0].volume > 0)
      assert.ok(stats[0].avgPrice > 0)
    })

    it('should get trade stats by hour', async () => {
      const stats = await repository.getTradeStatsByPeriod('BTC-USD', 'hour', 24)
      
      assert.ok(stats.length > 0)
      assert.ok(stats.every(s => s.tradeCount >= 0))
    })
  })

  describe('Top Trades', () => {
    beforeEach(async () => {
      const trades: Trade[] = [
        {
          id: 'top-pnl',
          orderId: 'order-top-pnl',
          symbol: 'BTC-USD',
          side: 'sell',
          price: 55000,
          size: 1,
          fee: 10,
          pnl: 5000,
          executedAt: new Date(Date.now() - 3600000)
        },
        {
          id: 'top-size',
          orderId: 'order-top-size',
          symbol: 'BTC-USD',
          side: 'buy',
          price: 50000,
          size: 5,
          fee: 50,
          pnl: 0,
          executedAt: new Date(Date.now() - 7200000)
        },
        {
          id: 'recent',
          orderId: 'order-recent',
          symbol: 'BTC-USD',
          side: 'buy',
          price: 51000,
          size: 0.1,
          fee: 5,
          pnl: -50,
          executedAt: new Date(Date.now() - 60000)
        }
      ]
      
      await repository.recordTradesBatch(trades)
    })

    it('should get top trades by P&L', async () => {
      const topPnL = await repository.getTopTrades('BTC-USD', 10, 'pnl')
      
      assert.ok(topPnL.length > 0)
      assert.ok(topPnL[0])
      assert.equal(topPnL[0].id, 'top-pnl')
      assert.equal(topPnL[0].pnl, 5000)
    })

    it('should get top trades by size', async () => {
      const topSize = await repository.getTopTrades('BTC-USD', 10, 'size')
      
      assert.ok(topSize[0])
      assert.equal(topSize[0].id, 'top-size')
      assert.equal(topSize[0].size, 5)
    })

    it('should get recent trades', async () => {
      const recent = await repository.getTopTrades('BTC-USD', 10, 'recent')
      
      assert.ok(recent[0])
      assert.equal(recent[0].id, 'recent')
    })
  })

  describe('Cleanup', () => {
    it('should cleanup old trades', async () => {
      const oldTrade: Trade = {
        id: 'old-trade',
        orderId: 'old-order',
        symbol: 'BTC-USD',
        side: 'buy',
        price: 40000,
        size: 0.1,
        fee: 4,
        executedAt: new Date(Date.now() - 400 * 86400000)
      }
      
      const recentTrade: Trade = {
        id: 'recent-trade',
        orderId: 'recent-order',
        symbol: 'BTC-USD',
        side: 'buy',
        price: 50000,
        size: 0.1,
        fee: 5,
        executedAt: new Date()
      }
      
      await repository.recordTrade(oldTrade)
      await repository.recordTrade(recentTrade)
      
      const deleted = await repository.cleanup(365)
      
      assert.equal(deleted, 1)
      
      const oldRetrieved = await repository.getTrade('old-trade')
      assert.equal(oldRetrieved, null)
      
      const recentRetrieved = await repository.getTrade('recent-trade')
      assert.ok(recentRetrieved)
    })
  })
})