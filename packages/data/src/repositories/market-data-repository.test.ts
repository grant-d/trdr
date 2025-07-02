import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { MarketDataRepository } from './market-data-repository'
import { createConnectionManager } from '../db/connection-manager'
import type { ConnectionManager } from '../db/connection-manager'
import type { Candle, PriceTick } from '../types/market-data'

describe('MarketDataRepository', () => {
  let repository: MarketDataRepository
  let connectionManager: ConnectionManager

  beforeEach(async () => {
    connectionManager = createConnectionManager({ databasePath: ':memory:' })
    await connectionManager.initialize()
    
    // Create tables
    await connectionManager.execute(`
      CREATE TABLE candles (
        id BIGINT PRIMARY KEY,
        symbol VARCHAR NOT NULL,
        interval VARCHAR NOT NULL,
        open_time TIMESTAMP NOT NULL,
        close_time TIMESTAMP NOT NULL,
        open DECIMAL(20, 8) NOT NULL,
        high DECIMAL(20, 8) NOT NULL,
        low DECIMAL(20, 8) NOT NULL,
        close DECIMAL(20, 8) NOT NULL,
        volume DECIMAL(20, 8) NOT NULL,
        quote_volume DECIMAL(20, 8),
        trades_count INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `)
    
    await connectionManager.execute(`
      CREATE TABLE market_ticks (
        id BIGINT PRIMARY KEY,
        symbol VARCHAR NOT NULL,
        price DECIMAL(20, 8) NOT NULL,
        volume DECIMAL(20, 8) NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        bid DECIMAL(20, 8),
        ask DECIMAL(20, 8),
        bid_size DECIMAL(20, 8),
        ask_size DECIMAL(20, 8),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `)
    
    repository = new MarketDataRepository(connectionManager)
  })

  afterEach(async () => {
    await connectionManager.close()
  })

  describe('Candle Operations', () => {
    const testCandle: Candle = {
      symbol: 'BTC-USD',
      interval: '1h',
      timestamp: new Date(Date.now() - 3600000),
      openTime: new Date(Date.now() - 3600000),
      closeTime: new Date(),
      open: 50000,
      high: 51000,
      low: 49500,
      close: 50500,
      volume: 1000,
      quoteVolume: 50500000,
      tradesCount: 5000
    }

    it('should save and retrieve a candle', async () => {
      await repository.saveCandle(testCandle)
      
      const retrieved = await repository.getLatestCandle('BTC-USD', '1h')
      
      assert.ok(retrieved)
      assert.equal(retrieved.symbol, testCandle.symbol)
      assert.equal(retrieved.open, testCandle.open)
      assert.equal(retrieved.close, testCandle.close)
      assert.equal(retrieved.volume, testCandle.volume)
    })

    it('should save candles in batch', async () => {
      const candles: Candle[] = Array.from({ length: 10 }, (_, i) => ({
        symbol: 'BTC-USD',
        interval: '1h',
        timestamp: new Date(Date.now() - (10 - i) * 3600000),
        openTime: new Date(Date.now() - (10 - i) * 3600000),
        closeTime: new Date(Date.now() - (9 - i) * 3600000),
        open: 50000 + i * 100,
        high: 50100 + i * 100,
        low: 49900 + i * 100,
        close: 50050 + i * 100,
        volume: 1000 + i * 10
      }))
      
      await repository.saveCandlesBatch(candles)
      
      const retrieved = await repository.getCandles(
        'BTC-USD',
        '1h',
        new Date(Date.now() - 11 * 3600000),
        new Date()
      )
      
      assert.equal(retrieved.length, 10)
      assert.ok(retrieved[0])
      assert.equal(retrieved[0].open, 50000)
      assert.ok(retrieved[9])
      assert.equal(retrieved[9].open, 50900)
    })

    it('should get candles within time range', async () => {
      const now = Date.now()
      const candles: Candle[] = [
        {
          ...testCandle,
          openTime: new Date(now - 7200000),
          closeTime: new Date(now - 3600000)
        },
        {
          ...testCandle,
          openTime: new Date(now - 3600000),
          closeTime: new Date(now)
        }
      ]
      
      await repository.saveCandlesBatch(candles)
      
      const retrieved = await repository.getCandles(
        'BTC-USD',
        '1h',
        new Date(now - 5400000), // 1.5 hours ago
        new Date()
      )
      
      assert.equal(retrieved.length, 1)
    })

    it('should return null for non-existent candle', async () => {
      const retrieved = await repository.getLatestCandle('ETH-USD', '1h')
      assert.equal(retrieved, null)
    })
  })

  describe('Tick Operations', () => {
    const testTick: PriceTick = {
      symbol: 'BTC-USD',
      timestamp: new Date(),
      price: 50000,
      volume: 0.1,
      bid: 49999,
      ask: 50001,
      bidSize: 1.5,
      askSize: 2.0
    }

    it('should save and retrieve a tick', async () => {
      await repository.saveTick(testTick)
      
      const retrieved = await repository.getLatestTick('BTC-USD')
      
      assert.ok(retrieved)
      assert.equal(retrieved.symbol, testTick.symbol)
      assert.equal(retrieved.price, testTick.price)
      assert.equal(retrieved.bid, testTick.bid)
      assert.equal(retrieved.ask, testTick.ask)
    })

    it('should save ticks in batch', async () => {
      const ticks: PriceTick[] = Array.from({ length: 5 }, (_, i) => ({
        symbol: 'BTC-USD',
        timestamp: new Date(Date.now() - (5 - i) * 1000),
        price: 50000 + i,
        volume: 0.1
      }))
      
      await repository.saveTicksBatch(ticks)
      
      const retrieved = await repository.getTicks(
        'BTC-USD',
        new Date(Date.now() - 10000),
        new Date()
      )
      
      assert.equal(retrieved.length, 5)
    })

    it('should handle empty batch', async () => {
      await repository.saveTicksBatch([])
      // Should not throw
    })
  })

  describe('Market Statistics', () => {
    it('should calculate market stats', async () => {
      const candles: Candle[] = Array.from({ length: 30 }, (_, i) => ({
        symbol: 'BTC-USD',
        interval: '1d',
        timestamp: new Date(Date.now() - (30 - i) * 86400000),
        openTime: new Date(Date.now() - (30 - i) * 86400000),
        closeTime: new Date(Date.now() - (29 - i) * 86400000),
        open: 50000 + Math.random() * 1000,
        high: 51000 + Math.random() * 1000,
        low: 49000 + Math.random() * 1000,
        close: 50000 + Math.random() * 1000,
        volume: 1000 + Math.random() * 100
      }))
      
      await repository.saveCandlesBatch(candles)
      
      const stats = await repository.getMarketStats('BTC-USD', '1d', 30)
      
      assert.ok(stats.avgVolume > 0)
      assert.ok(stats.avgPrice > 0)
      assert.ok(stats.priceRange.min > 0)
      assert.ok(stats.priceRange.max > stats.priceRange.min)
      assert.ok(stats.volatility >= 0)
    })
  })

  describe('Cleanup Operations', () => {
    it('should cleanup old data', async () => {
      const oldCandle: Candle = {
        symbol: 'BTC-USD',
        interval: '1h',
        timestamp: new Date(Date.now() - 100 * 86400000),
        openTime: new Date(Date.now() - 100 * 86400000),
        closeTime: new Date(Date.now() - 100 * 86400000 + 3600000),
        open: 40000,
        high: 41000,
        low: 39000,
        close: 40500,
        volume: 1000
      }
      
      const recentCandle: Candle = {
        ...oldCandle,
        timestamp: new Date(Date.now() - 3600000),
        openTime: new Date(Date.now() - 3600000),
        closeTime: new Date()
      }
      
      await repository.saveCandle(oldCandle)
      await repository.saveCandle(recentCandle)
      
      const cleanup = await repository.cleanup(90)
      
      assert.equal(cleanup.candlesDeleted, 1)
      
      const remaining = await repository.getLatestCandle('BTC-USD', '1h')
      assert.ok(remaining)
      assert.equal(remaining.timestamp.getTime(), recentCandle.timestamp.getTime())
    })

    it('should cleanup old ticks', async () => {
      const oldTick: PriceTick = {
        symbol: 'BTC-USD',
        timestamp: new Date(Date.now() - 100 * 86400000),
        price: 40000,
        volume: 0.1
      }
      
      await repository.saveTick(oldTick)
      
      const cleanup = await repository.cleanup(90)
      
      assert.equal(cleanup.ticksDeleted, 1)
    })
  })
})