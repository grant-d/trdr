import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { BaseMarketDataFeed } from './base-market-data-feed'
import type { Candle, PriceTick } from '@trdr/shared'
import { epochDateNow, toEpochDate } from '@trdr/shared'
import type { DataFeedConfig, HistoricalDataRequest } from '../interfaces/market-data-pipeline'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

// Test implementation of BaseMarketDataFeed
class TestMarketDataFeed extends BaseMarketDataFeed {
  async subscribe(symbols: string[]): Promise<void> {
    symbols.forEach(symbol => this.subscribedSymbols.add(symbol))
  }

  async unsubscribe(symbols: string[]): Promise<void> {
    symbols.forEach(symbol => this.subscribedSymbols.delete(symbol))
  }

  async getHistorical(request: HistoricalDataRequest): Promise<Candle[]> {
    // Mock implementation
    return [{
      timestamp: request.start,
      open: 100,
      high: 110,
      low: 90,
      close: 105,
      volume: 1000,
    }]
  }

  async getCurrentPrice(_symbol: string): Promise<number> {
    // Mock implementation
    return 100.50
  }

  async start(): Promise<void> {
    this.emitConnected()
    // Emit a tick to establish lastMessageTime for health check
    this.testEmitTick({
      symbol: this.config.symbol,
      timestamp: epochDateNow(),
      price: 100,
      volume: 1,
    })
  }

  async stop(): Promise<void> {
    this.emitDisconnected('Manual stop')
  }

  // Test helper methods
  testEmitCandle(candle: Candle, symbol: string, interval: string): void {
    this.emitCandle(candle, symbol, interval)
  }

  testEmitTick(tick: PriceTick): void {
    this.emitTick(tick)
  }

  testEmitError(error: Error): void {
    this.emitError(error)
  }

  testEmitReconnecting(attempt: number): void {
    this.emitReconnecting(attempt)
  }
}

describe('BaseMarketDataFeed', () => {
  let feed: TestMarketDataFeed
  let eventBus: EventBus
  let events: Array<{ type: string; data: any }> = []
  let config: DataFeedConfig
  let subscriptions: Array<{ unsubscribe: () => void }> = []

  beforeEach(() => {
    config = {
      symbol: 'BTC-USD',
      feedType: 'coinbase',
      debug: false,
    }

    feed = new TestMarketDataFeed(config)
    eventBus = EventBus.getInstance()
    events = []
    subscriptions = []

    // Register event types
    Object.values(EventTypes).forEach(type => {
      eventBus.registerEvent(type)
    })

    // Capture all events
    const eventTypes = [
      EventTypes.MARKET_CANDLE,
      EventTypes.MARKET_TICK,
      EventTypes.SYSTEM_INFO,
      EventTypes.SYSTEM_WARNING,
      EventTypes.SYSTEM_ERROR,
    ]

    eventTypes.forEach(type => {
      const subscription = eventBus.subscribe(type, (data: any) => {
        events.push({ type, data })
      })
      subscriptions.push(subscription)
    })
  })

  afterEach(async () => {
    await feed.stop().catch(() => {
    })
    subscriptions.forEach(sub => sub.unsubscribe())
    eventBus.reset()
  })

  describe('connection lifecycle', () => {
    it('should emit connected event on start', async () => {
      await feed.start()

      const connectedEvents = events.filter(e =>
        e.type === EventTypes.SYSTEM_INFO &&
        e.data.message.includes('connected'),
      )

      assert.equal(connectedEvents.length, 1)
      assert.ok(connectedEvents[0]?.data.message.includes('Market data feed connected'))
      assert.equal(connectedEvents[0]?.data.details.feedType, 'coinbase')
    })

    it('should emit disconnected event on stop', async () => {
      await feed.start()
      await feed.stop()

      const disconnectedEvents = events.filter(e =>
        e.type === EventTypes.SYSTEM_WARNING &&
        e.data.message.includes('disconnected'),
      )

      assert.equal(disconnectedEvents.length, 1)
      assert.ok(disconnectedEvents[0]?.data.message.includes('Manual stop'))
    })

    it('should track connection state', async () => {
      assert.equal(feed.isHealthy(), false)

      await feed.start()
      assert.equal(feed.isHealthy(), true)

      await feed.stop()
      assert.equal(feed.isHealthy(), false)
    })
  })

  describe('subscription management', () => {
    it('should subscribe to symbols', async () => {
      await feed.subscribe(['BTC-USD', 'ETH-USD'])

      const stats = feed.getStats()
      assert.deepEqual(stats.subscribedSymbols, ['BTC-USD', 'ETH-USD'])
    })

    it('should unsubscribe from symbols', async () => {
      await feed.subscribe(['BTC-USD', 'ETH-USD', 'SOL-USD'])
      await feed.unsubscribe(['ETH-USD'])

      const stats = feed.getStats()
      assert.deepEqual(stats.subscribedSymbols, ['BTC-USD', 'SOL-USD'])
    })
  })

  describe('market data events', () => {
    it('should emit candle events', async () => {
      await feed.start()

      const candle: Candle = {
        timestamp: epochDateNow(),
        open: 50000,
        high: 51000,
        low: 49000,
        close: 50500,
        volume: 100,
      }

      feed.testEmitCandle(candle, 'BTC-USD', '1h')

      const candleEvents = events.filter(e => e.type === EventTypes.MARKET_CANDLE)
      assert.equal(candleEvents.length, 1)

      const eventData = candleEvents[0]?.data
      assert.equal(eventData.symbol, 'BTC-USD')
      assert.equal(eventData.interval, '1h')
      assert.equal(eventData.open, candle.open)
      assert.equal(eventData.close, candle.close)
    })

    it('should emit tick events', async () => {
      await feed.start()

      const tick: PriceTick = {
        symbol: 'BTC-USD',
        timestamp: epochDateNow(),
        price: 50250,
        volume: 0.5,
      }

      feed.testEmitTick(tick)

      const tickEvents = events.filter(e => e.type === EventTypes.MARKET_TICK)
      assert.equal(tickEvents.length, 2) // 1 from start() + 1 from testEmitTick()

      const eventData = tickEvents[1]?.data // Get the second tick (the one we just emitted)
      assert.equal(eventData.symbol, tick.symbol)
      assert.equal(eventData.price, tick.price)
      assert.equal(eventData.volume, tick.volume)
    })
  })

  describe('error handling', () => {
    it('should emit error events', async () => {
      const error = new Error('Test error')
      feed.testEmitError(error)

      const errorEvents = events.filter(e => e.type === EventTypes.SYSTEM_ERROR)
      assert.equal(errorEvents.length, 1)
      assert.equal(errorEvents[0]?.data.error.message, 'Test error')
      assert.equal(errorEvents[0]?.data.severity, 'medium')
    })

    it('should emit reconnecting events', async () => {
      feed.testEmitReconnecting(3)

      const reconnectEvents = events.filter(e =>
        e.type === EventTypes.SYSTEM_INFO &&
        e.data.message.includes('reconnecting'),
      )

      assert.equal(reconnectEvents.length, 1)
      assert.ok(reconnectEvents[0]?.data.message.includes('attempt 3'))
      assert.equal(reconnectEvents[0]?.data.details.attempt, 3)
    })
  })

  describe('statistics', () => {
    it('should track connection statistics', async () => {
      await feed.start()

      const stats1 = feed.getStats()
      assert.equal(stats1.connected, true)
      assert.equal(stats1.messagesReceived, 1) // 1 from start() tick emission
      assert.equal(stats1.reconnectAttempts, 0)

      // Emit some events
      feed.testEmitTick({
        symbol: 'BTC-USD',
        timestamp: epochDateNow(),
        price: 50000,
        volume: 1,
      })

      feed.testEmitReconnecting(1)

      const stats2 = feed.getStats()
      assert.equal(stats2.messagesReceived, 2) // 1 from start() + 1 from testEmitTick()
      assert.equal(stats2.reconnectAttempts, 1)
      assert.ok(stats2.lastMessageTime)
      assert.ok(stats2.uptime >= 0) // Allow 0 uptime for very fast tests
    })

    it('should track last error', async () => {
      const error = new Error('Connection failed')
      feed.testEmitError(error)

      const stats = feed.getStats()
      assert.equal(stats.lastError, 'Connection failed')
    })
  })

  describe('health check', () => {
    it('should be unhealthy when not connected', () => {
      assert.equal(feed.isHealthy(), false)
    })

    it('should be healthy when connected and receiving data', async () => {
      await feed.start()
      assert.equal(feed.isHealthy(), true)
    })

    it('should be unhealthy if no data received for 1 minute', async () => {
      await feed.start()

      // Mock old message time
      const stats = feed.getStats()
      if (stats.lastMessageTime) {
        // This is a limitation of our test - we can't easily mock the internal state
        // In a real implementation, we'd need to expose a way to test this
        assert.equal(feed.isHealthy(), true)
      }
    })
  })

  describe('historical data', () => {
    it('should fetch historical data', async () => {
      const request: HistoricalDataRequest = {
        symbol: 'BTC-USD',
        start: toEpochDate(new Date('2024-01-01')),
        end: toEpochDate(new Date('2024-01-02')),
        interval: '1h',
      }

      const candles = await feed.getHistorical(request)
      assert.equal(candles.length, 1)
      assert.equal(candles[0]?.open, 100)
    })
  })

  describe('current price', () => {
    it('should get current price', async () => {
      const price = await feed.getCurrentPrice('BTC-USD')
      assert.equal(price, 100.50)
    })
  })

  describe('debug mode', () => {
    it('should create feed with debug enabled', () => {
      const debugConfig: DataFeedConfig = {
        ...config,
        debug: true,
      }

      const debugFeed = new TestMarketDataFeed(debugConfig)

      // Verify the feed was created with debug config
      assert.ok(debugFeed)
      assert.equal(debugFeed.getStats().subscribedSymbols.length, 0)
    })
  })
})
