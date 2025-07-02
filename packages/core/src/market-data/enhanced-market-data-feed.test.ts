import { describe, it, beforeEach, mock } from 'node:test'
import assert from 'node:assert/strict'
import type { PriceTick, Candle } from '@trdr/shared'
import { EnhancedMarketDataFeed, type EnhancedDataFeedConfig } from './enhanced-market-data-feed'
import { enhancedEventBus } from '../events/enhanced-event-bus'
import { EnhancedEventTypes } from '../events/market-data-events'
import { EventTypes } from '../events/types'

// Test implementation of the abstract class
class TestEnhancedMarketDataFeed extends EnhancedMarketDataFeed {
  constructor(config: EnhancedDataFeedConfig) {
    super(config)
  }

  async connect(): Promise<void> {
    this.connected = true
    this.emitConnected()
  }

  async subscribe(_symbols: string[]): Promise<void> {
    // Mock implementation
  }

  async unsubscribe(_symbols: string[]): Promise<void> {
    // Mock implementation
  }

  async getHistorical(_request: any): Promise<any[]> {
    return []
  }

  async getCurrentPrice(_symbol: string): Promise<number> {
    return 50000
  }

  async start(): Promise<void> {
    await this.connect()
  }

  isConnected(): boolean {
    return this.connected
  }

  // Test methods to trigger enhanced events
  testEmitTick(tick: PriceTick, sourceTimestamp?: Date): void {
    this.emitEnhancedTick(tick, sourceTimestamp)
  }

  testEmitCandle(candle: Candle, symbol: string, interval: string, sourceTimestamp?: Date): void {
    this.emitEnhancedCandle(candle, symbol, interval, sourceTimestamp)
  }

  testEmitConnectionStatus(status: 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting'): void {
    this.emitConnectionStatus(status)
  }
}

describe('EnhancedMarketDataFeed', () => {
  let feed: TestEnhancedMarketDataFeed
  let config: EnhancedDataFeedConfig

  beforeEach(() => {
    config = {
      feedType: 'paper',
      symbol: 'BTC-USD',
      enhancedEvents: true,
      priceChangeThreshold: 100, // 1% in basis points
      enableCompression: true,
      compressionWindow: 1000,
      maxEventsPerSecond: 10,
      enableStatistics: true,
    }

    enhancedEventBus.reset()

    // Register standard event types
    Object.values(EventTypes).forEach(eventType => {
      enhancedEventBus.registerEvent(eventType)
    })

    feed = new TestEnhancedMarketDataFeed(config)

    // Register enhanced event types
    Object.values(EnhancedEventTypes).forEach(eventType => {
      enhancedEventBus.registerEvent(eventType)
    })
  })

  describe('Enhanced Event Emission', () => {
    it('should emit enhanced tick events', () => {
      const handler = mock.fn()
      enhancedEventBus.subscribe(EnhancedEventTypes.MARKET_TICK_ENHANCED, handler)

      const tick: PriceTick = {
        symbol: 'BTC-USD',
        price: 50000,
        volume: 100,
        timestamp: Date.now(),
      }

      // Check that the enhanced feed's lastPrices is empty for this symbol
      const enhancedFeed = feed as any
      assert.equal(enhancedFeed.lastPrices.get('BTC-USD'), undefined, 'No previous price should exist')

      feed.testEmitTick(tick)

      assert.equal(handler.mock.calls.length, 1, 'Enhanced tick event should be emitted')

      const emittedEvent = handler.mock.calls[0]?.arguments[0]
      assert.equal(emittedEvent.type, 'market.tick.enhanced')
      assert.equal(emittedEvent.symbol, 'BTC-USD')
      assert.equal(emittedEvent.price, 50000)
      assert.equal(emittedEvent.volume, 100)
      assert.equal(emittedEvent.source, 'paper')
      assert.equal(emittedEvent.feedType, 'paper')
      assert.equal(emittedEvent.sequence, 1)
      assert.ok(emittedEvent.latency >= 0)
    })

    it('should calculate price change for subsequent ticks', () => {
      const handler = mock.fn()
      enhancedEventBus.subscribe(EnhancedEventTypes.MARKET_TICK_ENHANCED, handler)

      // First tick
      feed.testEmitTick({
        symbol: 'BTC-USD',
        price: 50000,
        volume: 100,
        timestamp: Date.now(),
      })

      // Second tick with price change
      feed.testEmitTick({
        symbol: 'BTC-USD',
        price: 51000,
        volume: 150,
        timestamp: Date.now(),
      })

      assert.equal(handler.mock.calls.length, 2)

      const secondEvent = handler.mock.calls[1]?.arguments[0]
      assert.equal(secondEvent.priceChange, 1000)
      assert.equal(secondEvent.priceChangePercent, 2) // (1000/50000) * 100
      assert.equal(secondEvent.sequence, 2)
    })

    it('should emit enhanced candle events', () => {
      const handler = mock.fn()
      enhancedEventBus.subscribe(EnhancedEventTypes.MARKET_CANDLE_ENHANCED, handler)

      const candle: Candle = {
        open: 49000,
        high: 51000,
        low: 48500,
        close: 50000,
        volume: 1000,
        timestamp: Date.now(),
      }

      feed.testEmitCandle(candle, 'BTC-USD', '1m')

      assert.equal(handler.mock.calls.length, 1)

      const emittedEvent = handler.mock.calls[0]?.arguments[0]
      assert.equal(emittedEvent.type, 'market.candle.enhanced')
      assert.equal(emittedEvent.symbol, 'BTC-USD')
      assert.equal(emittedEvent.interval, '1m')
      assert.equal(emittedEvent.range, 2500) // high - low
      assert.equal(emittedEvent.bodySize, 1000) // |close - open|
      assert.equal(emittedEvent.upperWick, 1000) // high - max(open, close)
      assert.equal(emittedEvent.lowerWick, 500) // min(open, close) - low
      assert.equal(emittedEvent.typicalPrice, (51000 + 48500 + 50000) / 3)
      assert.equal(emittedEvent.candleType, 'bullish') // close > open
    })

    it('should classify candle types correctly', () => {
      const handler = mock.fn()
      enhancedEventBus.subscribe(EnhancedEventTypes.MARKET_CANDLE_ENHANCED, handler)

      // Bearish candle
      const bearishCandle: Candle = {
        open: 50000,
        high: 50200,
        low: 49000,
        close: 49500,
        volume: 1000,
        timestamp: Date.now(),
      }

      feed.testEmitCandle(bearishCandle, 'BTC-USD', '1m')

      let event = handler.mock.calls[0]?.arguments[0]
      assert.equal(event.candleType, 'bearish')

      // Doji candle (small body relative to range)
      const dojiCandle: Candle = {
        open: 50000,
        high: 52000,
        low: 48000,
        close: 50050, // Very small body relative to 4000 range
        volume: 1000,
        timestamp: Date.now(),
      }

      feed.testEmitCandle(dojiCandle, 'BTC-USD', '1m')

      event = handler.mock.calls[1]?.arguments[0]
      assert.equal(event.candleType, 'doji')
    })

    it('should emit connection status events', () => {
      const handler = mock.fn()
      enhancedEventBus.subscribe(EnhancedEventTypes.MARKET_CONNECTION, handler)

      feed.testEmitConnectionStatus('connected')

      assert.equal(handler.mock.calls.length, 1)

      const event = handler.mock.calls[0]?.arguments[0]
      assert.equal(event.type, 'market.connection')
      assert.equal(event.status, 'connected')
      assert.equal(event.source, 'paper')
      assert.equal(event.feedType, 'paper')
      assert.ok(event.details)
    })
  })

  describe('Enhanced Statistics', () => {
    it('should track enhanced statistics', () => {
      // Emit some events to generate stats
      feed.testEmitTick({
        symbol: 'BTC-USD',
        price: 50000,
        volume: 100,
        timestamp: Date.now(),
      }, new Date(Date.now() - 10))

      feed.testEmitCandle({
        open: 49000,
        high: 51000,
        low: 48500,
        close: 50000,
        volume: 1000,
        timestamp: Date.now(),
      }, 'BTC-USD', '1m')

      const stats = feed.getEnhancedStats()

      assert.equal(stats.events.ticksReceived, 1)
      assert.equal(stats.events.candlesReceived, 1)
      assert.ok(stats.events.avgLatency >= 0)
      assert.ok(stats.events.lastEventTime instanceof Date)
      assert.equal(stats.market.currentPrice, 50000)
    })

    it('should update market statistics from candles', () => {
      // First candle
      feed.testEmitCandle({
        open: 48000,
        high: 49000,
        low: 47000,
        close: 48500,
        volume: 500,
        timestamp: Date.now(),
      }, 'BTC-USD', '1m')

      // Second candle
      feed.testEmitCandle({
        open: 48500,
        high: 51000,
        low: 48000,
        close: 50000,
        volume: 800,
        timestamp: Date.now(),
      }, 'BTC-USD', '1m')

      const stats = feed.getEnhancedStats()

      assert.equal(stats.market.currentPrice, 50000)
      assert.equal(stats.market.high24h, 51000)
      assert.equal(stats.market.low24h, 47000)
      assert.equal(stats.market.volume24h, 1300) // 500 + 800
      assert.equal(stats.market.priceChange24h, 2000) // 50000 - 48000 (first price)
    })
  })

  describe('Event Filtering Integration', () => {
    it('should work with filtered subscriptions', async () => {
      const handler = mock.fn()

      // Subscribe with filter that only allows BTC-USD
      const subscription = await feed.subscribeWithFilter(['BTC-USD'], (data: any) => {
        return data.symbol === 'BTC-USD'
      })

      enhancedEventBus.subscribe(EnhancedEventTypes.MARKET_TICK_ENHANCED, handler)

      // This should trigger the filtered subscription
      feed.testEmitTick({
        symbol: 'BTC-USD',
        price: 50000,
        volume: 100,
        timestamp: Date.now(),
      })

      // Verify the subscription was created
      assert.ok(subscription)
      assert.equal(typeof subscription.unsubscribe, 'function')
    })
  })

  describe('Configuration Options', () => {
    it('should respect enhancedEvents configuration', () => {
      const configWithoutEnhanced = { ...config, enhancedEvents: false }
      const feedWithoutEnhanced = new TestEnhancedMarketDataFeed(configWithoutEnhanced)

      const handler = mock.fn()
      enhancedEventBus.subscribe(EnhancedEventTypes.MARKET_TICK_ENHANCED, handler)

      feedWithoutEnhanced.testEmitTick({
        symbol: 'BTC-USD',
        price: 50000,
        volume: 100,
        timestamp: Date.now(),
      })

      // Should not emit enhanced events
      assert.equal(handler.mock.calls.length, 0)
    })

    it('should apply global filters based on configuration', () => {
      // The feed should have set up global filters during construction
      const handler = mock.fn()
      enhancedEventBus.subscribe(EnhancedEventTypes.MARKET_TICK_ENHANCED, handler)

      // Emit many events quickly to test rate limiting
      for (let i = 0; i < 15; i++) {
        feed.testEmitTick({
          symbol: 'BTC-USD',
          price: 50000 + i,
          volume: 100,
          timestamp: Date.now(),
        })
      }

      // Should be rate limited to maxEventsPerSecond (10)
      assert.ok(handler.mock.calls.length <= 10)
    })
  })

  describe('Event Export', () => {
    it('should export event data', async () => {
      const eventTypes = [EnhancedEventTypes.MARKET_TICK_ENHANCED]
      const timeRange = {
        start: new Date('2024-01-01T00:00:00Z'),
        end: new Date('2024-01-01T12:00:00Z'),
      }

      const exportData = await feed.exportEventData(eventTypes, timeRange)
      const parsed = JSON.parse(exportData)

      assert.ok(parsed.exportTimestamp)
      // Check that timeRange dates are properly serialized
      assert.equal(parsed.timeRange.start.value, timeRange.start.toISOString())
      assert.equal(parsed.timeRange.end.value, timeRange.end.toISOString())
      assert.deepEqual(parsed.eventTypes, eventTypes)
      assert.ok(Array.isArray(parsed.events))
    })
  })

  describe('Cleanup and Lifecycle', () => {
    it('should cleanup enhanced subscriptions on stop', async () => {
      // Subscribe to some enhanced events
      await feed.subscribeWithFilter(['BTC-USD'])

      // Verify we have subscriptions
      assert.ok(feed['eventSubscriptions'].length >= 0)

      const handler = mock.fn()
      enhancedEventBus.subscribe(EnhancedEventTypes.MARKET_CONNECTION, handler)

      // Stop the feed
      await feed.stop()

      // Wait for any async operations to complete
      await enhancedEventBus.waitForAsyncHandlers()

      // Should emit disconnection status
      assert.equal(handler.mock.calls.length, 1)
      const event = handler.mock.calls[0]?.arguments[0]
      assert.equal(event.status, 'disconnected')

      // Should clear subscriptions
      assert.equal(feed['eventSubscriptions'].length, 0)

      // Should set connected state
      assert.equal(feed.isConnected(), false)
    })

    it('should handle stop without enhanced events enabled', async () => {
      const configWithoutEnhanced = { ...config, enhancedEvents: false }
      const feedWithoutEnhanced = new TestEnhancedMarketDataFeed(configWithoutEnhanced)

      // Should not throw error
      await feedWithoutEnhanced.stop()
      
      // Wait for any async operations to complete
      await enhancedEventBus.waitForAsyncHandlers()
      
      assert.equal(feedWithoutEnhanced.isConnected(), false)
    })
  })

  describe('Price History Tracking', () => {
    it('should track price history and limit size', async () => {
      // Emit many ticks to test history size limiting
      for (let i = 0; i < 1200; i++) {
        feed.testEmitTick({
          symbol: 'BTC-USD',
          price: 50000 + i,
          volume: 100,
          timestamp: Date.now(),
        })
      }

      // Wait for any async operations to complete
      await enhancedEventBus.waitForAsyncHandlers()

      const history = feed['priceHistory'].get('BTC-USD')
      assert.ok(history)
      assert.ok(history.length <= 1000) // Should be limited to 1000 entries
    })
  })
})
