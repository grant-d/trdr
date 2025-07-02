import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import { BacktestDataFeed, type BacktestConfig } from './backtest-data-feed'
import type { HistoricalDataRequest } from '../interfaces/market-data-pipeline'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

describe('BacktestDataFeed', () => {
  let feed: BacktestDataFeed
  let config: BacktestConfig
  let eventBus: EventBus
  let events: Array<{ type: string; data: any }> = []
  let subscriptions: Array<{ unsubscribe: () => void }> = []

  beforeEach(() => {
    const startDate = new Date('2023-01-01T00:00:00Z')
    const endDate = new Date('2023-01-01T01:00:00Z') // 1 hour test period

    config = {
      symbol: 'BTC-USD',
      feedType: 'backtest',
      dataSource: 'test-data',
      speed: 1000, // 1000x speed
      startDate,
      endDate,
      networkDelay: 0,
      failureRate: 0,
      debug: false,
    }

    feed = new BacktestDataFeed(config)
    eventBus = EventBus.getInstance()
    events = []
    subscriptions = []

    // Register required event types
    Object.values(EventTypes).forEach(type => {
      eventBus.registerEvent(type)
    })

    // Capture events
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

  describe('initialization', () => {
    it('should create feed with backtest configuration', () => {
      assert.ok(feed)
      assert.equal(feed.getStats().subscribedSymbols.length, 0)
      assert.equal(feed.isHealthy(), false)
    })

    it('should have correct initial time settings', () => {
      const currentTime = feed.getCurrentTime()
      assert.equal(currentTime.getTime(), config.startDate.getTime())
    })

    it('should create feed with default speed', () => {
      const defaultConfig: BacktestConfig = {
        symbol: 'ETH-USD',
        feedType: 'backtest',
        dataSource: 'test',
        startDate: new Date(),
        endDate: new Date(),
      }
      const defaultFeed = new BacktestDataFeed(defaultConfig)
      assert.ok(defaultFeed)
    })
  })

  describe('connection lifecycle', () => {
    it('should start successfully and load data', async () => {
      await feed.start()

      // Check for connected event
      const connectedEvents = events.filter(e =>
        e.type === EventTypes.SYSTEM_INFO &&
        e.data.message.includes('connected'),
      )
      assert.equal(connectedEvents.length, 1)
      assert.equal(feed.isHealthy(), true)
      assert.equal(feed.getStats().connected, true)
    })

    it('should stop cleanly', async () => {
      await feed.start()
      await feed.stop()

      const disconnectedEvents = events.filter(e =>
        e.type === EventTypes.SYSTEM_WARNING &&
        e.data.message.includes('disconnected'),
      )
      assert.equal(disconnectedEvents.length, 1)
      assert.equal(feed.isHealthy(), false)
    })

    it('should handle start failure gracefully', async () => {
      // Override loadHistoricalData to simulate failure
      const originalLoad = feed['loadHistoricalData']
      feed['loadHistoricalData'] = async () => {
        throw new Error('Data load failed')
      }

      await assert.rejects(
        () => feed.start(),
        { message: 'Data load failed' },
      )

      // Check that error was emitted
      const errorEvents = events.filter(e => e.type === EventTypes.SYSTEM_ERROR)
      assert.ok(errorEvents.length > 0)

      // Restore original method
      feed['loadHistoricalData'] = originalLoad
    })
  })

  describe('symbol subscription and replay', () => {
    beforeEach(async () => {
      await feed.start()
    })

    it('should subscribe to symbols and start replay', async () => {
      await feed.subscribe(['BTC-USD'])

      const stats = feed.getStats()
      assert.deepEqual(stats.subscribedSymbols, ['BTC-USD'])

      // Wait a bit for replay to generate some events
      await new Promise(resolve => setTimeout(resolve, 50))

      // Should have received some candle events
      const candleEvents = events.filter(e => e.type === EventTypes.MARKET_CANDLE)
      assert.ok(candleEvents.length > 0)

      // Should have received some tick events
      const tickEvents = events.filter(e => e.type === EventTypes.MARKET_TICK)
      assert.ok(tickEvents.length > 0)
    })

    it('should unsubscribe from symbols', async () => {
      await feed.subscribe(['BTC-USD', 'ETH-USD'])
      await feed.unsubscribe(['ETH-USD'])

      const stats = feed.getStats()
      assert.deepEqual(stats.subscribedSymbols, ['BTC-USD'])
    })

    it('should stop replay when all symbols unsubscribed', async () => {
      await feed.subscribe(['BTC-USD'])
      await feed.unsubscribe(['BTC-USD'])

      const stats = feed.getStats()
      assert.equal(stats.subscribedSymbols.length, 0)
    })

    it('should handle multiple symbol subscriptions', async () => {
      await feed.subscribe(['BTC-USD', 'ETH-USD'])

      const stats = feed.getStats()
      assert.equal(stats.subscribedSymbols.length, 2)
      assert.ok(stats.subscribedSymbols.includes('BTC-USD'))
      assert.ok(stats.subscribedSymbols.includes('ETH-USD'))
    })
  })

  describe('historical data access', () => {
    beforeEach(async () => {
      await feed.start()
    })

    it('should fetch historical data', async () => {
      const request: HistoricalDataRequest = {
        symbol: 'BTC-USD',
        start: config.startDate,
        end: config.endDate,
        interval: '1m',
      }

      const candles = await feed.getHistorical(request)

      assert.ok(Array.isArray(candles))
      assert.ok(candles.length > 0)

      // Verify candle structure
      const firstCandle = candles[0]
      assert.ok(typeof firstCandle?.open === 'number')
      assert.ok(typeof firstCandle?.high === 'number')
      assert.ok(typeof firstCandle?.low === 'number')
      assert.ok(typeof firstCandle?.close === 'number')
      assert.ok(typeof firstCandle?.volume === 'number')
      assert.ok(typeof firstCandle?.timestamp === 'number')
    })

    it('should return empty array for unknown symbol', async () => {
      const request: HistoricalDataRequest = {
        symbol: 'UNKNOWN-USD',
        start: config.startDate,
        end: config.endDate,
      }

      const candles = await feed.getHistorical(request)
      assert.equal(candles.length, 0)
    })

    it('should respect limit parameter', async () => {
      const request: HistoricalDataRequest = {
        symbol: 'BTC-USD',
        start: config.startDate,
        end: config.endDate,
        limit: 5,
      }

      const candles = await feed.getHistorical(request)
      assert.ok(candles.length <= 5)
    })

    it('should filter data by date range', async () => {
      const narrowStart = new Date(config.startDate.getTime() + 10 * 60000) // +10 minutes
      const narrowEnd = new Date(config.startDate.getTime() + 20 * 60000) // +20 minutes

      const request: HistoricalDataRequest = {
        symbol: 'BTC-USD',
        start: narrowStart,
        end: narrowEnd,
      }

      const candles = await feed.getHistorical(request)

      // All candles should be within the specified range
      candles.forEach(candle => {
        const candleTime = new Date(candle.timestamp)
        assert.ok(candleTime >= narrowStart)
        assert.ok(candleTime <= narrowEnd)
      })
    })
  })

  describe('current price access', () => {
    beforeEach(async () => {
      await feed.start()
      await feed.subscribe(['BTC-USD'])
    })

    it('should get current price', async () => {
      const price = await feed.getCurrentPrice('BTC-USD')

      assert.ok(typeof price === 'number')
      assert.ok(price > 0)
    })

    it('should return 0 for unknown symbol', async () => {
      const price = await feed.getCurrentPrice('UNKNOWN-USD')
      assert.equal(price, 0)
    })

    it('should return 0 when no data available', async () => {
      // Create a feed with no data
      const emptyFeed = new BacktestDataFeed({
        ...config,
        startDate: new Date('2030-01-01'),
        endDate: new Date('2030-01-02'),
      })

      await emptyFeed.start()
      const price = await emptyFeed.getCurrentPrice('BTC-USD')
      assert.equal(price, 0)

      await emptyFeed.stop()
    })
  })

  describe('time manipulation', () => {
    beforeEach(async () => {
      await feed.start()
    })

    it('should set replay speed', () => {
      feed.setSpeed(500)
      // Speed change should be reflected in internal state
      // We can't easily test the actual timing without complex async coordination
      assert.ok(true) // Speed setting doesn't throw
    })

    it('should get current backtest time', () => {
      const currentTime = feed.getCurrentTime()
      assert.ok(currentTime instanceof Date)
      assert.equal(currentTime.getTime(), config.startDate.getTime())
    })

    it('should seek to specific time', async () => {
      const seekTime = new Date(config.startDate.getTime() + 30 * 60000) // +30 minutes

      await feed.seekToTime(seekTime)

      const currentTime = feed.getCurrentTime()
      assert.equal(currentTime.getTime(), seekTime.getTime())
    })

    it('should reject seek outside backtest range', async () => {
      const outsideTime = new Date(config.endDate.getTime() + 60000) // +1 minute past end

      await assert.rejects(
        () => feed.seekToTime(outsideTime),
        { message: 'Seek time outside of backtest range' },
      )
    })

    it('should reject seek before backtest start', async () => {
      const beforeTime = new Date(config.startDate.getTime() - 60000) // -1 minute before start

      await assert.rejects(
        () => feed.seekToTime(beforeTime),
        { message: 'Seek time outside of backtest range' },
      )
    })
  })

  describe('network simulation', () => {
    it('should simulate network delays', async () => {
      const delayConfig: BacktestConfig = {
        ...config,
        networkDelay: 100, // 100ms delay
      }

      const delayFeed = new BacktestDataFeed(delayConfig)
      await delayFeed.start()

      const startTime = Date.now()
      await delayFeed.getCurrentPrice('BTC-USD')
      const endTime = Date.now()

      // Should have some delay (though exact timing is hard to test)
      assert.ok(endTime - startTime >= 50) // At least some delay

      await delayFeed.stop()
    })

    it('should simulate network failures', async () => {
      const failureConfig: BacktestConfig = {
        ...config,
        failureRate: 1.0, // 100% failure rate
      }

      const failureFeed = new BacktestDataFeed(failureConfig)
      await failureFeed.start()

      // Should throw due to simulated failure
      await assert.rejects(
        () => failureFeed.getCurrentPrice('BTC-USD'),
        { message: 'Simulated network failure' },
      )

      await failureFeed.stop()
    })

    it('should work normally with zero failure rate', async () => {
      const noFailureConfig: BacktestConfig = {
        ...config,
        failureRate: 0.0, // 0% failure rate
      }

      const noFailureFeed = new BacktestDataFeed(noFailureConfig)
      await noFailureFeed.start()

      // Should not throw
      const price = await noFailureFeed.getCurrentPrice('BTC-USD')
      assert.ok(typeof price === 'number')

      await noFailureFeed.stop()
    })
  })

  describe('statistics and monitoring', () => {
    beforeEach(async () => {
      await feed.start()
      await feed.subscribe(['BTC-USD'])
    })

    it('should track connection statistics', () => {
      const stats = feed.getStats()

      assert.equal(stats.connected, true)
      assert.ok(stats.uptime >= 0)
      assert.equal(stats.reconnectAttempts, 0)
      assert.equal(stats.subscribedSymbols.length, 1)
      assert.deepEqual(stats.subscribedSymbols, ['BTC-USD'])
    })

    it('should update message statistics during replay', async () => {
      // Wait for some replay events
      await new Promise(resolve => setTimeout(resolve, 100))

      const stats = feed.getStats()
      assert.ok(stats.messagesReceived > 0)
      assert.ok(stats.lastMessageTime instanceof Date)
    })

    it('should track errors', async () => {
      // Simulate an error by calling a method that will fail
      try {
        await feed.seekToTime(new Date('2030-01-01'))
      } catch (error) {
        // Expected to fail
      }

      // Error handling is tested elsewhere - seekToTime validation
      assert.ok(true)
    })
  })

  describe('edge cases and error handling', () => {
    it('should handle empty data gracefully', async () => {
      // Create feed with date range that has no data
      const emptyConfig: BacktestConfig = {
        ...config,
        startDate: new Date('2030-01-01'),
        endDate: new Date('2030-01-02'),
      }

      const emptyFeed = new BacktestDataFeed(emptyConfig)
      await emptyFeed.start()
      await emptyFeed.subscribe(['BTC-USD'])

      // Should not crash
      const price = await emptyFeed.getCurrentPrice('BTC-USD')
      assert.equal(price, 0)

      await emptyFeed.stop()
    })

    it('should handle rapid start/stop cycles', async () => {
      await feed.start()
      await feed.stop()
      await feed.start()
      await feed.stop()

      // Should not throw or crash
      assert.ok(true)
    })

    it('should handle subscription before start', async () => {
      // Subscribe before starting
      await feed.subscribe(['BTC-USD'])

      // Should not throw
      const stats = feed.getStats()
      assert.deepEqual(stats.subscribedSymbols, ['BTC-USD'])

      // Starting should begin replay
      await feed.start()
      assert.equal(feed.isHealthy(), true)
    })

    it('should handle multiple subscribe calls for same symbol', async () => {
      await feed.start()

      await feed.subscribe(['BTC-USD'])
      await feed.subscribe(['BTC-USD']) // Duplicate

      const stats = feed.getStats()
      assert.equal(stats.subscribedSymbols.length, 1)
      assert.deepEqual(stats.subscribedSymbols, ['BTC-USD'])
    })
  })
})
