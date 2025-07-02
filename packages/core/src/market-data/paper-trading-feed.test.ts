import { describe, it, beforeEach, afterEach } from 'node:test'
import assert from 'node:assert/strict'
import {
  PaperTradingFeed,
  type PaperTradingConfig,
  type MarketScenario,
  type ExecutionSimulation,
} from './paper-trading-feed'
import type { HistoricalDataRequest } from '../interfaces/market-data-pipeline'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

describe('PaperTradingFeed', () => {
  let feed: PaperTradingFeed
  let config: PaperTradingConfig
  let eventBus: EventBus
  let events: Array<{ type: string; data: any }> = []
  let subscriptions: Array<{ unsubscribe: () => void }> = []

  beforeEach(() => {
    config = {
      symbol: 'BTC-USD',
      feedType: 'paper',
      baseDataSource: 'coinbase',
      baseFeedConfig: {
        symbol: 'BTC-USD',
        feedType: 'coinbase',
        debug: false,
      },
      slippage: 20, // 0.2%
      executionDelay: 50, // 50ms
      maxPriceImpact: 100, // 1%
      liquidityMultiplier: 1.0,
      acceleratedTime: false,
      timeAcceleration: 1,
      enableCustomScenarios: true,
      debug: false,
    }

    feed = new PaperTradingFeed(config)
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
    it('should create feed with paper trading configuration', () => {
      assert.ok(feed)
      assert.equal(feed.getStats().subscribedSymbols.length, 0)
      assert.equal(feed.isHealthy(), false)
    })

    it('should create feed with default configuration values', () => {
      const minimalConfig: PaperTradingConfig = {
        symbol: 'ETH-USD',
        feedType: 'paper',
        baseDataSource: 'coinbase',
        baseFeedConfig: {
          symbol: 'ETH-USD',
          feedType: 'coinbase',
        },
      }

      const defaultFeed = new PaperTradingFeed(minimalConfig)
      assert.ok(defaultFeed)
    })

    it('should initialize with accelerated time configuration', () => {
      const acceleratedConfig: PaperTradingConfig = {
        ...config,
        acceleratedTime: true,
        timeAcceleration: 5,
      }

      const acceleratedFeed = new PaperTradingFeed(acceleratedConfig)
      assert.ok(acceleratedFeed)
    })
  })

  describe('connection lifecycle', () => {
    it('should start successfully', async () => {
      // Mock the base feed to avoid actual network calls
      const originalStart = feed['baseFeed'].start
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
        feed['baseFeed']['startTime'] = new Date()
      }

      await feed.start()

      // Check for connected event
      const connectedEvents = events.filter(e =>
        e.type === EventTypes.SYSTEM_INFO &&
        e.data.message && e.data.message.includes('connected'),
      )
      assert.ok(connectedEvents.length > 0)
      assert.equal(feed.isHealthy(), true)

      // Restore original method
      feed['baseFeed'].start = originalStart
    })

    it('should stop cleanly', async () => {
      // Mock base feed
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].stop = async () => {
        feed['baseFeed']['connected'] = false
      }

      await feed.start()
      await feed.stop()

      const disconnectedEvents = events.filter(e =>
        e.type === EventTypes.SYSTEM_WARNING &&
        e.data.message && e.data.message.includes('disconnected'),
      )
      assert.ok(disconnectedEvents.length > 0)
      assert.equal(feed.isHealthy(), false)
    })

    it('should handle start failure gracefully', async () => {
      // Mock base feed to throw error
      feed['baseFeed'].start = async () => {
        throw new Error('Base feed connection failed')
      }

      await assert.rejects(
        () => feed.start(),
        { message: 'Base feed connection failed' },
      )

      // Check that error was emitted
      const errorEvents = events.filter(e => e.type === EventTypes.SYSTEM_ERROR)
      assert.ok(errorEvents.length > 0)
    })
  })

  describe('symbol subscription', () => {
    beforeEach(async () => {
      // Mock base feed
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].subscribe = async () => {
      }
      feed['baseFeed'].unsubscribe = async () => {
      }

      await feed.start()
    })

    it('should subscribe to symbols', async () => {
      await feed.subscribe(['BTC-USD', 'ETH-USD'])

      const stats = feed.getStats()
      assert.equal(stats.subscribedSymbols.length, 2)
      assert.ok(stats.subscribedSymbols.includes('BTC-USD'))
      assert.ok(stats.subscribedSymbols.includes('ETH-USD'))
    })

    it('should unsubscribe from symbols', async () => {
      await feed.subscribe(['BTC-USD', 'ETH-USD'])
      await feed.unsubscribe(['ETH-USD'])

      const stats = feed.getStats()
      assert.equal(stats.subscribedSymbols.length, 1)
      assert.deepEqual(stats.subscribedSymbols, ['BTC-USD'])
    })
  })

  describe('historical data access', () => {
    beforeEach(async () => {
      // Mock base feed
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].getHistorical = async () => [{
        timestamp: Date.now(),
        open: 50000,
        high: 51000,
        low: 49000,
        close: 50500,
        volume: 100,
      }]

      await feed.start()
    })

    it('should fetch historical data from base feed', async () => {
      const request: HistoricalDataRequest = {
        symbol: 'BTC-USD',
        start: new Date('2023-01-01'),
        end: new Date('2023-01-02'),
        interval: '1h',
      }

      const candles = await feed.getHistorical(request)

      assert.ok(Array.isArray(candles))
      assert.equal(candles.length, 1)
      assert.equal(candles[0]?.close, 50500)
    })
  })

  describe('current price access', () => {
    beforeEach(async () => {
      // Mock base feed
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].getCurrentPrice = async () => 50000

      await feed.start()
    })

    it('should get current price from base feed', async () => {
      const price = await feed.getCurrentPrice('BTC-USD')

      assert.ok(typeof price === 'number')
      assert.equal(price, 50000)
    })

    it('should apply market scenarios to price', async () => {
      // Add a bullish scenario
      const scenario: MarketScenario = {
        id: 'bullish-test',
        startTime: new Date(Date.now() - 1000),
        duration: 10000,
        volatilityMultiplier: 1.5,
        trendDirection: 1,
        liquidityImpact: 1.0,
      }

      feed.addMarketScenario(scenario)
      const price = await feed.getCurrentPrice('BTC-USD')

      assert.ok(typeof price === 'number')
      // Price should be adjusted by scenario (exact value depends on timing)
      assert.ok(price !== 50000)
    })
  })

  describe('order execution simulation', () => {
    beforeEach(async () => {
      // Mock base feed
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].getCurrentPrice = async () => 50000

      await feed.start()
    })

    it('should simulate market order execution', async () => {
      const execution: ExecutionSimulation = {
        size: 1.0,
        side: 'buy',
        timestamp: new Date(),
      }

      const result = await feed.simulateExecution('BTC-USD', execution)

      assert.ok(typeof result.executedPrice === 'number')
      assert.equal(result.executedSize, 1.0)
      assert.ok(result.slippage >= 0)
      assert.ok(result.priceImpact >= 0)
      assert.ok(result.executionDelay >= config.executionDelay!)
      assert.ok(result.executionTime instanceof Date)

      // Buy order should execute at higher price due to slippage
      assert.ok(result.executedPrice > 50000)
    })

    it('should simulate limit order execution', async () => {
      const execution: ExecutionSimulation = {
        size: 1.0,
        side: 'buy',
        limitPrice: 51000,
        timestamp: new Date(),
      }

      const result = await feed.simulateExecution('BTC-USD', execution)

      // Should not exceed limit price
      assert.ok(result.executedPrice <= 51000)
    })

    it('should simulate sell order execution', async () => {
      const execution: ExecutionSimulation = {
        size: 1.0,
        side: 'sell',
        timestamp: new Date(),
      }

      const result = await feed.simulateExecution('BTC-USD', execution)

      // Sell order should execute at lower price due to slippage
      assert.ok(result.executedPrice < 50000)
    })

    it('should apply size-based slippage', async () => {
      const smallOrder: ExecutionSimulation = {
        size: 0.1,
        side: 'buy',
        timestamp: new Date(),
      }

      const largeOrder: ExecutionSimulation = {
        size: 10.0,
        side: 'buy',
        timestamp: new Date(),
      }

      const smallResult = await feed.simulateExecution('BTC-USD', smallOrder)
      const largeResult = await feed.simulateExecution('BTC-USD', largeOrder)

      // Large order should have higher slippage
      assert.ok(largeResult.slippage > smallResult.slippage)
      assert.ok(largeResult.executionDelay >= smallResult.executionDelay)
    })

    it('should emit execution events', async () => {
      const execution: ExecutionSimulation = {
        size: 1.0,
        side: 'buy',
        timestamp: new Date(),
      }

      await feed.simulateExecution('BTC-USD', execution)

      const executionEvents = events.filter(e =>
        e.type === EventTypes.SYSTEM_INFO &&
        e.data.message === 'Paper trade executed',
      )
      assert.equal(executionEvents.length, 1)

      const eventData = executionEvents[0]?.data
      assert.equal(eventData.details.symbol, 'BTC-USD')
      assert.equal(eventData.details.side, 'buy')
      assert.equal(eventData.details.size, 1.0)
    })
  })

  describe('market scenarios', () => {
    beforeEach(async () => {
      // Mock base feed
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].getCurrentPrice = async () => 50000

      await feed.start()
    })

    it('should add and remove market scenarios', () => {
      const scenario: MarketScenario = {
        id: 'test-scenario',
        startTime: new Date(),
        duration: 5000,
        volatilityMultiplier: 2.0,
        trendDirection: 1,
        liquidityImpact: 0.5,
      }

      feed.addMarketScenario(scenario)
      // Accessing private property for testing
      assert.equal(feed['activeScenarios'].size, 1)

      feed.removeMarketScenario('test-scenario')
      assert.equal(feed['activeScenarios'].size, 0)
    })

    it('should throw error when scenarios disabled', () => {
      const scenarioFeed = new PaperTradingFeed({
        ...config,
        enableCustomScenarios: false,
      })

      const scenario: MarketScenario = {
        id: 'test',
        startTime: new Date(),
        duration: 1000,
        volatilityMultiplier: 1.0,
        trendDirection: 0,
        liquidityImpact: 1.0,
      }

      assert.throws(
        () => scenarioFeed.addMarketScenario(scenario),
        { message: 'Custom scenarios are not enabled' },
      )
    })

    it('should apply liquidity impact to slippage', async () => {
      // Add scenario with reduced liquidity
      const scenario: MarketScenario = {
        id: 'low-liquidity',
        startTime: new Date(Date.now() - 1000),
        duration: 10000,
        volatilityMultiplier: 1.0,
        trendDirection: 0,
        liquidityImpact: 0.5, // 50% liquidity
      }

      feed.addMarketScenario(scenario)

      // First, simulate without scenario to get baseline
      const baselineExecution: ExecutionSimulation = {
        size: 2000, // Use larger size for meaningful slippage
        side: 'buy',
        timestamp: new Date(),
      }

      // Remove scenario temporarily to get baseline
      feed.removeMarketScenario('low-liquidity')
      const baselineResult = await feed.simulateExecution('BTC-USD', baselineExecution)

      // Re-add scenario and test
      feed.addMarketScenario(scenario)
      const scenarioResult = await feed.simulateExecution('BTC-USD', baselineExecution)

      // Should have higher slippage due to reduced liquidity
      assert.ok(scenarioResult.slippage > baselineResult.slippage)
    })
  })

  describe('time acceleration', () => {
    it('should support accelerated time mode', () => {
      const acceleratedFeed = new PaperTradingFeed({
        ...config,
        acceleratedTime: true,
        timeAcceleration: 5,
      })

      // Mock start time
      acceleratedFeed['startTime'] = new Date(Date.now() - 1000)

      const virtualTime = acceleratedFeed.getVirtualTime()
      const realTime = new Date()

      // Virtual time should be ahead of real time
      assert.ok(virtualTime.getTime() > realTime.getTime())
    })

    it('should set time acceleration factor', () => {
      feed.setTimeAcceleration(10)
      assert.equal(feed['timeAcceleration'], 10)

      // Should clamp to valid range
      feed.setTimeAcceleration(200)
      assert.equal(feed['timeAcceleration'], 100)

      feed.setTimeAcceleration(0.01)
      assert.equal(feed['timeAcceleration'], 0.1)
    })

    it('should return real time when acceleration disabled', () => {
      const normalTime = feed.getVirtualTime()
      const realTime = new Date()

      // Should be approximately equal (within 100ms)
      assert.ok(Math.abs(normalTime.getTime() - realTime.getTime()) < 100)
    })
  })

  describe('health monitoring', () => {
    it('should report healthy when base feed is healthy', async () => {
      // Mock base feed
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].isHealthy = () => true

      await feed.start()
      assert.equal(feed.isHealthy(), true)
    })

    it('should report unhealthy when base feed is unhealthy', async () => {
      // Mock base feed
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].isHealthy = () => false

      await feed.start()
      assert.equal(feed.isHealthy(), false)
    })

    it('should report unhealthy when not connected', () => {
      assert.equal(feed.isHealthy(), false)
    })
  })

  describe('statistics', () => {
    beforeEach(async () => {
      // Mock base feed
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].getStats = () => ({
        connected: true,
        uptime: 1000,
        reconnectAttempts: 0,
        messagesReceived: 10,
        lastMessageTime: new Date(),
        subscribedSymbols: ['BTC-USD'],
      })

      await feed.start()
      await feed.subscribe(['BTC-USD'])
    })

    it('should combine base feed stats with paper trading stats', () => {
      const stats = feed.getStats()

      assert.equal(stats.connected, true)
      assert.ok(stats.uptime >= 0)
      assert.equal(stats.reconnectAttempts, 0)
      assert.equal(stats.messagesReceived, 10)
      assert.equal(stats.subscribedSymbols.length, 1)
      assert.deepEqual(stats.subscribedSymbols, ['BTC-USD'])
    })
  })

  describe('edge cases and error handling', () => {
    it('should handle execution with zero size', async () => {
      // Mock base feed
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].getCurrentPrice = async () => 50000

      await feed.start()

      const execution: ExecutionSimulation = {
        size: 0,
        side: 'buy',
        timestamp: new Date(),
      }

      const result = await feed.simulateExecution('BTC-USD', execution)
      assert.equal(result.executedSize, 0)
    })

    it('should handle very large orders', async () => {
      // Mock base feed
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].getCurrentPrice = async () => 50000

      await feed.start()

      const execution: ExecutionSimulation = {
        size: 1000000, // Very large order
        side: 'buy',
        timestamp: new Date(),
      }

      const result = await feed.simulateExecution('BTC-USD', execution)

      // Should have maximum slippage
      assert.ok(result.slippage > 0)
      assert.ok(result.priceImpact <= config.maxPriceImpact!)
    })

    it('should handle price calculation edge cases', async () => {
      // Mock base feed to return edge case prices
      feed['baseFeed'].start = async () => {
        feed['baseFeed']['connected'] = true
      }
      feed['baseFeed'].getCurrentPrice = async () => 0.001 // Very small price

      await feed.start()

      const execution: ExecutionSimulation = {
        size: 1.0,
        side: 'buy',
        timestamp: new Date(),
      }

      const result = await feed.simulateExecution('BTC-USD', execution)
      assert.ok(result.executedPrice > 0)
    })
  })
})
