import { epochDateNow, toEpochDate } from '@trdr/shared'
import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  EnhancedEventTypes,
  EventCompressor,
  EventPriorityClassifier,
  EventSerializer,
  type ConnectionStatusEvent,
  type EnhancedCandleEvent,
  type EnhancedOrderBookEvent,
  type EnhancedTickEvent,
  type MarketStatusEvent,
} from './market-data-events'

describe('MarketDataEvents', () => {
  describe('EventSerializer', () => {
    it('should serialize and deserialize events', () => {
      const event = {
        type: 'test.event',
        timestamp: epochDateNow(),
        sourceTimestamp: epochDateNow() - 1000,
        data: 'test',
      }

      const serialized = EventSerializer.serialize(event)
      const deserialized = EventSerializer.deserialize(serialized) as typeof event

      assert.equal(deserialized.type, 'test.event')
      assert.equal(deserialized.data, 'test')
      assert.equal(deserialized.timestamp, event.timestamp)
      assert.equal(deserialized.sourceTimestamp, event.sourceTimestamp)
    })

    it('should serialize and deserialize complex objects', () => {
      const event = {
        type: 'test.complex',
        timestamp: epochDateNow(),
        nested: {
          data: 'nested value',
          number: 42,
          array: [1, 2, 3],
        },
      }

      const serialized = EventSerializer.serialize(event)
      const deserialized = EventSerializer.deserialize(serialized) as typeof event

      assert.equal(deserialized.type, 'test.complex')
      assert.equal(deserialized.nested.data, 'nested value')
      assert.equal(deserialized.nested.number, 42)
      assert.deepEqual(deserialized.nested.array, [1, 2, 3])
    })




  })

  describe('EventCompressor', () => {
    it('should compress multiple tick events', () => {
      const baseTime = new Date('2024-01-01T12:00:00Z')

      const ticks: EnhancedTickEvent[] = [
        {
          type: 'market.tick.enhanced',
          timestamp: toEpochDate(baseTime.getTime()),
          symbol: 'BTC-USD',
          price: 50000,
          volume: 10,
          source: 'test',
          feedType: 'live',
          priority: 'normal',
        },
        {
          type: 'market.tick.enhanced',
          timestamp: toEpochDate(baseTime.getTime() + 100),
          symbol: 'BTC-USD',
          price: 50050,
          volume: 15,
          source: 'test',
          feedType: 'live',
          priority: 'normal',
        },
        {
          type: 'market.tick.enhanced',
          timestamp: toEpochDate(baseTime.getTime() + 200),
          symbol: 'BTC-USD',
          price: 50100,
          volume: 20,
          source: 'test',
          feedType: 'live',
          priority: 'normal',
        },
      ]

      const compressed = EventCompressor.compressTickEvents(ticks)

      assert.ok(compressed)
      assert.equal(compressed.type, 'market.tick.enhanced')
      assert.equal(compressed.symbol, 'BTC-USD')
      assert.equal(compressed.price, 50100) // Last price
      assert.equal(compressed.volume, 45) // Sum of volumes
      assert.equal(compressed.priceChange, 100) // 50100 - 50000
      assert.equal(compressed.priceChangePercent, 0.2) // (100/50000) * 100
      assert.equal(compressed.timeSinceLastTick, 200) // Time difference
      assert.equal(compressed.source, 'compressed')
    })

    it('should handle single tick event', () => {
      const tick: EnhancedTickEvent = {
        type: 'market.tick.enhanced',
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
        volume: 10,
        source: 'test',
        feedType: 'live',
        priority: 'normal',
      }

      const compressed = EventCompressor.compressTickEvents([tick])

      assert.equal(compressed, tick)
    })

    it('should handle empty tick array', () => {
      const compressed = EventCompressor.compressTickEvents([])

      assert.equal(compressed, null)
    })

    it('should handle undefined/null ticks gracefully', () => {
      const ticks = [undefined, null] as any
      const compressed = EventCompressor.compressTickEvents(ticks)

      assert.equal(compressed, null)
    })

    it('should detect compressible events', () => {
      const baseTime = new Date()

      const events = [
        {
          type: 'test.event',
          timestamp: new Date(baseTime.getTime()),
          symbol: 'BTC-USD',
        },
        {
          type: 'test.event',
          timestamp: new Date(baseTime.getTime() + 500), // 500ms later
          symbol: 'BTC-USD',
        },
      ] as any[]

      assert.equal(EventCompressor.canCompress(events, 1000), true) // Within 1s window
      assert.equal(EventCompressor.canCompress(events, 200), false) // Outside 200ms window
    })

    it('should handle compression edge cases', () => {
      assert.equal(EventCompressor.canCompress([], 1000), false) // Empty array
      assert.equal(EventCompressor.canCompress([{ timestamp: new Date() }] as any, 1000), false) // Single event

      // Events with null/undefined
      const badEvents = [null, undefined] as any
      assert.equal(EventCompressor.canCompress(badEvents, 1000), false)
    })
  })

  describe('EventPriorityClassifier', () => {
    it('should classify market status events as critical', () => {
      const statusEvent = {
        type: 'market.status',
        symbol: 'BTC-USD',
        timestamp: new Date(),
      } as any

      const priority = EventPriorityClassifier.classifyPriority(statusEvent as any)
      assert.equal(priority, 'critical')
    })

    it('should classify connection events as high priority', () => {
      const connectionEvent = {
        type: 'market.connection',
        symbol: 'BTC-USD',
        timestamp: new Date(),
      } as any

      const priority = EventPriorityClassifier.classifyPriority(connectionEvent)
      assert.equal(priority, 'high')
    })

    it('should classify large price movements as high priority', () => {
      const priceEvent = {
        type: 'market.tick.enhanced',
        symbol: 'BTC-USD',
        priceChangePercent: 8.5, // > 5%
        timestamp: new Date(),
      } as any

      const priority = EventPriorityClassifier.classifyPriority(priceEvent)
      assert.equal(priority, 'high')
    })

    it('should classify high volume events as high priority', () => {
      const volumeEvent = {
        type: 'market.tick.enhanced',
        symbol: 'BTC-USD',
        volume: 15000, // > 10000
        timestamp: new Date(),
      } as any

      const priority = EventPriorityClassifier.classifyPriority(volumeEvent)
      assert.equal(priority, 'high')
    })

    it('should classify wide spreads as high priority', () => {
      const spreadEvent = {
        type: 'market.orderbook.enhanced',
        symbol: 'BTC-USD',
        spread: 600, // Large spread
        price: 50000, // 1.2% spread
        timestamp: new Date(),
      } as any

      const priority = EventPriorityClassifier.classifyPriority(spreadEvent)
      assert.equal(priority, 'high')
    })

    it('should classify normal events as normal priority', () => {
      const normalEvent = {
        type: 'market.tick.enhanced',
        symbol: 'BTC-USD',
        priceChangePercent: 1.2, // Small change
        volume: 500, // Low volume
        timestamp: new Date(),
      } as any

      const priority = EventPriorityClassifier.classifyPriority(normalEvent)
      assert.equal(priority, 'normal')
    })

    it('should handle events without price or volume data', () => {
      const basicEvent = {
        type: 'custom.event',
        symbol: 'BTC-USD',
        timestamp: new Date(),
      } as any

      const priority = EventPriorityClassifier.classifyPriority(basicEvent)
      assert.equal(priority, 'normal')
    })
  })

  describe('Enhanced Event Types', () => {
    it('should have all required enhanced event types', () => {
      const expectedTypes = [
        'market.tick.enhanced',
        'market.candle.enhanced',
        'market.orderbook.enhanced',
        'market.status',
        'market.connection',
        'market.summary',
        'market.statistics',
      ]

      const actualTypes = Object.values(EnhancedEventTypes)

      expectedTypes.forEach(type => {
        assert.ok(actualTypes.includes(type as any), `Missing event type: ${type}`)
      })
    })

    it('should validate enhanced tick event structure', () => {
      const tickEvent: EnhancedTickEvent = {
        type: 'market.tick.enhanced',
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
        volume: 100,
        source: 'test',
        feedType: 'live',
        priority: 'normal',
        sequence: 1,
        priceChange: 100,
        priceChangePercent: 0.2,
        bid: 49950,
        ask: 50050,
        spread: 100,
        sourceTimestamp: epochDateNow(),
        latency: 10,
        timeSinceLastTick: 1000,
      }

      // Test that all required fields are present
      assert.equal(tickEvent.type, 'market.tick.enhanced')
      assert.equal(tickEvent.symbol, 'BTC-USD')
      assert.equal(tickEvent.price, 50000)
      assert.equal(tickEvent.feedType, 'live')
      assert.equal(tickEvent.priority, 'normal')
    })

    it('should validate enhanced candle event structure', () => {
      const candleEvent: EnhancedCandleEvent = {
        type: 'market.candle.enhanced',
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        open: 49000,
        high: 51000,
        low: 48500,
        close: 50000,
        volume: 1000,
        interval: '1m',
        source: 'test',
        feedType: 'live',
        priority: 'normal',
        range: 2500,
        bodySize: 1000,
        upperWick: 1000,
        lowerWick: 500,
        vwap: 49750,
        tradeCount: 150,
        typicalPrice: 49833.33,
        candleType: 'bullish',
      }

      assert.equal(candleEvent.type, 'market.candle.enhanced')
      assert.equal(candleEvent.candleType, 'bullish')
      assert.equal(candleEvent.range, 2500)
      assert.equal(candleEvent.bodySize, 1000)
    })

    it('should validate enhanced order book event structure', () => {
      const orderBookEvent: EnhancedOrderBookEvent = {
        type: 'market.orderbook.enhanced',
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        source: 'test',
        feedType: 'live',
        priority: 'normal',
        bids: [
          { price: 49950, size: 10 },
          { price: 49900, size: 15 },
        ],
        asks: [
          { price: 50050, size: 8 },
          { price: 50100, size: 12 },
        ],
        bestBid: 49950,
        bestAsk: 50050,
        spread: 100,
        midPrice: 50000,
        totalBidVolume: 25,
        totalAskVolume: 20,
        depth: 2,
        imbalance: 0.111, // (25-20)/(25+20)
      }

      assert.equal(orderBookEvent.type, 'market.orderbook.enhanced')
      assert.equal(orderBookEvent.bestBid, 49950)
      assert.equal(orderBookEvent.bestAsk, 50050)
      assert.equal(orderBookEvent.depth, 2)
    })

    it('should validate market status event structure', () => {
      const statusEvent: MarketStatusEvent = {
        type: 'market.status',
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        source: 'test',
        feedType: 'live',
        priority: 'critical',
        status: 'open',
        session: {
          name: 'Regular Trading',
          start: new Date('2024-01-01T09:30:00Z'),
          end: new Date('2024-01-01T16:00:00Z'),
        },
        stats: {
          volume24h: 1000000,
          high24h: 52000,
          low24h: 48000,
          priceChange24h: 1000,
          priceChangePercent24h: 2.0,
        },
      }

      assert.equal(statusEvent.type, 'market.status')
      assert.equal(statusEvent.status, 'open')
      assert.equal(statusEvent.priority, 'critical')
    })

    it('should validate connection status event structure', () => {
      const connectionEvent: ConnectionStatusEvent = {
        type: 'market.connection',
        timestamp: epochDateNow(),
        source: 'test',
        feedType: 'live',
        status: 'connected',
        details: {
          reconnectAttempts: 0,
          uptime: 3600000, // 1 hour
          lastError: undefined,
          subscriptions: ['BTC-USD', 'ETH-USD'],
        },
      }

      assert.equal(connectionEvent.type, 'market.connection')
      assert.equal(connectionEvent.status, 'connected')
      assert.equal(connectionEvent.details?.uptime, 3600000)
    })
  })
})
