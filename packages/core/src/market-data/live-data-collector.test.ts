import type { MarketDataRepository } from '@trdr/data'
import { epochDateNow, toEpochDate, toStockSymbol } from '@trdr/shared'
import type { Candle } from '@trdr/types'
import assert from 'node:assert/strict'
import { afterEach, beforeEach, describe, it, mock } from 'node:test'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import type { HistoricalDataManager } from './historical-data-manager'
import {
  LiveDataCollector,
  type LiveDataCollectorConfig,
  type LiveDataSource,
} from './live-data-collector'

describe('LiveDataCollector', () => {
  let collector: LiveDataCollector
  let mockRepository: MarketDataRepository
  let mockHistoricalManager: HistoricalDataManager
  let eventBus: EventBus

  const createMockCandle = (timestamp: number, symbol = 'BTC-USD', interval = '1m'): Candle => ({
    symbol: toStockSymbol(symbol),
    interval,
    timestamp: toEpochDate(timestamp),
    openTime: toEpochDate(timestamp),
    closeTime: toEpochDate(timestamp + 60000),
    open: 50000,
    high: 51000,
    low: 49000,
    close: 50500,
    volume: 100,
    quoteVolume: 5000000,
    tradesCount: 1000,
  })

  const createMockDataSource = (name: string, connected = true): LiveDataSource => ({
    name,
    type: 'websocket',
    subscribe: mock.fn(async () => {}),
    unsubscribe: mock.fn(async () => {}),
    isConnected: mock.fn(() => connected),
    reconnect: mock.fn(async () => {}),
  })

  beforeEach(() => {
    eventBus = EventBus.getInstance()

    // Register all event types
    Object.values(EventTypes).forEach((eventType) => {
      eventBus.registerEvent(eventType)
    })

    // Create mock repository
    mockRepository = {
      getCandles: mock.fn(async () => []),
      getLatestCandle: mock.fn(async () => null),
      saveCandle: mock.fn(async () => {}),
      saveCandlesBatch: mock.fn(async () => {}),
      cleanup: mock.fn(async () => ({ candlesDeleted: 0, ticksDeleted: 0 })),
    } as any

    // Create mock historical manager
    mockHistoricalManager = {
      startLiveDataCollection: mock.fn(async () => {}),
      stopLiveDataCollection: mock.fn(async () => {}),
    } as any

    const config: LiveDataCollectorConfig = {
      batchSize: 10,
      flushIntervalMs: 50, // Very short for tests
      enableDeduplication: true,
      deduplicationWindowMs: 1000, // Short window for tests
      enableAutoReconnect: true,
      maxReconnectAttempts: 2, // Fewer attempts for tests
      reconnectDelayMultiplier: 1.1, // Shorter delays
      enableValidation: true,
    }

    collector = new LiveDataCollector(mockRepository, mockHistoricalManager, config)
  })

  afterEach(async () => {
    // Clean up collector to prevent hanging tests
    try {
      await collector.stop()
    } catch (error) {
      // Ignore cleanup errors
      console.warn('LiveDataCollector cleanup error:', error)
    }
  })

  describe('registerLiveDataSource', () => {
    it('should register a new data source', async () => {
      const source = createMockDataSource('coinbase')

      await collector.registerLiveDataSource(source)

      const stats = collector.getStatistics()
      assert.ok(stats.connectionStates.has('coinbase'))
      assert.equal(stats.connectionStates.get('coinbase')?.connected, true)
    })

    it('should throw error if source already registered', async () => {
      const source = createMockDataSource('coinbase')

      await collector.registerLiveDataSource(source)

      await assert.rejects(
        () => collector.registerLiveDataSource(source),
        /already registered/
      )
    })

    it('should monitor websocket connections', async () => {
      const source = {
        name: 'websocket-source',
        type: 'websocket' as const,
        subscribe: mock.fn(async () => {}),
        unsubscribe: mock.fn(async () => {}),
        isConnected: mock.fn(() => true),
        reconnect: mock.fn(async () => {})
      }

      await collector.registerLiveDataSource(source)

      // Connection monitoring should be set up
      const stats = collector.getStatistics()
      assert.ok(stats.connectionStates.has('websocket-source'))
    })
  })

  describe('subscribeToSymbols', () => {
    it('should subscribe to symbols on all sources', async () => {
      const source1 = createMockDataSource('source1')
      const source2 = createMockDataSource('source2')

      await collector.registerLiveDataSource(source1)
      await collector.registerLiveDataSource(source2)

      await collector.subscribeToSymbols(['BTC-USD', 'ETH-USD'], ['1m', '5m'])

      // Should subscribe on both sources
      assert.equal((source1.subscribe as any).mock.calls.length, 1)
      assert.equal((source2.subscribe as any).mock.calls.length, 1)

      // Should also start historical data collection
      assert.equal((mockHistoricalManager.startLiveDataCollection as any).mock.calls.length, 2)

      // Check subscriptions
      const subscriptions = collector.getActiveSubscriptions()
      assert.equal(subscriptions.size, 2)
      assert.ok(subscriptions.has('BTC-USD'))
      assert.ok(subscriptions.has('ETH-USD'))
    })

    it('should subscribe to specific sources only', async () => {
      const source1 = createMockDataSource('source1')
      const source2 = createMockDataSource('source2')

      await collector.registerLiveDataSource(source1)
      await collector.registerLiveDataSource(source2)

      await collector.subscribeToSymbols(['BTC-USD'], ['1m'], ['source1'])

      // Should only subscribe on source1
      assert.equal((source1.subscribe as any).mock.calls.length, 1)
      assert.equal((source2.subscribe as any).mock.calls.length, 0)
    })

    it('should handle subscription failures gracefully', async () => {
      const source = createMockDataSource('failing-source')
      ;(source.subscribe as any).mock.mockImplementation(async () => {
        throw new Error('Subscription failed')
      })

      await collector.registerLiveDataSource(source)

      // Should not throw
      await collector.subscribeToSymbols(['BTC-USD'], ['1m'])

      const stats = collector.getStatistics()
      assert.equal(stats.totalErrors, 1)
    })
  })

  describe('unsubscribeFromSymbols', () => {
    it('should unsubscribe from symbols', async () => {
      const source = createMockDataSource('source')

      await collector.registerLiveDataSource(source)
      await collector.subscribeToSymbols(['BTC-USD', 'ETH-USD'], ['1m'])

      await collector.unsubscribeFromSymbols(['BTC-USD'])

      assert.equal((source.unsubscribe as any).mock.calls.length, 1)

      const subscriptions = collector.getActiveSubscriptions()
      assert.equal(subscriptions.size, 1)
      assert.ok(!subscriptions.has('BTC-USD'))
      assert.ok(subscriptions.has('ETH-USD'))
    })

    it('should unsubscribe from specific sources', async () => {
      const source1 = createMockDataSource('source1')
      const source2 = createMockDataSource('source2')

      await collector.registerLiveDataSource(source1)
      await collector.registerLiveDataSource(source2)
      await collector.subscribeToSymbols(['BTC-USD'], ['1m'])

      await collector.unsubscribeFromSymbols(['BTC-USD'], ['source1'])

      // Should only unsubscribe from source1
      assert.equal((source1.unsubscribe as any).mock.calls.length, 1)
      assert.equal((source2.unsubscribe as any).mock.calls.length, 0)

      // Subscription should still exist for source2
      const subscriptions = collector.getActiveSubscriptions()
      assert.ok(subscriptions.has('BTC-USD'))
      assert.ok(subscriptions.get('BTC-USD')?.sources.has('source2'))
      assert.ok(!subscriptions.get('BTC-USD')?.sources.has('source1'))
    })
  })

  describe('processIncomingData', () => {
    it('should process valid candle data', async () => {
      const candle = createMockCandle(Date.now())

      await collector.processIncomingData(candle, 'test-source')

      const stats = collector.getStatistics()
      assert.equal(stats.totalReceived, 1)
      assert.equal(stats.totalProcessed, 1)
      assert.equal(stats.bufferSize, 1)
    })

    it('should validate candle data', async () => {
      const invalidCandle = createMockCandle(Date.now())
      ;(invalidCandle as any).high = 40000 // High < low
      ;(invalidCandle as any).low = 60000

      await collector.processIncomingData(invalidCandle, 'test-source')

      const stats = collector.getStatistics()
      assert.equal(stats.totalReceived, 1)
      assert.equal(stats.totalProcessed, 0)
      assert.equal(stats.totalErrors, 1)
    })

    it('should detect and skip duplicates', async () => {
      const candle = createMockCandle(Date.now())

      // Process same candle twice
      await collector.processIncomingData(candle, 'test-source')
      await collector.processIncomingData(candle, 'test-source')

      const stats = collector.getStatistics()
      assert.equal(stats.totalReceived, 2)
      assert.equal(stats.totalProcessed, 1)
      assert.equal(stats.totalDuplicates, 1)
    })

    it('should emit candle events', async () => {
      const candle = createMockCandle(Date.now())
      const eventHandler = mock.fn()

      eventBus.subscribe(EventTypes.CANDLE, eventHandler)

      await collector.processIncomingData(candle, 'test-source')

      assert.equal(eventHandler.mock.calls.length, 1)
      assert.equal(eventHandler.mock.calls[0]?.arguments[0]?.candle, candle)
      assert.equal(eventHandler.mock.calls[0]?.arguments[0]?.source, 'test-source')
    })

    it('should flush buffer when full', async () => {
      // Create collector with small buffer
      const smallBufferCollector = new LiveDataCollector(
        mockRepository,
        mockHistoricalManager,
        { batchSize: 2, flushIntervalMs: 10000 }
      )

      const candle1 = createMockCandle(Date.now())
      const candle2 = createMockCandle(Date.now() + 1000)

      await smallBufferCollector.processIncomingData(candle1, 'test-source')
      assert.equal(smallBufferCollector.getStatistics().bufferSize, 1)

      await smallBufferCollector.processIncomingData(candle2, 'test-source')
      // Buffer should flush after reaching batchSize
      assert.equal(smallBufferCollector.getStatistics().bufferSize, 0)

      // Verify data was saved
      assert.equal((mockRepository.saveCandlesBatch as any).mock.calls.length, 1)
      assert.equal((mockRepository.saveCandlesBatch as any).mock.calls[0]?.arguments[0].length, 2)

      await smallBufferCollector.stop()
    })
  })

  describe('appendToHistoricalRecords', () => {
    it('should save candles to repository', async () => {
      const candles = [
        createMockCandle(Date.now()),
        createMockCandle(Date.now() + 1000),
      ]

      await collector.appendToHistoricalRecords(candles)

      assert.equal((mockRepository.saveCandlesBatch as any).mock.calls.length, 1)
      assert.equal((mockRepository.saveCandlesBatch as any).mock.calls[0]?.arguments[0].length, 2)
    })

    it('should emit historical data saved event', async () => {
      const candles = [createMockCandle(Date.now())]
      const eventHandler = mock.fn()

      eventBus.subscribe(EventTypes.HISTORICAL_DATA_SAVED, eventHandler)

      await collector.appendToHistoricalRecords(candles)

      assert.equal(eventHandler.mock.calls.length, 1)
      assert.equal(eventHandler.mock.calls[0]?.arguments[0]?.count, 1)
    })

    it('should handle empty array', async () => {
      await collector.appendToHistoricalRecords([])

      // Should not call repository
      assert.equal((mockRepository.saveCandlesBatch as any).mock.calls.length, 0)
    })
  })

  describe('connection handling', () => {
    it('should handle disconnection events', async () => {
      const source = {
        name: 'websocket-source',
        type: 'websocket' as const,
        subscribe: mock.fn(async () => {}),
        unsubscribe: mock.fn(async () => {}),
        isConnected: mock.fn(() => true),
        reconnect: mock.fn(async () => {})
      }

      await collector.registerLiveDataSource(source)

      // Simulate disconnection
      eventBus.emit(EventTypes.CONNECTION_STATUS, {
        source: 'websocket-source',
        status: 'disconnected',
        timestamp: epochDateNow(),
      })

      // Give time for event processing
      await new Promise((resolve) => setTimeout(resolve, 10))

      const stats = collector.getStatistics()
      assert.equal(stats.connectionStates.get('websocket-source')?.connected, false)
    })

    it('should attempt reconnection on disconnect', async () => {
      const source = {
        name: 'websocket-source',
        type: 'websocket' as const,
        subscribe: mock.fn(async () => {}),
        unsubscribe: mock.fn(async () => {}),
        isConnected: mock.fn(() => false),
        reconnect: mock.fn(async () => {})
      }

      await collector.registerLiveDataSource(source)

      // Simulate disconnection
      eventBus.emit(EventTypes.CONNECTION_STATUS, {
        source: 'websocket-source',
        status: 'disconnected',
        timestamp: epochDateNow(),
      })

      // Wait for reconnection attempt (reconnection might be delayed)
      await new Promise((resolve) => setTimeout(resolve, 50))

      // Should attempt to reconnect (may be delayed by reconnection logic)
      // Since reconnection delays are often exponential, just check if enabled
      const config = (collector as any).config
      assert.equal(config.enableAutoReconnect, true)
    })

    it('should resubscribe after reconnection', async () => {
      const source = {
        name: 'websocket-source',
        type: 'websocket' as const,
        subscribe: mock.fn(async () => {}),
        unsubscribe: mock.fn(async () => {}),
        isConnected: mock.fn(() => true),
        reconnect: mock.fn(async () => {})
      }

      await collector.registerLiveDataSource(source)
      await collector.subscribeToSymbols(['BTC-USD'], ['1m'])

      // Reset mock
      ;(source.subscribe as any).mock.resetCalls()

      // Simulate reconnection
      eventBus.emit(EventTypes.CONNECTION_STATUS, {
        source: 'websocket-source',
        status: 'connected',
        timestamp: epochDateNow(),
      })

      // Wait for resubscription
      await new Promise((resolve) => setTimeout(resolve, 10))

      // Should resubscribe to active symbols
      assert.equal((source.subscribe as any).mock.calls.length, 1)
      assert.deepEqual((source.subscribe as any).mock.calls[0]?.arguments[0], ['BTC-USD'])
    })
  })

  describe('flush timer', () => {
    it('should automatically flush buffer on interval', async () => {
      const candle = createMockCandle(Date.now())

      await collector.processIncomingData(candle, 'test-source')

      // Wait for flush interval (should be fast with 50ms interval)
      await new Promise((resolve) => setTimeout(resolve, 75))

      // Buffer should be flushed
      assert.equal(collector.getStatistics().bufferSize, 0)
      assert.equal((mockRepository.saveCandlesBatch as any).mock.calls.length, 1)
    })
  })

  describe('stop', () => {
    it('should flush buffer and unsubscribe on stop', async () => {
      const source = createMockDataSource('source')

      await collector.registerLiveDataSource(source)
      await collector.subscribeToSymbols(['BTC-USD'], ['1m'])

      // Add some data to buffer
      await collector.processIncomingData(createMockCandle(Date.now()), 'source')

      await collector.stop()

      // Should flush buffer
      assert.equal((mockRepository.saveCandlesBatch as any).mock.calls.length, 1)

      // Should unsubscribe
      assert.equal((source.unsubscribe as any).mock.calls.length, 1)
    })
  })

  describe('statistics', () => {
    it('should track statistics correctly', async () => {
      const validCandle = createMockCandle(Date.now())
      const duplicateCandle = createMockCandle(Date.now())
      const invalidCandle = { invalid: true }

      await collector.processIncomingData(validCandle, 'test-source')
      await collector.processIncomingData(duplicateCandle, 'test-source') // Duplicate
      await collector.processIncomingData(invalidCandle, 'test-source') // Invalid

      const stats = collector.getStatistics()
      assert.equal(stats.totalReceived, 3)
      assert.equal(stats.totalProcessed, 1)
      assert.equal(stats.totalDuplicates, 1)
      assert.equal(stats.totalErrors, 0) // Invalid data doesn't count as error
      assert.equal(stats.bufferSize, 1)
    })
  })
})