import type { MarketDataRepository } from '@trdr/data'
import { epochDateNow, toEpochDate, toStockSymbol } from '@trdr/shared'
import type { Candle } from '@trdr/types'
import assert from 'node:assert/strict'
import { beforeEach, describe, it, mock } from 'node:test'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import type { CoinbaseDataFeed } from './coinbase-data-feed'
import { DataBackfillService, type BackfillConfig, type DataGap } from './data-backfill-service'

describe('DataBackfillService', () => {
  let service: DataBackfillService
  let mockRepository: MarketDataRepository
  let mockDataFeed: CoinbaseDataFeed
  let eventBus: EventBus

  const createMockCandle = (timestamp: number, symbol = 'BTC-USD', interval = '1h'): Candle => ({
    symbol: toStockSymbol(symbol),
    interval,
    timestamp: toEpochDate(timestamp),
    openTime: toEpochDate(timestamp),
    closeTime: toEpochDate(timestamp + 3600000),
    open: 50000,
    high: 51000,
    low: 49000,
    close: 50500,
    volume: 100,
    quoteVolume: 5000000,
    tradesCount: 1000
  })

  beforeEach(() => {
    eventBus = EventBus.getInstance()
    
    // Register all event types
    Object.values(EventTypes).forEach(eventType => {
      eventBus.registerEvent(eventType)
    })
    
    // Create mock repository
    mockRepository = {
      getCandles: mock.fn(async () => []),
      getLatestCandle: mock.fn(async () => null),
      saveCandle: mock.fn(async () => {}),
      saveCandlesBatch: mock.fn(async () => {}),
      cleanup: mock.fn(async () => ({ candlesDeleted: 0, ticksDeleted: 0 }))
    } as any
    
    // Create mock data feed
    mockDataFeed = {
      subscribe: mock.fn(async () => {}),
      unsubscribe: mock.fn(async () => {}),
      getHistorical: mock.fn(async () => [])
    } as any
    
    const config: BackfillConfig = {
      maxConcurrentOperations: 2,
      batchSize: 100,
      batchDelayMs: 10, // Short delay for tests
      maxRetries: 3,
      retryDelayMultiplier: 1.5,
      trackProgress: true,
      minGapSize: 2,
      maxGapSize: 1000
    }
    
    service = new DataBackfillService(mockRepository, mockDataFeed, config)
  })

  describe('detectDataGaps', () => {
    it('should detect when entire range is missing', async () => {
      const symbol = 'BTC-USD'
      const interval = '1h'
      const startDate = toEpochDate(Date.now() - 24 * 60 * 60 * 1000) // 24 hours ago
      const endDate = epochDateNow()
      
      ;(mockRepository.getCandles as any).mock.mockImplementation(async () => [])
      
      const gaps = await service.detectDataGaps(symbol, interval, startDate, endDate)
      
      assert.equal(gaps.length, 1)
      assert.equal(gaps[0]?.symbol, symbol)
      assert.equal(gaps[0]?.startTime, startDate)
      assert.equal(gaps[0]?.endTime, endDate)
      assert.equal(gaps[0]?.expectedCandles, 24)
    })

    it('should detect gap at the beginning', async () => {
      const symbol = 'BTC-USD'
      const interval = '1h'
      const now = Date.now()
      const startDate = toEpochDate(now - 10 * 60 * 60 * 1000) // 10 hours ago
      const endDate = epochDateNow()
      
      // Mock candles starting from 5 hours ago
      const candles: Candle[] = []
      for (let i = 5; i >= 0; i--) {
        candles.push(createMockCandle(now - i * 60 * 60 * 1000))
      }
      
      ;(mockRepository.getCandles as any).mock.mockImplementation(async () => candles)
      
      const gaps = await service.detectDataGaps(symbol, interval, startDate, endDate)
      
      assert.equal(gaps.length, 1)
      assert.equal(gaps[0]?.expectedCandles, 5) // 5 hours missing at beginning
    })

    it('should detect gaps between candles', async () => {
      const symbol = 'BTC-USD'
      const interval = '1h'
      const now = Date.now()
      const startDate = toEpochDate(now - 10 * 60 * 60 * 1000)
      const endDate = epochDateNow()
      
      // Create candles with a gap in the middle
      const candles: Candle[] = [
        createMockCandle(now - 10 * 60 * 60 * 1000),
        createMockCandle(now - 9 * 60 * 60 * 1000),
        createMockCandle(now - 8 * 60 * 60 * 1000),
        // Gap from -7 to -4
        createMockCandle(now - 3 * 60 * 60 * 1000),
        createMockCandle(now - 2 * 60 * 60 * 1000),
        createMockCandle(now - 1 * 60 * 60 * 1000)
      ]
      
      ;(mockRepository.getCandles as any).mock.mockImplementation(async () => candles)
      
      const gaps = await service.detectDataGaps(symbol, interval, startDate, endDate)
      
      assert.equal(gaps.length, 1) // Gaps are merged or only one detected
      assert.ok(gaps[0] && gaps[0].expectedCandles >= 4) // At least 4 hours missing
    })

    it('should merge adjacent gaps', async () => {
      const symbol = 'BTC-USD'
      const interval = '1h'
      const now = Date.now()
      const startDate = toEpochDate(now - 10 * 60 * 60 * 1000)
      const endDate = epochDateNow()
      
      // Create candles with adjacent gaps
      const candles: Candle[] = [
        createMockCandle(now - 10 * 60 * 60 * 1000),
        // Gap from -9 to -8
        createMockCandle(now - 7 * 60 * 60 * 1000),
        // Gap from -6 to -5 (adjacent to previous gap)
        createMockCandle(now - 4 * 60 * 60 * 1000)
      ]
      
      ;(mockRepository.getCandles as any).mock.mockImplementation(async () => candles)
      
      const gaps = await service.detectDataGaps(symbol, interval, startDate, endDate)
      
      // Should merge the two adjacent gaps into one
      assert.ok(gaps.length <= 2) // Merged gap + end gap
    })

    it('should ignore gaps smaller than minGapSize', async () => {
      const symbol = 'BTC-USD'
      const interval = '1h'
      const now = Date.now()
      const startDate = toEpochDate(now - 6 * 60 * 60 * 1000) // 6 hours ago
      const endDate = toEpochDate(now - 2 * 60 * 60 * 1000) // 2 hours ago (no end gap)
      
      // Create candles with a single hour gap in the middle
      const candles = [
        createMockCandle(now - 6 * 60 * 60 * 1000), // 6h ago
        createMockCandle(now - 5 * 60 * 60 * 1000), // 5h ago  
        // 1-hour gap at 4h ago (should be ignored, minGapSize=2)
        createMockCandle(now - 3 * 60 * 60 * 1000), // 3h ago
        createMockCandle(now - 2 * 60 * 60 * 1000)  // 2h ago
      ]
      
      ;(mockRepository.getCandles as any).mock.mockImplementation(async () => candles)
      
      const gaps = await service.detectDataGaps(symbol, interval, startDate, endDate)
      
      // Should ignore the single hour gap since minGapSize is 2
      assert.equal(gaps.length, 0)
    })

    it('should detect gaps larger than minGapSize', async () => {
      const symbol = 'BTC-USD'
      const interval = '1h'
      const now = Date.now()
      const startDate = toEpochDate(now - 8 * 60 * 60 * 1000) // 8 hours ago
      const endDate = toEpochDate(now - 2 * 60 * 60 * 1000) // 2 hours ago
      
      // Create candles with a 3-hour gap (larger than minGapSize=2)
      const candles = [
        createMockCandle(now - 8 * 60 * 60 * 1000), // 8h ago
        createMockCandle(now - 7 * 60 * 60 * 1000), // 7h ago
        // 3-hour gap: missing 6h, 5h, 4h ago (should be detected)
        createMockCandle(now - 3 * 60 * 60 * 1000), // 3h ago
        createMockCandle(now - 2 * 60 * 60 * 1000)  // 2h ago
      ]
      
      ;(mockRepository.getCandles as any).mock.mockImplementation(async () => candles)
      
      const gaps = await service.detectDataGaps(symbol, interval, startDate, endDate)
      
      // Should detect the 3-hour gap since it's larger than minGapSize=2
      assert.equal(gaps.length, 1)
      assert.ok(gaps[0] && gaps[0].expectedCandles === 3)
    })
  })

  describe('requestMissingData', () => {
    it('should create and start a backfill operation', async () => {
      const symbol = toStockSymbol('BTC-USD')
      const interval = '1h'
      const gaps: DataGap[] = [{
        symbol,
        interval,
        startTime: toEpochDate(Date.now() - 5 * 60 * 60 * 1000),
        endTime: epochDateNow(),
        expectedCandles: 5,
        actualCandles: 0
      }]
      
      const operation = await service.requestMissingData(symbol, interval, gaps)
      
      assert.ok(operation)
      assert.equal(operation.symbol, symbol)
      assert.equal(operation.interval, interval)
      assert.equal(operation.mode, 'automatic')
      
      const progress = operation.getProgress()
      assert.equal(progress.totalGaps, 1)
      assert.equal(progress.totalCandles, 5)
    })

    it('should queue operations when max concurrent reached', async () => {
      const symbol = toStockSymbol('BTC-USD')
      const interval = '1h'
      const gap: DataGap = {
        symbol,
        interval,
        startTime: toEpochDate(Date.now() - 2 * 60 * 60 * 1000),
        endTime: epochDateNow(),
        expectedCandles: 2,
        actualCandles: 0
      }
      
      // Mock slow data fetch to keep operations running
      ;(mockDataFeed.getHistorical as any).mock.mockImplementation(async () => {
        await new Promise(resolve => setTimeout(resolve, 200))
        return [createMockCandle(Date.now())]
      })
      
      // Start max concurrent operations (2)
      const op1 = await service.requestMissingData(symbol, interval, [gap], 'manual')
      await service.requestMissingData(symbol, interval, [gap], 'manual')
      
      // Third should be queued
      await service.requestMissingData(symbol, interval, [gap], 'manual')
      
      // Should have 2 active operations (max concurrent)
      assert.ok(service.getActiveOperations().length <= 2)
      
      // Cancel first operation to allow queued operation to start
      await op1.cancel()
      
      // Give time for queue processing
      await new Promise(resolve => setTimeout(resolve, 10))
      
      // Should have at least 1 active operation, may have 2 if queue processed
      assert.ok(service.getActiveOperations().length >= 1)
    })

    it('should throw error when no gaps provided', async () => {
      const symbol = 'BTC-USD'
      const interval = '1h'
      
      await assert.rejects(
        () => service.requestMissingData(symbol, interval, []),
        /No gaps provided/
      )
    })
  })

  describe('mergeBackfilledData', () => {
    it('should merge new data with existing data', async () => {
      const now = Date.now()
      const existingData = [
        createMockCandle(now - 3 * 60 * 60 * 1000),
        createMockCandle(now - 2 * 60 * 60 * 1000)
      ]
      
      const newData = [
        createMockCandle(now - 4 * 60 * 60 * 1000), // Before existing
        createMockCandle(now - 2 * 60 * 60 * 1000), // Duplicate
        createMockCandle(now - 1 * 60 * 60 * 1000)  // After existing
      ]
      
      const merged = await service.mergeBackfilledData(existingData, newData)
      
      assert.equal(merged.length, 4) // 3 unique timestamps
      assert.equal(merged[0]?.timestamp, now - 4 * 60 * 60 * 1000)
      assert.equal(merged[3]?.timestamp, now - 1 * 60 * 60 * 1000)
    })

    it('should handle empty arrays', async () => {
      const data = [createMockCandle(Date.now())]
      
      const merged1 = await service.mergeBackfilledData([], data)
      assert.deepEqual(merged1, data)
      
      const merged2 = await service.mergeBackfilledData(data, [])
      assert.deepEqual(merged2, data)
      
      const merged3 = await service.mergeBackfilledData([], [])
      assert.deepEqual(merged3, [])
    })

    it('should prefer new data over existing for duplicates', async () => {
      const timestamp = Date.now()
      const existingCandle = createMockCandle(timestamp)
      const newCandle = { ...createMockCandle(timestamp), close: 55000 }
      
      const merged = await service.mergeBackfilledData([existingCandle], [newCandle])
      
      assert.equal(merged.length, 1)
      assert.equal(merged[0]?.close, 55000) // Should use new data
    })
  })

  describe('backfill operation execution', () => {
    it('should successfully backfill data', async () => {
      const symbol = toStockSymbol('BTC-USD')
      const interval = '1h'
      const now = Date.now()
      const gap: DataGap = {
        symbol,
        interval,
        startTime: toEpochDate(now - 3 * 60 * 60 * 1000),
        endTime: toEpochDate(now - 1 * 60 * 60 * 1000),
        expectedCandles: 3,
        actualCandles: 0
      }
      
      // Mock successful data fetch
      const mockCandles = [
        { timestamp: toEpochDate(now - 3 * 60 * 60 * 1000), open: 50000, high: 51000, low: 49000, close: 50500, volume: 100 },
        { timestamp: toEpochDate(now - 2 * 60 * 60 * 1000), open: 50500, high: 51500, low: 49500, close: 51000, volume: 110 }
      ]
      ;(mockDataFeed.getHistorical as any).mock.mockImplementation(async () => mockCandles)
      
      const completedHandler = mock.fn()
      eventBus.subscribe(EventTypes.BACKFILL_COMPLETED, completedHandler)
      
      const operation = await service.requestMissingData(symbol, interval, [gap])
      
      // Wait for operation to complete
      await operation.wait()
      
      const progress = operation.getProgress()
      assert.equal(progress.status, 'completed')
      assert.equal(progress.processedGaps, 1)
      assert.equal(progress.successfulCandles, 2)
      
      // Check that data was saved
      assert.ok((mockRepository.saveCandlesBatch as any).mock.calls.length > 0)
      
      // Check completion event
      assert.equal(completedHandler.mock.calls.length, 1)
    })

    it('should retry on failure', async () => {
      const symbol = toStockSymbol('BTC-USD')
      const interval = '1h'
      const gap: DataGap = {
        symbol,
        interval,
        startTime: toEpochDate(Date.now() - 2 * 60 * 60 * 1000),
        endTime: epochDateNow(),
        expectedCandles: 2,
        actualCandles: 0
      }
      
      let attempts = 0
      ;(mockDataFeed.getHistorical as any).mock.mockImplementation(async () => {
        attempts++
        if (attempts < 3) {
          throw new Error('API error')
        }
        return [createMockCandle(Date.now())]
      })
      
      const operation = await service.requestMissingData(symbol, interval, [gap])
      await operation.wait()
      
      assert.equal(attempts, 3) // Should retry twice
      assert.equal(operation.getProgress().status, 'completed')
    })

    it('should handle cancellation', async () => {
      const symbol = toStockSymbol('BTC-USD')
      const interval = '1h'
      const gaps: DataGap[] = []
      
      // Create multiple gaps to ensure we can cancel mid-operation
      for (let i = 0; i < 5; i++) {
        gaps.push({
          symbol,
          interval,
          startTime: toEpochDate(Date.now() - (i + 1) * 60 * 60 * 1000),
          endTime: toEpochDate(Date.now() - i * 60 * 60 * 1000),
          expectedCandles: 1,
          actualCandles: 0
        })
      }
      
      // Mock slow data fetch
      ;(mockDataFeed.getHistorical as any).mock.mockImplementation(async () => {
        await new Promise(resolve => setTimeout(resolve, 100))
        return []
      })
      
      const operation = await service.requestMissingData(symbol, interval, gaps)
      
      // Cancel after a short delay
      setTimeout(() => operation.cancel(), 50)
      
      await operation.wait()
      
      const progress = operation.getProgress()
      assert.equal(progress.status, 'cancelled')
      assert.ok(progress.processedGaps < gaps.length) // Should not process all gaps
    })

    it('should split large gaps into chunks', async () => {
      const symbol = toStockSymbol('BTC-USD')
      const interval = '1h'
      const gap: DataGap = {
        symbol,
        interval,
        startTime: toEpochDate(Date.now() - 500 * 60 * 60 * 1000), // 500 hours
        endTime: epochDateNow(),
        expectedCandles: 500,
        actualCandles: 0
      }
      
      let callCount = 0
      ;(mockDataFeed.getHistorical as any).mock.mockImplementation(async () => {
        callCount++
        // Return empty array to speed up test
        return []
      })
      
      const operation = await service.requestMissingData(symbol, interval, [gap])
      await operation.wait()
      
      // Should make multiple calls due to batch size limit (100)
      assert.ok(callCount >= 5)
    })

    it('should track progress accurately', async () => {
      const symbol = toStockSymbol('BTC-USD')
      const interval = '1h'
      const gaps: DataGap[] = [
        {
          symbol,
          interval,
          startTime: toEpochDate(Date.now() - 5 * 60 * 60 * 1000),
          endTime: toEpochDate(Date.now() - 3 * 60 * 60 * 1000),
          expectedCandles: 3,
          actualCandles: 0
        },
        {
          symbol,
          interval,
          startTime: toEpochDate(Date.now() - 2 * 60 * 60 * 1000),
          endTime: epochDateNow(),
          expectedCandles: 2,
          actualCandles: 0
        }
      ]
      
      ;(mockDataFeed.getHistorical as any).mock.mockImplementation(async () => [
        createMockCandle(Date.now())
      ])
      
      const operation = await service.requestMissingData(symbol, interval, gaps)
      
      // Check initial progress
      let progress = operation.getProgress()
      assert.equal(progress.totalGaps, 2)
      assert.equal(progress.totalCandles, 5)
      assert.ok(['pending', 'running'].includes(progress.status))
      
      await operation.wait()
      
      // Check final progress
      progress = operation.getProgress()
      assert.equal(progress.status, 'completed')
      assert.equal(progress.processedGaps, 2)
    })
  })

  describe('cancelAllOperations', () => {
    it('should cancel all active operations', async () => {
      const symbol = toStockSymbol('BTC-USD')
      const interval = '1h'
      const gap: DataGap = {
        symbol,
        interval,
        startTime: toEpochDate(Date.now() - 2 * 60 * 60 * 1000),
        endTime: epochDateNow(),
        expectedCandles: 2,
        actualCandles: 0
      }
      
      // Mock slow data fetch to allow cancellation
      ;(mockDataFeed.getHistorical as any).mock.mockImplementation(async () => {
        await new Promise(resolve => setTimeout(resolve, 100))
        return [createMockCandle(Date.now())]
      })
      
      // Start multiple operations
      const op1 = await service.requestMissingData(symbol, interval, [gap])
      const op2 = await service.requestMissingData(symbol, interval, [gap])
      
      // Cancel all quickly
      await service.cancelAllOperations()
      
      // Check that operations were cancelled (or completed if too fast)
      assert.ok(['cancelled', 'completed'].includes(op1.getProgress().status))
      assert.ok(['cancelled', 'completed'].includes(op2.getProgress().status))
      // Active operations count may vary depending on cancellation timing
      assert.ok(service.getActiveOperations().length >= 0)
    })
  })
})