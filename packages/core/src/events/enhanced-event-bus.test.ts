import { EpochDate, epochDateNow, StockSymbol, toStockSymbol } from '@trdr/shared'
import assert from 'node:assert/strict'
import { beforeEach, describe, it, mock } from 'node:test'
import { enhancedEventBus } from './enhanced-event-bus'
import { FilterBuilder, MarketDataFilters } from './event-filter'
import type { EventData } from './types'

interface TestEvent extends EventData {
  readonly symbol: StockSymbol
  readonly price: number
  readonly volume?: number
}

describe('EnhancedEventBus', () => {
  beforeEach(() => {
    enhancedEventBus.reset()
  })

  describe('Basic Event Operations', () => {
    it('should register and emit events', () => {
      enhancedEventBus.registerEvent('test.event')

      const handler = mock.fn()
      enhancedEventBus.subscribe('test.event', handler)

      const eventData: TestEvent = {
        timestamp: epochDateNow(),
        symbol: toStockSymbol('BTC-USD'),
        price: 50000,
      }

      enhancedEventBus.emit('test.event', eventData)

      assert.equal(handler.mock.calls.length, 1)
      assert.equal(handler.mock.calls[0]?.arguments[0].symbol, 'BTC-USD')
    })

    it('should handle event filtering', () => {
      enhancedEventBus.registerEvent('test.filtered')

      const handler = mock.fn()
      const filter = MarketDataFilters.bySymbol(['BTC-USD'])

      enhancedEventBus.subscribeWithFilter('test.filtered', handler, { filter })

      // Should pass filter
      enhancedEventBus.emitWithFiltering('test.filtered', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      // Should not pass filter
      enhancedEventBus.emitWithFiltering('test.filtered', {
        timestamp: epochDateNow(),
        symbol: 'ETH-USD',
        price: 3000,
      })

      assert.equal(handler.mock.calls.length, 1)
      assert.equal(handler.mock.calls[0]?.arguments[0].symbol, 'BTC-USD')
    })

    it('should handle priority in filtered subscriptions', () => {
      enhancedEventBus.registerEvent('test.priority')

      const callOrder: number[] = []

      enhancedEventBus.subscribeWithFilter('test.priority', () => {
        callOrder.push(3)
      }, { priority: 0 })
      enhancedEventBus.subscribeWithFilter('test.priority', () => {
        callOrder.push(1)
      }, { priority: 10 })
      enhancedEventBus.subscribeWithFilter('test.priority', () => {
        callOrder.push(2)
      }, { priority: 5 })

      enhancedEventBus.emitWithFiltering('test.priority', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      assert.deepEqual(callOrder, [1, 2, 3])
    })

    it('should handle async filtered subscriptions', async () => {
      enhancedEventBus.registerEvent('test.async.filtered')

      let resolved = false

      enhancedEventBus.subscribeWithFilter('test.async.filtered', async () => {
        await new Promise(resolve => setTimeout(resolve, 10))
        resolved = true
      }, { isAsync: true })

      enhancedEventBus.emitWithFiltering('test.async.filtered', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      assert.equal(resolved, false)

      await enhancedEventBus.waitForAsyncHandlers()
      assert.equal(resolved, true)
    })
  })

  describe('Global Filters', () => {
    it('should apply global filters', () => {
      enhancedEventBus.registerEvent('test.global.filter')

      // Add global filter that only allows BTC-USD
      enhancedEventBus.addGlobalFilter({
        eventType: 'test.global.filter',
        filter: MarketDataFilters.bySymbol(['BTC-USD']),
        priority: 100,
        name: 'btc-only',
      })

      const handler = mock.fn()
      enhancedEventBus.subscribe('test.global.filter', handler)

      // Should pass global filter
      enhancedEventBus.emitWithFiltering('test.global.filter', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      // Should not pass global filter
      enhancedEventBus.emitWithFiltering('test.global.filter', {
        timestamp: epochDateNow(),
        symbol: 'ETH-USD',
        price: 3000,
      })

      assert.equal(handler.mock.calls.length, 1)
      assert.equal(handler.mock.calls[0]?.arguments[0].symbol, 'BTC-USD')
    })

    it('should handle multiple global filters with priority', () => {
      enhancedEventBus.registerEvent('test.multi.global')

      const results: string[] = []

      // Lower priority filter (executed after higher priority)
      enhancedEventBus.addGlobalFilter({
        eventType: 'test.multi.global',
        filter: (data: any) => {
          results.push('low-priority')
          return data.price > 1000
        },
        priority: 50,
        name: 'price-filter',
      })

      // Higher priority filter (executed first)
      enhancedEventBus.addGlobalFilter({
        eventType: 'test.multi.global',
        filter: (data: any) => {
          results.push('high-priority')
          return data.symbol === 'BTC-USD'
        },
        priority: 100,
        name: 'symbol-filter',
      })

      const handler = mock.fn()
      enhancedEventBus.subscribe('test.multi.global', handler)

      enhancedEventBus.emitWithFiltering('test.multi.global', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      assert.deepEqual(results, ['high-priority', 'low-priority'])
      assert.equal(handler.mock.calls.length, 1)
    })

    it('should stop filtering if global filter fails', () => {
      enhancedEventBus.registerEvent('test.filter.fail')

      const results: string[] = []

      // This filter will fail
      enhancedEventBus.addGlobalFilter({
        eventType: 'test.filter.fail',
        filter: (_data: any) => {
          results.push('first-filter')
          return false // Fail
        },
        priority: 100,
        name: 'failing-filter',
      })

      // This filter should not execute
      enhancedEventBus.addGlobalFilter({
        eventType: 'test.filter.fail',
        filter: (_data: any) => {
          results.push('second-filter')
          return true
        },
        priority: 50,
        name: 'second-filter',
      })

      const handler = mock.fn()
      enhancedEventBus.subscribe('test.filter.fail', handler)

      enhancedEventBus.emitWithFiltering('test.filter.fail', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      assert.deepEqual(results, ['first-filter']) // Only first filter executed
      assert.equal(handler.mock.calls.length, 0) // Handler not called
    })

    it('should remove global filters', () => {
      enhancedEventBus.registerEvent('test.remove.filter')

      enhancedEventBus.addGlobalFilter({
        eventType: 'test.remove.filter',
        filter: MarketDataFilters.bySymbol(['BTC-USD']),
        priority: 100,
        name: 'removable-filter',
      })

      const handler = mock.fn()
      enhancedEventBus.subscribe('test.remove.filter', handler)

      // Should pass filter
      enhancedEventBus.emitWithFiltering('test.remove.filter', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      // Remove filter
      enhancedEventBus.removeGlobalFilter('test.remove.filter', 'removable-filter')

      // Should now pass even with different symbol
      enhancedEventBus.emitWithFiltering('test.remove.filter', {
        timestamp: epochDateNow(),
        symbol: 'ETH-USD',
        price: 3000,
      })

      assert.equal(handler.mock.calls.length, 2)
    })
  })

  describe('Event Metrics', () => {
    it('should track event metrics', () => {
      enhancedEventBus.registerEvent('test.metrics')
      enhancedEventBus.setDebugMode(true)

      const handler = mock.fn()
      enhancedEventBus.subscribeWithFilter('test.metrics', handler)

      // Emit several events
      for (let i = 0; i < 5; i++) {
        enhancedEventBus.emit('test.metrics', {
          timestamp: epochDateNow(),
          symbol: 'BTC-USD',
          price: 50000 + i,
        })
      }

      const metrics = enhancedEventBus.getEventMetrics('test.metrics') as { emitted: number; filtered: number; handled: number; errors: number; lastEmitted?: EpochDate }

      assert.equal(metrics.emitted, 5)
      assert.equal(metrics.handled, 5)
      assert.ok(typeof metrics.lastEmitted === 'number')
    })

    it('should track global metrics', () => {
      enhancedEventBus.setDebugMode(true)

      enhancedEventBus.registerEvent('test.metrics.1')
      enhancedEventBus.registerEvent('test.metrics.2')

      enhancedEventBus.subscribe('test.metrics.1', mock.fn())
      enhancedEventBus.subscribe('test.metrics.2', mock.fn())

      enhancedEventBus.emit('test.metrics.1', { timestamp: epochDateNow() })
      enhancedEventBus.emit('test.metrics.2', { timestamp: epochDateNow() })

      const allMetrics = enhancedEventBus.getEventMetrics() as Map<string, { emitted: number; filtered: number; handled: number; errors: number; lastEmitted?: EpochDate }>

      assert.ok(allMetrics.has('test.metrics.1'))
      assert.ok(allMetrics.has('test.metrics.2'))
      assert.equal(allMetrics.get('test.metrics.1')?.emitted, 1)
      assert.equal(allMetrics.get('test.metrics.2')?.emitted, 1)
    })

    it('should always track metrics regardless of debug mode', () => {
      enhancedEventBus.registerEvent('test.always.metrics')
      // Debug mode is disabled by default but metrics should still be tracked

      enhancedEventBus.subscribeWithFilter('test.always.metrics', mock.fn())
      enhancedEventBus.emit('test.always.metrics', { timestamp: epochDateNow() })

      const metrics = enhancedEventBus.getEventMetrics('test.always.metrics') as { emitted: number; filtered: number; handled: number; errors: number; lastEmitted?: EpochDate }

      assert.equal(metrics.emitted, 1)
      assert.equal(metrics.handled, 1)
    })
  })

  describe('Complex Filtering Scenarios', () => {
    it('should combine subscription and global filters', () => {
      enhancedEventBus.registerEvent('test.combined.filters')

      // Global filter: only BTC-USD
      enhancedEventBus.addGlobalFilter({
        eventType: 'test.combined.filters',
        filter: MarketDataFilters.bySymbol(['BTC-USD']),
        priority: 100,
        name: 'global-symbol-filter',
      })

      // Subscription filter: price > 40000
      const subscriptionFilter = MarketDataFilters.byPriceRange(40000, Number.MAX_VALUE)

      const handler = mock.fn()
      enhancedEventBus.subscribeWithFilter('test.combined.filters', handler, {
        filter: subscriptionFilter,
      })

      // Should pass both filters
      enhancedEventBus.emitWithFiltering('test.combined.filters', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      // Should fail global filter
      enhancedEventBus.emitWithFiltering('test.combined.filters', {
        timestamp: epochDateNow(),
        symbol: 'ETH-USD',
        price: 50000,
      })

      // Should fail subscription filter
      enhancedEventBus.emitWithFiltering('test.combined.filters', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 30000,
      })

      assert.equal(handler.mock.calls.length, 1)
      assert.equal(handler.mock.calls[0]?.arguments[0].price, 50000)
    })

    it('should handle rate limiting with global filters', () => {
      enhancedEventBus.registerEvent('test.rate.limit')

      const eventTimes = new Map<string, number[]>()
      enhancedEventBus.addGlobalFilter({
        eventType: 'test.rate.limit',
        filter: MarketDataFilters.rateLimit(2, eventTimes), // 2 events per second
        priority: 100,
        name: 'rate-limiter',
      })

      const handler = mock.fn()
      enhancedEventBus.subscribe('test.rate.limit', handler)

      // First two should pass
      enhancedEventBus.emitWithFiltering('test.rate.limit', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      enhancedEventBus.emitWithFiltering('test.rate.limit', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50001,
      })

      // Third should be rate limited
      enhancedEventBus.emitWithFiltering('test.rate.limit', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50002,
      })

      assert.equal(handler.mock.calls.length, 2)
    })
  })

  describe('Error Handling', () => {
    it('should handle filter errors gracefully', () => {
      enhancedEventBus.registerEvent('test.filter.error')
      enhancedEventBus.registerEvent('system.error')

      const errorHandler = mock.fn()
      enhancedEventBus.subscribe('system.error', errorHandler)

      // Add filter that throws error
      enhancedEventBus.addGlobalFilter({
        eventType: 'test.filter.error',
        filter: (_data: any) => {
          throw new Error('Filter error')
        },
        priority: 100,
        name: 'error-filter',
      })

      const handler = mock.fn()
      enhancedEventBus.subscribeWithFilter('test.filter.error', handler)

      enhancedEventBus.emitWithFiltering('test.filter.error', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      // Enhanced handler should not be called due to filter error
      assert.equal(handler.mock.calls.length, 0)

      // Error should be emitted
      assert.equal(errorHandler.mock.calls.length, 1)
    })

    it('should handle subscription unsubscribe', () => {
      enhancedEventBus.registerEvent('test.unsubscribe')

      const handler = mock.fn()
      const subscription = enhancedEventBus.subscribeWithFilter('test.unsubscribe', handler, {
        filter: MarketDataFilters.bySymbol(['BTC-USD']),
      })

      enhancedEventBus.emitWithFiltering('test.unsubscribe', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      assert.equal(handler.mock.calls.length, 1)

      subscription.unsubscribe()

      enhancedEventBus.emitWithFiltering('test.unsubscribe', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50001,
      })

      assert.equal(handler.mock.calls.length, 1) // Still 1, not called again
    })
  })

  describe('Integration with FilterBuilder', () => {
    it('should work with complex FilterBuilder filters', () => {
      enhancedEventBus.registerEvent('test.filter.builder')

      // Create complex filter: (BTC-USD OR ETH-USD) AND price > 1000
      const symbolFilter = FilterBuilder.create<TestEvent>()
        .and(MarketDataFilters.bySymbol(['BTC-USD']))
        .or(FilterBuilder.create<TestEvent>().and(MarketDataFilters.bySymbol(['ETH-USD'])))

      const complexFilter = FilterBuilder.create<TestEvent>()
        .and(symbolFilter.build())
        .and(MarketDataFilters.byPriceRange(1000, Number.MAX_VALUE))
        .build()

      const handler = mock.fn()
      enhancedEventBus.subscribeWithFilter('test.filter.builder', handler, {
        filter: complexFilter,
      })

      // Should pass
      enhancedEventBus.emitWithFiltering('test.filter.builder', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 50000,
      })

      // Should pass
      enhancedEventBus.emitWithFiltering('test.filter.builder', {
        timestamp: epochDateNow(),
        symbol: 'ETH-USD',
        price: 3000,
      })

      // Should fail (wrong symbol)
      enhancedEventBus.emitWithFiltering('test.filter.builder', {
        timestamp: epochDateNow(),
        symbol: 'DOGE-USD',
        price: 2000,
      })

      // Should fail (price too low)
      enhancedEventBus.emitWithFiltering('test.filter.builder', {
        timestamp: epochDateNow(),
        symbol: 'BTC-USD',
        price: 500,
      })

      assert.equal(handler.mock.calls.length, 2)
    })
  })
})
