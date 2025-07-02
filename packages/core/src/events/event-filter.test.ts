import { epochDateNow, toEpochDate } from '@trdr/shared'
import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  FilterBuilder,
  MarketDataFilters,
  SystemFilters,
  TradingFilters,
} from './event-filter'

describe('EventFilter', () => {
  describe('MarketDataFilters', () => {
    it('should filter by symbol', () => {
      const filter = MarketDataFilters.bySymbol(['BTC-USD', 'ETH-USD'])

      assert.equal(filter({ symbol: 'BTC-USD', timestamp: epochDateNow() }), true)
      assert.equal(filter({ symbol: 'ETH-USD', timestamp: epochDateNow() }), true)
      assert.equal(filter({ symbol: 'DOGE-USD', timestamp: epochDateNow() }), false)
      assert.equal(filter({ timestamp: epochDateNow() }), true) // No symbol property
    })

    it('should filter by price range', () => {
      const filter = MarketDataFilters.byPriceRange(100, 200)

      assert.equal(filter({ price: 150, timestamp: epochDateNow() }), true)
      assert.equal(filter({ price: 100, timestamp: epochDateNow() }), true)
      assert.equal(filter({ price: 200, timestamp: epochDateNow() }), true)
      assert.equal(filter({ price: 50, timestamp: epochDateNow() }), false)
      assert.equal(filter({ price: 250, timestamp: epochDateNow() }), false)
      assert.equal(filter({ timestamp: epochDateNow() }), true) // No price property
    })

    it('should filter by volume threshold', () => {
      const filter = MarketDataFilters.byVolumeThreshold(1000)

      assert.equal(filter({ volume: 1500, timestamp: epochDateNow() }), true)
      assert.equal(filter({ volume: 1000, timestamp: epochDateNow() }), true)
      assert.equal(filter({ volume: 500, timestamp: epochDateNow() }), false)
      assert.equal(filter({ timestamp: epochDateNow() }), true) // No volume property
    })

    it('should filter by time range', () => {
      const start = toEpochDate(new Date('2024-01-01T00:00:00Z'))
      const end = toEpochDate(new Date('2024-01-01T12:00:00Z'))
      const filter = MarketDataFilters.byTimeRange(start, end)

      const withinRange = toEpochDate(new Date('2024-01-01T06:00:00Z'))
      const beforeRange = toEpochDate(new Date('2023-12-31T23:00:00Z'))
      const afterRange = toEpochDate(new Date('2024-01-01T13:00:00Z'))

      assert.equal(filter({ timestamp: withinRange }), true)
      assert.equal(filter({ timestamp: start }), true)
      assert.equal(filter({ timestamp: end }), true)
      assert.equal(filter({ timestamp: beforeRange }), false)
      assert.equal(filter({ timestamp: afterRange }), false)
      assert.equal(filter({ timestamp: withinRange }), true) // Same timestamp
      assert.equal(filter({ timestamp: epochDateNow() }), false) // Current timestamp is outside the 2024-01-01 range
    })

    it('should filter by price change threshold', () => {
      const lastPrices = new Map<string, number>()
      const filter = MarketDataFilters.byPriceChangeThreshold(0.05, lastPrices) // 5% threshold

      // First price for symbol - should pass
      assert.equal(filter({ symbol: 'BTC-USD', price: 100, timestamp: epochDateNow() }), true)
      assert.equal(lastPrices.get('BTC-USD'), 100)

      // Small change - should not pass
      assert.equal(filter({ symbol: 'BTC-USD', price: 102, timestamp: epochDateNow() }), false)
      assert.equal(lastPrices.get('BTC-USD'), 100) // Price not updated

      // Large change - should pass
      assert.equal(filter({ symbol: 'BTC-USD', price: 110, timestamp: epochDateNow() }), true)
      assert.equal(lastPrices.get('BTC-USD'), 110) // Price updated

      assert.equal(filter({ timestamp: epochDateNow() }), true) // No symbol or price
    })

    it('should filter by interval', () => {
      const filter = MarketDataFilters.byInterval(['1m', '5m', '1h'])

      assert.equal(filter({ interval: '1m', timestamp: epochDateNow() }), true)
      assert.equal(filter({ interval: '5m', timestamp: epochDateNow() }), true)
      assert.equal(filter({ interval: '15m', timestamp: epochDateNow() }), false)
      assert.equal(filter({ timestamp: epochDateNow() }), true) // No interval property
    })

    it('should rate limit events', () => {
      const eventTimes = new Map<string, number[]>()
      const filter = MarketDataFilters.rateLimit(2, eventTimes) // 2 events per second max

      // First two events should pass
      assert.equal(filter({ symbol: 'BTC-USD', timestamp: epochDateNow() }), true)
      assert.equal(filter({ symbol: 'BTC-USD', timestamp: epochDateNow() }), true)

      // Third event should fail
      assert.equal(filter({ symbol: 'BTC-USD', timestamp: epochDateNow() }), false)

      // Different symbol should have its own rate limit
      assert.equal(filter({ symbol: 'ETH-USD', timestamp: epochDateNow() }), true)
      assert.equal(filter({ symbol: 'ETH-USD', timestamp: epochDateNow() }), true)
      assert.equal(filter({ symbol: 'ETH-USD', timestamp: epochDateNow() }), false)
    })
  })

  describe('SystemFilters', () => {
    it('should filter by severity', () => {
      const filter = SystemFilters.bySeverity('high')

      assert.equal(filter({ severity: 'critical', timestamp: epochDateNow() }), true)
      assert.equal(filter({ severity: 'high', timestamp: epochDateNow() }), true)
      assert.equal(filter({ severity: 'medium', timestamp: epochDateNow() }), false)
      assert.equal(filter({ severity: 'low', timestamp: epochDateNow() }), false)
      assert.equal(filter({ timestamp: epochDateNow() }), true) // No severity property
    })

    it('should filter by context', () => {
      const filter = SystemFilters.byContext(['trading', 'market-data'])

      assert.equal(filter({ context: 'trading', timestamp: epochDateNow() }), true)
      assert.equal(filter({ context: 'market-data', timestamp: epochDateNow() }), true)
      assert.equal(filter({ context: 'portfolio', timestamp: epochDateNow() }), false)
      assert.equal(filter({ timestamp: epochDateNow() }), true) // No context property
    })
  })

  describe('TradingFilters', () => {
    it('should filter by side', () => {
      const filter = TradingFilters.bySide(['buy'])

      assert.equal(filter({ side: 'buy', timestamp: epochDateNow() }), true)
      assert.equal(filter({ side: 'sell', timestamp: epochDateNow() }), false)
      assert.equal(filter({ timestamp: epochDateNow() }), true) // No side property
    })

    it('should filter by size range', () => {
      const filter = TradingFilters.bySizeRange(0.1, 1.0)

      assert.equal(filter({ size: 0.5, timestamp: epochDateNow() }), true)
      assert.equal(filter({ size: 0.1, timestamp: epochDateNow() }), true)
      assert.equal(filter({ size: 1.0, timestamp: epochDateNow() }), true)
      assert.equal(filter({ size: 0.05, timestamp: epochDateNow() }), false)
      assert.equal(filter({ size: 1.5, timestamp: epochDateNow() }), false)
      assert.equal(filter({ timestamp: epochDateNow() }), true) // No size property
    })

    it('should filter by status', () => {
      const filter = TradingFilters.byStatus(['pending', 'filled'])

      assert.equal(filter({ status: 'pending', timestamp: epochDateNow() }), true)
      assert.equal(filter({ status: 'filled', timestamp: epochDateNow() }), true)
      assert.equal(filter({ status: 'cancelled', timestamp: epochDateNow() }), false)
      assert.equal(filter({ timestamp: epochDateNow() }), true) // No status property
    })
  })

  describe('FilterBuilder', () => {
    it('should build AND filters', () => {
      const filter = FilterBuilder.create<any>()
        .and(MarketDataFilters.bySymbol(['BTC-USD']))
        .and(MarketDataFilters.byPriceRange(100, 200))
        .build()

      assert.equal(filter({ symbol: 'BTC-USD', price: 150, timestamp: epochDateNow() }), true)
      assert.equal(filter({ symbol: 'BTC-USD', price: 250, timestamp: epochDateNow() }), false)
      assert.equal(filter({ symbol: 'ETH-USD', price: 150, timestamp: epochDateNow() }), false)
    })

    it('should build OR filters', () => {
      const builder1 = FilterBuilder.create<any>()
        .and(MarketDataFilters.bySymbol(['BTC-USD']))

      const builder2 = FilterBuilder.create<any>()
        .and(MarketDataFilters.bySymbol(['ETH-USD']))

      const filter = builder1.or(builder2).build()

      assert.equal(filter({ symbol: 'BTC-USD', timestamp: epochDateNow() }), true)
      assert.equal(filter({ symbol: 'ETH-USD', timestamp: epochDateNow() }), true)
      assert.equal(filter({ symbol: 'DOGE-USD', timestamp: epochDateNow() }), false)
    })

    it('should build NOT filters', () => {
      const filter = FilterBuilder.create<any>()
        .and(MarketDataFilters.bySymbol(['BTC-USD']))
        .not()
        .build()

      assert.equal(filter({ symbol: 'BTC-USD', timestamp: epochDateNow() }), false)
      assert.equal(filter({ symbol: 'ETH-USD', timestamp: epochDateNow() }), true)
    })

    it('should handle empty filter builder', () => {
      const filter = FilterBuilder.create<any>().build()

      assert.equal(filter({ anything: 'value', timestamp: epochDateNow() }), true)
    })

    it('should create complex composite filters', () => {
      // (BTC-USD OR ETH-USD) AND (price > 100) AND NOT (volume < 1000)
      const symbolFilter = FilterBuilder.create<any>()
        .and(MarketDataFilters.bySymbol(['BTC-USD']))
        .or(FilterBuilder.create<any>().and(MarketDataFilters.bySymbol(['ETH-USD'])))

      const volumeFilter = FilterBuilder.create<any>()
        .and(MarketDataFilters.byVolumeThreshold(1000))
        .not()

      const filter = FilterBuilder.create<any>()
        .and(symbolFilter.build())
        .and(MarketDataFilters.byPriceRange(100, Number.MAX_VALUE))
        .and(volumeFilter.build())
        .build()

      assert.equal(filter({ symbol: 'BTC-USD', price: 150, volume: 500, timestamp: epochDateNow() }), true)
      assert.equal(filter({ symbol: 'BTC-USD', price: 150, volume: 1500, timestamp: epochDateNow() }), false)
      assert.equal(filter({ symbol: 'DOGE-USD', price: 150, volume: 500, timestamp: epochDateNow() }), false)
      assert.equal(filter({ symbol: 'BTC-USD', price: 50, volume: 500, timestamp: epochDateNow() }), false)
    })
  })
})
