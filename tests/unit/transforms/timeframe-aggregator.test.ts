import { deepStrictEqual, ok, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import type { OhlcvDto } from '../../../src/models'
import { TimeframeAggregator } from '../../../src/transforms'
import type { TimeframeAggregationParams } from '../../../src/transforms'

describe('TimeframeAggregator', () => {
  // Helper to create test data
  const createTestData = (count: number, intervalMs = 60000): OhlcvDto[] => {
    const baseTime = new Date('2024-01-01T10:00:00Z').getTime()
    const data: OhlcvDto[] = []

    for (let i = 0; i < count; i++) {
      data.push({
        timestamp: baseTime + i * intervalMs,
        symbol: 'BTCUSD',
        exchange: 'test',
        open: 100 + i,
        high: 102 + i,
        low: 99 + i,
        close: 101 + i,  // i=0: 101, i=1: 102, i=2: 103, i=3: 104, i=4: 105
        volume: 1000 + i * 10,
      })
    }

    return data
  }

  // Helper to convert array to async iterator
  async function* arrayToAsyncIterator<T>(array: T[]): AsyncIterator<T> {
    for (const item of array) {
      yield item
    }
  }

  // Helper to collect async iterator results
  async function collectResults(iterator: AsyncIterator<OhlcvDto>): Promise<OhlcvDto[]> {
    const results: OhlcvDto[] = []
    for await (const item of { [Symbol.asyncIterator]: () => iterator }) {
      results.push(item)
    }
    return results
  }

  describe('constructor and validation', () => {
    it('should create instance with valid parameters', () => {
      const params: TimeframeAggregationParams = {
        targetTimeframe: '5m',
      }

      const aggregator = new TimeframeAggregator(params)
      ok(aggregator)
      strictEqual(aggregator.type, 'timeframeAggregation')
      strictEqual(aggregator.name, 'Timeframe Aggregator')
    })

    it('should set default parameters', () => {
      const aggregator = new TimeframeAggregator({ targetTimeframe: '5m' })

      strictEqual(aggregator.params.alignToMarketOpen, false)
      strictEqual(aggregator.params.marketOpenTime, '09:30')
      strictEqual(aggregator.params.marketTimezone, 'America/New_York')
      strictEqual(aggregator.params.incompleteBarBehavior, 'drop')
    })

    it('should validate target timeframe is provided', () => {
      const aggregator = new TimeframeAggregator({ targetTimeframe: '' })
      throws(() => aggregator.validate(), /Target timeframe is required/)
    })

    it('should validate target timeframe is valid', () => {
      const aggregator = new TimeframeAggregator({ targetTimeframe: '7m' })
      throws(() => aggregator.validate(), /Invalid target timeframe/)
    })

    it('should validate market open time format', () => {
      const aggregator = new TimeframeAggregator({
        targetTimeframe: '5m',
        marketOpenTime: '25:00',
      })
      throws(() => aggregator.validate(), /Market open time must be in HH:MM format/)
    })

    it('should validate incomplete bar behavior', () => {
      const aggregator = new TimeframeAggregator({
        targetTimeframe: '5m',
        incompleteBarBehavior: 'invalid' as any, // Intentionally invalid
      })
      throws(() => aggregator.validate(), /incompleteBarBehavior must be either/)
    })
  })

  describe('aggregation functionality', () => {
    it('should aggregate 1m bars to 5m bars', async () => {
      const testData = createTestData(10, 60000) // 10 1-minute bars
      const aggregator = new TimeframeAggregator({ targetTimeframe: '5m' })

      const result = await aggregator.apply(arrayToAsyncIterator(testData))
      const aggregated = await collectResults(result.data)

      // With drop behavior, only complete bars are emitted
      // Since we don't have data for an 11th minute to trigger emission of the second bar,
      // we only get 1 complete bar
      strictEqual(aggregated.length, 1)

      // Check first 5m bar (aggregates minutes 0-4)
      const firstBar = aggregated[0]!
      strictEqual(firstBar.symbol, 'BTCUSD')
      strictEqual(firstBar.exchange, 'test')
      strictEqual(firstBar.open, 100) // Open of first 1m bar (index 0)
      strictEqual(firstBar.high, 106) // Max high (102+4)
      strictEqual(firstBar.low, 99) // Min low (99+0)
      strictEqual(firstBar.close, 105) // Close of 5th 1m bar (index 4): 101 + 4 = 105
      strictEqual(firstBar.volume, 5100) // Sum of volumes
    })

    it('should aggregate 1m bars to 5m bars with emit behavior', async () => {
      const testData = createTestData(10, 60000) // 10 1-minute bars
      const aggregator = new TimeframeAggregator({ 
        targetTimeframe: '5m',
        incompleteBarBehavior: 'emit',
      })

      const result = await aggregator.apply(arrayToAsyncIterator(testData))
      const aggregated = await collectResults(result.data)

      // With emit behavior, we get both complete and incomplete bars
      strictEqual(aggregated.length, 2)

      // Check first 5m bar (aggregates minutes 0-4)
      const firstBar = aggregated[0]!
      strictEqual(firstBar.open, 100) // Open of first 1m bar
      strictEqual(firstBar.close, 105) // Close of 5th 1m bar
      strictEqual(firstBar.volume, 5100) // Sum of volumes

      // Check second 5m bar (aggregates minutes 5-9)
      const secondBar = aggregated[1]!
      strictEqual(secondBar.open, 105) // Open of 6th 1m bar (index 5)
      strictEqual(secondBar.high, 111) // Max high (102+9)
      strictEqual(secondBar.low, 104) // Min low (99+5)
      strictEqual(secondBar.close, 110) // Close of 10th 1m bar (index 9): 101 + 9 = 110
      strictEqual(secondBar.volume, 5350) // Sum of volumes
    })

    it('should handle incomplete bars with drop behavior', async () => {
      const testData = createTestData(7, 60000) // 7 1-minute bars
      const aggregator = new TimeframeAggregator({
        targetTimeframe: '5m',
        incompleteBarBehavior: 'drop',
      })

      const result = await aggregator.apply(arrayToAsyncIterator(testData))
      const aggregated = await collectResults(result.data)

      // Should have only 1 complete 5m bar (first 5 minutes)
      strictEqual(aggregated.length, 1)
    })

    it('should handle incomplete bars with emit behavior', async () => {
      const testData = createTestData(7, 60000) // 7 1-minute bars
      const aggregator = new TimeframeAggregator({
        targetTimeframe: '5m',
        incompleteBarBehavior: 'emit',
      })

      const result = await aggregator.apply(arrayToAsyncIterator(testData))
      const aggregated = await collectResults(result.data)

      // Should have 2 bars: 1 complete and 1 incomplete
      strictEqual(aggregated.length, 2)

      // Check incomplete bar
      const incompleteBar = aggregated[1]!
      strictEqual(incompleteBar.open, 105) // Open of 6th 1m bar
      strictEqual(incompleteBar.close, 107) // Close of 7th 1m bar (index 6): 101 + 6 = 107
      
      // Volume calculation: bar 5 (index 5) = 1050, bar 6 (index 6) = 1060
      // Total = 1050 + 1060 = 2110
      strictEqual(incompleteBar.volume, 2110) // Sum of 2 bars
    })

    it('should aggregate to hourly bars', async () => {
      const testData = createTestData(121, 60000) // 121 1-minute bars (2 hours + 1 minute to trigger emission)
      const aggregator = new TimeframeAggregator({ targetTimeframe: '1h' })

      const result = await aggregator.apply(arrayToAsyncIterator(testData))
      const aggregated = await collectResults(result.data)

      // Should have 2 hourly bars (the 121st minute triggers emission of the second hour)
      strictEqual(aggregated.length, 2)

      // Check first hourly bar
      const firstHour = aggregated[0]!
      strictEqual(firstHour.open, 100) // Open of first minute
      strictEqual(firstHour.close, 160) // Close of 60th minute (index 59): 101 + 59 = 160
      // Volume: sum of 1000 + i*10 for i=0 to 59
      // = 60*1000 + 10*(0+1+...+59) = 60000 + 10*1770 = 60000 + 17700 = 77700
      strictEqual(firstHour.volume, 77700) // Sum of 60 minutes
    })

    it('should handle multiple symbols', async () => {
      const testData: OhlcvDto[] = []
      const baseTime = new Date('2024-01-01T10:00:00Z').getTime()

      // Interleave data for two symbols
      for (let i = 0; i < 10; i++) {
        testData.push({
          timestamp: baseTime + i * 60000,
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 100 + i,
          high: 102 + i,
          low: 99 + i,
          close: 101 + i,
          volume: 1000 + i * 10,
        })
        testData.push({
          timestamp: baseTime + i * 60000,
          symbol: 'ETHUSD',
          exchange: 'test',
          open: 50 + i,
          high: 52 + i,
          low: 49 + i,
          close: 51 + i,
          volume: 500 + i * 5,
        })
      }

      const aggregator = new TimeframeAggregator({ 
        targetTimeframe: '5m',
        incompleteBarBehavior: 'emit',
      })
      const result = await aggregator.apply(arrayToAsyncIterator(testData))
      const aggregated = await collectResults(result.data)

      // Should have 4 bars total (2 symbols × 2 5m bars each)
      strictEqual(aggregated.length, 4)

      // Check we have bars for both symbols
      const btcBars = aggregated.filter(bar => bar.symbol === 'BTCUSD')
      const ethBars = aggregated.filter(bar => bar.symbol === 'ETHUSD')

      strictEqual(btcBars.length, 2)
      strictEqual(ethBars.length, 2)

      // Verify BTC aggregation
      strictEqual(btcBars[0]!.open, 100)
      strictEqual(btcBars[0]!.close, 105)

      // Verify ETH aggregation
      strictEqual(ethBars[0]!.open, 50)
      strictEqual(ethBars[0]!.close, 55)
    })

    it('should handle empty data', async () => {
      const aggregator = new TimeframeAggregator({ targetTimeframe: '5m' })
      const result = await aggregator.apply(arrayToAsyncIterator([]))
      const aggregated = await collectResults(result.data)

      strictEqual(aggregated.length, 0)
    })

    it('should handle single data point', async () => {
      const testData = createTestData(1)
      const aggregator = new TimeframeAggregator({
        targetTimeframe: '5m',
        incompleteBarBehavior: 'emit',
      })

      const result = await aggregator.apply(arrayToAsyncIterator(testData))
      const aggregated = await collectResults(result.data)

      strictEqual(aggregated.length, 1)
      deepStrictEqual(aggregated[0], {
        timestamp: Math.floor(testData[0]!.timestamp / (5 * 60000)) * (5 * 60000),
        symbol: 'BTCUSD',
        exchange: 'test',
        open: 100,
        high: 102,
        low: 99,
        close: 101,
        volume: 1000,
      })
    })
  })

  describe('getOutputFields and getRequiredFields', () => {
    it('should return empty output fields', () => {
      const aggregator = new TimeframeAggregator({ targetTimeframe: '5m' })
      const fields = aggregator.getOutputFields()
      strictEqual(fields.length, 0)
    })

    it('should return required fields', () => {
      const aggregator = new TimeframeAggregator({ targetTimeframe: '5m' })
      const fields = aggregator.getRequiredFields()
      deepStrictEqual(fields, ['timestamp', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume'])
    })
  })

  describe('withParams', () => {
    it('should create new instance with updated params', () => {
      const original = new TimeframeAggregator({ targetTimeframe: '5m' })
      const updated = original.withParams({ targetTimeframe: '1h' })

      strictEqual(original.params.targetTimeframe, '5m')
      strictEqual(updated.params.targetTimeframe, '1h')
      ok(original !== updated)
    })

    it('should preserve other parameters', () => {
      const original = new TimeframeAggregator({
        targetTimeframe: '5m',
        incompleteBarBehavior: 'emit',
      })
      const updated = original.withParams({ targetTimeframe: '1h' })

      strictEqual(updated.params.incompleteBarBehavior, 'emit')
    })
  })

  describe('edge cases', () => {
    it('should handle out-of-order timestamps', async () => {
      const baseTime = new Date('2024-01-01T10:00:00Z').getTime()
      const testData: OhlcvDto[] = [
        {
          timestamp: baseTime + 2 * 60000, // 3rd minute
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 102,
          high: 104,
          low: 101,
          close: 103,
          volume: 1020,
        },
        {
          timestamp: baseTime, // 1st minute
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 100,
          high: 102,
          low: 99,
          close: 101,
          volume: 1000,
        },
        {
          timestamp: baseTime + 60000, // 2nd minute
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 101,
          high: 103,
          low: 100,
          close: 102,
          volume: 1010,
        },
      ]

      const aggregator = new TimeframeAggregator({
        targetTimeframe: '5m',
        incompleteBarBehavior: 'emit',
      })
      const result = await aggregator.apply(arrayToAsyncIterator(testData))
      const aggregated = await collectResults(result.data)

      strictEqual(aggregated.length, 1)
      
      // Should still aggregate correctly despite out-of-order data
      const bar = aggregated[0]!
      strictEqual(bar.open, 102) // First received (not earliest timestamp)
      strictEqual(bar.high, 104) // Maximum high
      strictEqual(bar.low, 99) // Minimum low
      strictEqual(bar.close, 102) // Last received
      strictEqual(bar.volume, 3030) // Sum of all volumes
    })

    it('should handle gaps in data with drop behavior', async () => {
      const baseTime = new Date('2024-01-01T10:00:00Z').getTime()
      const testData: OhlcvDto[] = [
        {
          timestamp: baseTime, // 10:00
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 100,
          high: 102,
          low: 99,
          close: 101,
          volume: 1000,
        },
        {
          timestamp: baseTime + 10 * 60000, // 10:10 (10 minute gap)
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 110,
          high: 112,
          low: 109,
          close: 111,
          volume: 1100,
        },
      ]

      const aggregator = new TimeframeAggregator({ targetTimeframe: '5m' })
      const result = await aggregator.apply(arrayToAsyncIterator(testData))
      const aggregated = await collectResults(result.data)

      // With drop behavior, only the first bar is complete (0-5 min)
      // The second data point at 10 min starts a new bar (10-15 min) which is incomplete
      strictEqual(aggregated.length, 1)
      
      // First bar should only contain first data point
      strictEqual(aggregated[0]!.volume, 1000)
    })

    it('should handle gaps in data with emit behavior', async () => {
      const baseTime = new Date('2024-01-01T10:00:00Z').getTime()
      const testData: OhlcvDto[] = [
        {
          timestamp: baseTime, // 10:00
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 100,
          high: 102,
          low: 99,
          close: 101,
          volume: 1000,
        },
        {
          timestamp: baseTime + 10 * 60000, // 10:10 (10 minute gap)
          symbol: 'BTCUSD',
          exchange: 'test',
          open: 110,
          high: 112,
          low: 109,
          close: 111,
          volume: 1100,
        },
      ]

      const aggregator = new TimeframeAggregator({ 
        targetTimeframe: '5m',
        incompleteBarBehavior: 'emit',
      })
      const result = await aggregator.apply(arrayToAsyncIterator(testData))
      const aggregated = await collectResults(result.data)

      // With emit behavior, we get both bars (complete and incomplete)
      strictEqual(aggregated.length, 2)
      
      // First bar should only contain first data point
      strictEqual(aggregated[0]!.volume, 1000)
      
      // Second bar should only contain second data point
      strictEqual(aggregated[1]!.volume, 1100)
    })
  })
})