import type { Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import { BaseTransform } from './base-transform'
import type { TimeframeAggregationParams } from './transform-params'

/**
 * Aggregates OHLCV data from one timeframe to another
 * Supports standard timeframes: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 1w, 1M
 */
export class TimeframeAggregator extends BaseTransform<TimeframeAggregationParams> {
  private static readonly TIMEFRAME_MAPPINGS: Record<string, number> = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '2h': 2 * 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '6h': 6 * 60 * 60 * 1000,
    '8h': 8 * 60 * 60 * 1000,
    '12h': 12 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
    '1w': 7 * 24 * 60 * 60 * 1000,
    '1M': 30 * 24 * 60 * 60 * 1000, // Approximate
  }

  constructor(params: TimeframeAggregationParams) {
    super(
      'timeframeAggregation',
      'Timeframe Aggregator',
      'Aggregates OHLCV data from one timeframe to another',
      {
        alignToMarketOpen: false,
        marketOpenTime: '09:30',
        marketTimezone: 'America/New_York',
        incompleteBarBehavior: 'drop',
        ...params,
      },
      false
    )
  }

  /**
   * Transform a stream of OHLCV data to a different timeframe
   */
  protected async* transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    const targetMillis = this.getTimeframeMillis(this.params.targetTimeframe)
    const bars = new Map<string, Map<number, AggregationBar>>()
    const completedBars: Array<[string, number, AggregationBar]> = []

    for await (const ohlcv of this.iterateData(data)) {
      const barKey = this.getBarKey(ohlcv.timestamp, targetMillis)
      const symbolKey = `${ohlcv.symbol}-${ohlcv.exchange}`

      // Get or create symbol bars map
      if (!bars.has(symbolKey)) {
        bars.set(symbolKey, new Map())
      }
      const symbolBars = bars.get(symbolKey)!

      // Get or create bar for this timeframe
      if (!symbolBars.has(barKey)) {
        symbolBars.set(barKey, {
          timestamp: barKey,
          symbol: ohlcv.symbol,
          exchange: ohlcv.exchange,
          open: ohlcv.open,
          high: ohlcv.high,
          low: ohlcv.low,
          close: ohlcv.close,
          volume: ohlcv.volume,
          count: 1,
          firstTimestamp: ohlcv.timestamp,
          lastTimestamp: ohlcv.timestamp,
        })
      } else {
        // Update existing bar
        const bar = symbolBars.get(barKey)!
        bar.high = Math.max(bar.high, ohlcv.high)
        bar.low = Math.min(bar.low, ohlcv.low)
        bar.close = ohlcv.close
        bar.volume += ohlcv.volume
        bar.count++
        bar.lastTimestamp = ohlcv.timestamp
      }

      // Check all symbols for completed bars
      for (const [symbol, symbolBars] of bars) {
        for (const [key, bar] of symbolBars) {
          if (key < barKey) {
            // This bar is complete, add to completed list
            completedBars.push([symbol, key, bar])
          }
        }
      }

      // Emit and remove completed bars
      if (completedBars.length > 0) {
        // Sort by timestamp to maintain order
        completedBars.sort((a, b) => a[1] - b[1])

        for (const [symbol, key, bar] of completedBars) {
          yield this.barToOhlcv(bar)
          bars.get(symbol)?.delete(key)
        }

        completedBars.length = 0
      }
    }

    // Handle remaining bars based on incompleteBarBehavior
    if (this.params.incompleteBarBehavior === 'emit') {
      const remainingBars: AggregationBar[] = []

      for (const symbolBars of bars.values()) {
        for (const bar of symbolBars.values()) {
          remainingBars.push(bar)
        }
      }

      // Sort by timestamp
      remainingBars.sort((a, b) => a.timestamp - b.timestamp)

      for (const bar of remainingBars) {
        yield this.barToOhlcv(bar)
      }
    }
  }

  /**
   * Validate the transform parameters
   */
  public validate(): void {
    super.validate()

    if (!this.params.targetTimeframe) {
      throw new Error('Target timeframe is required')
    }

    if (!TimeframeAggregator.TIMEFRAME_MAPPINGS[this.params.targetTimeframe]) {
      throw new Error(
        `Invalid target timeframe: ${this.params.targetTimeframe}. Valid values: ${Object.keys(
          TimeframeAggregator.TIMEFRAME_MAPPINGS,
        ).join(', ')}`
      )
    }

    if (this.params.marketOpenTime) {
      const timeRegex = /^([0-1][0-9]|2[0-3]):[0-5][0-9]$/
      if (!timeRegex.test(this.params.marketOpenTime)) {
        throw new Error('Market open time must be in HH:MM format')
      }
    }

    if (this.params.incompleteBarBehavior &&
      !['emit', 'drop'].includes(this.params.incompleteBarBehavior)) {
      throw new Error('incompleteBarBehavior must be either "emit" or "drop"')
    }
  }

  /**
   * Get output fields - same as input for aggregation
   */
  public getOutputFields(): string[] {
    return []
  }

  /**
   * Get required fields for aggregation
   */
  public getRequiredFields(): string[] {
    return ['timestamp', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume']
  }

  /**
   * Create a copy with new parameters
   */
  public withParams(params: Partial<TimeframeAggregationParams>): Transform<TimeframeAggregationParams> {
    return new TimeframeAggregator({ ...this.params, ...params })
  }

  /**
   * Convert a timestamp to a bar key based on the target timeframe
   */
  private getBarKey(timestamp: number, targetMillis: number): number {
    if (this.params.alignToMarketOpen) {
      return this.alignToMarketOpen(timestamp, targetMillis)
    }
    return Math.floor(timestamp / targetMillis) * targetMillis
  }

  /**
   * Align timestamp to market open time
   */
  private alignToMarketOpen(timestamp: number, targetMillis: number): number {
    // This is a simplified implementation
    // In production, you'd want to use a proper timezone library
    const date = new Date(timestamp)
    const [hours, minutes] = this.params.marketOpenTime!.split(':').map(Number)

    // Set to market open time
    date.setHours(hours!, minutes, 0, 0)

    // Find the correct bar based on market open
    const marketOpenMillis = date.getTime()
    const barsSinceOpen = Math.floor((timestamp - marketOpenMillis) / targetMillis)

    return marketOpenMillis + (barsSinceOpen * targetMillis)
  }

  /**
   * Get the millisecond value for a timeframe string
   */
  private getTimeframeMillis(timeframe: string): number {
    const millis = TimeframeAggregator.TIMEFRAME_MAPPINGS[timeframe]
    if (!millis) {
      throw new Error(`Invalid timeframe: ${timeframe}`)
    }
    return millis
  }

  /**
   * Convert an aggregation bar to OhlcvDto
   */
  private barToOhlcv(bar: AggregationBar): OhlcvDto {
    return {
      timestamp: bar.timestamp,
      symbol: bar.symbol,
      exchange: bar.exchange,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
      volume: bar.volume,
    }
  }

  /**
   * Helper to iterate over async data
   */
  private async* iterateData(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let result = await data.next()
    while (!result.done) {
      yield result.value
      result = await data.next()
    }
  }
}

/**
 * Internal interface for tracking aggregation state
 */
interface AggregationBar {
  timestamp: number
  symbol: string
  exchange: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  count: number
  firstTimestamp: number
  lastTimestamp: number
}
