import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import type { BaseBarState } from './base-bar-generator'
import { BaseBarGenerator, baseBarSchema } from './base-bar-generator'

/**
 * Schema for time bar configuration
 * @property {string} timeframe - Target timeframe (e.g., '5m', '1h', '4h', '1d')
 * @property {boolean} [alignToMarketOpen] - Whether to align timestamps to market open
 * @property {string} [marketOpenTime] - Market open time in HH:MM format (default: '09:30')
 */
const txSchema = z.object({
  timeframe: z.enum([
    '1m',
    '5m',
    '15m',
    '30m',
    '1h',
    '2h',
    '4h',
    '6h',
    '8h',
    '12h',
    '1d',
    '1w',
    '1M'
  ]),
  alignToMarketOpen: z.boolean().default(false),
  marketOpenTime: z
    .string()
    .regex(/^([0-1][0-9]|2[0-3]):[0-5][0-9]$/)
    .default('09:30')
  // marketTimezone: z.string().default('America/New_York')
})

/**
 * Main schema for TimeBarGenerator transform
 */
const schema = baseBarSchema.extend({
  tx: z.union([txSchema, z.array(txSchema)])
})

interface TimeBarParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface TimeBarState extends BaseBarState {
  barKey: number;
}

/**
 * Time Bar Generator
 *
 * Generates bars based on fixed time intervals. This is the most common bar type
 * used in traditional technical analysis and charting applications.
 *
 * **Algorithm**:
 * 1. Calculate bar key based on timestamp and timeframe
 * 2. Accumulate ticks within the same time window
 * 3. When timestamp crosses into new window, complete the bar
 * 4. Start new bar for the new time window
 *
 * **Supported Timeframes**:
 * - Minutes: 1m, 5m, 15m, 30m
 * - Hours: 1h, 2h, 4h, 6h, 8h, 12h
 * - Days: 1d
 * - Weeks: 1w
 * - Months: 1M (approximate 30 days)
 *
 * **Key Properties**:
 * - Fixed time intervals regardless of market activity
 * - Predictable bar timestamps
 * - May contain zero volume during quiet periods
 * - Standard for most charting applications
 *
 * **Use Cases**:
 * - Traditional technical analysis
 * - Chart pattern recognition
 * - Time-based indicators (moving averages, etc.)
 * - Multi-timeframe analysis
 * - Backtesting with historical data
 *
 * @example
 * ```typescript
 * // Generate 5-minute bars
 * const fiveMinBars = new TimeBarGenerator({
 *   tx: { timeframe: '5m' }
 * }, inputSlice)
 *
 * // Generate daily bars aligned to market open
 * const dailyBars = new TimeBarGenerator({
 *   tx: {
 *     timeframe: '1d',
 *     alignToMarketOpen: true,
 *     marketOpenTime: '09:30'
 *   }
 * }, inputSlice)
 * ```
 *
 * @note Empty bars are not generated - only bars with data
 * @note State is maintained across batch boundaries
 * @note Market alignment is approximate without proper timezone handling
 */
export class TimeBarGenerator extends BaseBarGenerator<
  TimeBarParams,
  TimeBarState
> {
  private static readonly TIMEFRAME_MAPPINGS: Readonly<Record<string, number>> =
    {
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
      '1M': 30 * 24 * 60 * 60 * 1000 // Approximate
    }

  // Configuration
  private readonly _timeframeMs: number
  private readonly _alignToMarketOpen: boolean
  private readonly _marketOpenTime: string

  constructor(config: TimeBarParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'timeBars',
      'TimeBars',
      config.description || 'Time Bar Generator',
      parsed,
      inputSlice
    )

    // Use first config
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    const txConfig = tx[0]
    if (!txConfig) {
      throw new Error('At least one configuration is required')
    }

    this._timeframeMs = TimeBarGenerator.TIMEFRAME_MAPPINGS[txConfig.timeframe]!
    this._alignToMarketOpen = txConfig.alignToMarketOpen
    this._marketOpenTime = txConfig.marketOpenTime
  }

  /**
   * Create a new bar from the first tick
   */
  protected createNewBar(tick: any, _rid: number): TimeBarState {
    const barKey = this.getBarKey(tick.timestamp)

    return {
      open: tick.open,
      high: tick.high,
      low: tick.low,
      close: tick.close,
      volume: tick.volume,
      firstTimestamp: tick.timestamp,
      lastTimestamp: tick.timestamp,
      tickCount: 1,
      barKey
    }
  }

  /**
   * Update the current bar with new tick data
   */
  protected updateBar(bar: TimeBarState, tick: any, _rid: number): void {
    // Update OHLCV values
    bar.high = Math.max(bar.high, tick.high)
    bar.low = Math.min(bar.low, tick.low)
    bar.close = tick.close
    bar.volume += tick.volume
    bar.lastTimestamp = tick.timestamp
    bar.tickCount++
  }

  /**
   * Check if the current bar is complete
   */
  protected isBarComplete(bar: TimeBarState, tick: any, _rid: number): boolean {
    const currentBarKey = this.getBarKey(tick.timestamp)
    return currentBarKey !== bar.barKey
  }

  /**
   * Override to set timestamp to bar key instead of last tick timestamp
   */
  protected emitBar(sourceRid: number, bar: TimeBarState): void {
    // Create new row object with aggregated bar values
    const newRow: Record<string, number> = {}

    // Copy all columns from source row first to preserve metadata
    const columnNames = this.inputSlice.getColumns()
    for (const colName of columnNames) {
      const colDef = this.inputSlice.getColumn(colName)
      if (colDef) {
        newRow[colName] =
          this.inputSlice.getValue(sourceRid, colDef.index) || 0
      }
    }

    // Then update with aggregated bar values
    // Use bar key as timestamp for consistent time-based bars
    newRow.timestamp = bar.barKey
    newRow.open = bar.open
    newRow.high = bar.high
    newRow.low = bar.low
    newRow.close = bar.close
    newRow.volume = bar.volume

    // Push to output buffer
    this._outputBuffer.push(newRow)
  }

  /**
   * Convert a timestamp to a bar key based on the target timeframe
   */
  private getBarKey(timestamp: number): number {
    if (this._alignToMarketOpen) {
      return this.alignToMarketOpen(timestamp)
    }
    return Math.floor(timestamp / this._timeframeMs) * this._timeframeMs
  }

  /**
   * Align timestamp to market open time
   */
  private alignToMarketOpen(timestamp: number): number {
    // This is a simplified implementation
    // In production, you'd want to use a proper timezone library
    const date = new Date(timestamp)
    const [hours, minutes] = this._marketOpenTime.split(':').map(Number)

    // Set to market open time
    date.setHours(hours!, minutes, 0, 0)

    // Find the correct bar based on market open
    const marketOpenMillis = date.getTime()
    const barsSinceOpen = Math.floor(
      (timestamp - marketOpenMillis) / this._timeframeMs
    )

    return marketOpenMillis + barsSinceOpen * this._timeframeMs
  }
}
