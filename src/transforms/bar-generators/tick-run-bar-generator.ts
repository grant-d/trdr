import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import type { BaseBarState } from './base-bar-generator'
import { BaseBarGenerator, baseBarSchema } from './base-bar-generator'

/**
 * Schema for tick run bar configuration
 * @property {number} runLength - Number of consecutive ticks in same direction to trigger a new bar
 * @property {boolean} [useVolume] - Use volume runs instead of tick count runs
 */
const txSchema = z.object({
  runLength: z.number().positive(),
  useVolume: z.boolean().default(false)
})

/**
 * Main schema for TickRunBarGenerator transform
 */
const schema = baseBarSchema.extend({
  tx: z.union([txSchema, z.array(txSchema)])
})

interface TickRunBarParams
  extends z.infer<typeof schema>,
          BaseTransformParams {
}

interface TickRunBarState extends BaseBarState {
  currentRunLength: number;
  currentRunDirection: 'up' | 'down' | 'neutral';
  previousClose: number;
  volumeInRun: number;
}

/**
 * Tick Run Bar Generator
 *
 * Generates bars based on tick runs - consecutive ticks moving in the same direction.
 * This advanced bar type is particularly effective for capturing momentum periods
 * and directional market moves.
 *
 * **Algorithm**:
 * 1. Classify each tick direction: up, down, or neutral
 * 2. Track consecutive ticks in the same direction
 * 3. When run length >= threshold, complete the bar
 * 4. Start new bar with fresh run tracking
 *
 * **Direction Classification**:
 * - Up: current price > previous price
 * - Down: current price < previous price
 * - Neutral: current price = previous price (doesn't break runs)
 *
 * **Key Properties**:
 * - Bars form during sustained directional movement
 * - Natural breakpoints at momentum changes
 * - More bars during trending periods
 * - Each bar represents cohesive directional move
 *
 * **Use Cases**:
 * - Momentum and trend analysis
 * - Breakout detection
 * - Market microstructure studies
 * - Reversal identification
 * - High-frequency trading strategies
 *
 * @example
 * ```typescript
 * // Capture runs of 10 consecutive directional ticks
 * const tickRunBars = new TickRunBarGenerator({
 *   tx: {
 *     runLength: 10,
 *     useVolume: false
 *   }
 * }, inputBuffer)
 *
 * // Volume-weighted runs for institutional momentum
 * const volumeRunBars = new TickRunBarGenerator({
 *   tx: {
 *     runLength: 50000,
 *     useVolume: true
 *   }
 * }, inputBuffer)
 * ```
 *
 * @note Neutral ticks maintain but don't extend runs
 * @note First tick has no previous close for comparison
 * @note State maintained across batch boundaries
 */
export class TickRunBarGenerator extends BaseBarGenerator<
  TickRunBarParams,
  TickRunBarState
> {
  // Configuration
  private readonly _runLength: number
  private readonly _useVolume: boolean
  // Track previous close across all bars
  private _lastClose: number | undefined

  constructor(config: TickRunBarParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'tickRunBars',
      'TickRunBars',
      config.description || 'Tick Run Bar Generator',
      parsed,
      inputSlice
    )

    // Use first config
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    const txConfig = tx[0]
    if (!txConfig) {
      throw new Error('At least one configuration is required')
    }

    this._runLength = txConfig.runLength
    this._useVolume = txConfig.useVolume
  }

  /**
   * Create a new bar from the first tick
   */
  protected createNewBar(tick: any, _rid: number): TickRunBarState {
    const tickDirection = this.getTickDirection(tick.close, this._lastClose)

    return {
      open: tick.open,
      high: tick.high,
      low: tick.low,
      close: tick.close,
      volume: tick.volume,
      firstTimestamp: tick.timestamp,
      lastTimestamp: tick.timestamp,
      tickCount: 1,
      currentRunLength: tickDirection === 'neutral' ? 0 : 1,
      currentRunDirection: tickDirection,
      previousClose: tick.close,
      volumeInRun: tick.volume
    }
  }

  /**
   * Update the current bar with new tick data
   */
  protected updateBar(bar: TickRunBarState, tick: any, _rid: number): void {
    const tickDirection = this.getTickDirection(tick.close, bar.previousClose)

    // Update OHLCV values
    bar.high = Math.max(bar.high, tick.high)
    bar.low = Math.min(bar.low, tick.low)
    bar.close = tick.close
    bar.volume += tick.volume
    bar.lastTimestamp = tick.timestamp
    bar.tickCount++

    // Update run tracking
    if (tickDirection !== 'neutral') {
      if (
        tickDirection === bar.currentRunDirection ||
        bar.currentRunDirection === 'neutral'
      ) {
        // Same direction or first direction, increment run
        if (bar.currentRunDirection === 'neutral') {
          bar.currentRunDirection = tickDirection
        }
        bar.currentRunLength++
        bar.volumeInRun += tick.volume
      } else {
        // Direction changed, reset run
        bar.currentRunDirection = tickDirection
        bar.currentRunLength = 1
        bar.volumeInRun = tick.volume
      }
    } else {
      // Neutral tick, just add volume
      bar.volumeInRun += tick.volume
    }

    // Update previous close for next tick
    bar.previousClose = tick.close
  }

  /**
   * Check if the current bar is complete
   */
  protected isBarComplete(
    bar: TickRunBarState,
    _tick: any,
    _rid: number
  ): boolean {
    if (this._useVolume) {
      // For volume runs, check volume units
      const avgVolumePerTick = bar.volume / bar.tickCount
      const volumeRunUnits = bar.volumeInRun / avgVolumePerTick
      return volumeRunUnits >= this._runLength
    } else {
      // For tick count runs, check if we've reached the threshold
      return bar.currentRunLength >= this._runLength
    }
  }

  /**
   * Override to track last close across bars
   */
  protected emitBar(sourceRid: number, bar: TickRunBarState): void {
    // Store last close for next bar's first tick classification
    this._lastClose = bar.close

    // Call parent implementation
    super.emitBar(sourceRid, bar)
  }

  /**
   * Override to add run metrics to emitted bars
   */
  protected addAdditionalBarFields(
    row: Record<string, number>,
    bar: TickRunBarState
  ): void {
    // Add run metrics
    row.run_length = bar.currentRunLength
    row.run_direction =
      bar.currentRunDirection === 'up'
        ? 1
        : bar.currentRunDirection === 'down'
          ? -1
          : 0
    row.volume_in_run = bar.volumeInRun
  }

  /**
   * Classify tick direction based on price movement
   */
  private getTickDirection(
    currentPrice: number,
    previousPrice: number | undefined
  ): 'up' | 'down' | 'neutral' {
    if (previousPrice === undefined) {
      return 'neutral'
    }

    if (currentPrice > previousPrice) {
      return 'up'
    } else if (currentPrice < previousPrice) {
      return 'down'
    } else {
      return 'neutral'
    }
  }
}
