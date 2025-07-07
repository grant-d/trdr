import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import type { BaseBarState } from './base-bar-generator'
import { BaseBarGenerator, baseBarSchema } from './base-bar-generator'

/**
 * Schema for tick bar configuration
 * @property {number} ticks - Number of ticks/trades that trigger bar completion
 */
const txSchema = z.object({
  ticks: z.number().int().positive()
})

/**
 * Main schema for TickBarGenerator transform
 */
const schema = baseBarSchema.extend({
  tx: z.union([txSchema, z.array(txSchema)])
})

interface TickBarParams extends z.infer<typeof schema>, BaseTransformParams {
}

/**
 * Tick Bar Generator
 *
 * Generates bars based on a fixed number of ticks or trades. This is one of the most basic
 * alternative bar types that provides equal representation of market activity rather than time.
 *
 * **Algorithm**:
 * 1. Count each incoming tick/trade
 * 2. When tick count reaches threshold, complete the bar
 * 3. Start a new bar with count reset to 1 for the next tick
 *
 * **Key Properties**:
 * - Each bar contains exactly the same number of trades
 * - Bars form based on market activity, not time
 * - More bars during active periods, fewer during quiet times
 * - No empty periods during market closures
 *
 * **Use Cases**:
 * - Market microstructure analysis
 * - High-frequency trading strategies
 * - Activity-based technical analysis
 * - Noise reduction in low-activity periods
 * - Fair comparison across different time periods
 *
 * @example
 * ```typescript
 * // Create bars every 50 ticks
 * const tickBars = new TickBarGenerator({
 *   tx: { ticks: 50 }
 * }, inputBuffer)
 *
 * // For high-frequency analysis
 * const microBars = new TickBarGenerator({
 *   tx: { ticks: 5 }
 * }, inputBuffer)
 * ```
 *
 * @note Each bar contains exactly the configured number of ticks
 * @note Time between bars varies based on market activity
 * @note State is maintained across batch boundaries
 */
export class TickBarGenerator extends BaseBarGenerator<TickBarParams> {
  // Configuration
  private readonly _ticksPerBar: number

  constructor(config: TickBarParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'tickBars',
      'TickBars',
      config.description || 'Tick Bar Generator',
      parsed,
      inputSlice
    )

    // Use first config
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    const txConfig = tx[0]
    if (!txConfig) {
      throw new Error('At least one configuration is required')
    }

    this._ticksPerBar = txConfig.ticks

    if (!Number.isInteger(this._ticksPerBar)) {
      throw new Error('Ticks per bar must be an integer')
    }
  }

  /**
   * Create a new bar from the first tick
   */
  protected createNewBar(tick: any, _rid: number): BaseBarState {
    return {
      open: tick.open,
      high: tick.high,
      low: tick.low,
      close: tick.close,
      volume: tick.volume,
      firstTimestamp: tick.timestamp,
      lastTimestamp: tick.timestamp,
      tickCount: 1
    }
  }

  /**
   * Update the current bar with new tick data
   */
  protected updateBar(bar: BaseBarState, tick: any, _rid: number): void {
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
  protected isBarComplete(
    bar: BaseBarState,
    _tick: any,
    _rid: number
  ): boolean {
    return bar.tickCount >= this._ticksPerBar
  }
}
