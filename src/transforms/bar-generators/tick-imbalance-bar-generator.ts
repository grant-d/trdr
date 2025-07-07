import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import type { BaseBarState } from './base-bar-generator'
import { BaseBarGenerator, baseBarSchema } from './base-bar-generator'

/**
 * Schema for tick imbalance bar configuration
 * @property {number} imbalanceThreshold - Imbalance threshold that triggers bar completion
 * @property {boolean} [useVolume] - Use volume imbalance instead of tick count imbalance
 */
const txSchema = z.object({
  imbalanceThreshold: z.number().positive(),
  useVolume: z.boolean().default(false)
})

/**
 * Main schema for TickImbalanceBarGenerator transform
 */
const schema = baseBarSchema.extend({
  tx: z.union([txSchema, z.array(txSchema)])
})

interface TickImbalanceBarParams
  extends z.infer<typeof schema>,
          BaseTransformParams {
}

interface TickImbalanceBarState extends BaseBarState {
  buyTicks: number;
  sellTicks: number;
  buyVolume: number;
  sellVolume: number;
  previousClose: number;
}

/**
 * Tick Imbalance Bar Generator
 *
 * Generates bars based on tick imbalance - the difference between buy-side and sell-side
 * activity. This advanced bar type is particularly useful for detecting order flow imbalances
 * and market microstructure patterns.
 *
 * **Algorithm**:
 * 1. Classify each tick as buy, sell, or neutral based on price movement
 *    - Buy tick: current price > previous price
 *    - Sell tick: current price < previous price
 *    - Neutral: current price = previous price
 * 2. Track imbalance using either:
 *    - Tick Count: |buyTicks - sellTicks|
 *    - Volume: |buyVolume - sellVolume|
 * 3. When imbalance >= threshold, complete the bar
 * 4. Start new bar with fresh imbalance tracking
 *
 * **Key Properties**:
 * - Bars form at natural inflection points
 * - Captures periods of strong directional flow
 * - More bars during periods of order flow imbalance
 * - Better microstructure insights than time bars
 *
 * **Use Cases**:
 * - Order flow analysis and detection
 * - Market microstructure studies
 * - Momentum and pressure detection
 * - High-frequency trading strategies
 * - Liquidity and market maker analysis
 *
 * @example
 * ```typescript
 * // Detect imbalances of 20 ticks
 * const tickImbalanceBars = new TickImbalanceBarGenerator({
 *   tx: {
 *     imbalanceThreshold: 20,
 *     useVolume: false
 *   }
 * }, inputBuffer)
 *
 * // Volume-weighted imbalance for institutional flow
 * const volumeImbalanceBars = new TickImbalanceBarGenerator({
 *   tx: {
 *     imbalanceThreshold: 100000,
 *     useVolume: true
 *   }
 * }, inputBuffer)
 * ```
 *
 * @note Neutral ticks don't contribute to imbalance
 * @note First tick has no previous close for comparison
 * @note State maintained across batch boundaries
 */
export class TickImbalanceBarGenerator extends BaseBarGenerator<
  TickImbalanceBarParams,
  TickImbalanceBarState
> {
  // Configuration
  private readonly _imbalanceThreshold: number
  private readonly _useVolume: boolean
  // Track previous close across all bars
  private _lastClose: number | undefined

  constructor(config: TickImbalanceBarParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'tickImbalanceBars',
      'TickImbalanceBars',
      config.description || 'Tick Imbalance Bar Generator',
      parsed,
      inputSlice
    )

    // Use first config
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    const txConfig = tx[0]
    if (!txConfig) {
      throw new Error('At least one configuration is required')
    }

    this._imbalanceThreshold = txConfig.imbalanceThreshold
    this._useVolume = txConfig.useVolume
  }

  /**
   * Create a new bar from the first tick
   */
  protected createNewBar(tick: any, _rid: number): TickImbalanceBarState {
    const tickType = this.classifyTick(tick.close, this._lastClose)

    return {
      open: tick.open,
      high: tick.high,
      low: tick.low,
      close: tick.close,
      volume: tick.volume,
      firstTimestamp: tick.timestamp,
      lastTimestamp: tick.timestamp,
      tickCount: 1,
      buyTicks: tickType === 'buy' ? 1 : 0,
      sellTicks: tickType === 'sell' ? 1 : 0,
      buyVolume: tickType === 'buy' ? tick.volume : 0,
      sellVolume: tickType === 'sell' ? tick.volume : 0,
      previousClose: tick.close
    }
  }

  /**
   * Update the current bar with new tick data
   */
  protected updateBar(
    bar: TickImbalanceBarState,
    tick: any,
    _rid: number
  ): void {
    const tickType = this.classifyTick(tick.close, bar.previousClose)

    // Update tick counts and volumes
    if (tickType === 'buy') {
      bar.buyTicks++
      bar.buyVolume += tick.volume
    } else if (tickType === 'sell') {
      bar.sellTicks++
      bar.sellVolume += tick.volume
    }

    // Update OHLCV values
    bar.high = Math.max(bar.high, tick.high)
    bar.low = Math.min(bar.low, tick.low)
    bar.close = tick.close
    bar.volume += tick.volume
    bar.lastTimestamp = tick.timestamp
    bar.tickCount++

    // Update previous close for next tick
    bar.previousClose = tick.close
  }

  /**
   * Check if the current bar is complete
   */
  protected isBarComplete(
    bar: TickImbalanceBarState,
    _tick: any,
    _rid: number
  ): boolean {
    // Calculate imbalance
    const imbalance = this._useVolume
      ? Math.abs(bar.buyVolume - bar.sellVolume)
      : Math.abs(bar.buyTicks - bar.sellTicks)

    return imbalance >= this._imbalanceThreshold
  }

  /**
   * Override to track last close across bars
   */
  protected emitBar(sourceRid: number, bar: TickImbalanceBarState): void {
    // Store last close for next bar's first tick classification
    this._lastClose = bar.close

    // Call parent implementation
    super.emitBar(sourceRid, bar)
  }

  /**
   * Override to add imbalance metrics to emitted bars
   */
  protected addAdditionalBarFields(
    row: Record<string, number>,
    bar: TickImbalanceBarState
  ): void {
    // Add imbalance metrics
    row.buy_ticks = bar.buyTicks
    row.sell_ticks = bar.sellTicks
    row.buy_volume = bar.buyVolume
    row.sell_volume = bar.sellVolume
    row.tick_imbalance = bar.buyTicks - bar.sellTicks
    row.volume_imbalance = bar.buyVolume - bar.sellVolume
  }

  /**
   * Classify tick direction based on price movement
   */
  private classifyTick(
    currentPrice: number,
    previousPrice: number | undefined
  ): 'buy' | 'sell' | 'neutral' {
    if (previousPrice === undefined) {
      return 'neutral'
    }

    if (currentPrice > previousPrice) {
      return 'buy'
    } else if (currentPrice < previousPrice) {
      return 'sell'
    } else {
      return 'neutral'
    }
  }
}
