import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import type { BaseBarState } from './base-bar-generator'
import { BaseBarGenerator, baseBarSchema } from './base-bar-generator'

/**
 * Schema for Lorentzian distance bar configuration
 * @property {number} cFactor - Scaling factor for time component (min: 0.1)
 * @property {number} threshold - Distance threshold for bar completion (min: 10.0)
 */
const txSchema = z.object({
  cFactor: z.number().min(0.1),
  threshold: z.number().min(10.0)
})

/**
 * Main schema for LorentzianDistanceBarGenerator transform
 */
const schema = baseBarSchema.extend({
  tx: z.union([txSchema, z.array(txSchema)])
})

interface LorentzianDistanceBarParams
  extends z.infer<typeof schema>,
          BaseTransformParams {
}

interface LorentzianDistanceBarState extends BaseBarState {
  /** Anchor point price for distance calculation */
  anchorPrice: number;
  /** Anchor point volume for distance calculation */
  anchorVolume: number;
  /** Anchor point time (bar index) for distance calculation */
  anchorTime: number;
  /** Current time index */
  currentTime: number;
}

/**
 * Lorentzian Distance Bar Generator
 *
 * Calculates Lorentzian distance between current state and anchor point
 * in price-time-volume space using relativistic geometry.
 *
 * **Formula**: d = √(c²Δt² - Δp² - Δv²) where c is a scaling factor
 *
 * **Algorithm**:
 * 1. Set anchor point at bar start (price, volume, time)
 * 2. For each new tick, calculate deltas from anchor
 * 3. Compute Lorentzian distance using relativistic formula
 * 4. If space-like interval (negative), use Euclidean fallback
 * 5. Create new bar when distance exceeds threshold
 *
 * **Key Properties**:
 * - Captures relativistic "warping" of market spacetime
 * - More sensitive to rapid moves than Euclidean distance
 * - Handles extreme market conditions with relativistic approach
 * - Excels at capturing acceleration/deceleration dynamics
 *
 * **Use Cases**:
 * - High-frequency trading algorithms
 * - Volatility regime detection
 * - Acceleration/deceleration analysis
 * - Market microstructure studies
 * - Temporal dynamics analysis
 *
 * @example
 * ```typescript
 * const lorentzianBars = new LorentzianDistanceBarGenerator({
 *   tx: {
 *     cFactor: 1.0,      // Time scaling factor
 *     threshold: 50.0    // Distance threshold
 *   }
 * }, inputBuffer)
 * ```
 *
 * @note Space-like intervals use Euclidean fallback
 * @note Time index increments with each tick
 * @note State maintained across batch boundaries
 */
export class LorentzianDistanceBarGenerator extends BaseBarGenerator<
  LorentzianDistanceBarParams,
  LorentzianDistanceBarState
> {
  // Configuration
  private readonly _cFactor: number
  private readonly _threshold: number
  // Track time index
  private _timeIndex = 0

  constructor(config: LorentzianDistanceBarParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'lorentzianBars',
      'LorentzianBars',
      config.description || 'Lorentzian Distance Bar Generator',
      parsed,
      inputSlice
    )

    // Use first config
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    const txConfig = tx[0]
    if (!txConfig) {
      throw new Error('At least one configuration is required')
    }

    this._cFactor = txConfig.cFactor
    this._threshold = txConfig.threshold
  }

  /**
   * Override processBatch to track time index
   */
  protected processBatch(): { from: number; to: number } {
    let firstValidRow = -1
    const rowCount = this.inputSlice.length()

    for (let rid = 0; rid < rowCount; rid++) {
      this._timeIndex++
      this._totalRowsProcessed++

      // Get tick data
      const tickData = this.extractTickData(rid)

      // Process tick
      if (!this._currentBar) {
        // Start new bar
        this._currentBar = this.createNewBar(tickData, rid)
      } else {
        // Update current bar
        this.updateBar(this._currentBar, tickData, rid)

        // Check if bar is complete
        if (this.isBarComplete(this._currentBar, tickData, rid)) {
          // Emit completed bar to output buffer
          this.emitBar(rid, this._currentBar)

          // Track first valid row (in absolute buffer coordinates)
          if (firstValidRow === -1) {
            firstValidRow = this.inputSlice.from + rid
          }

          // Reset for next bar
          this._currentBar = undefined
          this._barsGenerated++
        }
      }
    }

    // Mark as ready after processing first batch
    this._isReady = true

    // Return the range of rows that were processed
    return {
      from: firstValidRow === -1 ? this.inputSlice.to : firstValidRow,
      to: this.inputSlice.to
    }
  }

  /**
   * Create a new bar from the first tick
   */
  protected createNewBar(tick: any, _rid: number): LorentzianDistanceBarState {
    return {
      open: tick.open,
      high: tick.high,
      low: tick.low,
      close: tick.close,
      volume: tick.volume,
      firstTimestamp: tick.timestamp,
      lastTimestamp: tick.timestamp,
      tickCount: 1,
      anchorPrice: tick.close,
      anchorVolume: tick.volume,
      anchorTime: this._timeIndex,
      currentTime: this._timeIndex
    }
  }

  /**
   * Update the current bar with new tick data
   */
  protected updateBar(
    bar: LorentzianDistanceBarState,
    tick: any,
    _rid: number
  ): void {
    // Update current time
    bar.currentTime = this._timeIndex

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
    bar: LorentzianDistanceBarState,
    tick: any,
    _rid: number
  ): boolean {
    // Calculate Lorentzian distance from anchor point
    const distance = this.calculateLorentzianDistance(tick, bar)
    return distance >= this._threshold
  }

  /**
   * Override to reset anchor point when new bar starts
   */
  protected emitBar(sourceRid: number, bar: LorentzianDistanceBarState): void {
    // Call parent implementation
    super.emitBar(sourceRid, bar)
  }

  /**
   * Override to add distance metrics to emitted bars
   */
  protected addAdditionalBarFields(
    row: Record<string, number>,
    bar: LorentzianDistanceBarState
  ): void {
    // Calculate final distance for the bar
    const finalDistance = this.calculateLorentzianDistance(
      { close: bar.close, volume: bar.volume } as any,
      bar
    )

    // Add distance metrics
    row.lorentzian_distance = finalDistance
    row.time_elapsed = bar.currentTime - bar.anchorTime
  }

  /**
   * Calculate Lorentzian distance from anchor point
   */
  private calculateLorentzianDistance(
    tick: any,
    state: LorentzianDistanceBarState
  ): number {
    // Calculate deltas from anchor point
    const deltaTime = state.currentTime - state.anchorTime

    // Normalize price delta as percentage
    const deltaPrice =
      state.anchorPrice !== 0
        ? ((tick.close - state.anchorPrice) / state.anchorPrice) * 100
        : 0

    // Log-normalize volume delta to handle large variations
    const deltaVolume =
      state.anchorVolume > 0
        ? Math.log(tick.volume / state.anchorVolume + 1) * 10
        : 0

    // Calculate Lorentzian distance: d = √(c²Δt² - Δp² - Δv²)
    const lorentzianComponent =
      this._cFactor * this._cFactor * deltaTime * deltaTime -
      deltaPrice * deltaPrice -
      deltaVolume * deltaVolume

    // If component is negative (space-like interval), use Euclidean distance in price-volume space
    if (lorentzianComponent > 0) {
      return Math.sqrt(lorentzianComponent)
    } else {
      // Euclidean fallback
      return Math.sqrt(deltaPrice * deltaPrice + deltaVolume * deltaVolume)
    }
  }
}
