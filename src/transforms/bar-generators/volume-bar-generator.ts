import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import type { BaseBarState } from './base-bar-generator'
import { BaseBarGenerator, baseBarSchema } from './base-bar-generator'

/**
 * Schema for volume bar configuration
 * @property {number} volume - Volume threshold that triggers bar completion
 * @property {string} [volumeField] - Field name for volume data (default: 'volume')
 */
const txSchema = z.object({
  volume: z.number().positive(),
  volumeField: z
    .string()
    .regex(/^[a-zA-Z0-9_]{1,20}$/)
    .default('volume')
})

/**
 * Main schema for VolumeBarGenerator transform
 */
const schema = baseBarSchema.extend({
  tx: z.union([txSchema, z.array(txSchema)])
})

interface VolumeBarParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface VolumeBarState extends BaseBarState {
  accumulatedVolume: number;
}

/**
 * Volume Bar Generator
 *
 * Generates bars based on accumulated volume rather than time or tick count. This creates
 * bars that represent equal liquidity consumption, making them particularly useful for
 * analyzing market microstructure and institutional trading patterns.
 *
 * **Algorithm**:
 * 1. Accumulate volume from each incoming tick
 * 2. When accumulated volume ≥ threshold, complete the bar
 * 3. Start new bar with volume accumulation reset
 *
 * **Key Properties**:
 * - Each bar represents equal liquidity consumption
 * - More bars during high-volume periods
 * - Better reflects actual market participation
 * - Highlights institutional trading activity
 *
 * **Use Cases**:
 * - Liquidity analysis and profiling
 * - Institutional flow detection
 * - Market impact studies
 * - TWAP/VWAP algorithm development
 * - Volume cluster analysis
 *
 * @example
 * ```typescript
 * // Create bars when 50,000 shares are traded
 * const volumeBars = new VolumeBarGenerator({
 *   tx: { volume: 50000 }
 * }, inputBuffer)
 *
 * // Using custom volume field
 * const customVolumeBars = new VolumeBarGenerator({
 *   tx: { volume: 100000, volumeField: 'vol' }
 * }, inputBuffer)
 *
 * // For high-volume assets
 * const cryptoBars = new VolumeBarGenerator({
 *   tx: { volume: 1000000, volumeField: 'v' } // 1M units
 * }, inputBuffer)
 * ```
 *
 * @note Volume doesn't guarantee price movement
 * @note Each bar may contain different numbers of ticks
 * @note State is maintained across batch boundaries
 */
export class VolumeBarGenerator extends BaseBarGenerator<
  VolumeBarParams,
  VolumeBarState
> {
  // Configuration
  private readonly _volumePerBar: number
  private readonly _volumeFieldName: string
  private readonly _volumeFieldIndex: number

  constructor(config: VolumeBarParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'volumeBars',
      'VolumeBars',
      config.description || 'Volume Bar Generator',
      parsed,
      inputSlice
    )

    // Use first config
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    const txConfig = tx[0]
    if (!txConfig) {
      throw new Error('At least one configuration is required')
    }

    this._volumePerBar = txConfig.volume
    this._volumeFieldName = txConfig.volumeField

    // Get volume field index
    const volumeCol = inputSlice.getColumn(this._volumeFieldName)
    if (!volumeCol) {
      throw new Error(
        `Volume field '${this._volumeFieldName}' not found in input slice`
      )
    }
    this._volumeFieldIndex = volumeCol.index
  }

  /**
   * Extract tick data including custom volume field
   */
  protected extractTickData(rid: number): any {
    const baseData = super.extractTickData(rid)

    // Override volume with custom field if different
    if (this._volumeFieldName !== 'volume') {
      baseData.customVolume = this.inputSlice.getValue(
        rid,
        this._volumeFieldIndex
      )!
    }

    return baseData
  }

  /**
   * Create a new bar from the first tick
   */
  protected createNewBar(tick: any, _rid: number): VolumeBarState {
    const volume = tick.customVolume ?? tick.volume

    return {
      open: tick.open,
      high: tick.high,
      low: tick.low,
      close: tick.close,
      volume: volume,
      firstTimestamp: tick.timestamp,
      lastTimestamp: tick.timestamp,
      tickCount: 1,
      accumulatedVolume: volume
    }
  }

  /**
   * Update the current bar with new tick data
   */
  protected updateBar(bar: VolumeBarState, tick: any, _rid: number): void {
    const volume = tick.customVolume ?? tick.volume

    // Update OHLCV values
    bar.high = Math.max(bar.high, tick.high)
    bar.low = Math.min(bar.low, tick.low)
    bar.close = tick.close
    bar.volume += volume
    bar.lastTimestamp = tick.timestamp
    bar.tickCount++
    bar.accumulatedVolume += volume
  }

  /**
   * Check if the current bar is complete
   */
  protected isBarComplete(
    bar: VolumeBarState,
    _tick: any,
    _rid: number
  ): boolean {
    return bar.accumulatedVolume >= this._volumePerBar
  }
}
