import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import type { BaseBarState } from './base-bar-generator'
import { BaseBarGenerator, baseBarSchema } from './base-bar-generator'

/**
 * Schema for VWAP bar configuration
 * @property {number} notionalValue - Notional dollar value threshold (price * volume)
 * @property {string} [priceField] - Price field for VWAP calculation: 'typical', 'close', 'vwap' (default: 'typical')
 * @property {string} [volumeField] - Volume field name (default: 'volume')
 */
const txSchema = z.object({
  notionalValue: z.number().positive(),
  priceField: z.enum(['typical', 'close', 'vwap']).default('typical'),
  volumeField: z
    .string()
    .regex(/^[a-zA-Z0-9_]{1,20}$/)
    .default('volume')
})

/**
 * Main schema for VwapBarGenerator transform
 */
const schema = baseBarSchema.extend({
  tx: z.union([txSchema, z.array(txSchema)])
})

interface VwapBarParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface VwapBarState extends BaseBarState {
  priceVolumeSum: number; // Sum of (price * volume) for VWAP calculation
  volumeSum: number; // Sum of volume for VWAP calculation
  notionalSum: number; // Accumulated notional value for bar completion
}

/**
 * VWAP Bar Generator (Volume-Weighted Average Price Bars)
 *
 * Generates bars based on notional value (price × volume) while tracking VWAP.
 * Unlike simple volume bars, these bars account for both price and volume,
 * creating more economically meaningful aggregations.
 *
 * **Algorithm**:
 * 1. Calculate notional value for each tick: price × volume
 * 2. Track running VWAP: Σ(price × volume) / Σ(volume)
 * 3. When accumulated notional ≥ threshold, complete the bar
 * 4. Start new bar with fresh VWAP calculation
 *
 * **VWAP Calculation**:
 * - VWAP = Σ(Price × Volume) / Σ(Volume)
 * - Resets with each new bar
 * - Represents average execution price weighted by volume
 *
 * **Key Properties**:
 * - Bars represent equal notional value traded
 * - Each bar has its own VWAP calculation
 * - More bars during high price × volume periods
 * - Captures both liquidity and price movement
 *
 * **Use Cases**:
 * - Institutional execution analysis
 * - VWAP trading strategies
 * - Market impact studies
 * - Liquidity-adjusted technical analysis
 * - Cross-asset comparison with notional normalization
 *
 * @example
 * ```typescript
 * // Generate bars every $1M notional
 * const vwapBars = new VwapBarGenerator({
 *   tx: {
 *     notionalValue: 1000000,
 *     priceField: 'typical'
 *   }
 * }, inputBuffer)
 *
 * // Using actual VWAP field if available
 * const vwapFieldBars = new VwapBarGenerator({
 *   tx: {
 *     notionalValue: 5000000,
 *     priceField: 'vwap',
 *     volumeField: 'vol'
 *   }
 * }, inputBuffer)
 * ```
 *
 * @note Each bar tracks its own VWAP from constituent ticks
 * @note Notional value = price × volume at each tick
 * @note State is maintained across batch boundaries
 */
export class VwapBarGenerator extends BaseBarGenerator<
  VwapBarParams,
  VwapBarState
> {
  // Configuration
  private readonly _notionalThreshold: number
  private readonly _priceField: 'typical' | 'close' | 'vwap'
  private readonly _volumeFieldName: string
  private readonly _volumeFieldIndex: number

  constructor(config: VwapBarParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'vwapBars',
      'VwapBars',
      config.description || 'VWAP Bar Generator',
      parsed,
      inputSlice
    )

    // Use first config
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    const txConfig = tx[0]
    if (!txConfig) {
      throw new Error('At least one configuration is required')
    }

    this._notionalThreshold = txConfig.notionalValue
    this._priceField = txConfig.priceField
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
   * Initialize VWAP index if available
   */
  protected initializeAdditionalIndices(inputSlice: DataSlice): void {
    const vwapCol = inputSlice.getColumn('vwap')
    if (vwapCol) {
      this._indices.vwap = vwapCol.index
    }
  }

  /**
   * Extract tick data including custom volume field and VWAP
   */
  protected extractTickData(rid: number): any {
    const baseData = super.extractTickData(rid)

    // Add custom volume if different field
    if (this._volumeFieldName !== 'volume') {
      baseData.customVolume = this.inputSlice.getValue(
        rid,
        this._volumeFieldIndex
      )!
    }

    // Add VWAP if available
    if (this._indices.vwap !== undefined) {
      baseData.vwap = this.inputSlice.getValue(
        rid,
        this._indices.vwap
      )!
    }

    return baseData
  }

  /**
   * Create a new bar from the first tick
   */
  protected createNewBar(tick: any, _rid: number): VwapBarState {
    const volume = tick.customVolume ?? tick.volume
    const price = this.calculatePrice(tick)
    const priceVolume = price * volume
    const notional = priceVolume

    return {
      open: tick.open,
      high: tick.high,
      low: tick.low,
      close: tick.close,
      volume: volume,
      firstTimestamp: tick.timestamp,
      lastTimestamp: tick.timestamp,
      tickCount: 1,
      priceVolumeSum: priceVolume,
      volumeSum: volume,
      notionalSum: notional
    }
  }

  /**
   * Update the current bar with new tick data
   */
  protected updateBar(bar: VwapBarState, tick: any, _rid: number): void {
    const volume = tick.customVolume ?? tick.volume
    const price = this.calculatePrice(tick)
    const priceVolume = price * volume
    const notional = priceVolume

    // Update OHLCV values
    bar.high = Math.max(bar.high, tick.high)
    bar.low = Math.min(bar.low, tick.low)
    bar.close = tick.close
    bar.volume += volume
    bar.lastTimestamp = tick.timestamp
    bar.tickCount++

    // Update VWAP tracking
    bar.priceVolumeSum += priceVolume
    bar.volumeSum += volume
    bar.notionalSum += notional
  }

  /**
   * Check if the current bar is complete
   */
  protected isBarComplete(
    bar: VwapBarState,
    _tick: any,
    _rid: number
  ): boolean {
    return bar.notionalSum >= this._notionalThreshold
  }

  /**
   * Override to add VWAP to emitted bars
   */
  protected addAdditionalBarFields(
    row: Record<string, number>,
    bar: VwapBarState
  ): void {
    // Calculate and add VWAP for this bar
    if (bar.volumeSum > 0) {
      row.bar_vwap = bar.priceVolumeSum / bar.volumeSum
    }
  }

  /**
   * Calculate price based on configured field
   */
  private calculatePrice(tick: any): number {
    switch (this._priceField) {
      case 'vwap':
        return tick.vwap || (tick.high + tick.low + tick.close) / 3
      case 'typical':
        return (tick.high + tick.low + tick.close) / 3
      case 'close':
      default:
        return tick.close
    }
  }
}
