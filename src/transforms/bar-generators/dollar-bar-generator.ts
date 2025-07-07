import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import type { BaseBarState } from './base-bar-generator'
import { BaseBarGenerator, baseBarSchema } from './base-bar-generator'

/**
 * Schema for individual dollar bar configuration
 * @property {number} threshold - Dollar value threshold that triggers bar completion
 * @property {string} [priceField] - Price field for calculations: 'close', 'vwap', 'typical' (default: 'close')
 */
const txSchema = z.object({
  threshold: z.number().positive(),
  priceField: z.enum(['close', 'vwap', 'typical']).default('close')
})

/**
 * Main schema for DollarBarGenerator transform
 * @property {string} [description] - Optional description of the transform
 * @property {object|array} tx - Single dollar bar config or array of configs
 *
 * @example
 * // Single configuration
 * { tx: { threshold: 50000, priceField: "close" } }
 *
 * @example
 * // Multiple configurations with different thresholds
 * {
 *   tx: [
 *     { threshold: 1000000, priceField: "vwap" },
 *     { threshold: 5000000, priceField: "close" },
 *     { threshold: 100000, priceField: "typical" }
 *   ]
 * }
 */
const schema = baseBarSchema.extend({
  tx: z.union([txSchema, z.array(txSchema)])
})

interface DollarBarParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface DollarBarState extends BaseBarState {
  accumulatedValue: number;
}

interface State {
  readonly threshold: number;
  readonly priceField: 'close' | 'vwap' | 'typical';
}

/**
 * Dollar Bar Generator
 *
 * Generates bars based on accumulated dollar value (price × volume). This technique creates
 * bars of uniform economic significance rather than fixed time or tick counts, providing
 * better statistical properties for high-frequency trading and market microstructure analysis.
 *
 * **Algorithm**:
 * 1. Calculate dollar value for each tick: price × volume
 * 2. Accumulate dollar values within current bar
 * 3. When accumulated value ≥ threshold, emit completed bar
 * 4. Start new bar with next tick
 *
 * **Price Field Options**:
 * - `close`: Use closing price (default)
 * - `vwap`: Use VWAP if available, otherwise typical price
 * - `typical`: Use (high + low + close) / 3
 *
 * **Key Properties**:
 * - Equal economic weight per bar
 * - More bars during high activity/volatility
 * - Better captures true market microstructure
 * - Normalizes for both price and volume changes
 *
 * **Use Cases**:
 * - High-frequency trading strategies
 * - Market impact analysis
 * - Liquidity profiling
 * - Volume-weighted technical analysis
 * - Cross-asset comparison with dollar normalization
 *
 * @example
 * ```typescript
 * const dollarBars = new DollarBarGenerator({
 *   tx: {
 *     threshold: 1000000,
 *     priceField: "vwap"
 *   }
 * }, inputSlice)
 * ```
 *
 * @note This transform reshapes data - output buffer has fewer rows than input
 * @note Bars are emitted when threshold is reached, not at fixed intervals
 * @note State is maintained across batch boundaries - bars can span multiple input buffers
 */
export class DollarBarGenerator extends BaseBarGenerator<
  DollarBarParams,
  DollarBarState
> {
  // Configuration state
  private readonly _config: State

  constructor(config: DollarBarParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'dollarBars',
      'DollarBars',
      config.description || 'Dollar Bar Generator',
      parsed,
      inputSlice
    )

    // Initialize configuration - use first config
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    const txConfig = tx[0]
    if (!txConfig) {
      throw new Error('At least one configuration is required')
    }

    this._config = {
      threshold: txConfig.threshold,
      priceField: txConfig.priceField
    }
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
   * Extract tick data including VWAP if available
   */
  protected extractTickData(rid: number): any {
    const baseData = super.extractTickData(rid)

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
  protected createNewBar(tick: any, _rid: number): DollarBarState {
    const price = this.calculatePrice(tick)
    const dollarValue = price * tick.volume

    return {
      open: tick.open,
      high: tick.high,
      low: tick.low,
      close: tick.close,
      volume: tick.volume,
      firstTimestamp: tick.timestamp,
      lastTimestamp: tick.timestamp,
      tickCount: 1,
      accumulatedValue: dollarValue
    }
  }

  /**
   * Update the current bar with new tick data
   */
  protected updateBar(bar: DollarBarState, tick: any, _rid: number): void {
    const price = this.calculatePrice(tick)
    const dollarValue = price * tick.volume

    // Update OHLCV values
    bar.high = Math.max(bar.high, tick.high)
    bar.low = Math.min(bar.low, tick.low)
    bar.close = tick.close
    bar.volume += tick.volume
    bar.lastTimestamp = tick.timestamp
    bar.tickCount++
    bar.accumulatedValue += dollarValue
  }

  /**
   * Check if the current bar is complete
   */
  protected isBarComplete(
    bar: DollarBarState,
    _tick: any,
    _rid: number
  ): boolean {
    return bar.accumulatedValue >= this._config.threshold
  }

  /**
   * Calculate price based on configured field
   */
  private calculatePrice(tick: any): number {
    switch (this._config.priceField) {
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
