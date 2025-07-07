import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import type { BaseBarState } from './base-bar-generator'
import { BaseBarGenerator, baseBarSchema } from './base-bar-generator'

/**
 * Schema for Shannon information bar configuration
 * @property {number} lookback - Lookback period for building return distribution (min: 10)
 * @property {number} threshold - Information content threshold in bits (min: 1.0)
 * @property {number} decayRate - Exponential decay rate for cumulative information (0.8-0.99)
 */
const txSchema = z.object({
  lookback: z.number().min(10),
  threshold: z.number().min(1.0),
  decayRate: z.number().min(0.8).max(0.99)
})

/**
 * Main schema for ShannonInformationBarGenerator transform
 */
const schema = baseBarSchema.extend({
  tx: z.union([txSchema, z.array(txSchema)])
})

interface ShannonInformationBarParams
  extends z.infer<typeof schema>,
          BaseTransformParams {
}

interface ShannonInformationBarState extends BaseBarState {
  /** Historical returns for probability estimation */
  returnHistory: number[];
  /** Cumulative information content */
  cumulativeInformation: number;
  /** Previous price for return calculation */
  previousPrice: number;
}

/**
 * Shannon Information Bar Generator
 *
 * Uses Shannon information theory to measure the "surprise" content
 * of price movements. Higher information = more unexpected moves.
 *
 * **Algorithm**:
 * 1. Build probability distribution of price changes over rolling window
 * 2. Calculate entropy and surprise of each new price update
 * 3. Information = -log2(probability of observed move)
 * 4. Apply exponential decay to cumulative information
 * 5. Add new information content from current tick
 * 6. Create new bar when cumulative information exceeds threshold
 *
 * **Key Insight**: Rare price moves carry more information than common ones
 * - 5% move when volatility is low = high information
 * - 5% move when volatility is high = low information
 *
 * **Key Properties**:
 * - More responsive during news/events
 * - Quieter during normal drift periods
 * - Adaptive to volatility regimes
 * - Self-adjusting sensitivity
 *
 * **Use Cases**:
 * - News-driven trading strategies
 * - Event detection and reaction
 * - Adaptive position sizing based on information flow
 * - Market regime identification
 * - Volatility regime transitions
 *
 * @example
 * ```typescript
 * const infoBarGenerator = new ShannonInformationBarGenerator({
 *   tx: {
 *     lookback: 20,      // 20-period return distribution
 *     threshold: 5.0,    // 5 bits information threshold
 *     decayRate: 0.90    // 90% information retention
 *   }
 * }, inputBuffer)
 * ```
 *
 * @note Information content measured in bits
 * @note Requires minimum 10 periods of history
 * @note State maintained across batch boundaries
 */
export class ShannonInformationBarGenerator extends BaseBarGenerator<
  ShannonInformationBarParams,
  ShannonInformationBarState
> {
  // Configuration
  private readonly _lookback: number
  private readonly _threshold: number
  private readonly _decayRate: number
  // Track previous price across all bars
  private _lastPrice: number | undefined

  constructor(config: ShannonInformationBarParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'shannonInfoBars',
      'ShannonInfoBars',
      config.description || 'Shannon Information Bar Generator',
      parsed,
      inputSlice
    )

    // Use first config
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    const txConfig = tx[0]
    if (!txConfig) {
      throw new Error('At least one configuration is required')
    }

    this._lookback = txConfig.lookback
    this._threshold = txConfig.threshold
    this._decayRate = txConfig.decayRate
  }

  /**
   * Update the current bar with new tick data
   */
  protected updateBar(
    bar: ShannonInformationBarState,
    tick: any,
    _rid: number
  ): void {
    // Update OHLCV values
    bar.high = Math.max(bar.high, tick.high)
    bar.low = Math.min(bar.low, tick.low)
    bar.close = tick.close
    bar.volume += tick.volume
    bar.lastTimestamp = tick.timestamp
    bar.tickCount++

    // Apply exponential decay to existing information
    bar.cumulativeInformation *= this._decayRate

    // Calculate current return if we have a previous price
    if (bar.previousPrice > 0) {
      const currentReturn =
        (tick.close - bar.previousPrice) / bar.previousPrice

      // Update return history
      bar.returnHistory.push(currentReturn)

      // Keep only lookback periods
      if (bar.returnHistory.length > this._lookback) {
        bar.returnHistory = bar.returnHistory.slice(-this._lookback)
      }

      // Calculate information content if we have enough history
      if (bar.returnHistory.length >= 10) {
        const informationContent = this.calculateInformationContent(
          currentReturn,
          bar.returnHistory
        )
        bar.cumulativeInformation += informationContent
      }
    }

    // Update previous price for next tick
    bar.previousPrice = tick.close
  }

  /**
   * Check if the current bar is complete
   */
  protected isBarComplete(
    bar: ShannonInformationBarState,
    _tick: any,
    _rid: number
  ): boolean {
    // Check if cumulative information exceeds threshold
    return bar.cumulativeInformation >= this._threshold
  }

  /**
   * Override to track last price across bars and preserve some return history
   */
  protected emitBar(sourceRid: number, bar: ShannonInformationBarState): void {
    // Store last price for next bar
    this._lastPrice = bar.close

    // Call parent implementation
    super.emitBar(sourceRid, bar)
  }

  /**
   * Override to preserve return history for next bar
   */
  protected createNewBar(tick: any, _rid: number): ShannonInformationBarState {
    // Get return history from previous bar if available
    const prevHistory = this._currentBar?.returnHistory || []

    // Keep partial history for continuity
    const keepHistory = Math.floor(this._lookback / 2)
    const carryOverHistory =
      prevHistory.length > keepHistory
        ? prevHistory.slice(-keepHistory)
        : prevHistory

    return {
      open: tick.open,
      high: tick.high,
      low: tick.low,
      close: tick.close,
      volume: tick.volume,
      firstTimestamp: tick.timestamp,
      lastTimestamp: tick.timestamp,
      tickCount: 1,
      returnHistory: carryOverHistory,
      cumulativeInformation: 0.0,
      previousPrice: this._lastPrice || tick.close
    }
  }

  /**
   * Override to add information metrics to emitted bars
   */
  protected addAdditionalBarFields(
    row: Record<string, number>,
    bar: ShannonInformationBarState
  ): void {
    // Add information metrics
    row.cumulative_information = bar.cumulativeInformation
    row.return_history_size = bar.returnHistory.length
  }

  private calculateInformationContent(
    currentReturn: number,
    returnHistory: number[]
  ): number {
    if (returnHistory.length < 2) return 0

    // Calculate standard deviation of return history
    const mean =
      returnHistory.reduce((sum, ret) => sum + ret, 0) / returnHistory.length
    const variance =
      returnHistory.reduce((sum, ret) => sum + (ret - mean) * (ret - mean), 0) /
      (returnHistory.length - 1)
    const stdev = Math.sqrt(variance)

    if (stdev <= 0) return 0

    // Calculate Z-score of current return
    const zScore = Math.abs(currentReturn / stdev)

    // Information content based on rarity (squared Z-score scaled)
    // Higher Z-scores (rarer events) contribute more information
    const informationContent = Math.pow(zScore, 2) * 0.5

    return informationContent
  }
}
