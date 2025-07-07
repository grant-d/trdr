import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import type { BaseBarState } from './base-bar-generator'
import { BaseBarGenerator, baseBarSchema } from './base-bar-generator'

/**
 * Schema for statistical regime bar configuration
 * @property {number} lookback - Lookback period for statistical calculations (min: 10)
 * @property {number} threshold - Z-score threshold for combined metrics (min: 1.0)
 */
const txSchema = z.object({
  lookback: z.number().min(10),
  threshold: z.number().min(1.0)
})

/**
 * Main schema for StatisticalRegimeBarGenerator transform
 */
const schema = baseBarSchema.extend({
  tx: z.union([txSchema, z.array(txSchema)])
})

interface StatisticalRegimeBarParams
  extends z.infer<typeof schema>,
          BaseTransformParams {
}

interface StatisticalRegimeBarState extends BaseBarState {
  /** Historical close prices for statistical calculations */
  priceHistory: number[];
  /** Historical returns for calculations */
  returnHistory: number[];
  /** Kurtosis values for Z-score calculation */
  kurtosisHistory: number[];
  /** Skewness values for Z-score calculation */
  skewnessHistory: number[];
  /** Hurst exponent values for Z-score calculation */
  hurstHistory: number[];
  /** Entropy values for Z-score calculation */
  entropyHistory: number[];
}

/**
 * Statistical Regime Bar Generator
 *
 * Monitors rolling window of multiple statistical moments to detect regime changes.
 * Creates new bar when combined Z-score of these metrics exceeds threshold.
 *
 * **Statistical Metrics**:
 * - Kurtosis: Measures extreme events and fat tails
 * - Skewness: Captures asymmetry in returns
 * - Hurst exponent: Trending vs mean-reverting behavior
 * - Entropy: Disorder/randomness in price movements
 *
 * **Algorithm**:
 * 1. Maintain rolling window of close prices and returns
 * 2. Calculate statistical moments for recent window
 * 3. Calculate Z-scores for each metric using rolling stats
 * 4. Combine Z-scores using Euclidean distance
 * 5. Create new bar when combined Z-score exceeds threshold
 *
 * **Key Properties**:
 * - Multi-dimensional statistical view
 * - Adapts to changing market personality
 * - Bars cluster during regime transitions
 * - Sparse bars during stable periods
 *
 * **Use Cases**:
 * - Regime change detection
 * - Risk management during transitions
 * - Adaptive strategy adjustment
 * - Market microstructure analysis
 * - Volatility regime identification
 *
 * @example
 * ```typescript
 * const regimeBars = new StatisticalRegimeBarGenerator({
 *   tx: {
 *     lookback: 20,      // 20-period statistical window
 *     threshold: 2.5     // Trigger at 2.5 combined Z-score
 *   }
 * }, inputBuffer)
 * ```
 *
 * @note Requires 2x lookback periods before generating bars
 * @note Statistical moments calculated on rolling basis
 * @note State maintained across batch boundaries
 */
export class StatisticalRegimeBarGenerator extends BaseBarGenerator<
  StatisticalRegimeBarParams,
  StatisticalRegimeBarState
> {
  // Configuration
  private readonly _lookback: number
  private readonly _threshold: number

  constructor(config: StatisticalRegimeBarParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'regimeBars',
      'RegimeBars',
      config.description || 'Statistical Regime Bar Generator',
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
  }

  /**
   * Create a new bar from the first tick
   */
  protected createNewBar(tick: any, _rid: number): StatisticalRegimeBarState {
    // Preserve some history from previous bar if available
    const prevHistory = this._currentBar?.priceHistory || []
    const keepHistory = Math.floor(this._lookback / 2)
    const carryOverPrices =
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
      priceHistory: [...carryOverPrices, tick.close],
      returnHistory: [],
      kurtosisHistory: [],
      skewnessHistory: [],
      hurstHistory: [],
      entropyHistory: []
    }
  }

  /**
   * Update the current bar with new tick data
   */
  protected updateBar(
    bar: StatisticalRegimeBarState,
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

    // Update price history
    bar.priceHistory.push(tick.close)

    // Calculate return if we have previous price
    if (bar.priceHistory.length > 1) {
      const prevPrice = bar.priceHistory[bar.priceHistory.length - 2]!
      const currentReturn = (tick.close - prevPrice) / prevPrice
      bar.returnHistory.push(currentReturn)
    }

    // Keep only lookback * 3 history for efficiency
    const maxHistory = this._lookback * 3
    if (bar.priceHistory.length > maxHistory) {
      bar.priceHistory = bar.priceHistory.slice(-maxHistory)
    }
    if (bar.returnHistory.length > maxHistory) {
      bar.returnHistory = bar.returnHistory.slice(-maxHistory)
    }

    // Update statistical metrics if we have enough data
    if (bar.priceHistory.length >= this._lookback) {
      this.updateStatisticalMetrics(bar)
    }
  }

  /**
   * Check if the current bar is complete
   */
  protected isBarComplete(
    bar: StatisticalRegimeBarState,
    _tick: any,
    _rid: number
  ): boolean {
    // Need enough data to calculate statistics
    if (bar.priceHistory.length < this._lookback * 2) {
      return false
    }

    // Calculate combined Z-score
    const combinedZ = this.calculateCombinedZScore(bar)
    return combinedZ > this._threshold
  }

  /**
   * Override to add statistical metrics to emitted bars
   */
  protected addAdditionalBarFields(
    row: Record<string, number>,
    bar: StatisticalRegimeBarState
  ): void {
    // Add latest statistical metrics if available
    if (bar.kurtosisHistory.length > 0) {
      row.kurtosis = bar.kurtosisHistory[bar.kurtosisHistory.length - 1]!
      row.skewness = bar.skewnessHistory[bar.skewnessHistory.length - 1]!
      row.hurst = bar.hurstHistory[bar.hurstHistory.length - 1]!
      row.entropy = bar.entropyHistory[bar.entropyHistory.length - 1]!
      row.combined_zscore = this.calculateCombinedZScore(bar)
    }
  }

  private updateStatisticalMetrics(state: StatisticalRegimeBarState): void {
    const prices = state.priceHistory.slice(-this._lookback)
    const returns = state.returnHistory.slice(-this._lookback)

    if (returns.length < this._lookback - 1) return

    // Calculate statistical moments
    const kurtosis = this.calculateKurtosis(returns)
    const skewness = this.calculateSkewness(returns)
    const hurst = this.calculateHurstExponent(prices)
    const entropy = this.calculateEntropy(returns)

    // Store in history
    state.kurtosisHistory.push(kurtosis)
    state.skewnessHistory.push(skewness)
    state.hurstHistory.push(hurst)
    state.entropyHistory.push(entropy)

    // Keep limited history
    const maxStatHistory = this._lookback * 2
    if (state.kurtosisHistory.length > maxStatHistory) {
      state.kurtosisHistory = state.kurtosisHistory.slice(-maxStatHistory)
    }
    if (state.skewnessHistory.length > maxStatHistory) {
      state.skewnessHistory = state.skewnessHistory.slice(-maxStatHistory)
    }
    if (state.hurstHistory.length > maxStatHistory) {
      state.hurstHistory = state.hurstHistory.slice(-maxStatHistory)
    }
    if (state.entropyHistory.length > maxStatHistory) {
      state.entropyHistory = state.entropyHistory.slice(-maxStatHistory)
    }
  }

  private calculateCombinedZScore(state: StatisticalRegimeBarState): number {
    if (state.kurtosisHistory.length < this._lookback) {
      return 0
    }

    // Get current values
    const currentKurtosis =
      state.kurtosisHistory[state.kurtosisHistory.length - 1]!
    const currentSkewness =
      state.skewnessHistory[state.skewnessHistory.length - 1]!
    const currentHurst = state.hurstHistory[state.hurstHistory.length - 1]!
    const currentEntropy =
      state.entropyHistory[state.entropyHistory.length - 1]!

    // Calculate Z-scores
    const zKurtosis = this.calculateZScore(
      currentKurtosis,
      state.kurtosisHistory
    )
    const zSkewness = this.calculateZScore(
      currentSkewness,
      state.skewnessHistory
    )
    const zHurst = this.calculateZScore(currentHurst, state.hurstHistory)
    const zEntropy = this.calculateZScore(currentEntropy, state.entropyHistory)

    // Combined Z-score (Euclidean distance in Z-space, normalized)
    return (
      Math.sqrt(
        zKurtosis * zKurtosis +
        zSkewness * zSkewness +
        zHurst * zHurst +
        zEntropy * zEntropy
      ) / 2
    )
  }

  private calculateZScore(value: number, history: number[]): number {
    if (history.length < 2) return 0

    const mean = history.reduce((sum, val) => sum + val, 0) / history.length
    const variance =
      history.reduce((sum, val) => sum + (val - mean) * (val - mean), 0) /
      (history.length - 1)
    const stdev = Math.sqrt(variance)

    return stdev > 0 ? (value - mean) / stdev : 0
  }

  private calculateKurtosis(returns: number[]): number {
    if (returns.length < 4) return 0

    const mean = returns.reduce((sum, val) => sum + val, 0) / returns.length
    const variance =
      returns.reduce((sum, val) => sum + (val - mean) * (val - mean), 0) /
      returns.length
    const stdev = Math.sqrt(variance)

    if (stdev === 0) return 0

    const sum4 = returns.reduce(
      (sum, val) => sum + Math.pow((val - mean) / stdev, 4),
      0
    )
    return sum4 / returns.length - 3.0 // Excess kurtosis
  }

  private calculateSkewness(returns: number[]): number {
    if (returns.length < 3) return 0

    const mean = returns.reduce((sum, val) => sum + val, 0) / returns.length
    const variance =
      returns.reduce((sum, val) => sum + (val - mean) * (val - mean), 0) /
      returns.length
    const stdev = Math.sqrt(variance)

    if (stdev === 0) return 0

    const sum3 = returns.reduce(
      (sum, val) => sum + Math.pow((val - mean) / stdev, 3),
      0
    )
    return sum3 / returns.length
  }

  private calculateHurstExponent(prices: number[]): number {
    if (prices.length < 10) return 0.5

    // Calculate returns
    const returns: number[] = []
    for (let i = 1; i < prices.length; i++) {
      if (prices[i - 1]! > 0) {
        returns.push(Math.log(prices[i]! / prices[i - 1]!))
      }
    }

    if (returns.length < 5) return 0.5

    const meanReturn =
      returns.reduce((sum, val) => sum + val, 0) / returns.length

    // Calculate cumulative deviations
    const cumDevs: number[] = []
    let cumSum = 0
    for (const ret of returns) {
      cumSum += ret - meanReturn
      cumDevs.push(cumSum)
    }

    // Calculate R (range) and S (standard deviation)
    const R = Math.max(...cumDevs) - Math.min(...cumDevs)
    const variance =
      returns.reduce(
        (sum, val) => sum + (val - meanReturn) * (val - meanReturn),
        0
      ) /
      (returns.length - 1)
    const S = Math.sqrt(variance)

    if (S === 0 || R === 0) return 0.5

    const RS = R / S
    return RS > 0 ? Math.log(RS) / Math.log(returns.length * 0.5) : 0.5
  }

  private calculateEntropy(returns: number[], bins = 10): number {
    if (returns.length < bins) return 0

    const minReturn = Math.min(...returns)
    const maxReturn = Math.max(...returns)
    const range = maxReturn - minReturn

    if (range === 0) return 0

    // Create histogram
    const counts = new Array(bins).fill(0)
    for (const ret of returns) {
      const binIndex = Math.floor(((ret - minReturn) / range) * (bins - 1))
      const clampedIndex = Math.max(0, Math.min(bins - 1, binIndex))
      counts[clampedIndex]++
    }

    // Calculate entropy
    let entropy = 0
    for (const count of counts) {
      if (count > 0) {
        const probability = count / returns.length
        entropy -= probability * Math.log2(probability)
      }
    }

    return entropy
  }
}
