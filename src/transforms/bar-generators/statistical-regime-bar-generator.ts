import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'
import type { Transform } from '../../interfaces'

export interface StatisticalRegimeBarParams extends BarGeneratorParams {
  /** 
   * Lookback period for statistical calculations 
   * @example 20 // Use 20 periods for statistical analysis
   * @minimum 10
   */
  lookback: number
  
  /** 
   * Z-score threshold for combined statistical metrics
   * @example 2.5 // Trigger new bar when combined Z-score exceeds 2.5
   * @minimum 1.0
   */
  threshold: number
}

interface StatisticalRegimeState extends BarState {
  /** Historical close prices for statistical calculations */
  priceHistory: number[]
  /** Historical returns for calculations */
  returnHistory: number[]
  /** Kurtosis values for Z-score calculation */
  kurtosisHistory: number[]
  /** Skewness values for Z-score calculation */
  skewnessHistory: number[]
  /** Hurst exponent values for Z-score calculation */
  hurstHistory: number[]
  /** Entropy values for Z-score calculation */
  entropyHistory: number[]
}

/**
 * Statistical Regime Bar Generator
 * 
 * Monitors rolling window of multiple statistical moments to detect regime changes:
 * - Kurtosis (tail risk) - measures extreme events and fat tails
 * - Skewness (directional bias) - captures asymmetry in returns  
 * - Hurst exponent (trending vs mean-reverting) - persistence of trends
 * - Entropy (randomness/uncertainty) - disorder in price movements
 * 
 * Creates new bar when combined Z-score of these metrics exceeds threshold.
 * Adapts to changing market personality rather than single metric.
 * Bars cluster during regime transitions, sparse during stable periods.
 * 
 * ## Algorithm
 * 1. Maintain rolling window of close prices and returns
 * 2. Calculate statistical moments: kurtosis, skewness, hurst, entropy
 * 3. Calculate Z-scores for each metric using rolling mean/stdev  
 * 4. Combine Z-scores using Euclidean distance in Z-space
 * 5. Create new bar when combined Z-score exceeds threshold
 * 
 * ## Use Cases
 * - Regime change detection in algorithmic trading
 * - Risk management during market transitions
 * - Adaptive strategy parameter adjustment
 * - Market microstructure analysis
 * 
 * ## Advantages
 * - Multi-dimensional statistical view of market state
 * - Adaptive to different market personalities  
 * - Early detection of regime shifts
 * - Robust to single-metric false signals
 * 
 * @example
 * ```typescript
 * const generator = new StatisticalRegimeBarGenerator({
 *   lookback: 20,        // 20-period statistical window
 *   threshold: 2.5       // Trigger at 2.5 combined Z-score
 * })
 * ```
 */
export class StatisticalRegimeBarGenerator extends BarGeneratorTransform<StatisticalRegimeBarParams> {
  constructor(params: StatisticalRegimeBarParams) {
    super(params, 'statisticalRegime', 'Statistical Regime Bar Generator')
    this.validate()
  }

  public validate(): void {
    super.validate()
    
    if (this.params.lookback < 10) {
      throw new Error('Statistical Regime Bar Generator: lookback must be at least 10')
    }
    
    if (this.params.threshold < 1.0) {
      throw new Error('Statistical Regime Bar Generator: threshold must be at least 1.0')
    }
  }

  isBarComplete(_symbol: string, _tick: OhlcvDto, state: StatisticalRegimeState): boolean {
    // Need enough data to calculate statistics
    if (state.priceHistory.length < this.params.lookback * 2) {
      return false
    }

    // Calculate combined Z-score
    const combinedZ = this.calculateCombinedZScore(state)
    return combinedZ > this.params.threshold
  }

  createNewBar(symbol: string, tick: OhlcvDto): StatisticalRegimeState {
    return {
      currentBar: {
        timestamp: tick.timestamp,
        symbol,
        exchange: tick.exchange,
        open: tick.close,
        high: tick.high,
        low: tick.low,
        close: tick.close,
        volume: tick.volume
      },
      complete: false,
      priceHistory: [tick.close],
      returnHistory: [],
      kurtosisHistory: [],
      skewnessHistory: [],
      hurstHistory: [],
      entropyHistory: []
    }
  }

  updateBar(_symbol: string, currentBar: OhlcvDto, tick: OhlcvDto, state: StatisticalRegimeState): OhlcvDto {
    // Update price history
    state.priceHistory.push(tick.close)
    
    // Calculate return if we have previous price
    if (state.priceHistory.length > 1) {
      const prevPrice = state.priceHistory[state.priceHistory.length - 2]!
      const currentReturn = (tick.close - prevPrice) / prevPrice
      state.returnHistory.push(currentReturn)
    }
    
    // Keep only lookback * 3 history for efficiency
    const maxHistory = this.params.lookback * 3
    if (state.priceHistory.length > maxHistory) {
      state.priceHistory = state.priceHistory.slice(-maxHistory)
    }
    if (state.returnHistory.length > maxHistory) {
      state.returnHistory = state.returnHistory.slice(-maxHistory)
    }
    
    // Update statistical metrics if we have enough data
    if (state.priceHistory.length >= this.params.lookback) {
      this.updateStatisticalMetrics(state)
    }
    
    // Update bar OHLCV
    return {
      timestamp: currentBar.timestamp,
      symbol: currentBar.symbol,
      exchange: currentBar.exchange,
      open: currentBar.open,
      high: Math.max(currentBar.high, tick.high),
      low: Math.min(currentBar.low, tick.low),
      close: tick.close,
      volume: currentBar.volume + tick.volume
    }
  }

  resetState(state: StatisticalRegimeState): void {
    // Keep some recent history for continuity
    const keepHistory = Math.floor(this.params.lookback / 2)
    state.priceHistory = state.priceHistory.slice(-keepHistory)
    state.returnHistory = state.returnHistory.slice(-keepHistory)
    state.kurtosisHistory = state.kurtosisHistory.slice(-keepHistory)
    state.skewnessHistory = state.skewnessHistory.slice(-keepHistory)
    state.hurstHistory = state.hurstHistory.slice(-keepHistory)
    state.entropyHistory = state.entropyHistory.slice(-keepHistory)
    state.complete = false
  }

  private updateStatisticalMetrics(state: StatisticalRegimeState): void {
    const prices = state.priceHistory.slice(-this.params.lookback)
    const returns = state.returnHistory.slice(-this.params.lookback)
    
    if (returns.length < this.params.lookback - 1) return
    
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
    const maxStatHistory = this.params.lookback * 2
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

  private calculateCombinedZScore(state: StatisticalRegimeState): number {
    if (state.kurtosisHistory.length < this.params.lookback) {
      return 0
    }
    
    // Get current values
    const currentKurtosis = state.kurtosisHistory[state.kurtosisHistory.length - 1]!
    const currentSkewness = state.skewnessHistory[state.skewnessHistory.length - 1]!
    const currentHurst = state.hurstHistory[state.hurstHistory.length - 1]!
    const currentEntropy = state.entropyHistory[state.entropyHistory.length - 1]!
    
    // Calculate Z-scores
    const zKurtosis = this.calculateZScore(currentKurtosis, state.kurtosisHistory)
    const zSkewness = this.calculateZScore(currentSkewness, state.skewnessHistory)
    const zHurst = this.calculateZScore(currentHurst, state.hurstHistory)
    const zEntropy = this.calculateZScore(currentEntropy, state.entropyHistory)
    
    // Combined Z-score (Euclidean distance in Z-space, normalized)
    return Math.sqrt(zKurtosis * zKurtosis + zSkewness * zSkewness + zHurst * zHurst + zEntropy * zEntropy) / 2
  }

  private calculateZScore(value: number, history: number[]): number {
    if (history.length < 2) return 0
    
    const mean = history.reduce((sum, val) => sum + val, 0) / history.length
    const variance = history.reduce((sum, val) => sum + (val - mean) * (val - mean), 0) / (history.length - 1)
    const stdev = Math.sqrt(variance)
    
    return stdev > 0 ? (value - mean) / stdev : 0
  }

  private calculateKurtosis(returns: number[]): number {
    if (returns.length < 4) return 0
    
    const mean = returns.reduce((sum, val) => sum + val, 0) / returns.length
    const variance = returns.reduce((sum, val) => sum + (val - mean) * (val - mean), 0) / returns.length
    const stdev = Math.sqrt(variance)
    
    if (stdev === 0) return 0
    
    const sum4 = returns.reduce((sum, val) => sum + Math.pow((val - mean) / stdev, 4), 0)
    return sum4 / returns.length - 3.0  // Excess kurtosis
  }

  private calculateSkewness(returns: number[]): number {
    if (returns.length < 3) return 0
    
    const mean = returns.reduce((sum, val) => sum + val, 0) / returns.length
    const variance = returns.reduce((sum, val) => sum + (val - mean) * (val - mean), 0) / returns.length
    const stdev = Math.sqrt(variance)
    
    if (stdev === 0) return 0
    
    const sum3 = returns.reduce((sum, val) => sum + Math.pow((val - mean) / stdev, 3), 0)
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
    
    const meanReturn = returns.reduce((sum, val) => sum + val, 0) / returns.length
    
    // Calculate cumulative deviations
    const cumDevs: number[] = []
    let cumSum = 0
    for (const ret of returns) {
      cumSum += ret - meanReturn
      cumDevs.push(cumSum)
    }
    
    // Calculate R (range) and S (standard deviation)
    const R = Math.max(...cumDevs) - Math.min(...cumDevs)
    const variance = returns.reduce((sum, val) => sum + (val - meanReturn) * (val - meanReturn), 0) / (returns.length - 1)
    const S = Math.sqrt(variance)
    
    if (S === 0 || R === 0) return 0.5
    
    const RS = R / S
    return RS > 0 ? Math.log(RS) / Math.log(returns.length * 0.5) : 0.5
  }

  private calculateEntropy(returns: number[], bins: number = 10): number {
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

  public withParams(params: Partial<StatisticalRegimeBarParams>): Transform<StatisticalRegimeBarParams> {
    return new StatisticalRegimeBarGenerator({ ...this.params, ...params })
  }
}