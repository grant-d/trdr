import type { Candle } from '@trdr/shared'
import type { Logger } from '@trdr/types'
import { ATRIndicator } from '../indicators/atr'

/**
 * Configuration for volatility-based grid spacing
 */
export interface VolatilitySpacingConfig {
  /** Base grid spacing percentage (used as minimum) */
  readonly baseSpacing: number
  /** Maximum grid spacing percentage (cap for high volatility) */
  readonly maxSpacing: number
  /** Volatility measurement method */
  readonly volatilityMethod: 'atr' | 'standard_deviation'
  /** Period for volatility calculation */
  readonly volatilityPeriod: number
  /** Sensitivity multiplier for volatility adjustments */
  readonly volatilitySensitivity: number
  /** Risk adjustment factor (0-1, higher = more conservative) */
  readonly riskAdjustment: number
  /** Enable adaptive spacing based on market conditions */
  readonly enableAdaptiveSpacing: boolean
}

/**
 * Market volatility metrics
 */
export interface VolatilityMetrics {
  /** Current volatility value */
  readonly currentVolatility: number
  /** Normalized volatility (0-1) */
  readonly normalizedVolatility: number
  /** Average volatility over the period */
  readonly averageVolatility: number
  /** Volatility percentile (0-100) */
  readonly volatilityPercentile: number
  /** Volatility trend (increasing/decreasing) */
  readonly volatilityTrend: 'increasing' | 'decreasing' | 'stable'
}

/**
 * Grid spacing calculation result
 */
export interface SpacingCalculationResult {
  /** Calculated optimal spacing percentage */
  readonly optimalSpacing: number
  /** Volatility metrics used for calculation */
  readonly volatilityMetrics: VolatilityMetrics
  /** Reasoning for the spacing decision */
  readonly reasoning: string
  /** Confidence level in the calculation (0-1) */
  readonly confidence: number
}

/**
 * Swing identification result
 */
interface SwingPoint {
  readonly timestamp: number
  readonly price: number
  readonly type: 'high' | 'low'
  readonly strength: number
}

/**
 * VolatilityGridSpacing implements volatility-based grid spacing algorithms
 * to optimize grid trading performance across different market conditions.
 * 
 * Features:
 * - Multiple volatility measurement methods (ATR, Standard Deviation)
 * - Adaptive spacing based on market conditions
 * - Risk-adjusted spacing calculations
 * - Historical volatility analysis
 * - Swing-based spacing validation
 */
export class VolatilityGridSpacing {
  private readonly config: Required<VolatilitySpacingConfig>
  private readonly atrIndicator: ATRIndicator
  private readonly logger?: Logger
  private readonly volatilityHistory: number[] = []

  constructor(config: Partial<VolatilitySpacingConfig> = {}, logger?: Logger) {
    this.config = {
      baseSpacing: config.baseSpacing ?? 1.0, // 1% base spacing
      maxSpacing: config.maxSpacing ?? 5.0, // 5% max spacing
      volatilityMethod: config.volatilityMethod ?? 'atr',
      volatilityPeriod: config.volatilityPeriod ?? 14,
      volatilitySensitivity: config.volatilitySensitivity ?? 1.5,
      riskAdjustment: config.riskAdjustment ?? 0.8,
      enableAdaptiveSpacing: config.enableAdaptiveSpacing ?? true
    }

    this.logger = logger
    this.atrIndicator = new ATRIndicator({
      period: this.config.volatilityPeriod,
      smoothing: 'wilder'
    })

    this.logger?.debug('VolatilityGridSpacing initialized', { config: this.config })
  }

  /**
   * Calculates optimal grid spacing based on current market volatility
   */
  async calculateOptimalSpacing(
    candles: readonly Candle[],
    currentPrice: number
  ): Promise<SpacingCalculationResult> {
    if (candles.length < this.config.volatilityPeriod + 1) {
      // Not enough data, use base spacing
      return {
        optimalSpacing: this.config.baseSpacing,
        volatilityMetrics: {
          currentVolatility: 0,
          normalizedVolatility: 0,
          averageVolatility: 0,
          volatilityPercentile: 50,
          volatilityTrend: 'stable'
        },
        reasoning: 'Insufficient historical data, using base spacing',
        confidence: 0.3
      }
    }

    // Calculate volatility metrics
    const volatilityMetrics = await this.calculateVolatilityMetrics(candles, currentPrice)

    // Calculate base spacing adjustment
    let spacingMultiplier = 1.0

    if (this.config.enableAdaptiveSpacing) {
      // Increase spacing during high volatility
      spacingMultiplier = 1 + (volatilityMetrics.normalizedVolatility * this.config.volatilitySensitivity)

      // Apply risk adjustment (more conservative = wider spacing)
      spacingMultiplier *= (1 + this.config.riskAdjustment * 0.5)

      // Consider volatility trend
      if (volatilityMetrics.volatilityTrend === 'increasing') {
        spacingMultiplier *= 1.2 // 20% wider during increasing volatility
      } else if (volatilityMetrics.volatilityTrend === 'decreasing') {
        spacingMultiplier *= 0.9 // 10% tighter during decreasing volatility
      }
    }

    // Calculate optimal spacing
    const optimalSpacing = Math.min(
      this.config.baseSpacing * spacingMultiplier,
      this.config.maxSpacing
    )

    // Generate reasoning
    const reasoning = this.generateSpacingReasoning(volatilityMetrics, spacingMultiplier, optimalSpacing)

    // Calculate confidence based on data quality and volatility stability
    const confidence = this.calculateConfidence(candles, volatilityMetrics)

    this.logger?.info('Optimal grid spacing calculated', {
      optimalSpacing,
      volatility: volatilityMetrics.currentVolatility,
      multiplier: spacingMultiplier,
      confidence
    })

    return {
      optimalSpacing,
      volatilityMetrics,
      reasoning,
      confidence
    }
  }

  /**
   * Validates spacing against historical swing patterns
   */
  async validateSpacingWithSwings(
    candles: readonly Candle[],
    proposedSpacing: number
  ): Promise<{ isValid: boolean; adjustedSpacing?: number; reason: string }> {
    const swings = this.identifySwings(candles)
    
    if (swings.length < 3) {
      return {
        isValid: true,
        reason: 'Insufficient swing data for validation'
      }
    }

    const averageSwingSize = this.calculateAverageSwing(swings)
    const currentPrice = candles[candles.length - 1]?.close || 0

    // Proposed spacing should be a reasonable fraction of average swing
    const spacingInPrice = (proposedSpacing / 100) * currentPrice
    const optimalFraction = 0.3 // 30% of average swing

    if (spacingInPrice > averageSwingSize * 0.5) {
      // Spacing too wide, might miss profitable opportunities
      const adjustedSpacing = (averageSwingSize * optimalFraction / currentPrice) * 100

      return {
        isValid: false,
        adjustedSpacing: Math.max(adjustedSpacing, this.config.baseSpacing),
        reason: `Spacing too wide relative to swing patterns (${spacingInPrice.toFixed(2)} vs avg swing ${averageSwingSize.toFixed(2)})`
      }
    }

    if (spacingInPrice < averageSwingSize * 0.1) {
      // Spacing too tight, might trigger too frequently
      const adjustedSpacing = (averageSwingSize * optimalFraction / currentPrice) * 100

      return {
        isValid: false,
        adjustedSpacing: Math.min(adjustedSpacing, this.config.maxSpacing),
        reason: `Spacing too tight relative to swing patterns (${spacingInPrice.toFixed(2)} vs avg swing ${averageSwingSize.toFixed(2)})`
      }
    }

    return {
      isValid: true,
      reason: `Spacing ${proposedSpacing.toFixed(2)}% is appropriate for current swing patterns`
    }
  }

  /**
   * Calculates comprehensive volatility metrics
   */
  private async calculateVolatilityMetrics(
    candles: readonly Candle[],
    currentPrice: number
  ): Promise<VolatilityMetrics> {
    let currentVolatility = 0

    if (this.config.volatilityMethod === 'atr') {
      // Use ATR for volatility measurement
      const atrResult = this.atrIndicator.calculate(candles)
      if (atrResult) {
        currentVolatility = atrResult.value / currentPrice // Normalize by price
      }
    } else {
      // Use standard deviation of returns
      currentVolatility = this.calculateStandardDeviationVolatility(candles)
    }

    // Update volatility history
    this.volatilityHistory.push(currentVolatility)
    if (this.volatilityHistory.length > this.config.volatilityPeriod * 2) {
      this.volatilityHistory.shift()
    }

    // Calculate average volatility
    const averageVolatility = this.volatilityHistory.length > 0
      ? this.volatilityHistory.reduce((sum, vol) => sum + vol, 0) / this.volatilityHistory.length
      : currentVolatility

    // Normalize volatility (0-1 scale)
    const maxHistoricalVol = Math.max(...this.volatilityHistory, currentVolatility)
    const minHistoricalVol = Math.min(...this.volatilityHistory, currentVolatility)
    const normalizedVolatility = maxHistoricalVol > minHistoricalVol
      ? (currentVolatility - minHistoricalVol) / (maxHistoricalVol - minHistoricalVol)
      : 0.5

    // Calculate volatility percentile
    const sortedVolatility = [...this.volatilityHistory].sort((a, b) => a - b)
    const percentileIndex = Math.floor(sortedVolatility.length * 0.5)
    const medianVolatility = sortedVolatility[percentileIndex] || currentVolatility
    const volatilityPercentile = currentVolatility > medianVolatility ? 75 : 25

    // Determine volatility trend
    let volatilityTrend: 'increasing' | 'decreasing' | 'stable' = 'stable'
    if (this.volatilityHistory.length >= 3) {
      const recentVol = this.volatilityHistory.slice(-3)
      const trend = recentVol[2]! - recentVol[0]!
      const trendThreshold = averageVolatility * 0.1

      if (trend > trendThreshold) {
        volatilityTrend = 'increasing'
      } else if (trend < -trendThreshold) {
        volatilityTrend = 'decreasing'
      }
    }

    return {
      currentVolatility,
      normalizedVolatility,
      averageVolatility,
      volatilityPercentile,
      volatilityTrend
    }
  }

  /**
   * Calculates standard deviation-based volatility
   */
  private calculateStandardDeviationVolatility(candles: readonly Candle[]): number {
    if (candles.length < 2) return 0

    // Calculate returns
    const returns: number[] = []
    for (let i = 1; i < candles.length; i++) {
      const currentPrice = candles[i]!.close
      const previousPrice = candles[i - 1]!.close
      const ret = (currentPrice - previousPrice) / previousPrice
      returns.push(ret)
    }

    // Calculate standard deviation
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length
    const standardDeviation = Math.sqrt(variance)

    // Annualize (assume daily data)
    return standardDeviation * Math.sqrt(365)
  }

  /**
   * Identifies swing points in price data
   */
  private identifySwings(candles: readonly Candle[]): SwingPoint[] {
    const swings: SwingPoint[] = []
    const lookback = 5 // Look 5 periods back and forward

    for (let i = lookback; i < candles.length - lookback; i++) {
      const current = candles[i]!
      const prevCandles = candles.slice(i - lookback, i)
      const nextCandles = candles.slice(i + 1, i + lookback + 1)

      // Check for swing high
      const isSwingHigh = prevCandles.every(c => c.high <= current.high) &&
                         nextCandles.every(c => c.high <= current.high)

      // Check for swing low
      const isSwingLow = prevCandles.every(c => c.low >= current.low) &&
                        nextCandles.every(c => c.low >= current.low)

      if (isSwingHigh) {
        swings.push({
          timestamp: current.timestamp,
          price: current.high,
          type: 'high',
          strength: this.calculateSwingStrength(candles, i, 'high')
        })
      }

      if (isSwingLow) {
        swings.push({
          timestamp: current.timestamp,
          price: current.low,
          type: 'low',
          strength: this.calculateSwingStrength(candles, i, 'low')
        })
      }
    }

    return swings
  }

  /**
   * Calculates the strength of a swing point
   */
  private calculateSwingStrength(
    candles: readonly Candle[],
    index: number,
    type: 'high' | 'low'
  ): number {
    const lookback = 10
    const start = Math.max(0, index - lookback)
    const end = Math.min(candles.length - 1, index + lookback)
    
    const current = candles[index]!
    const currentPrice = type === 'high' ? current.high : current.low
    
    let strength = 0
    for (let i = start; i <= end; i++) {
      if (i === index) continue
      
      const candle = candles[i]!
      const price = type === 'high' ? candle.high : candle.low
      
      if (type === 'high' && price < currentPrice) {
        strength += (currentPrice - price) / currentPrice
      } else if (type === 'low' && price > currentPrice) {
        strength += (price - currentPrice) / currentPrice
      }
    }
    
    return strength / (end - start)
  }

  /**
   * Calculates average swing size
   */
  private calculateAverageSwing(swings: SwingPoint[]): number {
    if (swings.length < 2) return 0

    let totalSwingSize = 0
    let swingCount = 0

    for (let i = 1; i < swings.length; i++) {
      const current = swings[i]!
      const previous = swings[i - 1]!
      
      // Only count swings between different types (high to low or low to high)
      if (current.type !== previous.type) {
        const swingSize = Math.abs(current.price - previous.price)
        totalSwingSize += swingSize
        swingCount++
      }
    }

    return swingCount > 0 ? totalSwingSize / swingCount : 0
  }

  /**
   * Generates human-readable reasoning for spacing decision
   */
  private generateSpacingReasoning(
    volatilityMetrics: VolatilityMetrics,
    spacingMultiplier: number,
    optimalSpacing: number
  ): string {
    const { currentVolatility, volatilityTrend, volatilityPercentile } = volatilityMetrics

    let reasoning = `Base spacing of ${this.config.baseSpacing}% adjusted to ${optimalSpacing.toFixed(2)}% `

    if (spacingMultiplier > 1.2) {
      reasoning += `(widened by ${((spacingMultiplier - 1) * 100).toFixed(1)}%) due to `
    } else if (spacingMultiplier < 0.9) {
      reasoning += `(tightened by ${((1 - spacingMultiplier) * 100).toFixed(1)}%) due to `
    } else {
      reasoning += `(minimal adjustment) due to `
    }

    if (volatilityPercentile > 75) {
      reasoning += `high volatility (${(currentVolatility * 100).toFixed(2)}%)`
    } else if (volatilityPercentile < 25) {
      reasoning += `low volatility (${(currentVolatility * 100).toFixed(2)}%)`
    } else {
      reasoning += `moderate volatility (${(currentVolatility * 100).toFixed(2)}%)`
    }

    if (volatilityTrend !== 'stable') {
      reasoning += ` with ${volatilityTrend} trend`
    }

    reasoning += `. Risk adjustment factor: ${this.config.riskAdjustment}`

    return reasoning
  }

  /**
   * Calculates confidence level for the spacing calculation
   */
  private calculateConfidence(
    candles: readonly Candle[],
    volatilityMetrics: VolatilityMetrics
  ): number {
    let confidence = 0.5 // Base confidence

    // More data = higher confidence
    const dataQuality = Math.min(candles.length / (this.config.volatilityPeriod * 2), 1)
    confidence += dataQuality * 0.3

    // Stable volatility = higher confidence
    if (volatilityMetrics.volatilityTrend === 'stable') {
      confidence += 0.2
    }

    // Moderate volatility = higher confidence (extreme values are less predictable)
    if (volatilityMetrics.volatilityPercentile >= 25 && volatilityMetrics.volatilityPercentile <= 75) {
      confidence += 0.2
    }

    // Sufficient volatility history = higher confidence
    if (this.volatilityHistory.length >= this.config.volatilityPeriod) {
      confidence += 0.1
    }

    return Math.min(confidence, 1.0)
  }

  /**
   * Resets volatility history (useful for backtesting)
   */
  resetHistory(): void {
    this.volatilityHistory.length = 0
    this.logger?.debug('Volatility history reset')
  }

  /**
   * Gets current volatility history for analysis
   */
  getVolatilityHistory(): readonly number[] {
    return [...this.volatilityHistory]
  }
}