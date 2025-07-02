import type { Candle } from '@trdr/shared'
import { createCacheKey, createCandleHash, IndicatorCache } from './cache'
import type { BollingerBandsConfig, BollingerBandsResult, IMultiValueIndicator } from './interfaces'
import { SMAIndicator } from './sma'

/**
 * Bollinger Bands indicator
 *
 * Consists of:
 * - Middle Band: Simple Moving Average (SMA)
 * - Upper Band: SMA + (Standard Deviation × Multiplier)
 * - Lower Band: SMA - (Standard Deviation × Multiplier)
 *
 * Also calculates:
 * - Bandwidth: (Upper Band - Lower Band) / Middle Band
 * - %B: (Close - Lower Band) / (Upper Band - Lower Band)
 */
export class BollingerBandsIndicator
  implements IMultiValueIndicator<BollingerBandsResult, BollingerBandsConfig>
{
  readonly name = 'BollingerBands'
  readonly config: BollingerBandsConfig
  private readonly cache?: IndicatorCache
  private readonly smaIndicator: SMAIndicator

  constructor(config: Partial<BollingerBandsConfig> = {}) {
    this.config = {
      period: config.period ?? 20,
      stdDevMultiplier: config.stdDevMultiplier ?? 2,
      cacheEnabled: config.cacheEnabled ?? true,
      cacheSize: config.cacheSize ?? 100,
    }

    if (this.config.cacheEnabled) {
      this.cache = new IndicatorCache(this.config.cacheSize)
    }

    if (this.config.period! < 2) {
      throw new Error('Bollinger Bands period must be at least 2')
    }

    if ((this.config.stdDevMultiplier ?? 2) <= 0) {
      throw new Error('Standard deviation multiplier must be positive')
    }

    // Create SMA indicator for middle band
    this.smaIndicator = new SMAIndicator({
      period: this.config.period!,
      cacheEnabled: false, // We handle caching at this level
    })
  }

  calculate(candles: readonly Candle[]): BollingerBandsResult | null {
    const period = this.config.period!
    const stdDevMultiplier = this.config.stdDevMultiplier ?? 2

    if (candles.length < period) {
      return null
    }

    // Check cache
    if (this.cache) {
      const cacheKey = createCacheKey(
        this.name,
        { period, stdDevMultiplier },
        createCandleHash(candles)
      )

      const cached = this.cache.get<BollingerBandsResult>(cacheKey)
      if (cached) {
        return cached
      }
    }

    // Calculate bands
    const relevantCandles = candles.slice(-period)
    const closes = relevantCandles.map((c) => c.close)
    const lastCandle = candles[candles.length - 1]!

    // Calculate middle band (SMA)
    const smaResult = this.smaIndicator.calculate(relevantCandles)
    if (!smaResult) {
      return null
    }
    const sma = smaResult.value

    // Calculate standard deviation
    const variance =
      closes.reduce((sum, close) => {
        const diff = close - sma
        return sum + diff * diff
      }, 0) / period
    const stdDev = Math.sqrt(variance)

    // Calculate bands
    const upper = sma + stdDev * stdDevMultiplier
    const lower = sma - stdDev * stdDevMultiplier

    // Calculate derived metrics
    const bandwidth = (upper - lower) / sma
    const percentB = (lastCandle.close - lower) / (upper - lower)

    const result: BollingerBandsResult = {
      value: sma, // Primary value is the middle band
      upper,
      middle: sma,
      lower,
      bandwidth,
      percentB,
      timestamp: lastCandle.timestamp,
    }

    // Store in cache
    if (this.cache) {
      const cacheKey = createCacheKey(
        this.name,
        { period, stdDevMultiplier },
        createCandleHash(candles)
      )
      this.cache.set(cacheKey, result, candles.length)
    }

    return result
  }

  calculateAll(candles: readonly Candle[]): readonly BollingerBandsResult[] {
    const results: BollingerBandsResult[] = []
    const period = this.config.period!

    if (candles.length < period) {
      return results
    }

    // Calculate bands for each valid position
    for (let i = period - 1; i < candles.length; i++) {
      const windowCandles = candles.slice(0, i + 1)
      const result = this.calculate(windowCandles)
      if (result) {
        results.push(result)
      }
    }

    return results
  }

  reset(): void {
    this.cache?.clear()
    this.smaIndicator.reset()
  }

  getMinimumCandles(): number {
    return this.config.period!
  }

  /**
   * Static helper to calculate Bollinger Bands without creating an instance
   */
  static calculate(
    candles: readonly Candle[],
    period = 20,
    stdDevMultiplier = 2
  ): BollingerBandsResult | null {
    if (candles.length < period || period < 2 || stdDevMultiplier <= 0) {
      return null
    }

    const indicator = new BollingerBandsIndicator({
      period,
      stdDevMultiplier,
      cacheEnabled: false,
    })
    return indicator.calculate(candles)
  }

  /**
   * Calculate Bollinger Bands using typed arrays for better performance
   */
  static calculateOptimized(
    closes: Float64Array,
    period = 20,
    stdDevMultiplier = 2
  ): {
    upper: Float64Array
    middle: Float64Array
    lower: Float64Array
    bandwidth: Float64Array
    percentB: Float64Array
  } | null {
    if (closes.length < period || period < 2 || stdDevMultiplier <= 0) {
      return null
    }

    const resultLength = closes.length - period + 1
    const upper = new Float64Array(resultLength)
    const middle = new Float64Array(resultLength)
    const lower = new Float64Array(resultLength)
    const bandwidth = new Float64Array(resultLength)
    const percentB = new Float64Array(resultLength)

    // Calculate for each window
    for (let i = 0; i < resultLength; i++) {
      // Calculate SMA
      let sum = 0
      for (let j = 0; j < period; j++) {
        sum += closes[i + j]!
      }
      const sma = sum / period
      middle[i] = sma

      // Calculate standard deviation
      let variance = 0
      for (let j = 0; j < period; j++) {
        const diff = closes[i + j]! - sma
        variance += diff * diff
      }
      const stdDev = Math.sqrt(variance / period)

      // Calculate bands
      upper[i] = sma + stdDev * stdDevMultiplier
      lower[i] = sma - stdDev * stdDevMultiplier

      // Calculate derived metrics
      const currentClose = closes[i + period - 1]!
      bandwidth[i] = (upper[i]! - lower[i]!) / sma
      percentB[i] = (currentClose - lower[i]!) / (upper[i]! - lower[i]!)
    }

    return { upper, middle, lower, bandwidth, percentB }
  }
}
