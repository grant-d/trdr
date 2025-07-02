import type { Candle } from '@trdr/shared'
import { createCacheKey, createCandleHash, IndicatorCache } from './cache'
import type { IIndicator, IndicatorResult, MovingAverageConfig } from './interfaces'
import { SMAIndicator } from './sma'

/**
 * Exponential Moving Average (EMA) indicator
 *
 * Calculates the exponentially weighted moving average, giving more weight to recent prices.
 * Formula: EMA = (Close - Previous EMA) × Multiplier + Previous EMA
 * where Multiplier = 2 / (Period + 1)
 */
export class EMAIndicator implements IIndicator<MovingAverageConfig> {
  readonly name = 'EMA'
  readonly config: MovingAverageConfig
  private readonly cache?: IndicatorCache
  private readonly multiplier: number

  constructor(config: MovingAverageConfig) {
    this.config = {
      ...config,
      cacheEnabled: config.cacheEnabled ?? true,
      cacheSize: config.cacheSize ?? 100,
    }

    if (this.config.cacheEnabled) {
      this.cache = new IndicatorCache(this.config.cacheSize)
    }

    if (config.period < 1) {
      throw new Error('EMA period must be at least 1')
    }

    // Calculate smoothing multiplier
    this.multiplier = 2 / (config.period + 1)
  }

  calculate(candles: readonly Candle[]): IndicatorResult | null {
    if (candles.length < this.config.period) {
      return null
    }

    // Check cache
    if (this.cache) {
      const cacheKey = createCacheKey(
        this.name,
        { period: this.config.period },
        createCandleHash(candles)
      )

      const cached = this.cache.get<IndicatorResult>(cacheKey)
      if (cached) {
        return cached
      }
    }

    // Calculate EMA
    const emaValues = this.calculateEMAValues(candles)
    if (emaValues.length === 0) {
      return null
    }

    const lastEMAValue = emaValues[emaValues.length - 1]!
    const lastCandle = candles[candles.length - 1]!
    const result: IndicatorResult = {
      value: lastEMAValue,
      timestamp: lastCandle.timestamp,
    }

    // Store in cache
    if (this.cache) {
      const cacheKey = createCacheKey(
        this.name,
        { period: this.config.period },
        createCandleHash(candles)
      )
      this.cache.set(cacheKey, result, candles.length)
    }

    return result
  }

  calculateAll(candles: readonly Candle[]): readonly IndicatorResult[] {
    const results: IndicatorResult[] = []

    if (candles.length < this.config.period) {
      return results
    }

    const emaValues = this.calculateEMAValues(candles)

    // Create results starting from where we have EMA values
    const startIndex = this.config.period - 1
    for (let i = 0; i < emaValues.length; i++) {
      const emaValue = emaValues[i]!
      const candle = candles[startIndex + i]!
      results.push({
        value: emaValue,
        timestamp: candle.timestamp,
      })
    }

    return results
  }

  /**
   * Calculate all EMA values for the given candles
   */
  private calculateEMAValues(candles: readonly Candle[]): number[] {
    if (candles.length < this.config.period) {
      return []
    }

    const emaValues: number[] = []

    // Use SMA for the first EMA value
    const firstSMA = SMAIndicator.calculate(
      candles.slice(0, this.config.period),
      this.config.period
    )
    if (firstSMA === null) {
      return []
    }

    emaValues.push(firstSMA)

    // Calculate subsequent EMA values
    for (let i = this.config.period; i < candles.length; i++) {
      const prevEMA = emaValues[emaValues.length - 1]!
      const currentCandle = candles[i]!
      const currentPrice = currentCandle.close
      const ema = (currentPrice - prevEMA) * this.multiplier + prevEMA
      emaValues.push(ema)
    }

    return emaValues
  }

  reset(): void {
    this.cache?.clear()
  }

  getMinimumCandles(): number {
    return this.config.period
  }

  /**
   * Static helper to calculate EMA without creating an instance
   */
  static calculate(candles: readonly Candle[], period: number): number | null {
    if (candles.length < period || period < 1) {
      return null
    }

    const multiplier = 2 / (period + 1)

    // Use SMA for the first EMA value
    const firstSMA = SMAIndicator.calculate(candles.slice(0, period), period)
    if (firstSMA === null) {
      return null
    }

    let ema = firstSMA

    // Calculate EMA for remaining candles
    for (let i = period; i < candles.length; i++) {
      const candle = candles[i]!
      ema = (candle.close - ema) * multiplier + ema
    }

    return ema
  }

  /**
   * Calculate EMA using typed arrays for better performance
   */
  static calculateOptimized(closes: Float64Array, period: number): Float64Array | null {
    if (closes.length < period || period < 1) {
      return null
    }

    const results = new Float64Array(closes.length - period + 1)
    const multiplier = 2 / (period + 1)

    // Calculate initial SMA
    let sum = 0
    for (let i = 0; i < period; i++) {
      sum += closes[i]!
    }
    results[0] = sum / period

    // Calculate EMA values
    for (let i = 1; i < results.length; i++) {
      const currentPrice = closes[period + i - 1]!
      const prevEMA = results[i - 1]!
      results[i] = (currentPrice - prevEMA) * multiplier + prevEMA
    }

    return results
  }
}
