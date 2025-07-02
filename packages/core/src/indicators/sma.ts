import type { Candle } from '@trdr/shared'
import { createCacheKey, createCandleHash, IndicatorCache } from './cache'
import type { IIndicator, IndicatorResult, MovingAverageConfig } from './interfaces'

/**
 * Simple Moving Average (SMA) indicator
 *
 * Calculates the arithmetic mean of closing prices over a specified period.
 * Formula: SMA = (P1 + P2 + ... + Pn) / n
 * where P = closing price and n = period
 */
export class SMAIndicator implements IIndicator<MovingAverageConfig> {
  readonly name = 'SMA'
  readonly config: MovingAverageConfig
  private readonly cache?: IndicatorCache

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
      throw new Error('SMA period must be at least 1')
    }
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

    // Calculate SMA
    const relevantCandles = candles.slice(-this.config.period)
    const sum = relevantCandles.reduce((acc, candle) => acc + candle.close, 0)
    const sma = sum / this.config.period

    const lastCandle = candles[candles.length - 1]
    const result: IndicatorResult = {
      value: sma,
      timestamp: lastCandle!.timestamp,
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

    // Need at least period candles to start
    if (candles.length < this.config.period) {
      return results
    }

    // Calculate SMA for each valid position
    for (let i = this.config.period - 1; i < candles.length; i++) {
      const slice = candles.slice(i - this.config.period + 1, i + 1)
      const sum = slice.reduce((acc, candle) => acc + candle.close, 0)
      const sma = sum / this.config.period

      const candle = candles[i]
      results.push({
        value: sma,
        timestamp: candle!.timestamp,
      })
    }

    return results
  }

  reset(): void {
    this.cache?.clear()
  }

  getMinimumCandles(): number {
    return this.config.period
  }

  /**
   * Static helper to calculate SMA without creating an instance
   */
  static calculate(candles: readonly Candle[], period: number): number | null {
    if (candles.length < period || period < 1) {
      return null
    }

    const relevantCandles = candles.slice(-period)
    const sum = relevantCandles.reduce((acc, candle) => acc + candle.close, 0)
    return sum / period
  }

  /**
   * Calculate SMA using typed arrays for better performance
   */
  static calculateOptimized(closes: Float64Array, period: number): Float64Array | null {
    if (closes.length < period || period < 1) {
      return null
    }

    const results = new Float64Array(closes.length - period + 1)

    // Calculate first SMA
    let sum = 0
    for (let i = 0; i < period; i++) {
      sum += closes[i]!
    }
    results[0] = sum / period

    // Calculate remaining SMAs using sliding window
    for (let i = 1; i < results.length; i++) {
      sum = sum - closes[i - 1]! + closes[i + period - 1]!
      results[i] = sum / period
    }

    return results
  }
}
