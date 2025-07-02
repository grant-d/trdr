import type { Candle } from '@trdr/shared'
import { createCacheKey, createCandleHash, IndicatorCache } from './cache'
import { EMAIndicator } from './ema'
import type { IMultiValueIndicator, MACDConfig, MACDResult } from './interfaces'

/**
 * MACD (Moving Average Convergence Divergence) indicator
 *
 * A trend-following momentum indicator that shows the relationship between
 * two moving averages of prices.
 *
 * MACD Line = 12-period EMA - 26-period EMA
 * Signal Line = 9-period EMA of MACD Line
 * MACD Histogram = MACD Line - Signal Line
 */
export class MACDIndicator implements IMultiValueIndicator<MACDResult, MACDConfig> {
  readonly name = 'MACD'
  readonly config: MACDConfig
  private readonly cache?: IndicatorCache
  private readonly fastEMA: EMAIndicator
  private readonly slowEMA: EMAIndicator
  private readonly signalEMA: EMAIndicator

  constructor(config: Partial<MACDConfig> = {}) {
    this.config = {
      fastPeriod: config.fastPeriod ?? 12,
      slowPeriod: config.slowPeriod ?? 26,
      signalPeriod: config.signalPeriod ?? 9,
      cacheEnabled: config.cacheEnabled ?? true,
      cacheSize: config.cacheSize ?? 50,
    }

    if (this.config.cacheEnabled) {
      this.cache = new IndicatorCache(this.config.cacheSize)
    }

    // Validate periods
    if (this.config.fastPeriod! >= this.config.slowPeriod!) {
      throw new Error('Fast period must be less than slow period')
    }

    // Create EMA indicators
    this.fastEMA = new EMAIndicator({
      period: this.config.fastPeriod!,
      cacheEnabled: false, // We handle caching at MACD level
    })

    this.slowEMA = new EMAIndicator({
      period: this.config.slowPeriod!,
      cacheEnabled: false,
    })

    this.signalEMA = new EMAIndicator({
      period: this.config.signalPeriod!,
      cacheEnabled: false,
    })
  }

  calculate(candles: readonly Candle[]): MACDResult | null {
    if (candles.length < this.getMinimumCandles()) {
      return null
    }

    // Check cache
    if (this.cache) {
      const cacheKey = createCacheKey(
        this.name,
        {
          fast: this.config.fastPeriod,
          slow: this.config.slowPeriod,
          signal: this.config.signalPeriod,
        },
        createCandleHash(candles)
      )

      const cached = this.cache.get<MACDResult>(cacheKey)
      if (cached) {
        return cached
      }
    }

    // Calculate MACD values
    const macdData = this.calculateMACDData(candles)
    if (!macdData || macdData.macdValues.length === 0) {
      return null
    }

    const lastIndex = macdData.macdValues.length - 1
    const lastCandle = candles[candles.length - 1]

    const result: MACDResult = {
      value: macdData.macdValues[lastIndex]!,
      macd: macdData.macdValues[lastIndex]!,
      signal: macdData.signalValues[lastIndex]!,
      histogram: macdData.histogramValues[lastIndex]!,
      timestamp: lastCandle!.timestamp,
    }

    // Store in cache
    if (this.cache) {
      const cacheKey = createCacheKey(
        this.name,
        {
          fast: this.config.fastPeriod,
          slow: this.config.slowPeriod,
          signal: this.config.signalPeriod,
        },
        createCandleHash(candles)
      )
      this.cache.set(cacheKey, result, candles.length)
    }

    return result
  }

  calculateAll(candles: readonly Candle[]): readonly MACDResult[] {
    const results: MACDResult[] = []

    if (candles.length < this.getMinimumCandles()) {
      return results
    }

    const macdData = this.calculateMACDData(candles)
    if (!macdData) {
      return results
    }

    // Create results for each valid MACD point
    const startIndex = candles.length - macdData.macdValues.length
    for (let i = 0; i < macdData.macdValues.length; i++) {
      const candleIndex = startIndex + i
      const candle = candles[candleIndex]

      results.push({
        value: macdData.macdValues[i]!,
        macd: macdData.macdValues[i]!,
        signal: macdData.signalValues[i]!,
        histogram: macdData.histogramValues[i]!,
        timestamp: candle!.timestamp,
      })
    }

    return results
  }

  /**
   * Calculate all MACD components
   */
  private calculateMACDData(candles: readonly Candle[]): { macdValues: number[]; signalValues: number[]; histogramValues: number[] } | null {
    // Calculate fast and slow EMAs
    const fastEMAResults = this.fastEMA.calculateAll(candles)
    const slowEMAResults = this.slowEMA.calculateAll(candles)

    if (fastEMAResults.length === 0 || slowEMAResults.length === 0) {
      return null
    }

    // Calculate MACD line (fast EMA - slow EMA)
    const macdValues: number[] = []
    const startOffset = this.config.slowPeriod! - 1

    for (let i = 0; i < slowEMAResults.length; i++) {
      const fastIndex = i + (this.config.slowPeriod! - this.config.fastPeriod!)
      const macd = fastEMAResults[fastIndex]!.value - slowEMAResults[i]!.value
      macdValues.push(macd)
    }

    // Create synthetic candles for signal line calculation
    const syntheticCandles: Candle[] = macdValues.map((value, i) => ({
      timestamp: candles[startOffset + i]!.timestamp,
      open: value,
      high: value,
      low: value,
      close: value,
      volume: 0,
    }))

    // Calculate signal line (EMA of MACD)
    const signalResults = this.signalEMA.calculateAll(syntheticCandles)

    if (signalResults.length === 0) {
      return null
    }

    // Align arrays and calculate histogram
    const signalOffset = this.config.signalPeriod! - 1
    const alignedMacdValues = macdValues.slice(signalOffset)
    const signalValues = signalResults.map((r) => r.value)
    const histogramValues = alignedMacdValues.map((macd, i) => macd - signalValues[i]!)

    return {
      macdValues: alignedMacdValues,
      signalValues,
      histogramValues,
    }
  }

  reset(): void {
    this.cache?.clear()
    this.fastEMA.reset()
    this.slowEMA.reset()
    this.signalEMA.reset()
  }

  getMinimumCandles(): number {
    // Need enough candles for slow EMA + signal period
    return this.config.slowPeriod! + this.config.signalPeriod! - 1
  }

  /**
   * Static helper to calculate MACD without creating an instance
   */
  static calculate(candles: readonly Candle[], config?: Partial<MACDConfig>): MACDResult | null {
    const indicator = new MACDIndicator(config)
    return indicator.calculate(candles)
  }
}
