import type { Candle } from '@trdr/shared'
import { createCacheKey, createCandleHash, IndicatorCache } from './cache'
import type { ATRConfig, IIndicator, IndicatorResult } from './interfaces'

/**
 * Average True Range (ATR) indicator
 *
 * Measures market volatility by decomposing the entire range of an asset price for that period.
 * The true range is the maximum of:
 * 1. Current high - current low
 * 2. Absolute value of (current high - previous close)
 * 3. Absolute value of (current low - previous close)
 *
 * ATR is the moving average of true ranges
 */
export class ATRIndicator implements IIndicator<ATRConfig> {
  readonly name = 'ATR'
  readonly config: ATRConfig
  private readonly cache?: IndicatorCache

  constructor(config: Partial<ATRConfig> = {}) {
    this.config = {
      period: config.period ?? 14,
      smoothing: config.smoothing ?? 'wilder',
      cacheEnabled: config.cacheEnabled ?? true,
      cacheSize: config.cacheSize ?? 100,
    }

    if (this.config.cacheEnabled) {
      this.cache = new IndicatorCache(this.config.cacheSize)
    }

    if (this.config.period! < 1) {
      throw new Error('ATR period must be at least 1')
    }
  }

  calculate(candles: readonly Candle[]): IndicatorResult | null {
    const period = this.config.period!
    const smoothing = this.config.smoothing!

    if (candles.length < period + 1) {
      return null
    }

    // Check cache
    if (this.cache) {
      const cacheKey = createCacheKey(this.name, { period, smoothing }, createCandleHash(candles))

      const cached = this.cache.get<IndicatorResult>(cacheKey)
      if (cached) {
        return cached
      }
    }

    // Calculate ATR
    const atrValues = this.calculateATRValues(candles)
    if (atrValues.length === 0) {
      return null
    }

    const lastATR = atrValues[atrValues.length - 1]!
    const lastCandle = candles[candles.length - 1]!
    const result: IndicatorResult = {
      value: lastATR,
      timestamp: lastCandle.timestamp,
    }

    // Store in cache
    if (this.cache) {
      const cacheKey = createCacheKey(this.name, { period, smoothing }, createCandleHash(candles))
      this.cache.set(cacheKey, result, candles.length)
    }

    return result
  }

  calculateAll(candles: readonly Candle[]): readonly IndicatorResult[] {
    const results: IndicatorResult[] = []
    const period = this.config.period!

    if (candles.length < period + 1) {
      return results
    }

    const atrValues = this.calculateATRValues(candles)

    // Create results starting from where we have ATR values
    const startIndex = period
    for (let i = 0; i < atrValues.length; i++) {
      const atrValue = atrValues[i]!
      const candle = candles[startIndex + i]!
      results.push({
        value: atrValue,
        timestamp: candle.timestamp,
      })
    }

    return results
  }

  /**
   * Calculate True Range for a candle
   */
  private calculateTrueRange(current: Candle, previous: Candle): number {
    const highLow = current.high - current.low
    const highPrevClose = Math.abs(current.high - previous.close)
    const lowPrevClose = Math.abs(current.low - previous.close)

    return Math.max(highLow, highPrevClose, lowPrevClose)
  }

  /**
   * Calculate all ATR values for the given candles
   */
  private calculateATRValues(candles: readonly Candle[]): number[] {
    const period = this.config.period!
    const smoothing = this.config.smoothing!

    if (candles.length < period + 1) {
      return []
    }

    const atrValues: number[] = []
    const trValues: number[] = []

    // Calculate True Ranges
    for (let i = 1; i < candles.length; i++) {
      const current = candles[i]!
      const previous = candles[i - 1]!
      trValues.push(this.calculateTrueRange(current, previous))
    }

    // Calculate initial ATR as simple average
    let sum = 0
    for (let i = 0; i < period; i++) {
      sum += trValues[i]!
    }
    let atr = sum / period
    atrValues.push(atr)

    // Calculate subsequent ATR values
    if (smoothing === 'wilder') {
      // Wilder's smoothing: ATR = ((n-1) * prevATR + TR) / n
      for (let i = period; i < trValues.length; i++) {
        const tr = trValues[i]!
        atr = ((period - 1) * atr + tr) / period
        atrValues.push(atr)
      }
    } else {
      // Simple moving average
      for (let i = period; i < trValues.length; i++) {
        sum = sum - trValues[i - period]! + trValues[i]!
        atr = sum / period
        atrValues.push(atr)
      }
    }

    return atrValues
  }

  reset(): void {
    this.cache?.clear()
  }

  getMinimumCandles(): number {
    return this.config.period! + 1
  }

  /**
   * Static helper to calculate ATR without creating an instance
   */
  static calculate(
    candles: readonly Candle[],
    period = 14,
    smoothing: 'wilder' | 'sma' = 'wilder'
  ): number | null {
    if (candles.length < period + 1 || period < 1) {
      return null
    }

    const indicator = new ATRIndicator({ period, smoothing, cacheEnabled: false })
    const result = indicator.calculate(candles)
    return result ? result.value : null
  }

  /**
   * Calculate ATR using typed arrays for better performance
   */
  static calculateOptimized(
    highs: Float64Array,
    lows: Float64Array,
    closes: Float64Array,
    period = 14,
    smoothing: 'wilder' | 'sma' = 'wilder'
  ): Float64Array | null {
    const length = Math.min(highs.length, lows.length, closes.length)

    if (length < period + 1 || period < 1) {
      return null
    }

    const trValues = new Float64Array(length - 1)

    // Calculate True Ranges
    for (let i = 1; i < length; i++) {
      const highLow = highs[i]! - lows[i]!
      const highPrevClose = Math.abs(highs[i]! - closes[i - 1]!)
      const lowPrevClose = Math.abs(lows[i]! - closes[i - 1]!)
      trValues[i - 1] = Math.max(highLow, highPrevClose, lowPrevClose)
    }

    const results = new Float64Array(trValues.length - period + 1)

    // Calculate initial ATR
    let sum = 0
    for (let i = 0; i < period; i++) {
      sum += trValues[i]!
    }
    let atr = sum / period
    results[0] = atr

    // Calculate subsequent ATR values
    if (smoothing === 'wilder') {
      for (let i = 1; i < results.length; i++) {
        const tr = trValues[period + i - 1]!
        atr = ((period - 1) * atr + tr) / period
        results[i] = atr
      }
    } else {
      for (let i = 1; i < results.length; i++) {
        sum = sum - trValues[i - 1]! + trValues[period + i - 1]!
        atr = sum / period
        results[i] = atr
      }
    }

    return results
  }
}
