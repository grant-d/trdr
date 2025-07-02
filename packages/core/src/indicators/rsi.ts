import type { Candle } from '@trdr/shared'
import { createCacheKey, createCandleHash, IndicatorCache } from './cache'
import type { IIndicator, IndicatorResult, RSIConfig } from './interfaces'

/**
 * Relative Strength Index (RSI) indicator
 *
 * Measures momentum by comparing the magnitude of recent gains to recent losses.
 * RSI = 100 - (100 / (1 + RS))
 * where RS = Average Gain / Average Loss
 *
 * Values:
 * - Above 70: Generally considered overbought
 * - Below 30: Generally considered oversold
 * - 50: Neutral momentum
 */
export class RSIIndicator implements IIndicator<RSIConfig> {
  readonly name = 'RSI'
  readonly config: RSIConfig
  private readonly cache?: IndicatorCache

  constructor(config: Partial<RSIConfig> = {}) {
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
      throw new Error('RSI period must be at least 1')
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

    // Calculate RSI
    const rsiValues = this.calculateRSIValues(candles)
    if (rsiValues.length === 0) {
      return null
    }

    const lastRSI = rsiValues[rsiValues.length - 1]!
    const lastCandle = candles[candles.length - 1]!
    const result: IndicatorResult = {
      value: lastRSI,
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

    const rsiValues = this.calculateRSIValues(candles)

    // Create results starting from where we have RSI values
    const startIndex = period
    for (let i = 0; i < rsiValues.length; i++) {
      const rsiValue = rsiValues[i]!
      const candle = candles[startIndex + i]!
      results.push({
        value: rsiValue,
        timestamp: candle.timestamp,
      })
    }

    return results
  }

  /**
   * Calculate all RSI values for the given candles
   */
  private calculateRSIValues(candles: readonly Candle[]): number[] {
    const period = this.config.period!
    const smoothing = this.config.smoothing!

    if (candles.length < period + 1) {
      return []
    }

    const rsiValues: number[] = []

    // Calculate price changes
    const changes: number[] = []
    for (let i = 1; i < candles.length; i++) {
      changes.push(candles[i]!.close - candles[i - 1]!.close)
    }

    // Separate gains and losses
    const gains = changes.map((change) => Math.max(0, change))
    const losses = changes.map((change) => Math.max(0, -change))

    // Calculate initial average gain and loss
    let avgGain = 0
    let avgLoss = 0
    for (let i = 0; i < period; i++) {
      avgGain += gains[i]!
      avgLoss += losses[i]!
    }
    avgGain /= period
    avgLoss /= period

    // Calculate first RSI
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss
    const rsi = avgLoss === 0 ? 100 : 100 - 100 / (1 + rs)
    rsiValues.push(rsi)

    // Calculate subsequent RSI values
    if (smoothing === 'wilder') {
      // Wilder's smoothing
      for (let i = period; i < changes.length; i++) {
        avgGain = ((period - 1) * avgGain + gains[i]!) / period
        avgLoss = ((period - 1) * avgLoss + losses[i]!) / period

        const currentRS = avgLoss === 0 ? 100 : avgGain / avgLoss
        const currentRSI = avgLoss === 0 ? 100 : 100 - 100 / (1 + currentRS)
        rsiValues.push(currentRSI)
      }
    } else {
      // Simple moving average
      for (let i = period; i < changes.length; i++) {
        // Calculate SMA of gains and losses
        let gainSum = 0
        let lossSum = 0
        for (let j = 0; j < period; j++) {
          gainSum += gains[i - period + 1 + j]!
          lossSum += losses[i - period + 1 + j]!
        }
        avgGain = gainSum / period
        avgLoss = lossSum / period

        const currentRS = avgLoss === 0 ? 100 : avgGain / avgLoss
        const currentRSI = avgLoss === 0 ? 100 : 100 - 100 / (1 + currentRS)
        rsiValues.push(currentRSI)
      }
    }

    return rsiValues
  }

  reset(): void {
    this.cache?.clear()
  }

  getMinimumCandles(): number {
    return this.config.period! + 1
  }

  /**
   * Static helper to calculate RSI without creating an instance
   */
  static calculate(
    candles: readonly Candle[],
    period = 14,
    smoothing: 'wilder' | 'sma' = 'wilder'
  ): number | null {
    if (candles.length < period + 1 || period < 1) {
      return null
    }

    const indicator = new RSIIndicator({ period, smoothing, cacheEnabled: false })
    const result = indicator.calculate(candles)
    return result ? result.value : null
  }

  /**
   * Calculate RSI using typed arrays for better performance
   */
  static calculateOptimized(
    closes: Float64Array,
    period = 14,
    smoothing: 'wilder' | 'sma' = 'wilder'
  ): Float64Array | null {
    if (closes.length < period + 1 || period < 1) {
      return null
    }

    const results = new Float64Array(closes.length - period)

    // Calculate price changes
    const changes = new Float64Array(closes.length - 1)
    for (let i = 1; i < closes.length; i++) {
      changes[i - 1] = closes[i]! - closes[i - 1]!
    }

    // Calculate initial average gain and loss
    let avgGain = 0
    let avgLoss = 0
    for (let i = 0; i < period; i++) {
      const change = changes[i]!
      if (change > 0) {
        avgGain += change
      } else {
        avgLoss += -change
      }
    }
    avgGain /= period
    avgLoss /= period

    // Calculate first RSI
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss
    results[0] = avgLoss === 0 ? 100 : 100 - 100 / (1 + rs)

    // Calculate subsequent RSI values
    if (smoothing === 'wilder') {
      for (let i = 1; i < results.length; i++) {
        const change = changes[period + i - 1]!
        const gain = Math.max(0, change)
        const loss = Math.max(0, -change)

        avgGain = ((period - 1) * avgGain + gain) / period
        avgLoss = ((period - 1) * avgLoss + loss) / period

        const currentRS = avgLoss === 0 ? 100 : avgGain / avgLoss
        results[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + currentRS)
      }
    } else {
      // SMA smoothing
      for (let i = 1; i < results.length; i++) {
        let gainSum = 0
        let lossSum = 0
        for (let j = 0; j < period; j++) {
          const change = changes[i + j]!
          if (change > 0) {
            gainSum += change
          } else {
            lossSum += -change
          }
        }
        avgGain = gainSum / period
        avgLoss = lossSum / period

        const currentRS = avgLoss === 0 ? 100 : avgGain / avgLoss
        results[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + currentRS)
      }
    }

    return results
  }
}
