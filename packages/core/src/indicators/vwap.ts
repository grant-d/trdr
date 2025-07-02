import type { Candle } from '@trdr/shared'
import { createCacheKey, createCandleHash, IndicatorCache } from './cache'
import type { IIndicator, IndicatorResult, VWAPConfig } from './interfaces'

/**
 * Volume Weighted Average Price (VWAP) indicator
 *
 * Calculates the average price weighted by volume over a specific time period.
 * VWAP = Σ(Price × Volume) / Σ(Volume)
 *
 * Typically resets at the start of each trading session, week, or month.
 * Used as a benchmark for trade execution and to identify value areas.
 */
export class VWAPIndicator implements IIndicator<VWAPConfig> {
  readonly name = 'VWAP'
  readonly config: VWAPConfig
  private readonly cache?: IndicatorCache

  constructor(config: Partial<VWAPConfig> = {}) {
    this.config = {
      anchorTime: config.anchorTime ?? 'session',
      cacheEnabled: config.cacheEnabled ?? true,
      cacheSize: config.cacheSize ?? 100,
    }

    if (this.config.cacheEnabled) {
      this.cache = new IndicatorCache(this.config.cacheSize)
    }
  }

  calculate(candles: readonly Candle[]): IndicatorResult | null {
    if (candles.length === 0) {
      return null
    }

    // Check cache
    if (this.cache) {
      const cacheKey = createCacheKey(
        this.name,
        { anchorTime: this.config.anchorTime },
        createCandleHash(candles)
      )

      const cached = this.cache.get<IndicatorResult>(cacheKey)
      if (cached) {
        return cached
      }
    }

    // Calculate VWAP
    const vwapValue = this.calculateVWAP(candles)
    if (vwapValue === null) {
      return null
    }

    const lastCandle = candles[candles.length - 1]!
    const result: IndicatorResult = {
      value: vwapValue,
      timestamp: lastCandle.timestamp,
    }

    // Store in cache
    if (this.cache) {
      const cacheKey = createCacheKey(
        this.name,
        { anchorTime: this.config.anchorTime },
        createCandleHash(candles)
      )
      this.cache.set(cacheKey, result, candles.length)
    }

    return result
  }

  calculateAll(candles: readonly Candle[]): readonly IndicatorResult[] {
    if (candles.length === 0) {
      return []
    }

    const results: IndicatorResult[] = []
    const anchorIndices = this.findAnchorPoints(candles)

    // Calculate VWAP for each period
    for (let i = 0; i < anchorIndices.length; i++) {
      const startIdx = anchorIndices[i]!
      const endIdx = i + 1 < anchorIndices.length ? anchorIndices[i + 1]! : candles.length

      let cumulativePV = 0
      let cumulativeVolume = 0

      for (let j = startIdx; j < endIdx; j++) {
        const candle = candles[j]!
        const typicalPrice = (candle.high + candle.low + candle.close) / 3
        cumulativePV += typicalPrice * candle.volume
        cumulativeVolume += candle.volume

        if (cumulativeVolume > 0) {
          const vwap = cumulativePV / cumulativeVolume
          results.push({
            value: vwap,
            timestamp: candle.timestamp,
          })
        }
      }
    }

    return results
  }

  /**
   * Calculate VWAP for the current period
   */
  private calculateVWAP(candles: readonly Candle[]): number | null {
    if (candles.length === 0) {
      return null
    }

    // Find the most recent anchor point
    const anchorIndex = this.findLastAnchorPoint(candles)
    const relevantCandles = candles.slice(anchorIndex)

    let cumulativePV = 0
    let cumulativeVolume = 0

    for (const candle of relevantCandles) {
      const typicalPrice = (candle.high + candle.low + candle.close) / 3
      cumulativePV += typicalPrice * candle.volume
      cumulativeVolume += candle.volume
    }

    if (cumulativeVolume === 0) {
      return null
    }

    return cumulativePV / cumulativeVolume
  }

  /**
   * Find all anchor points based on the anchor time configuration
   */
  private findAnchorPoints(candles: readonly Candle[]): number[] {
    const anchorIndices: number[] = [0]

    if (this.config.anchorTime === 'session') {
      // Detect session changes (assuming daily sessions)
      for (let i = 1; i < candles.length; i++) {
        const prevDate = new Date(candles[i - 1]!.timestamp)
        const currDate = new Date(candles[i]!.timestamp)

        if (prevDate.getUTCDate() !== currDate.getUTCDate()) {
          anchorIndices.push(i)
        }
      }
    } else if (this.config.anchorTime === 'week') {
      // Detect week changes
      for (let i = 1; i < candles.length; i++) {
        const prevWeek = this.getWeekNumber(new Date(candles[i - 1]!.timestamp))
        const currWeek = this.getWeekNumber(new Date(candles[i]!.timestamp))

        if (prevWeek !== currWeek) {
          anchorIndices.push(i)
        }
      }
    } else if (this.config.anchorTime === 'month') {
      // Detect month changes
      for (let i = 1; i < candles.length; i++) {
        const prevDate = new Date(candles[i - 1]!.timestamp)
        const currDate = new Date(candles[i]!.timestamp)

        if (
          prevDate.getUTCMonth() !== currDate.getUTCMonth() ||
          prevDate.getUTCFullYear() !== currDate.getUTCFullYear()
        ) {
          anchorIndices.push(i)
        }
      }
    }

    return anchorIndices
  }

  /**
   * Find the last anchor point for the current period
   */
  private findLastAnchorPoint(candles: readonly Candle[]): number {
    if (candles.length === 0) {
      return 0
    }

    const anchorIndices = this.findAnchorPoints(candles)
    return anchorIndices[anchorIndices.length - 1]!
  }

  /**
   * Get ISO week number for a date
   */
  private getWeekNumber(date: Date): number {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()))
    const dayNum = d.getUTCDay() || 7
    d.setUTCDate(d.getUTCDate() + 4 - dayNum)
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1))
    return Math.ceil(((d.getTime() - yearStart.getTime()) / 86400000 + 1) / 7)
  }

  reset(): void {
    this.cache?.clear()
  }

  getMinimumCandles(): number {
    return 1
  }

  /**
   * Static helper to calculate VWAP without creating an instance
   */
  static calculate(
    candles: readonly Candle[],
    anchorTime: 'session' | 'week' | 'month' = 'session'
  ): number | null {
    if (candles.length === 0) {
      return null
    }

    const indicator = new VWAPIndicator({ anchorTime, cacheEnabled: false })
    const result = indicator.calculate(candles)
    return result ? result.value : null
  }

  /**
   * Calculate VWAP using typed arrays for better performance
   */
  static calculateOptimized(
    highs: Float64Array,
    lows: Float64Array,
    closes: Float64Array,
    volumes: Float64Array,
    timestamps: Float64Array,
    anchorTime: 'session' | 'week' | 'month' = 'session'
  ): Float64Array | null {
    const length = Math.min(highs.length, lows.length, closes.length, volumes.length, timestamps.length)

    if (length === 0) {
      return null
    }

    const results = new Float64Array(length)
    const anchorIndices: number[] = [0]

    // Find anchor points based on timestamps
    if (anchorTime === 'session') {
      for (let i = 1; i < length; i++) {
        const prevDate = new Date(timestamps[i - 1]!)
        const currDate = new Date(timestamps[i]!)

        if (prevDate.getUTCDate() !== currDate.getUTCDate()) {
          anchorIndices.push(i)
        }
      }
    }
    // Similar logic for week and month...

    // Calculate VWAP for each period
    for (let i = 0; i < anchorIndices.length; i++) {
      const startIdx = anchorIndices[i]!
      const endIdx = i + 1 < anchorIndices.length ? anchorIndices[i + 1]! : length

      let cumulativePV = 0
      let cumulativeVolume = 0

      for (let j = startIdx; j < endIdx; j++) {
        const typicalPrice = (highs[j]! + lows[j]! + closes[j]!) / 3
        cumulativePV += typicalPrice * volumes[j]!
        cumulativeVolume += volumes[j]!

        if (cumulativeVolume > 0) {
          results[j] = cumulativePV / cumulativeVolume
        } else {
          results[j] = j > 0 ? results[j - 1]! : 0
        }
      }
    }

    return results
  }
}