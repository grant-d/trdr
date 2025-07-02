import type { Candle } from '@trdr/shared'
import { createCacheKey, createCandleHash, IndicatorCache } from './cache'
import type { SwingDetectionConfig, SwingPoint } from './interfaces'

/**
 * Swing High/Low Detector
 *
 * Identifies significant turning points in price action by detecting
 * local highs and lows. A swing high is a candle with a high that is
 * higher than the surrounding candles, and vice versa for swing lows.
 */
export class SwingDetector {
  readonly name = 'SwingDetector'
  readonly config: SwingDetectionConfig
  private readonly cache?: IndicatorCache

  constructor(config: Partial<SwingDetectionConfig> = {}) {
    this.config = {
      lookback: config.lookback ?? 5,
      lookforward: config.lookforward ?? 5,
      minSwingPercent: config.minSwingPercent ?? 0.001,
      includeWicks: config.includeWicks ?? true,
      cacheEnabled: config.cacheEnabled ?? true,
      cacheSize: config.cacheSize ?? 100,
    }

    if (this.config.cacheEnabled) {
      this.cache = new IndicatorCache(this.config.cacheSize)
    }

    if (this.config.lookback! < 1 || this.config.lookforward! < 1) {
      throw new Error('Lookback and lookforward must be at least 1')
    }
  }

  calculate(candles: readonly Candle[]): readonly SwingPoint[] | null {
    const minCandles = this.getMinimumCandles()
    if (candles.length < minCandles) {
      return null
    }

    // Check cache
    if (this.cache) {
      const cacheKey = createCacheKey(
        this.name,
        {
          lookback: this.config.lookback,
          lookforward: this.config.lookforward,
          minSwingPercent: this.config.minSwingPercent,
          includeWicks: this.config.includeWicks,
        },
        createCandleHash(candles)
      )

      const cached = this.cache.get<readonly SwingPoint[]>(cacheKey)
      if (cached) {
        return cached
      }
    }

    // Detect swing points
    const swingPoints = this.detectSwingPoints(candles)

    // Store in cache
    if (this.cache && swingPoints.length > 0) {
      const cacheKey = createCacheKey(
        this.name,
        {
          lookback: this.config.lookback,
          lookforward: this.config.lookforward,
          minSwingPercent: this.config.minSwingPercent,
          includeWicks: this.config.includeWicks,
        },
        createCandleHash(candles)
      )
      this.cache.set(cacheKey, swingPoints, candles.length)
    }

    return swingPoints
  }

  calculateAll(candles: readonly Candle[]): readonly SwingPoint[] {
    // For swing detection, calculateAll returns the same as calculate
    // since we detect all swing points in one pass
    return this.calculate(candles) ?? []
  }

  /**
   * Detect all swing points in the given candles
   */
  private detectSwingPoints(candles: readonly Candle[]): SwingPoint[] {
    const swingPoints: SwingPoint[] = []
    const lookback = this.config.lookback!
    const lookforward = this.config.lookforward!
    const minSwingPercent = this.config.minSwingPercent!
    const includeWicks = this.config.includeWicks!

    // Can only detect swings for candles that have enough surrounding candles
    const startIdx = lookback
    const endIdx = candles.length - lookforward

    for (let i = startIdx; i < endIdx; i++) {
      const currentCandle = candles[i]!

      // Check for swing high
      const swingHigh = this.isSwingHigh(candles, i, lookback, lookforward, includeWicks)
      if (swingHigh) {
        // Check if the swing is significant enough
        const highPrice = includeWicks ? currentCandle.high : currentCandle.close
        const surroundingAvg = this.calculateSurroundingAverage(
          candles,
          i,
          lookback,
          lookforward,
          'high',
          includeWicks
        )
        const swingPercent = (highPrice - surroundingAvg) / surroundingAvg

        if (swingPercent >= minSwingPercent) {
          swingPoints.push({
            type: 'high',
            index: i,
            price: highPrice,
            timestamp: currentCandle.timestamp,
            strength: swingPercent,
          })
        }
      }

      // Check for swing low
      const swingLow = this.isSwingLow(candles, i, lookback, lookforward, includeWicks)
      if (swingLow) {
        // Check if the swing is significant enough
        const lowPrice = includeWicks ? currentCandle.low : currentCandle.close
        const surroundingAvg = this.calculateSurroundingAverage(
          candles,
          i,
          lookback,
          lookforward,
          'low',
          includeWicks
        )
        const swingPercent = (surroundingAvg - lowPrice) / surroundingAvg

        if (swingPercent >= minSwingPercent) {
          swingPoints.push({
            type: 'low',
            index: i,
            price: lowPrice,
            timestamp: currentCandle.timestamp,
            strength: swingPercent,
          })
        }
      }
    }

    return swingPoints
  }

  /**
   * Check if a candle is a swing high
   */
  private isSwingHigh(
    candles: readonly Candle[],
    index: number,
    lookback: number,
    lookforward: number,
    includeWicks: boolean
  ): boolean {
    const currentCandle = candles[index]!
    const highPrice = includeWicks ? currentCandle.high : currentCandle.close

    // Check previous candles
    for (let i = 1; i <= lookback; i++) {
      const prevCandle = candles[index - i]!
      const prevHigh = includeWicks ? prevCandle.high : prevCandle.close
      if (prevHigh >= highPrice) {
        return false
      }
    }

    // Check following candles
    for (let i = 1; i <= lookforward; i++) {
      const nextCandle = candles[index + i]!
      const nextHigh = includeWicks ? nextCandle.high : nextCandle.close
      if (nextHigh >= highPrice) {
        return false
      }
    }

    return true
  }

  /**
   * Check if a candle is a swing low
   */
  private isSwingLow(
    candles: readonly Candle[],
    index: number,
    lookback: number,
    lookforward: number,
    includeWicks: boolean
  ): boolean {
    const currentCandle = candles[index]!
    const lowPrice = includeWicks ? currentCandle.low : currentCandle.close

    // Check previous candles
    for (let i = 1; i <= lookback; i++) {
      const prevCandle = candles[index - i]!
      const prevLow = includeWicks ? prevCandle.low : prevCandle.close
      if (prevLow <= lowPrice) {
        return false
      }
    }

    // Check following candles
    for (let i = 1; i <= lookforward; i++) {
      const nextCandle = candles[index + i]!
      const nextLow = includeWicks ? nextCandle.low : nextCandle.close
      if (nextLow <= lowPrice) {
        return false
      }
    }

    return true
  }

  /**
   * Calculate average price of surrounding candles
   */
  private calculateSurroundingAverage(
    candles: readonly Candle[],
    index: number,
    lookback: number,
    lookforward: number,
    priceType: 'high' | 'low',
    includeWicks: boolean
  ): number {
    let sum = 0
    let count = 0

    // Previous candles
    for (let i = 1; i <= lookback; i++) {
      const candle = candles[index - i]!
      if (priceType === 'high') {
        sum += includeWicks ? candle.high : candle.close
      } else {
        sum += includeWicks ? candle.low : candle.close
      }
      count++
    }

    // Following candles
    for (let i = 1; i <= lookforward; i++) {
      const candle = candles[index + i]!
      if (priceType === 'high') {
        sum += includeWicks ? candle.high : candle.close
      } else {
        sum += includeWicks ? candle.low : candle.close
      }
      count++
    }

    return sum / count
  }

  reset(): void {
    this.cache?.clear()
  }

  getMinimumCandles(): number {
    return this.config.lookback! + this.config.lookforward! + 1
  }

  /**
   * Static helper to detect swing points without creating an instance
   */
  static detect(
    candles: readonly Candle[],
    lookback = 5,
    lookforward = 5,
    minSwingPercent = 0.001,
    includeWicks = true
  ): readonly SwingPoint[] | null {
    const detector = new SwingDetector({
      lookback,
      lookforward,
      minSwingPercent,
      includeWicks,
      cacheEnabled: false,
    })
    return detector.calculate(candles)
  }

  /**
   * Find the most recent swing high
   */
  static findLastSwingHigh(
    candles: readonly Candle[],
    lookback = 5,
    lookforward = 5,
    minSwingPercent = 0.001,
    includeWicks = true
  ): SwingPoint | null {
    const swings = SwingDetector.detect(candles, lookback, lookforward, minSwingPercent, includeWicks)
    if (!swings) return null

    // Find the last swing high
    for (let i = swings.length - 1; i >= 0; i--) {
      if (swings[i]!.type === 'high') {
        return swings[i]!
      }
    }

    return null
  }

  /**
   * Find the most recent swing low
   */
  static findLastSwingLow(
    candles: readonly Candle[],
    lookback = 5,
    lookforward = 5,
    minSwingPercent = 0.001,
    includeWicks = true
  ): SwingPoint | null {
    const swings = SwingDetector.detect(candles, lookback, lookforward, minSwingPercent, includeWicks)
    if (!swings) return null

    // Find the last swing low
    for (let i = swings.length - 1; i >= 0; i--) {
      if (swings[i]!.type === 'low') {
        return swings[i]!
      }
    }

    return null
  }
}