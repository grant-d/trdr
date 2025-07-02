import type { Candle } from '@trdr/shared'
import { ATRIndicator } from './atr'
import { BollingerBandsIndicator } from './bollinger-bands'
import { IndicatorCache } from './cache'
import { EMAIndicator } from './ema'
import type {
  ATRConfig,
  BollingerBandsConfig,
  BollingerBandsResult,
  IIndicatorCalculator,
  IndicatorResult,
  MACDConfig,
  MACDResult,
  RSIConfig,
  SwingDetectionConfig,
  SwingPoint,
  VWAPConfig,
} from './interfaces'
import { MACDIndicator } from './macd'
import { RSIIndicator } from './rsi'
import { SMAIndicator } from './sma'
import { SwingDetector } from './swing-detector'
import { VWAPIndicator } from './vwap'

/**
 * Main calculator for technical indicators
 *
 * Provides a unified interface for calculating various technical indicators
 * with optimized caching and batch calculations.
 */
export class IndicatorCalculator implements IIndicatorCalculator {
  private readonly cache: IndicatorCache

  constructor(cacheSize = 1000) {
    this.cache = new IndicatorCache(cacheSize)
  }

  /**
   * Calculate Simple Moving Average
   */
  sma(candles: readonly Candle[], period: number): IndicatorResult | null {
    const indicator = new SMAIndicator({ period, cacheEnabled: true })
    return indicator.calculate(candles)
  }

  /**
   * Calculate Exponential Moving Average
   */
  ema(candles: readonly Candle[], period: number): IndicatorResult | null {
    const indicator = new EMAIndicator({ period, cacheEnabled: true })
    return indicator.calculate(candles)
  }

  /**
   * Calculate MACD
   */
  macd(candles: readonly Candle[], config?: Partial<MACDConfig>): MACDResult | null {
    const indicator = new MACDIndicator(config)
    return indicator.calculate(candles)
  }

  /**
   * Calculate ATR (Average True Range)
   */
  atr(
    candles: readonly Candle[],
    period = 14,
    config?: Partial<ATRConfig>
  ): IndicatorResult | null {
    const indicator = new ATRIndicator({ period, ...config, cacheEnabled: true })
    return indicator.calculate(candles)
  }

  /**
   * Calculate Bollinger Bands
   */
  bollingerBands(
    candles: readonly Candle[],
    config?: Partial<BollingerBandsConfig>
  ): BollingerBandsResult | null {
    const indicator = new BollingerBandsIndicator({ ...config, cacheEnabled: true })
    return indicator.calculate(candles)
  }

  /**
   * Calculate RSI (Relative Strength Index)
   */
  rsi(
    candles: readonly Candle[],
    period = 14,
    config?: Partial<RSIConfig>
  ): IndicatorResult | null {
    const indicator = new RSIIndicator({ period, ...config, cacheEnabled: true })
    return indicator.calculate(candles)
  }

  /**
   * Calculate VWAP (Volume Weighted Average Price)
   */
  vwap(candles: readonly Candle[], config?: Partial<VWAPConfig>): IndicatorResult | null {
    const indicator = new VWAPIndicator({ ...config, cacheEnabled: true })
    return indicator.calculate(candles)
  }

  /**
   * Detect swing highs and lows
   */
  swingHighLow(
    candles: readonly Candle[],
    config?: Partial<SwingDetectionConfig>
  ): readonly SwingPoint[] {
    const detector = new SwingDetector({ ...config, cacheEnabled: true })
    return detector.calculate(candles) ?? []
  }

  /**
   * Calculate all standard indicators in a single pass
   */
  calculateAll(candles: readonly Candle[]): ReturnType<IIndicatorCalculator['calculateAll']> {
    const results: ReturnType<IIndicatorCalculator['calculateAll']> = {}

    // Calculate trend indicators
    results.sma20 = this.sma(candles, 20) ?? undefined
    results.sma50 = this.sma(candles, 50) ?? undefined
    results.ema20 = this.ema(candles, 20) ?? undefined

    // Calculate momentum indicators
    results.rsi = this.rsi(candles) ?? undefined
    results.macd = this.macd(candles) ?? undefined

    // Calculate volatility indicators
    results.atr = this.atr(candles) ?? undefined
    results.bollingerBands = this.bollingerBands(candles) ?? undefined

    // Calculate volume indicators
    results.vwap = this.vwap(candles) ?? undefined

    return results
  }

  /**
   * Clear the indicator cache
   */
  clearCache(): void {
    this.cache.clear()
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { hits: number; misses: number; evictions: number; size: number } {
    return this.cache.getStats()
  }
}
