import type { Candle, EpochDate } from '@trdr/shared'

/**
 * Result of an indicator calculation
 */
export interface IndicatorResult {
  readonly value: number
  readonly timestamp: EpochDate // Unix timestamp in milliseconds
}

/**
 * Result with multiple values (e.g., Bollinger Bands)
 */
export interface MultiValueIndicatorResult extends IndicatorResult {
  readonly values: Record<string, number>
}

/**
 * MACD specific result
 */
export interface MACDResult extends IndicatorResult {
  readonly macd: number
  readonly signal: number
  readonly histogram: number
}

/**
 * Bollinger Bands result
 */
export interface BollingerBandsResult extends IndicatorResult {
  readonly upper: number
  readonly middle: number
  readonly lower: number
  readonly bandwidth: number
  readonly percentB: number
}

/**
 * Swing point detection result
 */
export interface SwingPoint {
  readonly type: 'high' | 'low'
  readonly price: number
  readonly timestamp: EpochDate // Unix timestamp in milliseconds
  readonly strength: number // 1-5, higher is stronger
  readonly index: number
}

/**
 * Base configuration for indicators
 */
export interface IndicatorConfig {
  readonly cacheEnabled?: boolean
  readonly cacheSize?: number
}

/**
 * Configuration for moving averages
 */
export interface MovingAverageConfig extends IndicatorConfig {
  readonly period: number
}

/**
 * Configuration for MACD
 */
export interface MACDConfig extends IndicatorConfig {
  readonly fastPeriod?: number
  readonly slowPeriod?: number
  readonly signalPeriod?: number
}

/**
 * Configuration for RSI
 */
export interface RSIConfig extends IndicatorConfig {
  readonly period?: number
  readonly overbought?: number
  readonly oversold?: number
  readonly smoothing?: 'wilder' | 'sma'
}

/**
 * Configuration for Bollinger Bands
 */
export interface BollingerBandsConfig extends IndicatorConfig {
  readonly period?: number
  readonly standardDeviations?: number
  readonly stdDevMultiplier?: number
}

/**
 * Configuration for ATR
 */
export interface ATRConfig extends IndicatorConfig {
  readonly period?: number
  readonly smoothing?: 'wilder' | 'sma'
}

/**
 * Configuration for VWAP
 */
export interface VWAPConfig extends IndicatorConfig {
  readonly anchorTime?: 'session' | 'week' | 'month'
}

/**
 * Configuration for Swing Detection
 */
export interface SwingDetectionConfig extends IndicatorConfig {
  readonly lookback?: number
  readonly lookforward?: number
  readonly minSwingPercent?: number
  readonly includeWicks?: boolean
}

/**
 * Base interface for all indicators
 */
export interface IIndicator<TConfig extends IndicatorConfig = IndicatorConfig> {
  readonly name: string
  readonly config: TConfig

  /**
   * Calculate indicator value for latest candle
   */
  calculate(candles: readonly Candle[]): IndicatorResult | null

  /**
   * Calculate indicator values for all candles
   */
  calculateAll(candles: readonly Candle[]): readonly IndicatorResult[]

  /**
   * Reset the indicator state and cache
   */
  reset(): void

  /**
   * Get minimum number of candles required
   */
  getMinimumCandles(): number
}

/**
 * Interface for indicators with multiple outputs
 */
export interface IMultiValueIndicator<
  TResult extends MultiValueIndicatorResult | IndicatorResult,
  TConfig extends IndicatorConfig = IndicatorConfig,
> extends IIndicator<TConfig> {
  calculate(candles: readonly Candle[]): TResult | null
  calculateAll(candles: readonly Candle[]): readonly TResult[]
}

/**
 * Interface for the main indicator calculator
 */
export interface IIndicatorCalculator {
  // Trend indicators
  sma(candles: readonly Candle[], period: number): IndicatorResult | null
  ema(candles: readonly Candle[], period: number): IndicatorResult | null
  macd(candles: readonly Candle[], config?: Partial<MACDConfig>): MACDResult | null

  // Volatility indicators
  atr(candles: readonly Candle[], period?: number): IndicatorResult | null
  bollingerBands(
    candles: readonly Candle[],
    config?: Partial<BollingerBandsConfig>
  ): BollingerBandsResult | null

  // Momentum indicators
  rsi(candles: readonly Candle[], period?: number): IndicatorResult | null

  // Volume indicators
  vwap(candles: readonly Candle[], config?: Partial<VWAPConfig>): IndicatorResult | null

  // Price action
  swingHighLow(
    candles: readonly Candle[],
    config?: Partial<SwingDetectionConfig>
  ): readonly SwingPoint[]

  // Batch calculations
  calculateAll(candles: readonly Candle[]): {
    sma20?: IndicatorResult
    sma50?: IndicatorResult
    ema20?: IndicatorResult
    rsi?: IndicatorResult
    macd?: MACDResult
    atr?: IndicatorResult
    bollingerBands?: BollingerBandsResult
    vwap?: IndicatorResult
  }
}

/**
 * Cache entry for indicator results
 */
export interface CacheEntry<T> {
  readonly key: string
  readonly value: T
  readonly timestamp: EpochDate
  readonly candleCount: number
}

/**
 * Interface for indicator cache
 */
export interface IIndicatorCache {
  get<T>(key: string): T | undefined
  set<T>(key: string, value: T, candleCount: number): void
  clear(): void
  invalidate(pattern?: string): void
  getSize(): number
  getStats(): {
    hits: number
    misses: number
    evictions: number
    size: number
  }
}
