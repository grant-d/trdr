import type { Candle } from '@trdr/shared'

export interface NormalizationParams {
  method: 'z-score' | 'min-max' | 'log' | 'percent-change' | 'frac-diff' | 'none'
  // Z-score params
  mean?: number
  stdDev?: number
  // Min-max params
  min?: number
  max?: number
  // Log params
  logBase?: number
  // Percent change params
  referencePrice?: number
  // Fractional differentiation params
  d?: number // differentiation order (0 < d < 1)
  threshold?: number // weight cutoff threshold
  windowSize?: number // max window for weights
  // Common params
  priceFields?: ('open' | 'high' | 'low' | 'close')[]
}

export interface NormalizedCandle extends Candle {
  // Original prices preserved
  originalOpen: number
  originalHigh: number
  originalLow: number
  originalClose: number
}

export class PriceNormalizer {
  private readonly params: Required<NormalizationParams>
  private isInitialized = false
  private readonly fracDiffWeights: number[] = []
  private readonly fracDiffHistory = new Map<string, number[]>()
  
  constructor(params: NormalizationParams = { method: 'none' }) {
    this.params = {
      method: params.method,
      mean: params.mean ?? 0,
      stdDev: params.stdDev ?? 1,
      min: params.min ?? 0,
      max: params.max ?? 1,
      logBase: params.logBase ?? Math.E,
      referencePrice: params.referencePrice ?? 1,
      d: params.d ?? 0.4,
      threshold: params.threshold ?? 1e-4,
      windowSize: params.windowSize ?? 100,
      priceFields: params.priceFields ?? ['open', 'high', 'low', 'close']
    }
    
    // Pre-calculate fractional diff weights if needed
    if (this.params.method === 'frac-diff') {
      this.fracDiffWeights = this.calculateFracDiffWeights()
    }
  }
  
  /**
   * Initialize normalization parameters from a sample of candles
   */
  initialize(candles: Candle[]): void {
    if (candles.length === 0 || this.params.method === 'none') {
      this.isInitialized = true
      return
    }
    
    // Collect all price values
    const allPrices: number[] = []
    for (const candle of candles) {
      if (this.params.priceFields.includes('open')) allPrices.push(candle.open)
      if (this.params.priceFields.includes('high')) allPrices.push(candle.high)
      if (this.params.priceFields.includes('low')) allPrices.push(candle.low)
      if (this.params.priceFields.includes('close')) allPrices.push(candle.close)
    }
    
    switch (this.params.method) {
      case 'z-score':
        this.params.mean = this.calculateMean(allPrices)
        this.params.stdDev = this.calculateStdDev(allPrices, this.params.mean)
        if (this.params.stdDev === 0) this.params.stdDev = 1 // Prevent division by zero
        break
        
      case 'min-max':
        this.params.min = Math.min(...allPrices)
        this.params.max = Math.max(...allPrices)
        if (this.params.min === this.params.max) {
          this.params.max = this.params.min + 1 // Prevent division by zero
        }
        break
        
      case 'log':
        // No initialization needed for log normalization
        break
        
      case 'percent-change':
        // Use the first candle's close as reference
        this.params.referencePrice = candles[0]?.close ?? 1
        break
        
      case 'frac-diff':
        // Initialize history for each price field
        for (const field of this.params.priceFields) {
          this.fracDiffHistory.set(field, [])
        }
        // Pre-populate history with initial prices
        for (const candle of candles.slice(0, Math.min(this.params.windowSize, candles.length))) {
          if (this.params.priceFields.includes('open')) {
            this.fracDiffHistory.get('open')!.push(candle.open)
          }
          if (this.params.priceFields.includes('high')) {
            this.fracDiffHistory.get('high')!.push(candle.high)
          }
          if (this.params.priceFields.includes('low')) {
            this.fracDiffHistory.get('low')!.push(candle.low)
          }
          if (this.params.priceFields.includes('close')) {
            this.fracDiffHistory.get('close')!.push(candle.close)
          }
        }
        break
    }
    
    this.isInitialized = true
  }
  
  /**
   * Normalize a single price value
   */
  normalizePrice(price: number, field?: string): number {
    switch (this.params.method) {
      case 'z-score':
        return (price - this.params.mean) / this.params.stdDev
        
      case 'min-max':
        return (price - this.params.min) / (this.params.max - this.params.min)
        
      case 'log':
        return Math.log(price) / Math.log(this.params.logBase)
        
      case 'percent-change':
        return (price - this.params.referencePrice) / this.params.referencePrice
        
      case 'frac-diff':
        // For frac-diff, we need the historical context
        // This is handled in normalizeCandle instead
        return price // Return original if called directly
        
      case 'none':
      default:
        return price
    }
  }
  
  /**
   * Denormalize a price back to original scale
   */
  denormalizePrice(normalizedPrice: number): number {
    switch (this.params.method) {
      case 'z-score':
        return normalizedPrice * this.params.stdDev + this.params.mean
        
      case 'min-max':
        return normalizedPrice * (this.params.max - this.params.min) + this.params.min
        
      case 'log':
        return Math.pow(this.params.logBase, normalizedPrice)
        
      case 'percent-change':
        return normalizedPrice * this.params.referencePrice + this.params.referencePrice
        
      case 'frac-diff':
        // Fractional differentiation is not directly reversible
        // Return the value as-is (caller should use originalPrice)
        return normalizedPrice
        
      case 'none':
      default:
        return normalizedPrice
    }
  }
  
  /**
   * Normalize a candle, preserving original prices
   */
  normalizeCandle(candle: Candle): NormalizedCandle {
    if (!this.isInitialized) {
      throw new Error('Normalizer not initialized. Call initialize() with sample data first.')
    }
    
    if (this.params.method === 'none') {
      return {
        ...candle,
        originalOpen: candle.open,
        originalHigh: candle.high,
        originalLow: candle.low,
        originalClose: candle.close
      }
    }
    
    // Handle fractional differentiation specially
    if (this.params.method === 'frac-diff') {
      // Update history with new prices
      if (this.params.priceFields.includes('open')) {
        const openHistory = this.fracDiffHistory.get('open')!
        openHistory.push(candle.open)
        if (openHistory.length > this.params.windowSize) openHistory.shift()
      }
      if (this.params.priceFields.includes('high')) {
        const highHistory = this.fracDiffHistory.get('high')!
        highHistory.push(candle.high)
        if (highHistory.length > this.params.windowSize) highHistory.shift()
      }
      if (this.params.priceFields.includes('low')) {
        const lowHistory = this.fracDiffHistory.get('low')!
        lowHistory.push(candle.low)
        if (lowHistory.length > this.params.windowSize) lowHistory.shift()
      }
      if (this.params.priceFields.includes('close')) {
        const closeHistory = this.fracDiffHistory.get('close')!
        closeHistory.push(candle.close)
        if (closeHistory.length > this.params.windowSize) closeHistory.shift()
      }
      
      return {
        timestamp: candle.timestamp,
        open: this.params.priceFields.includes('open') 
          ? this.applyFracDiff(this.fracDiffHistory.get('open')!)
          : candle.open,
        high: this.params.priceFields.includes('high')
          ? this.applyFracDiff(this.fracDiffHistory.get('high')!)
          : candle.high,
        low: this.params.priceFields.includes('low')
          ? this.applyFracDiff(this.fracDiffHistory.get('low')!)
          : candle.low,
        close: this.params.priceFields.includes('close')
          ? this.applyFracDiff(this.fracDiffHistory.get('close')!)
          : candle.close,
        volume: candle.volume,
        originalOpen: candle.open,
        originalHigh: candle.high,
        originalLow: candle.low,
        originalClose: candle.close
      }
    }
    
    return {
      timestamp: candle.timestamp,
      open: this.params.priceFields.includes('open') 
        ? this.normalizePrice(candle.open) 
        : candle.open,
      high: this.params.priceFields.includes('high')
        ? this.normalizePrice(candle.high)
        : candle.high,
      low: this.params.priceFields.includes('low')
        ? this.normalizePrice(candle.low)
        : candle.low,
      close: this.params.priceFields.includes('close')
        ? this.normalizePrice(candle.close)
        : candle.close,
      volume: candle.volume,
      originalOpen: candle.open,
      originalHigh: candle.high,
      originalLow: candle.low,
      originalClose: candle.close
    }
  }
  
  /**
   * Normalize an array of candles
   */
  normalizeCandles(candles: Candle[]): NormalizedCandle[] {
    if (!this.isInitialized) {
      this.initialize(candles)
    }
    
    return candles.map(candle => this.normalizeCandle(candle))
  }
  
  /**
   * Denormalize a candle back to original prices
   */
  denormalizeCandle(candle: NormalizedCandle): Candle {
    if (this.params.method === 'none') {
      return {
        timestamp: candle.timestamp,
        open: candle.originalOpen,
        high: candle.originalHigh,
        low: candle.originalLow,
        close: candle.originalClose,
        volume: candle.volume
      }
    }
    
    return {
      timestamp: candle.timestamp,
      open: this.params.priceFields.includes('open') 
        ? this.denormalizePrice(candle.open) 
        : candle.originalOpen,
      high: this.params.priceFields.includes('high')
        ? this.denormalizePrice(candle.high)
        : candle.originalHigh,
      low: this.params.priceFields.includes('low')
        ? this.denormalizePrice(candle.low)
        : candle.originalLow,
      close: this.params.priceFields.includes('close')
        ? this.denormalizePrice(candle.close)
        : candle.originalClose,
      volume: candle.volume
    }
  }
  
  /**
   * Get normalization parameters for serialization/persistence
   */
  getParams(): Required<NormalizationParams> {
    return { ...this.params }
  }
  
  /**
   * Create a normalizer from saved parameters
   */
  static fromParams(params: Required<NormalizationParams>): PriceNormalizer {
    const normalizer = new PriceNormalizer(params)
    normalizer.isInitialized = true
    return normalizer
  }
  
  /**
   * Calculate mean of an array
   */
  private calculateMean(values: number[]): number {
    if (values.length === 0) return 0
    return values.reduce((sum, val) => sum + val, 0) / values.length
  }
  
  /**
   * Calculate standard deviation of an array
   */
  private calculateStdDev(values: number[], mean: number): number {
    if (values.length === 0) return 1
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
    return Math.sqrt(variance)
  }
  
  /**
   * Calculate fractional differentiation weights
   */
  private calculateFracDiffWeights(): number[] {
    const weights: number[] = [1.0]
    let k = 1
    
    while (k < this.params.windowSize) {
      const prevWeight = weights[k - 1]
      if (!prevWeight) break
      
      const w_k = -prevWeight * (this.params.d - k + 1) / k
      if (Math.abs(w_k) < this.params.threshold) {
        break
      }
      weights.push(w_k)
      k++
    }
    
    return weights.reverse() // Reverse so most recent has highest index
  }
  
  /**
   * Apply fractional differentiation to a price series
   */
  private applyFracDiff(values: number[]): number {
    if (values.length === 0) return 0
    
    // Use pre-calculated weights
    const effectiveLength = Math.min(values.length, this.fracDiffWeights.length)
    let result = 0
    
    // Apply weights to values (most recent value gets weight[0])
    for (let i = 0; i < effectiveLength; i++) {
      const valueIdx = values.length - 1 - i
      const weightIdx = this.fracDiffWeights.length - 1 - i
      const value = values[valueIdx]
      const weight = this.fracDiffWeights[weightIdx]
      if (value !== undefined && weight !== undefined) {
        result += value * weight
      }
    }
    
    return result
  }
  
  /**
   * Normalize a price change (for stop loss, take profit, etc.)
   */
  normalizePriceChange(priceChange: number, referencePrice: number): number {
    switch (this.params.method) {
      case 'z-score':
        // Price changes in z-score are already in standard deviations
        return priceChange / this.params.stdDev
        
      case 'min-max':
        // Convert to normalized scale
        return priceChange / (this.params.max - this.params.min)
        
      case 'log':
        // Log of ratio
        return Math.log((referencePrice + priceChange) / referencePrice) / Math.log(this.params.logBase)
        
      case 'percent-change':
        // Already in percent terms
        return priceChange / this.params.referencePrice
        
      case 'none':
      default:
        return priceChange
    }
  }
  
  /**
   * Denormalize a price change back to original scale
   */
  denormalizePriceChange(normalizedChange: number, referencePrice: number): number {
    switch (this.params.method) {
      case 'z-score':
        return normalizedChange * this.params.stdDev
        
      case 'min-max':
        return normalizedChange * (this.params.max - this.params.min)
        
      case 'log':
        // Inverse of log transformation
        const ratio = Math.pow(this.params.logBase, normalizedChange)
        return referencePrice * (ratio - 1)
        
      case 'percent-change':
        return normalizedChange * this.params.referencePrice
        
      case 'none':
      default:
        return normalizedChange
    }
  }
  
  /**
   * Check if normalizer is properly initialized
   */
  get initialized(): boolean {
    return this.isInitialized
  }
  
  /**
   * Get a human-readable description of the normalization
   */
  describe(): string {
    switch (this.params.method) {
      case 'z-score':
        return `Z-score normalization (μ=${this.params.mean.toFixed(2)}, σ=${this.params.stdDev.toFixed(2)})`
        
      case 'min-max':
        return `Min-max normalization ([${this.params.min.toFixed(2)}, ${this.params.max.toFixed(2)}] → [0, 1])`
        
      case 'log':
        return `Log normalization (base ${this.params.logBase})`
        
      case 'percent-change':
        return `Percent change from ${this.params.referencePrice.toFixed(2)}`
        
      case 'frac-diff':
        return `Fractional differentiation (d=${this.params.d}, window=${this.params.windowSize})`
        
      case 'none':
        return 'No normalization'
        
      default:
        return 'Unknown normalization'
    }
  }
}

/**
 * Factory function to create appropriate normalizer based on data characteristics
 */
export function createAutoNormalizer(candles: Candle[]): PriceNormalizer {
  if (candles.length === 0) {
    return new PriceNormalizer({ method: 'none' })
  }
  
  // Calculate price statistics
  const prices = candles.map(c => c.close)
  const mean = prices.reduce((sum, p) => sum + p, 0) / prices.length
  const variance = prices.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / prices.length
  const stdDev = Math.sqrt(variance)
  const coefficientOfVariation = stdDev / mean
  
  // Detect price scale
  const minPrice = Math.min(...prices)
  const maxPrice = Math.max(...prices)
  const priceRange = maxPrice - minPrice
  const relativeRange = priceRange / mean
  
  // Calculate autocorrelation to detect if fractional diff would help
  const autocorrelation = calculateAutocorrelation(prices, 1)
  
  // Choose normalization method based on data characteristics
  let method: NormalizationParams['method']
  
  if (autocorrelation > 0.95 && candles.length > 100) {
    // High autocorrelation - fractional differentiation ideal
    // Choose d based on how much stationarity we need
    const d = autocorrelation > 0.98 ? 0.5 : 0.3
    const normalizer = new PriceNormalizer({ 
      method: 'frac-diff',
      d,
      windowSize: Math.min(100, Math.floor(candles.length / 2))
    })
    normalizer.initialize(candles)
    return normalizer
  } else if (coefficientOfVariation > 0.5 || relativeRange > 2) {
    // High volatility or large range - use log normalization
    method = 'log'
  } else if (mean > 10000 || mean < 0.01) {
    // Extreme price scales - use z-score
    method = 'z-score'
  } else if (relativeRange < 0.1) {
    // Small range - use min-max to spread out values
    method = 'min-max'
  } else {
    // Default to z-score for general use
    method = 'z-score'
  }
  
  const normalizer = new PriceNormalizer({ method })
  normalizer.initialize(candles)
  
  return normalizer
}

/**
 * Calculate autocorrelation for a series at given lag
 */
function calculateAutocorrelation(series: number[], lag: number): number {
  if (series.length < lag + 1) return 0
  
  const mean = series.reduce((sum, val) => sum + val, 0) / series.length
  
  let numerator = 0
  let denominator = 0
  
  for (let i = lag; i < series.length; i++) {
    const val1 = series[i]
    const val2 = series[i - lag]
    if (val1 !== undefined && val2 !== undefined) {
      const deviation1 = val1 - mean
      const deviation2 = val2 - mean
      numerator += deviation1 * deviation2
    }
  }
  
  for (let i = 0; i < series.length; i++) {
    const val = series[i]
    if (val !== undefined) {
      const deviation = val - mean
      denominator += deviation * deviation
    }
  }
  
  return denominator === 0 ? 0 : numerator / denominator
}