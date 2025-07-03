import type { Candle } from '@trdr/shared'

/**
 * Represents the current market regime with multiple dimensions of analysis
 */
export interface MarketRegime {
  /** Overall market trend direction based on multi-timeframe analysis */
  trend: 'bullish' | 'bearish' | 'neutral'
  /** Current volatility level compared to recent history */
  volatility: 'high' | 'normal' | 'low'
  /** Strength of price momentum */
  momentum: 'strong' | 'moderate' | 'weak'
  /** Volume trend compared to recent average */
  volume: 'increasing' | 'stable' | 'decreasing'
  /** Overall market regime classification */
  regime: 'trending' | 'ranging' | 'breakout' | 'reversal'
  /** Confidence level in the regime detection (0-1) */
  confidence: number
}

/**
 * Market Regime Detector
 * 
 * Analyzes market conditions across multiple dimensions to determine the current
 * market regime. This enables agents to adapt their strategies dynamically.
 * 
 * The detector analyzes:
 * - **Trend**: Uses multiple moving averages (10, 20, 50 periods) to determine direction
 * - **Volatility**: Calculates using both standard deviation and ATR
 * - **Momentum**: Measures rate of change and consecutive price movements
 * - **Volume**: Tracks volume trends compared to recent averages
 * - **Overall Regime**: Synthesizes all factors to classify market behavior
 * 
 * Regime Classifications:
 * - **Trending**: Clear directional movement with aligned indicators
 * - **Ranging**: Sideways movement with mean-reverting behavior
 * - **Breakout**: High volatility with increasing volume
 * - **Reversal**: Weakening momentum with potential trend change
 */
export class MarketRegimeDetector {
  private readonly shortPeriod = 10
  private readonly mediumPeriod = 20
  private readonly longPeriod = 50
  
  detectRegime(candles: readonly Candle[]): MarketRegime {
    if (candles.length < this.longPeriod) {
      return {
        trend: 'neutral',
        volatility: 'normal',
        momentum: 'moderate',
        volume: 'stable',
        regime: 'ranging',
        confidence: 0.3
      }
    }
    
    // Calculate trend using multiple timeframes
    const trend = this.detectTrend(candles)
    
    // Calculate volatility
    const volatility = this.detectVolatility(candles)
    
    // Calculate momentum
    const momentum = this.detectMomentum(candles)
    
    // Calculate volume trend
    const volumeTrend = this.detectVolumeTrend(candles)
    
    // Determine overall regime
    const regime = this.determineRegime(trend, volatility, momentum, volumeTrend)
    
    // Calculate confidence based on agreement between indicators
    const confidence = this.calculateConfidence(trend, volatility, momentum, regime)
    
    return {
      trend: trend.direction,
      volatility: volatility.level,
      momentum: momentum.strength,
      volume: volumeTrend.direction,
      regime: regime.type,
      confidence
    }
  }
  
  private detectTrend(candles: readonly Candle[]): { direction: 'bullish' | 'bearish' | 'neutral', strength: number } {
    // Short-term trend
    const shortCandles = candles.slice(-this.shortPeriod)
    const shortMA = shortCandles.reduce((sum, c) => sum + c.close, 0) / shortCandles.length
    
    // Medium-term trend
    const mediumCandles = candles.slice(-this.mediumPeriod)
    const mediumMA = mediumCandles.reduce((sum, c) => sum + c.close, 0) / mediumCandles.length
    
    // Long-term trend
    const longCandles = candles.slice(-this.longPeriod)
    const longMA = longCandles.reduce((sum, c) => sum + c.close, 0) / longCandles.length
    
    const currentPrice = candles[candles.length - 1]!.close
    
    // Determine trend direction
    let bullishScore = 0
    let bearishScore = 0
    
    // Price vs MAs
    if (currentPrice > shortMA) bullishScore++
    else bearishScore++
    
    if (currentPrice > mediumMA) bullishScore++
    else bearishScore++
    
    if (currentPrice > longMA) bullishScore++
    else bearishScore++
    
    // MA alignment
    if (shortMA > mediumMA && mediumMA > longMA) bullishScore += 2
    else if (shortMA < mediumMA && mediumMA < longMA) bearishScore += 2
    
    // Price action
    const recentHigh = Math.max(...shortCandles.map(c => c.high))
    const recentLow = Math.min(...shortCandles.map(c => c.low))
    const pricePosition = (currentPrice - recentLow) / (recentHigh - recentLow)
    
    if (pricePosition > 0.7) bullishScore++
    else if (pricePosition < 0.3) bearishScore++
    
    // Determine direction
    let direction: 'bullish' | 'bearish' | 'neutral' = 'neutral'
    if (bullishScore > bearishScore + 1) direction = 'bullish'
    else if (bearishScore > bullishScore + 1) direction = 'bearish'
    
    const strength = Math.abs(bullishScore - bearishScore) / 6
    
    return { direction, strength }
  }
  
  private detectVolatility(candles: readonly Candle[]): { level: 'high' | 'normal' | 'low', value: number } {
    const returns: number[] = []
    
    for (let i = 1; i < candles.length; i++) {
      const ret = (candles[i]!.close - candles[i - 1]!.close) / candles[i - 1]!.close
      returns.push(ret)
    }
    
    // Calculate standard deviation
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length
    const stdDev = Math.sqrt(variance)
    
    // Calculate ATR-based volatility
    const atrValues: number[] = []
    for (let i = 1; i < candles.length; i++) {
      const high = candles[i]!.high
      const low = candles[i]!.low
      const prevClose = candles[i - 1]!.close
      
      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      )
      atrValues.push(tr)
    }
    
    const atr = atrValues.slice(-14).reduce((sum, v) => sum + v, 0) / 14
    const avgPrice = candles.slice(-14).reduce((sum, c) => sum + c.close, 0) / 14
    const atrPercent = atr / avgPrice
    
    // Determine volatility level
    let level: 'high' | 'normal' | 'low' = 'normal'
    if (stdDev > 0.03 || atrPercent > 0.04) level = 'high'
    else if (stdDev < 0.01 || atrPercent < 0.015) level = 'low'
    
    return { level, value: stdDev }
  }
  
  private detectMomentum(candles: readonly Candle[]): { strength: 'strong' | 'moderate' | 'weak', value: number } {
    // Rate of change
    const roc10 = (candles[candles.length - 1]!.close - candles[candles.length - 11]!.close) / candles[candles.length - 11]!.close
    const roc20 = candles.length > 20 ? (candles[candles.length - 1]!.close - candles[candles.length - 21]!.close) / candles[candles.length - 21]!.close : roc10
    
    // Consecutive moves
    let consecutiveUps = 0
    let consecutiveDowns = 0
    
    for (let i = candles.length - 5; i < candles.length; i++) {
      if (candles[i]!.close > candles[i - 1]!.close) {
        consecutiveUps++
        consecutiveDowns = 0
      } else {
        consecutiveDowns++
        consecutiveUps = 0
      }
    }
    
    // Determine momentum strength
    const momentumScore = Math.abs(roc10) + Math.abs(roc20) / 2 + Math.max(consecutiveUps, consecutiveDowns) * 0.01
    
    let strength: 'strong' | 'moderate' | 'weak' = 'moderate'
    if (momentumScore > 0.1) strength = 'strong'
    else if (momentumScore < 0.03) strength = 'weak'
    
    return { strength, value: momentumScore }
  }
  
  private detectVolumeTrend(candles: readonly Candle[]): { direction: 'increasing' | 'stable' | 'decreasing', ratio: number } {
    const recentVolumes = candles.slice(-10).map(c => c.volume)
    const olderVolumes = candles.slice(-20, -10).map(c => c.volume)
    
    const recentAvg = recentVolumes.reduce((sum, v) => sum + v, 0) / recentVolumes.length
    const olderAvg = olderVolumes.reduce((sum, v) => sum + v, 0) / olderVolumes.length
    
    const ratio = recentAvg / olderAvg
    
    let direction: 'increasing' | 'stable' | 'decreasing' = 'stable'
    if (ratio > 1.3) direction = 'increasing'
    else if (ratio < 0.7) direction = 'decreasing'
    
    return { direction, ratio }
  }
  
  private determineRegime(
    trend: { direction: string, strength: number },
    volatility: { level: string, value: number },
    momentum: { strength: string, value: number },
    volume: { direction: string, ratio: number }
  ): { type: 'trending' | 'ranging' | 'breakout' | 'reversal' } {
    // Strong trend with momentum = trending
    if (trend.strength > 0.6 && momentum.strength === 'strong') {
      return { type: 'trending' }
    }
    
    // High volatility with volume increase = potential breakout
    if (volatility.level === 'high' && volume.direction === 'increasing') {
      return { type: 'breakout' }
    }
    
    // Weak momentum with trend change = potential reversal
    if (momentum.strength === 'weak' && trend.strength < 0.3 && volatility.level !== 'low') {
      return { type: 'reversal' }
    }
    
    // Default to ranging
    return { type: 'ranging' }
  }
  
  private calculateConfidence(
    trend: { direction: string, strength: number },
    volatility: { level: string, value: number },
    momentum: { strength: string, value: number },
    regime: { type: string }
  ): number {
    let confidence = 0.5
    
    // Trend strength contributes to confidence
    confidence += trend.strength * 0.2
    
    // Clear momentum adds confidence
    if (momentum.strength === 'strong') confidence += 0.15
    else if (momentum.strength === 'weak') confidence -= 0.1
    
    // Normal volatility is more predictable
    if (volatility.level === 'normal') confidence += 0.1
    else if (volatility.level === 'high') confidence -= 0.1
    
    // Trending and ranging are more reliable than breakout/reversal
    if (regime.type === 'trending' || regime.type === 'ranging') confidence += 0.1
    
    return Math.max(0.2, Math.min(0.95, confidence))
  }
}