import type { AgentSignal, MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import { AdaptiveBaseAgent } from './adaptive-base-agent'
import type { MarketRegime } from './market-regime-detector'
import type { Candle } from '@trdr/shared/src/types/market-data'
import type { Logger } from '@trdr/types'

interface AdaptiveVolumeProfileConfig {
  baseVolumeThreshold?: number
  basePriceResolution?: number
  baseLookbackPeriod?: number
  baseSupportResistanceThreshold?: number
}

/**
 * Adaptive Volume Profile Agent
 * 
 * A sophisticated volume analysis agent that builds price-volume profiles and
 * dynamically adapts to market conditions. This agent identifies high-volume
 * areas, support/resistance levels, and volume spikes with regime-aware interpretation.
 * 
 * ## Core Features:
 * - Dynamic volume profile construction with adaptive resolution
 * - Volume spike detection with significance scoring
 * - Support/resistance identification based on volume clusters
 * - Point of Control (POC) and value area analysis
 * - Price touch counting for level validation
 * 
 * ## Adaptive Parameters:
 * 
 * ### By Market Regime
 * 
 * **Trending Markets**:
 * - Volume Threshold: 120% of base (larger spikes needed)
 * - Price Resolution: 80% of base (fewer, clearer levels)
 * - Lookback Period: 120% of base (capture trend structure)
 * - S/R Threshold: 110% of base (stronger levels only)
 * 
 * **Ranging Markets**:
 * - Volume Threshold: 90% of base (sensitive to smaller moves)
 * - Price Resolution: 120% of base (detailed profile)
 * - Lookback Period: Standard
 * - S/R Threshold: 90% of base (more levels identified)
 * 
 * **Breakout Markets**:
 * - Volume Threshold: 80% of base (catch early moves)
 * - Price Resolution: Standard
 * - Lookback Period: 70% of base (recent activity focus)
 * - S/R Threshold: Standard
 * 
 * **Reversal Markets**:
 * - Volume Threshold: Standard
 * - Price Resolution: 130% of base (precise levels)
 * - Lookback Period: 90% of base
 * - S/R Threshold: 80% of base (developing levels)
 * 
 * ### Volatility Adjustments
 * - **High**: Spike detection period 15, threshold +10%
 * - **Low**: Spike detection period 25, threshold -10%
 * 
 * ### Volume Trend Adjustments
 * - **Increasing**: Threshold +15% (already elevated baseline)
 * - **Decreasing**: Threshold -15% (lower baseline)
 * 
 * ## Signal Generation:
 * 
 * ### Priority 1: Volume Spikes (>60% significance)
 * - Base confidence: 70% + significance × 20%
 * - **Trending**: +15% if aligned with trend
 * - **Breakout**: +10% for any spike
 * - **Reversal**: +5% if against trend
 * 
 * ### Priority 2: Support/Resistance Levels
 * - Must be within 0.3% of price
 * - Base confidence: 60% + strength × 25% + touches × 5%
 * - **Ranging**: +10% confidence (more reliable)
 * - **Strong Trends**: ×0.8 confidence (likely to break)
 * 
 * ### Priority 3: Volume Profile Patterns
 * 
 * **Low Volume Breakouts**:
 * - In breakout regime with directional spike
 * - 75% confidence
 * 
 * **High Volume Node Rejection**:
 * - In ranging markets with price reversal
 * - 70% confidence
 * 
 * ## Volume Profile Types:
 * - **POC**: Point of Control (highest volume price)
 * - **High**: >70% of POC volume
 * - **Low**: <30% of POC volume
 * - **Normal**: Between high and low
 * 
 * ## Support/Resistance Detection:
 * - Identifies levels with >threshold% of POC volume
 * - Counts historical price touches
 * - Ranks by proximity to current price
 * - Maximum 5 levels tracked
 */
export class AdaptiveVolumeProfileAgent extends AdaptiveBaseAgent {
  // Base configuration
  private readonly baseConfig: Required<AdaptiveVolumeProfileConfig>
  
  // Adaptive parameters
  private volumeThreshold: number
  private priceResolution: number
  private lookbackPeriod: number
  private supportResistanceThreshold: number
  private spikeDetectionPeriod: number
  
  // Volume tracking
  private priceHistory: number[] = []
  private readonly volumeProfile = new Map<number, number>()
  private readonly historyLength = 100
  
  constructor(metadata: AgentMetadata, logger?: Logger, config?: AdaptiveVolumeProfileConfig) {
    super(metadata, logger, {
      adaptationRate: 0.15,
      regimeMemory: 10,
      parameterBounds: {
        volumeThreshold: { min: 1.5, max: 3.5 },
        priceResolution: { min: 20, max: 100 },
        lookbackPeriod: { min: 50, max: 200 },
        supportResistanceThreshold: { min: 0.5, max: 0.9 }
      }
    })
    
    this.baseConfig = {
      baseVolumeThreshold: config?.baseVolumeThreshold ?? 2.0,
      basePriceResolution: config?.basePriceResolution ?? 50,
      baseLookbackPeriod: config?.baseLookbackPeriod ?? 100,
      baseSupportResistanceThreshold: config?.baseSupportResistanceThreshold ?? 0.7
    }
    
    // Initialize with base values
    this.volumeThreshold = this.baseConfig.baseVolumeThreshold
    this.priceResolution = this.baseConfig.basePriceResolution
    this.lookbackPeriod = this.baseConfig.baseLookbackPeriod
    this.supportResistanceThreshold = this.baseConfig.baseSupportResistanceThreshold
    this.spikeDetectionPeriod = 20
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Adaptive Volume Profile Agent initialized', {
      baseVolumeThreshold: this.baseConfig.baseVolumeThreshold,
      basePriceResolution: this.baseConfig.basePriceResolution,
      baseLookbackPeriod: this.baseConfig.baseLookbackPeriod
    })
  }
  
  protected async adaptParameters(regime: MarketRegime): Promise<void> {
    // Adapt based on market regime
    switch (regime.regime) {
      case 'trending':
        // In trending markets, look for larger volume spikes
        this.volumeThreshold = this.baseConfig.baseVolumeThreshold * 1.2
        // Use fewer price levels for clearer S/R
        this.priceResolution = Math.floor(this.baseConfig.basePriceResolution * 0.8)
        // Longer lookback to capture trend structure
        this.lookbackPeriod = Math.floor(this.baseConfig.baseLookbackPeriod * 1.2)
        // Higher threshold for S/R in trends
        this.supportResistanceThreshold = this.baseConfig.baseSupportResistanceThreshold * 1.1
        break
        
      case 'ranging':
        // In ranging markets, be sensitive to smaller volume changes
        this.volumeThreshold = this.baseConfig.baseVolumeThreshold * 0.9
        // More price levels for detailed profile
        this.priceResolution = Math.floor(this.baseConfig.basePriceResolution * 1.2)
        // Standard lookback
        this.lookbackPeriod = this.baseConfig.baseLookbackPeriod
        // Lower threshold to identify more S/R levels
        this.supportResistanceThreshold = this.baseConfig.baseSupportResistanceThreshold * 0.9
        break
        
      case 'breakout':
        // During breakouts, focus on significant volume
        this.volumeThreshold = this.baseConfig.baseVolumeThreshold * 0.8
        // Standard resolution
        this.priceResolution = this.baseConfig.basePriceResolution
        // Shorter lookback for recent activity
        this.lookbackPeriod = Math.floor(this.baseConfig.baseLookbackPeriod * 0.7)
        // Standard threshold
        this.supportResistanceThreshold = this.baseConfig.baseSupportResistanceThreshold
        break
        
      case 'reversal':
        // During reversals, look for volume divergences
        this.volumeThreshold = this.baseConfig.baseVolumeThreshold
        // High resolution for precise levels
        this.priceResolution = Math.floor(this.baseConfig.basePriceResolution * 1.3)
        // Medium lookback
        this.lookbackPeriod = Math.floor(this.baseConfig.baseLookbackPeriod * 0.9)
        // Lower threshold to catch developing levels
        this.supportResistanceThreshold = this.baseConfig.baseSupportResistanceThreshold * 0.8
        break
    }
    
    // Volatility adjustments
    if (regime.volatility === 'high') {
      // Shorter detection period in high volatility
      this.spikeDetectionPeriod = 15
      // Higher volume threshold
      this.volumeThreshold *= 1.1
    } else if (regime.volatility === 'low') {
      // Longer detection period in low volatility
      this.spikeDetectionPeriod = 25
      // Lower volume threshold
      this.volumeThreshold *= 0.9
    } else {
      this.spikeDetectionPeriod = 20
    }
    
    // Volume trend adjustments
    if (regime.volume === 'increasing') {
      // Raise thresholds when volume is already elevated
      this.volumeThreshold *= 1.15
    } else if (regime.volume === 'decreasing') {
      // Lower thresholds when volume is declining
      this.volumeThreshold *= 0.85
    }
    
    // Apply bounds
    this.volumeThreshold = Math.max(1.5, Math.min(3.5, this.volumeThreshold))
    this.priceResolution = Math.max(20, Math.min(100, this.priceResolution))
    this.lookbackPeriod = Math.max(50, Math.min(200, this.lookbackPeriod))
    this.supportResistanceThreshold = Math.max(0.5, Math.min(0.9, this.supportResistanceThreshold))
    
    this.logger?.debug('Volume Profile parameters adapted', {
      regime: regime.regime,
      volatility: regime.volatility,
      volume: regime.volume,
      volumeThreshold: this.volumeThreshold,
      priceResolution: this.priceResolution,
      lookbackPeriod: this.lookbackPeriod
    })
  }
  
  protected async performAdaptiveAnalysis(context: MarketContext, regime: MarketRegime): Promise<AgentSignal> {
    const { candles, currentPrice } = context
    
    if (candles.length < this.spikeDetectionPeriod) {
      return this.createSignal('hold', 0.3, 'Insufficient data for volume analysis')
    }
    
    // Update price history
    this.priceHistory.push(currentPrice)
    if (this.priceHistory.length > this.historyLength) {
      this.priceHistory.shift()
    }
    
    // Build volume profile
    const volumeProfile = this.buildVolumeProfile(candles)
    
    // Detect volume spikes
    const volumeSpike = this.detectVolumeSpike(candles)
    
    // Identify support/resistance levels
    const supportResistance = this.identifySupportResistance(volumeProfile, currentPrice)
    
    // Calculate average volume
    const recentCandles = (candles).slice(-this.spikeDetectionPeriod)
    const averageVolume = recentCandles.reduce((sum, c) => sum + c.volume, 0) / recentCandles.length
    
    // Generate regime-aware signal
    return this.synthesizeRegimeAwareSignal(
      volumeSpike,
      supportResistance,
      volumeProfile,
      currentPrice,
      averageVolume,
      regime
    )
  }
  
  private synthesizeRegimeAwareSignal(
    spike: {timestamp: number, volume: number, priceChange: number, significance: number, type: string} | null,
    supportResistance: Array<{level: number, strength: number, type: 'support' | 'resistance', touches: number}>,
    profile: Map<number, {volume: number, percentage: number, type: string}>,
    currentPrice: number,
    averageVolume: number,
    regime: MarketRegime
  ): AgentSignal {
    // Priority 1: Volume spike with regime alignment
    if (spike && spike.significance > 0.6) {
      let confidence = 0.7 + spike.significance * 0.2
      
      // Adjust confidence based on regime
      switch (regime.regime) {
        case 'trending':
          // Spikes in trend direction are stronger
          if ((regime.trend === 'bullish' && spike.type === 'bullish') ||
              (regime.trend === 'bearish' && spike.type === 'bearish')) {
            confidence = Math.min(0.95, confidence + 0.15)
          }
          break
        case 'breakout':
          // All spikes are significant during breakouts
          confidence = Math.min(0.9, confidence + 0.1)
          break
        case 'reversal':
          // Spikes against trend might signal reversal
          if ((regime.trend === 'bullish' && spike.type === 'bearish') ||
              (regime.trend === 'bearish' && spike.type === 'bullish')) {
            confidence = Math.min(0.85, confidence + 0.05)
          }
          break
      }
      
      if (spike.type === 'bullish') {
        return this.createAdaptiveSignal(
          'buy',
          confidence,
          `Bullish volume spike (${(spike.volume / averageVolume).toFixed(1)}x avg)`
        )
      } else if (spike.type === 'bearish') {
        return this.createAdaptiveSignal(
          'sell',
          confidence,
          `Bearish volume spike (${(spike.volume / averageVolume).toFixed(1)}x avg)`
        )
      }
    }
    
    // Priority 2: Support/Resistance with regime context
    const nearestLevel = supportResistance[0]
    if (nearestLevel) {
      const distance = Math.abs(nearestLevel.level - currentPrice) / currentPrice
      
      if (distance < 0.003) {
        let confidence = 0.6 + nearestLevel.strength * 0.25 + nearestLevel.touches * 0.05
        
        // Regime adjustments
        if (regime.regime === 'ranging') {
          // S/R is more reliable in ranges
          confidence = Math.min(0.9, confidence + 0.1)
        } else if (regime.regime === 'trending' && regime.momentum === 'strong') {
          // S/R might break in strong trends
          confidence *= 0.8
        }
        
        if (nearestLevel.type === 'support') {
          return this.createAdaptiveSignal(
            'buy',
            Math.min(0.85, confidence),
            `At volume support $${nearestLevel.level.toFixed(2)} (${regime.regime} market)`
          )
        } else {
          return this.createAdaptiveSignal(
            'sell',
            Math.min(0.85, confidence),
            `At volume resistance $${nearestLevel.level.toFixed(2)} (${regime.regime} market)`
          )
        }
      }
    }
    
    // Priority 3: Volume profile patterns
    const currentLevel = this.findNearestProfileLevel(profile, currentPrice)
    if (currentLevel && spike) {
      // Low volume area breakouts
      if (currentLevel.type === 'low' && spike.type !== 'neutral' && regime.regime === 'breakout') {
        const action = spike.type === 'bullish' ? 'buy' : 'sell'
        return this.createAdaptiveSignal(
          action,
          0.75,
          `${spike.type} breakout from low volume area`
        )
      }
      
      // High volume area rejection
      if (currentLevel.type === 'high' && regime.regime === 'ranging') {
        // Price likely to reverse from high volume nodes in ranges
        const priceMovement = this.calculateRecentPriceMovement()
        if (priceMovement > 0.01) {
          return this.createAdaptiveSignal(
            'sell',
            0.7,
            `Rejection from high volume node`
          )
        } else if (priceMovement < -0.01) {
          return this.createAdaptiveSignal(
            'buy',
            0.7,
            `Bounce from high volume node`
          )
        }
      }
    }
    
    // Default signal
    return this.createAdaptiveSignal(
      'hold',
      0.5,
      `No clear volume signal (${regime.regime} market)`
    )
  }
  
  private buildVolumeProfile(candles: readonly Candle[]): Map<number, {volume: number, percentage: number, type: 'poc' | 'high' | 'low' | 'normal'}> {
    const lookback = candles.slice(-this.lookbackPeriod)
    
    let minPrice = Infinity
    let maxPrice = -Infinity
    let totalVolume = 0
    
    for (const candle of lookback) {
      minPrice = Math.min(minPrice, candle.low)
      maxPrice = Math.max(maxPrice, candle.high)
      totalVolume += candle.volume
    }
    
    const priceStep = (maxPrice - minPrice) / this.priceResolution
    const volumeByPrice = new Map<number, number>()
    
    for (const candle of lookback) {
      if (!candle.volume || isNaN(candle.volume) || candle.volume <= 0) continue
      
      const candleRange = candle.high - candle.low
      if (candleRange === 0) continue
      
      const volumePerUnit = candle.volume / candleRange
      
      for (let price = candle.low; price <= candle.high; price += priceStep) {
        const bucketPrice = Math.round(price / priceStep) * priceStep
        const currentVolume = volumeByPrice.get(bucketPrice) || 0
        volumeByPrice.set(bucketPrice, currentVolume + volumePerUnit * priceStep)
      }
    }
    
    if (volumeByPrice.size === 0) {
      return new Map()
    }
    
    let pocPrice = 0
    let pocVolume = 0
    
    for (const [price, volume] of volumeByPrice.entries()) {
      if (volume > pocVolume) {
        pocVolume = volume
        pocPrice = price
      }
    }
    
    const profile = new Map<number, {volume: number, percentage: number, type: 'poc' | 'high' | 'low' | 'normal'}>()
    
    for (const [price, volume] of volumeByPrice.entries()) {
      const percentage = volume / pocVolume
      let type: 'poc' | 'high' | 'low' | 'normal' = 'normal'
      
      if (price === pocPrice) {
        type = 'poc'
      } else if (percentage > 0.7) {
        type = 'high'
      } else if (percentage < 0.3) {
        type = 'low'
      }
      
      profile.set(price, { volume, percentage, type })
    }
    
    return profile
  }
  
  private detectVolumeSpike(candles: readonly Candle[]): {timestamp: number, volume: number, priceChange: number, significance: number, type: 'bullish' | 'bearish' | 'neutral'} | null {
    if (candles.length < this.spikeDetectionPeriod + 1) return null
    
    const currentCandle = candles[candles.length - 1]!
    const previousCandles = candles.slice(-this.spikeDetectionPeriod - 1, -1)
    
    const avgVolume = previousCandles.reduce((sum, c) => sum + c.volume, 0) / previousCandles.length
    
    const volumeRatio = currentCandle.volume / avgVolume
    if (volumeRatio < this.volumeThreshold) return null
    
    const priceChange = (currentCandle.close - currentCandle.open) / currentCandle.open
    const candleRange = currentCandle.high - currentCandle.low
    const bodyRatio = Math.abs(currentCandle.close - currentCandle.open) / candleRange
    
    let type: 'bullish' | 'bearish' | 'neutral' = 'neutral'
    if (priceChange > 0.001 && bodyRatio > 0.6) {
      type = 'bullish'
    } else if (priceChange < -0.001 && bodyRatio > 0.6) {
      type = 'bearish'
    }
    
    const significance = Math.min(1, (volumeRatio - 1) / 3)
    
    return {
      timestamp: currentCandle.timestamp,
      volume: currentCandle.volume,
      priceChange,
      significance,
      type
    }
  }
  
  private identifySupportResistance(
    profile: Map<number, {volume: number, percentage: number, type: string}>,
    currentPrice: number
  ): Array<{level: number, strength: number, type: 'support' | 'resistance', touches: number}> {
    const levels: Array<{level: number, strength: number, type: 'support' | 'resistance', touches: number}> = []
    
    const sortedLevels = Array.from(profile.entries())
      .filter(([_, data]) => data.percentage >= this.supportResistanceThreshold)
      .sort((a, b) => b[1].volume - a[1].volume)
    
    for (const [price, data] of sortedLevels) {
      const distance = Math.abs(price - currentPrice) / currentPrice
      if (distance < 0.005) continue
      
      const type = price < currentPrice ? 'support' : 'resistance'
      const strength = data.percentage
      const touches = this.countPriceTouches(price, type)
      
      levels.push({
        level: price,
        strength,
        type,
        touches
      })
    }
    
    return levels.sort((a, b) => 
      Math.abs(a.level - currentPrice) - Math.abs(b.level - currentPrice)
    ).slice(0, 5)
  }
  
  private countPriceTouches(level: number, type: 'support' | 'resistance'): number {
    let touches = 0
    const tolerance = level * 0.002
    
    for (let i = 1; i < this.priceHistory.length; i++) {
      const prev = this.priceHistory[i - 1]!
      const curr = this.priceHistory[i]!
      
      if (type === 'support') {
        if (prev > level && curr <= level + tolerance && i + 1 < this.priceHistory.length) {
          const next = this.priceHistory[i + 1]!
          if (next > level) touches++
        }
      } else {
        if (prev < level && curr >= level - tolerance && i + 1 < this.priceHistory.length) {
          const next = this.priceHistory[i + 1]!
          if (next < level) touches++
        }
      }
    }
    
    return touches
  }
  
  private findNearestProfileLevel(
    profile: Map<number, {volume: number, percentage: number, type: string}>,
    currentPrice: number
  ): {volume: number, percentage: number, type: string} | null {
    let nearest = null
    let minDistance = Infinity
    
    for (const [price, data] of profile.entries()) {
      const distance = Math.abs(price - currentPrice)
      if (distance < minDistance) {
        minDistance = distance
        nearest = data
      }
    }
    
    return nearest
  }
  
  private calculateRecentPriceMovement(): number {
    if (this.priceHistory.length < 5) return 0
    
    const recent = this.priceHistory.slice(-5)
    const oldPrice = recent[0]!
    const newPrice = recent[recent.length - 1]!
    
    return (newPrice - oldPrice) / oldPrice
  }
  
  protected async onReset(): Promise<void> {
    this.priceHistory = []
    this.volumeProfile.clear()
    this.currentRegime = null
    this.regimeHistory = []
    
    // Reset to base parameters
    this.volumeThreshold = this.baseConfig.baseVolumeThreshold
    this.priceResolution = this.baseConfig.basePriceResolution
    this.lookbackPeriod = this.baseConfig.baseLookbackPeriod
    this.supportResistanceThreshold = this.baseConfig.baseSupportResistanceThreshold
    this.spikeDetectionPeriod = 20
  }
  
  // Override the base performAnalysis to use adaptive analysis
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    return this.analyze(context)
  }
}