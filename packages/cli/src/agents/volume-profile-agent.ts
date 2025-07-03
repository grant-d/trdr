import { BaseAgent } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import type { Candle } from '@trdr/shared/src/types/market-data'

interface VolumeProfileConfig {
  volumeThreshold?: number // Multiplier for average volume to detect spikes
  priceResolution?: number // Number of price levels for volume profile
  lookbackPeriod?: number // Candles to analyze for volume profile
  supportResistanceThreshold?: number // Min volume % to be considered S/R
  spikeDetectionPeriod?: number // Period for volume moving average
}

interface VolumeLevel {
  price: number
  volume: number
  percentage: number
  type: 'poc' | 'high' | 'low' | 'normal' // Point of Control, High Volume Node, etc.
}

interface VolumeSpike {
  timestamp: number
  volume: number
  priceChange: number
  significance: number // 0-1 scale
  type: 'bullish' | 'bearish' | 'neutral'
}

interface SupportResistance {
  level: number
  strength: number // 0-1 based on volume concentration
  type: 'support' | 'resistance'
  touches: number
}

export class VolumeProfileAgent extends BaseAgent {
  // Configuration
  protected readonly config: Required<VolumeProfileConfig>
  
  // Volume tracking
  private priceHistory: number[] = []
  private readonly volumeProfile = new Map<number, number>()
  private readonly historyLength = 100 // Keep last 100 price points for touch analysis
  
  // Analysis cache
  private lastAnalysis: {
    timestamp: number
    volumeSpike: VolumeSpike | null
    volumeProfile: VolumeLevel[]
    supportResistance: SupportResistance[]
    averageVolume: number
  } | null = null
  
  constructor(metadata: any, logger?: any, config?: VolumeProfileConfig) {
    super(metadata, logger)
    
    this.config = {
      volumeThreshold: config?.volumeThreshold ?? 2.0, // 2x average = spike
      priceResolution: config?.priceResolution ?? 50, // 50 price levels
      lookbackPeriod: config?.lookbackPeriod ?? 100, // 100 candles
      supportResistanceThreshold: config?.supportResistanceThreshold ?? 0.7, // 70% of POC
      spikeDetectionPeriod: config?.spikeDetectionPeriod ?? 20 // 20 period MA
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Volume Profile Agent initialized', {
      volumeThreshold: this.config.volumeThreshold,
      priceResolution: this.config.priceResolution,
      lookbackPeriod: this.config.lookbackPeriod,
      supportResistanceThreshold: this.config.supportResistanceThreshold
    })
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    try {
      const { candles, currentPrice } = context
      
      // Check minimum data requirements
      if (candles.length < this.config.spikeDetectionPeriod) {
        return this.createSignal('hold', 0.3, 'Insufficient data for volume analysis')
      }
      
      // Build volume profile
      const volumeProfile = this.buildVolumeProfile(candles as Candle[])
      
      // Detect volume spikes
      const volumeSpike = this.detectVolumeSpike(candles as Candle[])
      
      // Identify support/resistance levels
      const supportResistance = this.identifySupportResistance(volumeProfile, currentPrice)
      
      // Update price history for touch analysis
      this.priceHistory.push(currentPrice)
      if (this.priceHistory.length > this.historyLength) {
        this.priceHistory.shift()
      }
      
      // Calculate average volume
      const recentCandles = (candles as Candle[]).slice(-this.config.spikeDetectionPeriod)
      const averageVolume = recentCandles.reduce((sum, c) => sum + (isNaN(c.volume) ? 0 : c.volume), 0) / recentCandles.length
      
      // Check for invalid average volume
      if (isNaN(averageVolume) || averageVolume === 0) {
        throw new Error('Invalid volume data - all volumes are NaN or zero')
      }
      
      // Cache analysis - convert Map to array format
      const profileArray = Array.from(volumeProfile.entries()).map(([price, data]) => ({
        price,
        volume: data.volume,
        percentage: data.percentage,
        type: data.type
      }))
      
      this.lastAnalysis = {
        timestamp: Date.now(),
        volumeSpike,
        volumeProfile: profileArray,
        supportResistance,
        averageVolume
      }
      
      // Generate trading signal
      return this.synthesizeSignal(volumeSpike, supportResistance, volumeProfile, currentPrice, averageVolume)
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.logger?.error('Volume profile analysis failed', { error: errorMessage })
      return this.createSignal('hold', 0.2, `Volume analysis error: ${errorMessage}`)
    }
  }
  
  /**
   * Build volume profile from candle data
   */
  private buildVolumeProfile(candles: readonly Candle[]): Map<number, {volume: number, percentage: number, type: 'poc' | 'high' | 'low' | 'normal'}> {
    const lookback = candles.slice(-this.config.lookbackPeriod)
    
    // Find price range
    let minPrice = Infinity
    let maxPrice = -Infinity
    let totalVolume = 0
    
    for (const candle of lookback) {
      minPrice = Math.min(minPrice, candle.low)
      maxPrice = Math.max(maxPrice, candle.high)
      totalVolume += candle.volume
    }
    
    // Create price buckets
    const priceStep = (maxPrice - minPrice) / this.config.priceResolution
    const volumeByPrice = new Map<number, number>()
    
    // Distribute volume across price levels
    for (const candle of lookback) {
      // Skip candles with invalid volume
      if (!candle.volume || isNaN(candle.volume) || candle.volume <= 0) continue
      
      const candleRange = candle.high - candle.low
      if (candleRange === 0) continue
      
      // Distribute volume proportionally across the candle's range
      const volumePerUnit = candle.volume / candleRange
      
      for (let price = candle.low; price <= candle.high; price += priceStep) {
        const bucketPrice = Math.round(price / priceStep) * priceStep
        const currentVolume = volumeByPrice.get(bucketPrice) || 0
        volumeByPrice.set(bucketPrice, currentVolume + volumePerUnit * priceStep)
      }
    }
    
    // If no valid volume data, return empty profile
    if (volumeByPrice.size === 0) {
      return new Map()
    }
    
    // Find POC (Point of Control - highest volume price)
    let pocPrice = 0
    let pocVolume = 0
    
    for (const [price, volume] of volumeByPrice.entries()) {
      if (volume > pocVolume) {
        pocVolume = volume
        pocPrice = price
      }
    }
    
    // Calculate percentages and classify levels
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
  
  /**
   * Detect significant volume spikes
   */
  private detectVolumeSpike(candles: readonly Candle[]): VolumeSpike | null {
    if (candles.length < this.config.spikeDetectionPeriod + 1) return null
    
    const currentCandle = candles[candles.length - 1]!
    const previousCandles = candles.slice(-this.config.spikeDetectionPeriod - 1, -1)
    
    // Calculate average volume
    const avgVolume = previousCandles.reduce((sum, c) => sum + c.volume, 0) / previousCandles.length
    
    // Check for spike
    const volumeRatio = currentCandle.volume / avgVolume
    if (volumeRatio < this.config.volumeThreshold) return null
    
    // Determine spike type based on price action
    const priceChange = (currentCandle.close - currentCandle.open) / currentCandle.open
    const candleRange = currentCandle.high - currentCandle.low
    const bodyRatio = Math.abs(currentCandle.close - currentCandle.open) / candleRange
    
    let type: 'bullish' | 'bearish' | 'neutral' = 'neutral'
    if (priceChange > 0.001 && bodyRatio > 0.6) {
      type = 'bullish'
    } else if (priceChange < -0.001 && bodyRatio > 0.6) {
      type = 'bearish'
    }
    
    // Calculate significance (0-1)
    const significance = Math.min(1, (volumeRatio - 1) / 3) // Max out at 4x average
    
    return {
      timestamp: currentCandle.timestamp,
      volume: currentCandle.volume,
      priceChange,
      significance,
      type
    }
  }
  
  /**
   * Identify support and resistance levels based on volume profile
   */
  private identifySupportResistance(
    profile: Map<number, {volume: number, percentage: number, type: string}>,
    currentPrice: number
  ): SupportResistance[] {
    const levels: SupportResistance[] = []
    
    // Find high volume nodes (HVN) that can act as S/R
    const sortedLevels = Array.from(profile.entries())
      .filter(([_, data]) => data.percentage >= this.config.supportResistanceThreshold)
      .sort((a, b) => b[1].volume - a[1].volume)
    
    for (const [price, data] of sortedLevels) {
      // Skip if too close to current price (within 0.5%)
      const distance = Math.abs(price - currentPrice) / currentPrice
      if (distance < 0.005) continue
      
      const type = price < currentPrice ? 'support' : 'resistance'
      const strength = data.percentage // Use volume concentration as strength
      
      // Count historical touches (simplified)
      const touches = this.countPriceTouches(price, type)
      
      levels.push({
        level: price,
        strength,
        type,
        touches
      })
    }
    
    // Sort by proximity to current price
    return levels.sort((a, b) => 
      Math.abs(a.level - currentPrice) - Math.abs(b.level - currentPrice)
    ).slice(0, 5) // Keep top 5 levels
  }
  
  /**
   * Count how many times price has touched a level
   */
  private countPriceTouches(level: number, type: 'support' | 'resistance'): number {
    let touches = 0
    const tolerance = level * 0.002 // 0.2% tolerance
    
    for (let i = 1; i < this.priceHistory.length; i++) {
      const prev = this.priceHistory[i - 1]!
      const curr = this.priceHistory[i]!
      
      if (type === 'support') {
        // Support: price went below then back above
        if (prev > level && curr <= level + tolerance && i + 1 < this.priceHistory.length) {
          const next = this.priceHistory[i + 1]!
          if (next > level) touches++
        }
      } else {
        // Resistance: price went above then back below  
        if (prev < level && curr >= level - tolerance && i + 1 < this.priceHistory.length) {
          const next = this.priceHistory[i + 1]!
          if (next < level) touches++
        }
      }
    }
    
    return touches
  }
  
  /**
   * Synthesize trading signal from volume analysis
   */
  private synthesizeSignal(
    spike: VolumeSpike | null,
    supportResistance: SupportResistance[],
    profile: Map<number, {volume: number, percentage: number, type: string}>,
    currentPrice: number,
    averageVolume: number
  ): AgentSignal {
    // Priority 1: Volume spike with direction
    if (spike && spike.significance > 0.7) {
      if (spike.type === 'bullish') {
        return this.createSignal(
          'buy',
          Math.min(0.9, 0.7 + spike.significance * 0.2),
          `Bullish volume spike (${(spike.volume / averageVolume).toFixed(1)}x avg, price change: ${(spike.priceChange * 100).toFixed(2)}%)`
        )
      } else if (spike.type === 'bearish') {
        return this.createSignal(
          'sell',
          Math.min(0.9, 0.7 + spike.significance * 0.2),
          `Bearish volume spike (${(spike.volume / averageVolume).toFixed(1)}x avg, price change: ${(spike.priceChange * 100).toFixed(2)}%)`
        )
      }
    }
    
    // Priority 2: Price at strong support/resistance
    const nearestLevel = supportResistance[0]
    if (nearestLevel) {
      const distance = Math.abs(nearestLevel.level - currentPrice) / currentPrice
      
      // Check if we're very close to a level (within 0.3%)
      if (distance < 0.003) {
        const confidence = Math.min(0.85, 0.6 + nearestLevel.strength * 0.25 + nearestLevel.touches * 0.05)
        
        if (nearestLevel.type === 'support') {
          return this.createSignal(
            'buy',
            confidence,
            `At volume support $${nearestLevel.level.toFixed(2)} (strength: ${(nearestLevel.strength * 100).toFixed(0)}%, touches: ${nearestLevel.touches})`
          )
        } else {
          return this.createSignal(
            'sell', 
            confidence,
            `At volume resistance $${nearestLevel.level.toFixed(2)} (strength: ${(nearestLevel.strength * 100).toFixed(0)}%, touches: ${nearestLevel.touches})`
          )
        }
      }
    }
    
    // Priority 3: Volume profile accumulation/distribution
    const currentLevel = this.findNearestProfileLevel(profile, currentPrice)
    if (currentLevel && spike && spike.significance > 0.5) {
      if (currentLevel.type === 'low' && spike.type === 'bullish') {
        return this.createSignal(
          'buy',
          0.7,
          `Accumulation in low volume area with bullish spike`
        )
      } else if (currentLevel.type === 'high' && spike.type === 'bearish') {
        return this.createSignal(
          'sell',
          0.7,
          `Distribution in high volume area with bearish spike`
        )
      }
    }
    
    // Default: Hold
    return this.createSignal(
      'hold',
      0.5,
      `No clear volume signal (nearest S/R: ${nearestLevel ? `$${nearestLevel.level.toFixed(2)} (${nearestLevel.type})` : 'none'})`
    )
  }
  
  /**
   * Find the nearest volume profile level to current price
   */
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
  
  /**
   * Get detailed analysis for logging/debugging
   */
  public getLastAnalysis() {
    return this.lastAnalysis
  }
  
  /**
   * Reset agent state
   */
  protected async onReset(): Promise<void> {
    this.priceHistory = []
    this.volumeProfile.clear()
    this.lastAnalysis = null
  }
}