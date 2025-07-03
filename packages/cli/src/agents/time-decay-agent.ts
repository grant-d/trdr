import { BaseAgent } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { enforceNoShorting } from './helpers/position-aware'

interface TimeDecayConfig {
  /** Initial trail distance percentage */
  initialTrailDistance?: number
  /** Minimum trail distance (won't decay below this) */
  minTrailDistance?: number
  /** Maximum time at a level before considered "stale" (minutes) */
  staleThreshold?: number
  /** Decay rate per time unit */
  decayRate?: number
  /** Time units for decay calculation (minutes) */
  decayInterval?: number
  /** Boost factor when price breaks through stale level */
  breakoutBoost?: number
}

interface PriceLevel {
  price: number
  firstSeen: number
  lastSeen: number
  timeSpent: number // Total minutes at this level
  visits: number // Number of times price returned to this level
  volume: number // Cumulative volume at this level
  isStale: boolean
  decayFactor: number // Current decay multiplier (0-1)
}

interface TimeProfile {
  levels: Map<number, PriceLevel>
  currentLevel: PriceLevel | null
  staleLevels: PriceLevel[]
  averageTimePerLevel: number
  maxTimeAtLevel: number
}

/**
 * Time Decay Agent
 * 
 * Tracks how long price stays at specific levels and tightens trailing stops
 * on "stale" prices that have been tested multiple times. Based on the concept
 * that prices spending excessive time at a level are building energy for a breakout.
 * 
 * Key Concepts:
 * - **Price Staleness**: Time spent at a level indicates exhaustion
 * - **Trail Tightening**: Reduce trail distance as time increases at a level
 * - **Breakout Anticipation**: Stale levels likely to see strong moves
 * - **Volume Accumulation**: Track volume buildup at time-heavy levels
 * 
 * Trading Signals:
 * - Tighten trails when price stalls at a level (reduce slippage on breakout)
 * - Strong signals when breaking from stale levels
 * - Wider trails on fresh price levels (allow price discovery)
 * - Exit signals when returning to previously stale levels
 * 
 * @todo Implement price level tracking and time measurement
 * @todo Calculate time decay factors for each level
 * @todo Identify stale price zones
 * @todo Adjust trail distances based on time decay
 * @todo Detect breakouts from stale levels
 * @todo Track volume accumulation at time-heavy levels
 */
export class TimeDecayAgent extends BaseAgent {
  protected readonly config: Required<TimeDecayConfig>
  private timeProfile: TimeProfile = {
    levels: new Map(),
    currentLevel: null,
    staleLevels: [],
    averageTimePerLevel: 0,
    maxTimeAtLevel: 0
  }
  private priceHistory: { price: number, timestamp: number, volume: number }[] = []
  // @ts-ignore - unused variable (reserved for future use)
  private _lastUpdateTime = 0
  
  constructor(metadata: any, logger?: any, config?: TimeDecayConfig) {
    super(metadata, logger)
    
    this.config = {
      initialTrailDistance: config?.initialTrailDistance ?? 0.02, // 2%
      minTrailDistance: config?.minTrailDistance ?? 0.005, // 0.5%
      staleThreshold: config?.staleThreshold ?? 60, // 1 hour (more responsive)
      decayRate: config?.decayRate ?? 0.1, // 10% decay per interval
      decayInterval: config?.decayInterval ?? 30, // 30 minutes
      breakoutBoost: config?.breakoutBoost ?? 1.5 // 50% confidence boost
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Time Decay Agent initialized', this.config)
    this._lastUpdateTime = Date.now()
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { currentPrice, candles } = context
    const currentTime = Date.now()
    
    // Update price tracking
    this.updatePriceLevels(currentPrice, candles[candles.length - 1]?.volume ?? 0)
    this.priceHistory.push({ price: currentPrice, timestamp: currentTime, volume: candles[candles.length - 1]?.volume ?? 0 })
    
    // Check for insufficient data
    if (this.priceHistory.length < 5) {
      return this.createSignal('hold', 0.3, 'Building time profile')
    }
    
    // Keep history manageable
    if (this.priceHistory.length > 1000) {
      this.priceHistory = this.priceHistory.slice(-500)
    }
    
    // Identify stale levels
    const staleLevels = this.identifyStaleLevels()
    this.timeProfile.staleLevels = staleLevels
    
    // Check for breakout from stale level
    if (this.priceHistory.length > 1) {
      const previousPrice = this.priceHistory[this.priceHistory.length - 2]!.price
      const breakout = this.detectStaleBreakout(currentPrice, previousPrice, staleLevels)
      
      if (breakout.isBreakout && breakout.level) {
        const action = breakout.direction === 'up' ? 'buy' : 'sell'
        const confidence = Math.min(0.95, 0.7 + breakout.level.timeSpent / (this.config.staleThreshold * 60000) * 0.3)
        
        const signal = this.createSignal(
          action,
          confidence,
          `Breakout from stale level ${breakout.level.price.toFixed(2)} after ${(breakout.level.timeSpent / 60000).toFixed(0)} minutes`
        )
        
        return enforceNoShorting(signal, context)
      }
    }
    
    // Calculate current level staleness
    const currentLevel = this.timeProfile.currentLevel
    if (currentLevel) {
      const timeAtLevel = currentTime - currentLevel.firstSeen
      const staleness = timeAtLevel / (this.config.staleThreshold * 60000)
      
      // Adjust trail distance based on staleness
      const decayFactor = this.calculateDecayFactor(timeAtLevel)
      // @ts-ignore - unused variable (reserved for future use)
      const _adjustedTrailDistance = this.adjustTrailDistance(this.config.initialTrailDistance, decayFactor)
      
      // Stale price - prepare for breakout
      if (staleness > 0.6) { // Lower threshold
        // Look for micro-movements to predict direction
        const recentMoves = this.priceHistory.slice(-5).map(h => h.price)
        const microTrend = recentMoves.length > 1 ? (recentMoves[recentMoves.length - 1]! - recentMoves[0]!) / recentMoves[0]! : 0
        
        if (Math.abs(microTrend) > 0.001) {
          const action = microTrend > 0 ? 'buy' : 'sell'
          const signal = this.createSignal(
            action,
            0.8,
            `Stale price coiling at ${currentPrice.toFixed(2)} for ${(timeAtLevel / 60000).toFixed(0)} min, ${action} on micro-trend`
          )
          return enforceNoShorting(signal, context)
        }
      }
    }
    
    // Look for time-weighted volume patterns
    const timeWeightedVolume = this.getTimeWeightedVolume(this.timeProfile.levels)
    const maxVolumeLevel = Array.from(timeWeightedVolume.entries())
      .sort((a, b) => b[1] - a[1])[0]
    
    if (maxVolumeLevel) {
      const [price, volume] = maxVolumeLevel
      const distance = Math.abs(currentPrice - price) / price
      
      if (distance < 0.02) { // Within 2% of high time-volume level
        // Trade away from time-heavy levels
        const action = currentPrice > price ? 'sell' : 'buy'
        const signal = this.createSignal(
          action,
          0.75,
          `${action.toUpperCase()} away from time-heavy level ${price.toFixed(2)} (volume: ${volume.toFixed(0)})`
        )
        return enforceNoShorting(signal, context)
      }
    }
    
    // Default signal
    const avgTimePerLevel = this.timeProfile.averageTimePerLevel / 60000 // Convert to minutes
    return this.createSignal(
      'hold',
      0.4,
      `Monitoring time decay. Avg time/level: ${avgTimePerLevel.toFixed(1)} min, stale levels: ${staleLevels.length}`
    )
  }
  
  /**
   * Track price levels and time spent
   */
  private updatePriceLevels(price: number, volume: number): void {
    const currentTime = Date.now()
    const quantizedPrice = this.quantizePrice(price)
    
    // Check if we're still at the same level
    if (this.timeProfile.currentLevel && Math.abs(this.timeProfile.currentLevel.price - quantizedPrice) < 0.0001) {
      // Update current level
      this.timeProfile.currentLevel.lastSeen = currentTime
      this.timeProfile.currentLevel.timeSpent = currentTime - this.timeProfile.currentLevel.firstSeen
      this.timeProfile.currentLevel.volume += volume
    } else {
      // New level
      const newLevel: PriceLevel = {
        price: quantizedPrice,
        firstSeen: currentTime,
        lastSeen: currentTime,
        timeSpent: 0,
        visits: 1,
        volume,
        isStale: false,
        decayFactor: 1
      }
      
      // Check if we've been at this level before
      if (this.timeProfile.levels.has(quantizedPrice)) {
        const existingLevel = this.timeProfile.levels.get(quantizedPrice)!
        existingLevel.visits++
        existingLevel.lastSeen = currentTime
        existingLevel.volume += volume
        this.timeProfile.currentLevel = existingLevel
      } else {
        this.timeProfile.levels.set(quantizedPrice, newLevel)
        this.timeProfile.currentLevel = newLevel
      }
    }
    
    // Update average time per level
    const totalTime = Array.from(this.timeProfile.levels.values())
      .reduce((sum, level) => sum + level.timeSpent, 0)
    this.timeProfile.averageTimePerLevel = this.timeProfile.levels.size > 0 ? 
      totalTime / this.timeProfile.levels.size : 0
    
    // Track max time at any level
    this.timeProfile.maxTimeAtLevel = Math.max(
      this.timeProfile.maxTimeAtLevel,
      this.timeProfile.currentLevel?.timeSpent ?? 0
    )
  }
  
  /**
   * Calculate decay factor based on time at level
   */
  private calculateDecayFactor(timeAtLevel: number): number {
    // Exponential decay based on time
    const timeInIntervals = timeAtLevel / (this.config.decayInterval * 60000)
    return Math.pow(1 - this.config.decayRate, timeInIntervals)
  }
  
  /**
   * Identify stale price zones
   */
  private identifyStaleLevels(): PriceLevel[] {
    const staleLevels: PriceLevel[] = []
    const currentTime = Date.now()
    
    for (const level of this.timeProfile.levels.values()) {
      const timeSinceLastVisit = currentTime - level.lastSeen
      const totalTimeAtLevel = level.timeSpent
      
      // Mark as stale if spent too much time there or hasn't been visited recently
      if (totalTimeAtLevel > this.config.staleThreshold * 60000 || 
          (level.visits > 3 && timeSinceLastVisit > this.config.staleThreshold * 60000 / 2)) {
        level.isStale = true
        level.decayFactor = this.calculateDecayFactor(totalTimeAtLevel)
        staleLevels.push(level)
      }
    }
    
    return staleLevels.sort((a, b) => b.timeSpent - a.timeSpent)
  }
  
  /**
   * Adjust trail distance based on staleness
   */
  private adjustTrailDistance(baseDistance: number, decayFactor: number): number {
    // Tighten trail as decay increases (lower decay factor = tighter trail)
    const adjustedDistance = baseDistance * decayFactor
    return Math.max(this.config.minTrailDistance, adjustedDistance)
  }
  
  /**
   * Detect breakout from stale level
   */
  private detectStaleBreakout(
    currentPrice: number, 
    previousPrice: number,
    staleLevels: PriceLevel[]
  ): { isBreakout: boolean, level: PriceLevel | null, direction: 'up' | 'down' | null } {
    for (const level of staleLevels) {
      const levelRange = level.price * 0.002 // 0.2% range around level
      
      // Check if we crossed through a stale level
      if (previousPrice <= level.price + levelRange && currentPrice > level.price + levelRange) {
        // Upward breakout
        return { isBreakout: true, level, direction: 'up' }
      } else if (previousPrice >= level.price - levelRange && currentPrice < level.price - levelRange) {
        // Downward breakout
        return { isBreakout: true, level, direction: 'down' }
      }
    }
    
    return { isBreakout: false, level: null, direction: null }
  }
  
  /**
   * Calculate time-weighted volume profile
   */
  private getTimeWeightedVolume(levels: Map<number, PriceLevel>): Map<number, number> {
    const weightedVolume = new Map<number, number>()
    
    for (const [price, level] of levels.entries()) {
      // Weight volume by time spent at level (more time = more significant)
      const timeWeight = level.timeSpent / (this.config.staleThreshold * 60000)
      const weightedValue = level.volume * Math.min(2, 1 + timeWeight) // Cap at 2x weight
      weightedVolume.set(price, weightedValue)
    }
    
    return weightedVolume
  }
  
  
  /**
   * Get quantized price level for tracking
   */
  private quantizePrice(price: number): number {
    // Quantize to nearest 0.1% to group nearby prices
    const quantum = price * 0.001 // 0.1%
    return Math.round(price / quantum) * quantum
  }
  
  protected async onReset(): Promise<void> {
    this.timeProfile = {
      levels: new Map(),
      currentLevel: null,
      staleLevels: [],
      averageTimePerLevel: 0,
      maxTimeAtLevel: 0
    }
    this.priceHistory = []
    this._lastUpdateTime = Date.now()
  }
}