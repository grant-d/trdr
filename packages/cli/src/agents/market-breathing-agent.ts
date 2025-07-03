import { BaseAgent } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { enforceNoShorting } from './helpers/position-aware'

interface BreathingConfig {
  /** Window for measuring breathing cycles (minutes) */
  cycleWindow?: number
  /** Minimum amplitude for valid breath */
  minAmplitude?: number
  /** Number of breaths to track */
  breathHistory?: number
  /** Threshold for detecting rhythm changes */
  rhythmChangeThreshold?: number
  /** Sensitivity to volume changes during breathing */
  volumeSensitivity?: number
}

interface BreathCycle {
  inhaleStart: number // Timestamp
  inhaleEnd: number
  exhaleStart: number
  exhaleEnd: number
  inhalePriceLow: number
  inhalePriceHigh: number
  exhalePriceHigh: number
  exhalePriceLow: number
  amplitude: number // Price range of breath
  duration: number // Total breath time
  volume: number // Volume during cycle
  strength: number // Breath strength (0-1)
}

interface BreathingPattern {
  currentPhase: 'inhale' | 'exhale' | 'pause'
  rhythm: 'regular' | 'irregular' | 'accelerating' | 'decelerating'
  averageAmplitude: number
  averageDuration: number
  breathRate: number // Breaths per hour
  oxygenLevel: number // Market "oxygen" - liquidity/volume metric
  stress: number // Breathing stress indicator (0-1)
}

/**
 * Market Breathing Agent
 * 
 * Models market movement as breathing cycles - expansion (inhale) and contraction
 * (exhale) phases. Based on the observation that markets exhibit rhythmic patterns
 * similar to breathing, with periods of expansion and contraction.
 * 
 * Key Concepts:
 * - **Inhale Phase**: Price expansion, increasing volatility, volume surge
 * - **Exhale Phase**: Price contraction, decreasing volatility, volume decline
 * - **Breath Rhythm**: Regular vs irregular breathing indicates market health
 * - **Market Oxygen**: Liquidity/volume as "oxygen" - low oxygen = stressed market
 * - **Breath Depth**: Amplitude of cycles indicates market energy
 * 
 * Trading Signals:
 * - Buy at the end of exhale (maximum contraction)
 * - Sell at the end of inhale (maximum expansion)
 * - Adjust position size based on breath rhythm regularity
 * - Exit when breathing becomes erratic (market stress)
 * - Trail distance based on current breath amplitude
 * 
 * @todo Implement breath cycle detection from price oscillations
 * @todo Calculate breathing rhythm and regularity
 * @todo Measure market "oxygen" levels from volume
 * @todo Detect stressed/labored breathing patterns
 * @todo Identify breath phase transitions
 * @todo Generate signals aligned with breathing cycles
 */
export class MarketBreathingAgent extends BaseAgent {
  protected readonly config: Required<BreathingConfig>
  private breathHistory: BreathCycle[] = []
  private currentBreath: Partial<BreathCycle> = {}
  private breathingPattern: BreathingPattern = {
    currentPhase: 'pause',
    rhythm: 'regular',
    averageAmplitude: 0,
    averageDuration: 0,
    breathRate: 0,
    oxygenLevel: 1,
    stress: 0
  }
  private priceOscillations: { price: number, timestamp: number, volume: number }[] = []
  
  constructor(metadata: any, logger?: any, config?: BreathingConfig) {
    super(metadata, logger)
    
    this.config = {
      cycleWindow: config?.cycleWindow ?? 240, // 4 hours
      minAmplitude: config?.minAmplitude ?? 0.002, // 0.2% minimum breath
      breathHistory: config?.breathHistory ?? 20, // Track 20 breaths
      rhythmChangeThreshold: config?.rhythmChangeThreshold ?? 0.3, // 30% rhythm change
      volumeSensitivity: config?.volumeSensitivity ?? 0.7
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Market Breathing Agent initialized', this.config)
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { currentPrice, candles } = context
    const currentVolume = candles[candles.length - 1]?.volume ?? 0
    const timestamp = Date.now()
    
    // Update oscillation tracking
    this.priceOscillations.push({ price: currentPrice, timestamp, volume: currentVolume })
    if (this.priceOscillations.length > this.config.cycleWindow) {
      this.priceOscillations.shift()
    }
    
    // Need minimum data for phase detection
    if (this.priceOscillations.length < 10) {
      return this.createSignal('hold', 0.3, 'Insufficient data for breathing analysis')
    }
    
    // Detect current breathing phase
    const prices = this.priceOscillations.map(o => o.price)
    const timestamps = this.priceOscillations.map(o => o.timestamp)
    const currentPhase = this.detectBreathingPhase(prices, timestamps)
    
    // Update breath cycles
    this.updateBreathCycles(currentPrice, currentVolume, currentPhase)
    
    // Analyze breathing pattern
    const rhythmAnalysis = this.analyzeBreathingRhythm(this.breathHistory)
    this.breathingPattern.rhythm = rhythmAnalysis.rhythm
    
    // Calculate market oxygen
    const recentVolumes = this.priceOscillations.map(o => o.volume)
    const avgVolume = recentVolumes.reduce((a, b) => a + b, 0) / recentVolumes.length
    this.breathingPattern.oxygenLevel = this.calculateOxygenLevel(recentVolumes, avgVolume)
    
    // Detect stress
    this.breathingPattern.stress = this.detectBreathingStress(this.breathingPattern, this.breathHistory)
    
    // Find trading opportunity based on breath phase
    const phaseProgress = this.getPhaseProgress(
      currentPhase,
      this.currentBreath.inhaleStart ?? timestamp,
      this.breathingPattern.averageDuration
    )
    
    const opportunity = this.findBreathingOpportunity(currentPhase, phaseProgress, this.breathingPattern)
    
    // Generate signal - trade even with some stress
    if (this.breathingPattern.stress > 0.85) { // Only avoid at extreme stress
      // Even in stress, look for opportunities
      const recentPrices = prices.slice(-3)
      if (recentPrices.length > 1) {
        const microTrend = (recentPrices[recentPrices.length - 1]! - recentPrices[0]!) / recentPrices[0]!
        if (Math.abs(microTrend) > 0.002) {
          const action = microTrend > 0 ? 'buy' : 'sell'
          const signal = this.createSignal(
            action,
            0.7,
            `Stressed breathing (${(this.breathingPattern.stress * 100).toFixed(0)}%) but ${action} on momentum`
          )
          return enforceNoShorting(signal, context)
        }
      }
    }
    
    if (opportunity.action !== 'hold') {
      // Note: Trail distance could be calculated with this.calculateBreathingTrailDistance
      // using current breath amplitude, but is handled by the trading system
      
      const signal = this.createSignal(
        opportunity.action,
        opportunity.confidence,
        `${currentPhase} phase ${(phaseProgress * 100).toFixed(0)}% complete. Rhythm: ${this.breathingPattern.rhythm}, O2: ${(this.breathingPattern.oxygenLevel * 100).toFixed(0)}%`
      )
      
      return enforceNoShorting(signal, context)
    }
    
    // Default signal
    return this.createSignal(
      'hold',
      0.4,
      `Breathing ${currentPhase}, rhythm: ${this.breathingPattern.rhythm}, rate: ${this.breathingPattern.breathRate.toFixed(1)}/hr`
    )
  }
  
  /**
   * Detect breathing phase from price action
   */
  private detectBreathingPhase(
    prices: number[], 
    timestamps: number[]
  ): 'inhale' | 'exhale' | 'pause' {
    if (prices.length < 3) return 'pause'
    
    // Calculate price velocity (first derivative)
    const velocities: number[] = []
    for (let i = 1; i < prices.length; i++) {
      const priceDiff = prices[i]! - prices[i-1]!
      const timeDiff = (timestamps[i]! - timestamps[i-1]!) / 60000 // minutes
      velocities.push(timeDiff > 0 ? priceDiff / timeDiff : 0)
    }
    
    // Calculate acceleration (second derivative)
    const accelerations: number[] = []
    for (let i = 1; i < velocities.length; i++) {
      accelerations.push(velocities[i]! - velocities[i-1]!)
    }
    
    // Recent velocity and acceleration
    const recentVel = velocities[velocities.length - 1] ?? 0
    const recentAcc = accelerations[accelerations.length - 1] ?? 0
    
    // Determine phase
    if (Math.abs(recentVel) < prices[prices.length - 1]! * 0.0001) {
      return 'pause' // Very low velocity
    } else if (recentVel > 0 && recentAcc >= 0) {
      return 'inhale' // Expanding
    } else if (recentVel < 0 && recentAcc <= 0) {
      return 'exhale' // Contracting
    } else {
      return 'pause' // Transition
    }
  }
  
  /**
   * Track and complete breath cycles
   */
  private updateBreathCycles(
    currentPrice: number, 
    currentVolume: number, 
    phase: 'inhale' | 'exhale' | 'pause'
  ): void {
    const timestamp = Date.now()
    
    // Handle phase transitions
    if (phase !== this.breathingPattern.currentPhase) {
      // Complete previous phase
      if (this.breathingPattern.currentPhase === 'inhale' && phase === 'exhale') {
        // Inhale -> Exhale: Peak reached
        this.currentBreath.inhaleEnd = timestamp
        this.currentBreath.inhalePriceHigh = currentPrice
        this.currentBreath.exhaleStart = timestamp
        this.currentBreath.exhalePriceHigh = currentPrice
      } else if (this.breathingPattern.currentPhase === 'exhale' && phase === 'inhale') {
        // Exhale -> Inhale: Trough reached, complete full breath
        if (this.currentBreath.inhaleStart) {
          const completedBreath: BreathCycle = {
            inhaleStart: this.currentBreath.inhaleStart,
            inhaleEnd: this.currentBreath.inhaleEnd ?? timestamp,
            exhaleStart: this.currentBreath.exhaleStart ?? timestamp,
            exhaleEnd: timestamp,
            inhalePriceLow: this.currentBreath.inhalePriceLow ?? currentPrice,
            inhalePriceHigh: this.currentBreath.inhalePriceHigh ?? currentPrice,
            exhalePriceHigh: this.currentBreath.exhalePriceHigh ?? currentPrice,
            exhalePriceLow: currentPrice,
            amplitude: (this.currentBreath.inhalePriceHigh ?? currentPrice) - (this.currentBreath.inhalePriceLow ?? currentPrice),
            duration: timestamp - this.currentBreath.inhaleStart,
            volume: this.currentBreath.volume ?? currentVolume,
            strength: 0 // Calculate below
          }
          
          // Calculate breath strength
          const priceRange = Math.abs(completedBreath.inhalePriceHigh - completedBreath.inhalePriceLow)
          const avgPrice = (completedBreath.inhalePriceHigh + completedBreath.inhalePriceLow) / 2
          completedBreath.strength = Math.min(1, (priceRange / avgPrice) * 50) // 2% range = 1.0 strength
          
          // Add to history
          this.breathHistory.push(completedBreath)
          if (this.breathHistory.length > this.config.breathHistory) {
            this.breathHistory.shift()
          }
          
          // Update pattern metrics
          this.updateBreathingMetrics()
        }
        
        // Start new breath
        this.currentBreath = {
          inhaleStart: timestamp,
          inhalePriceLow: currentPrice,
          volume: 0
        }
      }
      
      // Update current phase
      this.breathingPattern.currentPhase = phase
    }
    
    // Update current breath data
    if (this.currentBreath.inhaleStart) {
      this.currentBreath.volume = (this.currentBreath.volume ?? 0) + currentVolume
      
      if (phase === 'inhale') {
        this.currentBreath.inhalePriceLow = Math.min(
          this.currentBreath.inhalePriceLow ?? currentPrice,
          currentPrice
        )
        this.currentBreath.inhalePriceHigh = Math.max(
          this.currentBreath.inhalePriceHigh ?? currentPrice,
          currentPrice
        )
      } else if (phase === 'exhale') {
        this.currentBreath.exhalePriceHigh = Math.max(
          this.currentBreath.exhalePriceHigh ?? currentPrice,
          currentPrice
        )
        this.currentBreath.exhalePriceLow = Math.min(
          this.currentBreath.exhalePriceLow ?? currentPrice,
          currentPrice
        )
      }
    }
  }
  
  /**
   * Update breathing pattern metrics from history
   */
  private updateBreathingMetrics(): void {
    if (this.breathHistory.length === 0) return
    
    // Calculate averages
    this.breathingPattern.averageAmplitude = 
      this.breathHistory.reduce((sum, b) => sum + b.amplitude, 0) / this.breathHistory.length
    
    this.breathingPattern.averageDuration = 
      this.breathHistory.reduce((sum, b) => sum + b.duration, 0) / this.breathHistory.length
    
    // Calculate breath rate (breaths per hour)
    if (this.breathHistory.length >= 2) {
      const firstBreath = this.breathHistory[0]!
      const lastBreath = this.breathHistory[this.breathHistory.length - 1]!
      const timeSpan = lastBreath.inhaleStart - firstBreath.inhaleStart
      if (timeSpan > 0) {
        this.breathingPattern.breathRate = (this.breathHistory.length - 1) / (timeSpan / 3600000)
      }
    }
  }
  
  /**
   * Calculate breathing rhythm regularity
   */
  private analyzeBreathingRhythm(cycles: BreathCycle[]): {
    rhythm: 'regular' | 'irregular' | 'accelerating' | 'decelerating'
    consistency: number
  } {
    if (cycles.length < 3) {
      return { rhythm: 'regular', consistency: 0.5 }
    }
    
    // Calculate duration variance
    const durations = cycles.map(c => c.duration)
    const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length
    const durationVariance = durations.reduce((sum, d) => sum + Math.pow(d - avgDuration, 2), 0) / durations.length
    const durationCV = Math.sqrt(durationVariance) / avgDuration // Coefficient of variation
    
    // Calculate amplitude variance
    const amplitudes = cycles.map(c => c.amplitude)
    const avgAmplitude = amplitudes.reduce((a, b) => a + b, 0) / amplitudes.length
    const amplitudeVariance = amplitudes.reduce((sum, a) => sum + Math.pow(a - avgAmplitude, 2), 0) / amplitudes.length
    const amplitudeCV = avgAmplitude > 0 ? Math.sqrt(amplitudeVariance) / avgAmplitude : 1
    
    // Detect rhythm pattern
    let rhythm: 'regular' | 'irregular' | 'accelerating' | 'decelerating' = 'regular'
    
    // Check for acceleration/deceleration in recent breaths
    if (cycles.length >= 5) {
      const recentDurations = durations.slice(-5)
      const durationTrend = this.calculateTrend(recentDurations)
      
      if (durationTrend < -0.2) {
        rhythm = 'accelerating' // Breaths getting shorter
      } else if (durationTrend > 0.2) {
        rhythm = 'decelerating' // Breaths getting longer
      } else if (durationCV > 0.3 || amplitudeCV > 0.4) {
        rhythm = 'irregular' // High variance
      }
    }
    
    // Calculate overall consistency (0-1)
    const consistency = Math.max(0, 1 - (durationCV + amplitudeCV) / 2)
    
    return { rhythm, consistency }
  }
  
  /**
   * Calculate linear trend in data series
   */
  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0
    
    // Simple linear regression
    const n = values.length
    const indices = Array.from({ length: n }, (_, i) => i)
    
    const sumX = indices.reduce((a, b) => a + b, 0)
    const sumY = values.reduce((a, b) => a + b, 0)
    const sumXY = indices.reduce((sum, x, i) => sum + x * values[i]!, 0)
    const sumX2 = indices.reduce((sum, x) => sum + x * x, 0)
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
    const avgY = sumY / n
    
    // Return normalized slope
    return avgY > 0 ? slope / avgY : 0
  }
  
  /**
   * Calculate market oxygen level from volume
   */
  private calculateOxygenLevel(
    recentVolumes: number[], 
    averageVolume: number
  ): number {
    if (recentVolumes.length === 0 || averageVolume === 0) return 0.5
    
    // Calculate recent average volume
    const recentAvg = recentVolumes.reduce((a, b) => a + b, 0) / recentVolumes.length
    
    // Calculate volume consistency
    const volumeVariance = recentVolumes.reduce((sum, v) => sum + Math.pow(v - recentAvg, 2), 0) / recentVolumes.length
    const volumeStdDev = Math.sqrt(volumeVariance)
    const volumeCV = recentAvg > 0 ? volumeStdDev / recentAvg : 1
    
    // Calculate volume ratio (recent vs average)
    const volumeRatio = recentAvg / averageVolume
    
    // Calculate volume trend
    const volumeTrend = this.calculateTrend(recentVolumes)
    
    // Oxygen level factors:
    // 1. Volume ratio (0.5-2.0 optimal, outside = less oxygen)
    // 2. Volume consistency (lower CV = more oxygen)
    // 3. Volume trend (positive = increasing oxygen)
    
    let oxygenLevel = 0.5 // Base level
    
    // Volume ratio component (40%)
    if (volumeRatio >= 0.5 && volumeRatio <= 2.0) {
      oxygenLevel += 0.4 * (1 - Math.abs(1 - volumeRatio))
    } else if (volumeRatio < 0.5) {
      oxygenLevel += 0.4 * volumeRatio // Low volume = low oxygen
    } else {
      oxygenLevel += 0.4 * (2.0 / volumeRatio) // Very high volume = turbulence
    }
    
    // Consistency component (30%)
    oxygenLevel += 0.3 * Math.max(0, 1 - volumeCV)
    
    // Trend component (30%)
    if (volumeTrend > 0) {
      oxygenLevel += 0.3 * Math.min(1, volumeTrend * 2) // Positive trend = more oxygen
    } else {
      oxygenLevel += 0.3 * Math.max(0, 1 + volumeTrend) // Negative trend = less oxygen
    }
    
    return Math.min(1, Math.max(0, oxygenLevel))
  }
  
  /**
   * Detect stressed/labored breathing
   */
  private detectBreathingStress(
    pattern: BreathingPattern,
    recentCycles: BreathCycle[]
  ): number {
    let stressLevel = 0
    
    // Factor 1: Irregular rhythm (25%)
    if (pattern.rhythm === 'irregular') {
      stressLevel += 0.25
    } else if (pattern.rhythm === 'accelerating') {
      stressLevel += 0.15 // Mild stress from acceleration
    }
    
    // Factor 2: Low oxygen (25%)
    const oxygenStress = 1 - pattern.oxygenLevel
    stressLevel += 0.25 * oxygenStress
    
    // Factor 3: Extreme breath rate (20%)
    if (pattern.breathRate > 0) {
      // Normal rate ~10-20 breaths/hour for crypto
      if (pattern.breathRate < 5) {
        stressLevel += 0.2 * (1 - pattern.breathRate / 5) // Too slow
      } else if (pattern.breathRate > 30) {
        stressLevel += 0.2 * Math.min(1, (pattern.breathRate - 30) / 30) // Too fast
      }
    }
    
    // Factor 4: Amplitude extremes (15%)
    if (recentCycles.length > 0 && pattern.averageAmplitude > 0) {
      const recentAmplitudes = recentCycles.slice(-5).map(c => c.amplitude)
      const maxAmplitude = Math.max(...recentAmplitudes)
      const minAmplitude = Math.min(...recentAmplitudes)
      
      // Check for extreme variations
      if (maxAmplitude > pattern.averageAmplitude * 2) {
        stressLevel += 0.15 * Math.min(1, (maxAmplitude / pattern.averageAmplitude - 2) / 2)
      }
      if (minAmplitude < pattern.averageAmplitude * 0.3) {
        stressLevel += 0.15 * (1 - minAmplitude / (pattern.averageAmplitude * 0.3))
      }
    }
    
    // Factor 5: Incomplete breaths (15%)
    if (recentCycles.length >= 3) {
      const incompleteBreaths = recentCycles.filter(c => c.strength < 0.3).length
      const incompleteRatio = incompleteBreaths / recentCycles.length
      stressLevel += 0.15 * incompleteRatio
    }
    
    return Math.min(1, Math.max(0, stressLevel))
  }
  
  /**
   * Find optimal entry/exit points in breath cycle
   */
  private findBreathingOpportunity(
    phase: 'inhale' | 'exhale' | 'pause',
    phaseProgress: number,
    pattern: BreathingPattern
  ): { action: 'buy' | 'sell' | 'hold', confidence: number } {
    let action: 'buy' | 'sell' | 'hold' = 'hold'
    let confidence = 0.5
    
    // Base confidence on rhythm regularity and oxygen level - more aggressive
    const baseConfidence = 0.6 + (pattern.oxygenLevel * 0.2) + 
                          (pattern.rhythm === 'regular' ? 0.2 : 0.1)
    
    if (phase === 'exhale' && phaseProgress > 0.4) { // Much earlier entry
      // Mid-to-late exhale - approaching trough (buy opportunity)
      action = 'buy'
      confidence = baseConfidence + 0.2 + (phaseProgress - 0.4) * 0.5
      
      // Boost confidence for regular rhythm
      if (pattern.rhythm === 'regular') {
        confidence += 0.1
      }
    } else if (phase === 'inhale' && phaseProgress > 0.4) { // Much earlier entry
      // Mid-to-late inhale - approaching peak (sell opportunity)
      action = 'sell'
      confidence = baseConfidence + 0.2 + (phaseProgress - 0.4) * 0.5
      
      // Boost confidence for regular rhythm
      if (pattern.rhythm === 'regular') {
        confidence += 0.1
      }
    } else if (phase === 'pause') {
      // Pause phase - look for breakout direction
      action = 'hold' // Default
      confidence = 0.5
      
      // Don't just hold - look for micro trends
      if (pattern.oxygenLevel > 0.3) {
        confidence = 0.7 // Be ready to act
      }
    } else {
      // Mid-phase - generally hold
      action = 'hold'
      confidence = baseConfidence
      
      // Reduce confidence during irregular breathing
      if (pattern.rhythm === 'irregular') {
        confidence *= 0.8
      }
    }
    
    // Adjust for stress
    if (pattern.stress > 0.5) {
      // High stress - reduce action confidence, increase hold confidence
      if (action !== 'hold') {
        confidence *= (1 - pattern.stress * 0.5)
      } else {
        confidence = Math.min(0.9, confidence + pattern.stress * 0.2)
      }
    }
    
    return { action, confidence: Math.min(0.95, Math.max(0.1, confidence)) }
  }
  
  /**
   * Calculate breath phase progress (0-1)
   */
  private getPhaseProgress(
    phase: 'inhale' | 'exhale' | 'pause',
    startTime: number,
    averageDuration: number
  ): number {
    if (averageDuration <= 0) return 0.5
    
    const currentTime = Date.now()
    const elapsed = currentTime - startTime
    
    // Estimate phase duration based on average breath duration
    let estimatedPhaseDuration: number
    
    if (phase === 'inhale' || phase === 'exhale') {
      // Inhale and exhale typically take ~40% of breath each
      estimatedPhaseDuration = averageDuration * 0.4
    } else {
      // Pause phases are typically shorter ~20%
      estimatedPhaseDuration = averageDuration * 0.2
    }
    
    // Calculate progress
    const progress = elapsed / estimatedPhaseDuration
    
    // Cap at 1.0 (can't be more than 100% complete)
    return Math.min(1, Math.max(0, progress))
  }
  
  /**
   * Adjust trail distance based on breath amplitude
   */
  // @ts-ignore - unused variable (reserved for future use)
  private _calculateBreathingTrailDistance(
    baseDistance: number,
    currentAmplitude: number,
    averageAmplitude: number
  ): number {
    if (averageAmplitude <= 0) return baseDistance
    
    // Calculate amplitude ratio
    const amplitudeRatio = currentAmplitude / averageAmplitude
    
    // Scale trail distance based on breath depth
    // Deeper breaths (larger amplitude) = wider trails
    // Shallow breaths (smaller amplitude) = tighter trails
    let scaleFactor: number
    
    if (amplitudeRatio < 0.5) {
      // Very shallow breathing - tighten trails significantly
      scaleFactor = 0.5 + amplitudeRatio
    } else if (amplitudeRatio > 1.5) {
      // Deep breathing - widen trails
      scaleFactor = 1 + (amplitudeRatio - 1) * 0.5
      // Cap at 2x base distance
      scaleFactor = Math.min(2, scaleFactor)
    } else {
      // Normal breathing - moderate adjustment
      scaleFactor = 0.8 + amplitudeRatio * 0.2
    }
    
    // Apply breathing pattern adjustments
    if (this.breathingPattern.rhythm === 'irregular') {
      // Irregular breathing - tighten trails for safety
      scaleFactor *= 0.8
    } else if (this.breathingPattern.rhythm === 'accelerating') {
      // Accelerating breathing - slightly tighter
      scaleFactor *= 0.9
    }
    
    // Apply oxygen level adjustment
    if (this.breathingPattern.oxygenLevel < 0.3) {
      // Low oxygen - tighten trails
      scaleFactor *= 0.7
    } else if (this.breathingPattern.oxygenLevel > 0.8) {
      // High oxygen - can afford wider trails
      scaleFactor *= 1.1
    }
    
    // Apply stress adjustment
    if (this.breathingPattern.stress > 0.7) {
      // High stress - tighten trails significantly
      scaleFactor *= 0.6
    }
    
    return baseDistance * scaleFactor
  }
  
  protected async onReset(): Promise<void> {
    this.breathHistory = []
    this.currentBreath = {}
    this.breathingPattern = {
      currentPhase: 'pause',
      rhythm: 'regular',
      averageAmplitude: 0,
      averageDuration: 0,
      breathRate: 0,
      oxygenLevel: 1,
      stress: 0
    }
    this.priceOscillations = []
  }
}