import { BaseAgent } from '@trdr/core'
import type { AgentMetadata, AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { enforceNoShorting } from './helpers/position-aware'
import type { Logger } from '@trdr/types'

interface RelativisticFieldConfig {
  /** Number of field points above and below current price */
  fieldPointCount?: number
  /** Field radius as percentage of price */
  fieldRadius?: number
  /** Threshold for critical field alignment */
  criticalFieldThreshold?: number
  /** Historical field states to compare against */
  fieldHistoryLength?: number
  /** Relativistic speed limit for momentum */
  momentumSpeedLimit?: number
  /** Field decay rate with distance */
  fieldDecayRate?: number
  /** Minimum field distortion for signals */
  minFieldDistortion?: number
}

interface FieldPoint {
  price: number
  distance: number // Lorentzian distance from current price
  momentumVector: { magnitude: number, direction: number }
  fieldStrength: number
  relativistcWeight: number // Based on "distance" from current price
}

interface MomentumField {
  centerPrice: number
  fieldPoints: FieldPoint[]
  fieldStrength: number
  fieldDistortion: number
  convergenceRate: number // Positive = converging, negative = diverging
  relativistcMomentum: number
  fieldAlignment: number // How aligned are the momentum vectors
  timestamp: number
}

interface FieldState {
  field: MomentumField
  priceDirection: number
  fieldTrend: 'converging' | 'diverging' | 'stable'
  criticalField: boolean
  distortionLevel: 'low' | 'medium' | 'high'
}

/**
 * Relativistic Field Agent
 * 
 * Treats the market as a relativistic momentum field in phase space, creating
 * virtual field points around the current price and measuring how momentum
 * propagates through this field using Lorentzian distance metrics.
 * 
 * **Core Innovation: Market as a Momentum Field**
 * The indicator creates a "field" around the current price and measures how 
 * momentum propagates through this field using Lorentzian distance. Think of 
 * it like gravity or electromagnetic fields in physics.
 * 
 * **Key Concepts:**
 * - **Field Points**: Creates 20 virtual points above and below current price
 * - **Field Vectors**: Each point has a momentum vector influenced by price distance
 * - **Field Distortion**: Measures irregularities in the field using Lorentzian distance
 * - **Field Strength**: Total momentum-field alignment (like magnetic field alignment)
 * - **Field Convergence/Divergence**: Detects when the field is contracting or expanding
 * 
 * **How Lorentzian Distance is Used:**
 * - **Field Decay**: Field strength decreases with Lorentzian distance from price
 * - **Distortion Measurement**: Compares adjacent field vectors using Lorentzian distance
 * - **Pattern Matching**: Finds similar historical field states in Lorentzian space
 * - **Relativistic Momentum**: Applies special relativity concepts to momentum
 * 
 * **Signals Generated When:**
 * - **Critical Field**: Momentum-field alignment exceeds threshold
 * - **Field Reversal**: Convergence with negative momentum or divergence with positive
 * - **Pattern Confirmation**: Similar field states historically led to profitable moves
 * 
 * @todo Implement field point generation around current price
 * @todo Calculate momentum vectors for each field point
 * @todo Measure field distortion using Lorentzian distance
 * @todo Apply relativistic momentum decay with distance
 * @todo Detect field convergence and divergence patterns
 * @todo Pattern match historical field states
 */
export class RelativisticFieldAgent extends BaseAgent {
  protected readonly config: Required<RelativisticFieldConfig>
  private fieldHistory: MomentumField[] = []
  private fieldStateHistory: FieldState[] = []
  private readonly momentumCache = new Map<string, number>()
  
  constructor(metadata: AgentMetadata, logger?: Logger, config?: RelativisticFieldConfig) {
    super(metadata, logger)
    
    this.config = {
      fieldPointCount: config?.fieldPointCount ?? 20, // 20 points above and below
      fieldRadius: config?.fieldRadius ?? 0.05, // 5% radius around price
      criticalFieldThreshold: config?.criticalFieldThreshold ?? 0.5, // Lower threshold for more signals
      fieldHistoryLength: config?.fieldHistoryLength ?? 50,
      momentumSpeedLimit: config?.momentumSpeedLimit ?? 1.0, // 1% max momentum (scaled to 100 internally)
      fieldDecayRate: config?.fieldDecayRate ?? 2.0,
      minFieldDistortion: config?.minFieldDistortion ?? 0.3
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Relativistic Field Agent initialized', this.config)
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { currentPrice, candles } = context
    
    if (candles.length < 15) {
      return this.createSignal('hold', 0.3, 'Insufficient data for relativistic field analysis')
    }
    
    // 1. Calculate momentum field around current price
    const momentumField = this.generateMomentumField(currentPrice, [...candles])
    
    // 2. Store field in history
    this.fieldHistory.push(momentumField)
    if (this.fieldHistory.length > this.config.fieldHistoryLength) {
      this.fieldHistory.shift()
    }
    
    // Need minimum field history
    if (this.fieldHistory.length < 10) {
      return this.createSignal('hold', 0.4, 'Building relativistic field history')
    }
    
    // 3. Analyze current field state
    const fieldState = this.analyzeFieldState(momentumField, this.fieldHistory)
    this.fieldStateHistory.push(fieldState)
    
    // 4. Check for critical field conditions
    if (fieldState.criticalField) {
      const criticalSignal = this.generateCriticalFieldSignal(fieldState, currentPrice, context)
      if (criticalSignal.action !== 'hold') {
        return criticalSignal
      }
    }
    
    // 5. Check for field reversal patterns
    const reversalSignal = this.detectFieldReversal(fieldState, this.fieldStateHistory)
    if (reversalSignal.action !== 'hold') {
      return enforceNoShorting(reversalSignal, context)
    }
    
    // 6. Pattern matching with historical field states
    const patternSignal = this.matchHistoricalFieldPatterns(fieldState, this.fieldStateHistory)
    if (patternSignal.action !== 'hold') {
      return enforceNoShorting(patternSignal, context)
    }
    
    // Default to monitoring field state
    return this.createSignal(
      'hold',
      0.5,
      `Relativistic field: ${fieldState.fieldTrend}, distortion: ${fieldState.distortionLevel}, ` +
      `alignment: ${momentumField.fieldAlignment.toFixed(3)}`
    )
  }
  
  /**
   * Generate momentum field around current price
   */
  private generateMomentumField(currentPrice: number, candles: any[]): MomentumField {
    const fieldPoints: FieldPoint[] = []
    const fieldRadius = currentPrice * this.config.fieldRadius
    const pointCount = this.config.fieldPointCount
    
    // Create field points above and below current price
    for (let i = 0; i < pointCount; i++) {
      // Points above current price
      const upPrice = currentPrice + (fieldRadius * (i + 1) / pointCount)
      const upPoint = this.createFieldPoint(upPrice, currentPrice, candles)
      fieldPoints.push(upPoint)
      
      // Points below current price
      const downPrice = currentPrice - (fieldRadius * (i + 1) / pointCount)
      const downPoint = this.createFieldPoint(downPrice, currentPrice, candles)
      fieldPoints.push(downPoint)
    }
    
    // Calculate field properties
    const fieldStrength = this.calculateFieldStrength(fieldPoints)
    const fieldDistortion = this.calculateFieldDistortion(fieldPoints)
    const convergenceRate = this.calculateConvergenceRate(fieldPoints)
    const relativistcMomentum = this.calculateRelativisticMomentum(fieldPoints)
    const fieldAlignment = this.calculateFieldAlignment(fieldPoints)
    
    return {
      centerPrice: currentPrice,
      fieldPoints,
      fieldStrength,
      fieldDistortion,
      convergenceRate,
      relativistcMomentum,
      fieldAlignment,
      timestamp: Date.now()
    }
  }
  
  /**
   * Create a single field point with momentum vector
   */
  private createFieldPoint(price: number, centerPrice: number, candles: any[]): FieldPoint {
    // Calculate Lorentzian distance from center
    const priceDistance = Math.abs(price - centerPrice) / centerPrice
    const lorentzianDistance = Math.log(1 + priceDistance * priceDistance)
    
    // Calculate momentum at this price level
    const momentum = this.calculateMomentumAtPrice(price, candles)
    
    // Calculate field strength with relativistic decay
    const fieldStrength = this.applyRelativisticDecay(momentum, lorentzianDistance)
    
    // Momentum vector has magnitude and direction
    const momentumVector = {
      magnitude: Math.abs(momentum),
      direction: momentum > 0 ? 1 : -1
    }
    
    // Relativistic weight decreases with distance
    const relativistcWeight = 1 / (1 + lorentzianDistance * this.config.fieldDecayRate)
    
    return {
      price,
      distance: lorentzianDistance,
      momentumVector,
      fieldStrength,
      relativistcWeight
    }
  }
  
  /**
   * Calculate momentum at a specific price level
   */
  private calculateMomentumAtPrice(targetPrice: number, candles: any[]): number {
    const cacheKey = `${targetPrice.toFixed(2)}-${candles.length}`
    if (this.momentumCache.has(cacheKey)) {
      return this.momentumCache.get(cacheKey)!
    }
    
    // Use recent price changes weighted by proximity to target price
    const recentCandles = candles.slice(-10)
    let weightedMomentum = 0
    let totalWeight = 0
    
    for (let i = 1; i < recentCandles.length; i++) {
      const price = recentCandles[i]!.close
      const prevPrice = recentCandles[i-1]!.close
      const priceChange = (price - prevPrice) / prevPrice * 100 // Scale to percentage
      
      // Weight by proximity to target price (inverse distance)
      const proximity = 1 / (1 + Math.abs(price - targetPrice) / targetPrice)
      const weight = proximity * proximity // Square for stronger weighting
      
      weightedMomentum += priceChange * weight
      totalWeight += weight
    }
    
    const momentum = totalWeight > 0 ? weightedMomentum / totalWeight : 0
    
    // Apply speed limit (like speed of light in relativity)
    const limitedMomentum = Math.sign(momentum) * 
      Math.min(Math.abs(momentum), this.config.momentumSpeedLimit * 100) // Adjust limit for percentage scale
    
    this.momentumCache.set(cacheKey, limitedMomentum)
    return limitedMomentum
  }
  
  /**
   * Apply relativistic decay to field strength
   */
  private applyRelativisticDecay(momentum: number, distance: number): number {
    // Field strength decreases with Lorentzian distance
    // Similar to how electromagnetic fields decay with distance
    const decayFactor = 1 / (1 + distance * this.config.fieldDecayRate)
    return momentum * decayFactor
  }
  
  /**
   * Calculate total field strength (like magnetic field strength)
   */
  private calculateFieldStrength(fieldPoints: FieldPoint[]): number {
    if (fieldPoints.length === 0) return 0
    
    let totalStrength = 0
    let totalWeight = 0
    
    for (const point of fieldPoints) {
      if (isFinite(point.fieldStrength) && isFinite(point.relativistcWeight)) {
        totalStrength += point.fieldStrength * point.relativistcWeight
        totalWeight += point.relativistcWeight
      }
    }
    
    return totalWeight > 0 ? totalStrength / totalWeight : 0
  }
  
  /**
   * Calculate field distortion using Lorentzian distance between adjacent vectors
   */
  private calculateFieldDistortion(fieldPoints: FieldPoint[]): number {
    if (fieldPoints.length < 2) return 0
    
    let totalDistortion = 0
    let comparisons = 0
    
    // Sort by price to get adjacent points
    const sortedPoints = [...fieldPoints].sort((a, b) => a.price - b.price)
    
    for (let i = 1; i < sortedPoints.length; i++) {
      const current = sortedPoints[i]!
      const previous = sortedPoints[i-1]!
      
      // Calculate Lorentzian distance between momentum vectors
      const momentumDiff = current.momentumVector.magnitude - previous.momentumVector.magnitude
      const directionDiff = current.momentumVector.direction - previous.momentumVector.direction
      const vectorDistance = Math.sqrt(momentumDiff * momentumDiff + directionDiff * directionDiff)
      const lorentzianDistortion = Math.log(1 + vectorDistance * vectorDistance)
      
      totalDistortion += lorentzianDistortion
      comparisons++
    }
    
    return comparisons > 0 ? totalDistortion / comparisons : 0
  }
  
  /**
   * Calculate field convergence rate
   */
  private calculateConvergenceRate(fieldPoints: FieldPoint[]): number {
    if (fieldPoints.length === 0) return 0
    
    // Positive = field converging, negative = diverging
    let convergingForce = 0
    let divergingForce = 0
    let validPoints = 0
    
    for (const point of fieldPoints) {
      if (isFinite(point.momentumVector.magnitude) && isFinite(point.relativistcWeight)) {
        if (point.momentumVector.direction > 0) {
          convergingForce += point.momentumVector.magnitude * point.relativistcWeight
        } else {
          divergingForce += point.momentumVector.magnitude * point.relativistcWeight
        }
        validPoints++
      }
    }
    
    return validPoints > 0 ? (convergingForce - divergingForce) / validPoints : 0
  }
  
  /**
   * Calculate relativistic momentum
   */
  private calculateRelativisticMomentum(fieldPoints: FieldPoint[]): number {
    if (fieldPoints.length === 0) return 0
    
    // Apply Lorentz factor to momentum based on "velocity"
    let totalMomentum = 0
    let validPoints = 0
    
    for (const point of fieldPoints) {
      const velocity = Math.abs(point.momentumVector.magnitude)
      const speedOfLight = this.config.momentumSpeedLimit * 100 // Scale to match percentage scale
      
      if (speedOfLight === 0 || !isFinite(velocity)) continue
      
      // Lorentz factor: γ = 1 / sqrt(1 - v²/c²)
      const velocityRatio = Math.min(0.99, velocity / speedOfLight) // Cap at 99% of c
      const lorentzFactor = 1 / Math.sqrt(1 - velocityRatio * velocityRatio)
      
      // Relativistic momentum: p = γmv
      const relativistcMomentum = lorentzFactor * point.momentumVector.magnitude * point.relativistcWeight
      if (isFinite(relativistcMomentum)) {
        totalMomentum += relativistcMomentum * point.momentumVector.direction
        validPoints++
      }
    }
    
    return validPoints > 0 ? totalMomentum / validPoints : 0
  }
  
  /**
   * Calculate field alignment (how aligned are momentum vectors)
   */
  private calculateFieldAlignment(fieldPoints: FieldPoint[]): number {
    if (fieldPoints.length === 0) return 0
    
    const directions = fieldPoints.map(p => p.momentumVector.direction * p.relativistcWeight)
    const avgDirection = directions.reduce((sum, dir) => sum + dir, 0) / directions.length
    
    // Calculate alignment as inverse of variance
    const variance = directions.reduce((sum, dir) => 
      sum + Math.pow(dir - avgDirection, 2), 0) / directions.length
    
    return Math.max(0, 1 - variance)
  }
  
  /**
   * Analyze current field state
   */
  private analyzeFieldState(field: MomentumField, _history: MomentumField[]): FieldState {
    const criticalField = field.fieldAlignment > this.config.criticalFieldThreshold
    
    const fieldTrend = field.convergenceRate > 0.1 ? 'converging' :
                      field.convergenceRate < -0.1 ? 'diverging' : 'stable'
    
    const distortionLevel = field.fieldDistortion > 0.7 ? 'high' :
                           field.fieldDistortion > 0.3 ? 'medium' : 'low'
    
    const priceDirection = field.relativistcMomentum > 0 ? 1 : -1
    
    return {
      field,
      priceDirection,
      fieldTrend,
      criticalField,
      distortionLevel
    }
  }
  
  /**
   * Generate signal for critical field conditions
   */
  private generateCriticalFieldSignal(
    fieldState: FieldState, 
    _currentPrice: number,
    context: MarketContext
  ): AgentSignal {
    const field = fieldState.field
    const momentum = field.relativistcMomentum
    
    if (Math.abs(momentum) > 0.005) { // Lowered threshold for more signals
      const action = momentum > 0 ? 'buy' : 'sell'
      const confidence = Math.min(0.9, 0.5 + Math.abs(momentum) * 10)
      
      const signal = this.createSignal(
        action,
        confidence,
        `Critical field: alignment ${field.fieldAlignment.toFixed(3)}, ` +
        `relativistic momentum ${momentum.toFixed(4)}`
      )
      
      return action === 'sell' ? enforceNoShorting(signal, context) : signal
    }
    
    return this.createSignal('hold', 0.5, 'Critical field detected but momentum insufficient')
  }
  
  /**
   * Detect field reversal patterns
   */
  private detectFieldReversal(
    currentState: FieldState, 
    stateHistory: FieldState[]
  ): AgentSignal {
    if (stateHistory.length < 3) {
      return this.createSignal('hold', 0.4, 'Insufficient field reversal history')
    }
    
    const field = currentState.field
    
    // Field reversal: convergence with negative momentum OR divergence with positive momentum
    const convergingNegative = currentState.fieldTrend === 'converging' && 
                              field.relativistcMomentum < -0.005
    const divergingPositive = currentState.fieldTrend === 'diverging' && 
                             field.relativistcMomentum > 0.005
    
    if (convergingNegative) {
      return this.createSignal(
        'sell',
        0.7,
        `Field reversal: converging with negative momentum (${field.relativistcMomentum.toFixed(4)})`
      )
    }
    
    if (divergingPositive) {
      return this.createSignal(
        'buy',
        0.7,
        `Field reversal: diverging with positive momentum (${field.relativistcMomentum.toFixed(4)})`
      )
    }
    
    return this.createSignal('hold', 0.5, 'No field reversal detected')
  }
  
  /**
   * Match historical field patterns
   */
  private matchHistoricalFieldPatterns(
    currentState: FieldState, 
    stateHistory: FieldState[]
  ): AgentSignal {
    if (stateHistory.length < 20) {
      return this.createSignal('hold', 0.4, 'Insufficient pattern history')
    }
    
    const current = currentState.field
    const matches: { distance: number, futureReturn: number }[] = []
    
    // Find similar field states using Lorentzian distance
    for (let i = 0; i < stateHistory.length - 5; i++) {
      const historical = stateHistory[i]!.field
      
      // Calculate Lorentzian distance between field states
      const alignmentDiff = current.fieldAlignment - historical.fieldAlignment
      const distortionDiff = current.fieldDistortion - historical.fieldDistortion
      const momentumDiff = current.relativistcMomentum - historical.relativistcMomentum
      
      const stateDiff = Math.sqrt(
        alignmentDiff * alignmentDiff + 
        distortionDiff * distortionDiff + 
        momentumDiff * momentumDiff
      )
      const lorentzianDistance = Math.log(1 + stateDiff * stateDiff)
      
      // If similar enough, check what happened next
      if (lorentzianDistance < 0.3) { // Similar field state
        const futureState = stateHistory[i + 5] // Look 5 periods ahead
        if (futureState) {
          const futureReturn = (futureState.field.centerPrice - historical.centerPrice) / historical.centerPrice
          matches.push({ distance: lorentzianDistance, futureReturn })
        }
      }
    }
    
    if (matches.length >= 3) {
      const avgReturn = matches.reduce((sum, m) => sum + m.futureReturn, 0) / matches.length
      const confidence = Math.min(0.85, 0.5 + matches.length * 0.05)
      
      if (Math.abs(avgReturn) > 0.005) { // 0.5% threshold
        const action = avgReturn > 0 ? 'buy' : 'sell'
        return this.createSignal(
          action,
          confidence,
          `Pattern match: ${matches.length} similar fields, avg return ${(avgReturn * 100).toFixed(2)}%`
        )
      }
    }
    
    return this.createSignal('hold', 0.5, `Pattern analysis: ${matches.length} matches found`)
  }
  
  protected async onReset(): Promise<void> {
    this.fieldHistory = []
    this.fieldStateHistory = []
    this.momentumCache.clear()
  }
}