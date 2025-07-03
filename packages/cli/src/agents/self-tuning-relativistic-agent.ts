import { SelfTuningBaseAgent } from './self-tuning-base-agent'
import type { AgentMetadata, AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { enforceNoShorting } from './helpers/position-aware'
import type { Logger } from '@trdr/types'

interface RelativisticFieldConfig {
  fieldPointCount?: number
  fieldRadius?: number
  criticalFieldThreshold?: number
  fieldHistoryLength?: number
  momentumSpeedLimit?: number
  fieldDecayRate?: number
  minFieldDistortion?: number
  enableSelfTuning?: boolean
}

// Copy interfaces from original
interface FieldPoint {
  price: number
  distance: number
  momentumVector: { magnitude: number, direction: number }
  fieldStrength: number
  relativistcWeight: number
}

interface MomentumField {
  centerPrice: number
  fieldPoints: FieldPoint[]
  fieldStrength: number
  fieldDistortion: number
  convergenceRate: number
  relativistcMomentum: number
  fieldAlignment: number
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
 * Self-Tuning Relativistic Field Agent
 * 
 * An enhanced version of the Relativistic Field Agent that automatically
 * adjusts its parameters based on performance metrics.
 */
export class SelfTuningRelativisticAgent extends SelfTuningBaseAgent {
  protected config: Required<RelativisticFieldConfig>
  private fieldHistory: MomentumField[] = []
  private fieldStateHistory: FieldState[] = []
  private readonly momentumCache = new Map<string, number>()
  
  constructor(metadata: AgentMetadata, logger?: Logger, config?: RelativisticFieldConfig) {
    super(metadata, logger, {
      enableSelfTuning: config?.enableSelfTuning ?? true,
      evaluationWindow: 20,
      minSignalsForTuning: 10,
      learningRate: 0.15,
      performanceThreshold: 0.5
    })
    
    this.config = {
      fieldPointCount: config?.fieldPointCount ?? 20,
      fieldRadius: config?.fieldRadius ?? 0.05,
      criticalFieldThreshold: config?.criticalFieldThreshold ?? 0.4, // Start lower
      fieldHistoryLength: config?.fieldHistoryLength ?? 50,
      momentumSpeedLimit: config?.momentumSpeedLimit ?? 1.0,
      fieldDecayRate: config?.fieldDecayRate ?? 2.0,
      minFieldDistortion: config?.minFieldDistortion ?? 0.2, // Start lower
      enableSelfTuning: config?.enableSelfTuning ?? true
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Self-Tuning Relativistic Field Agent initialized', this.config)
  }
  
  protected getTunableParameters(): Record<string, number> {
    return {
      criticalFieldThreshold: this.config.criticalFieldThreshold,
      minFieldDistortion: this.config.minFieldDistortion,
      fieldRadius: this.config.fieldRadius,
      fieldDecayRate: this.config.fieldDecayRate,
      momentumThreshold: 0.005 // Add explicit momentum threshold
    }
  }
  
  protected applyTunedParameters(params: Record<string, number>): void {
    // Apply with bounds checking
    if (params.criticalFieldThreshold !== undefined) {
      this.config.criticalFieldThreshold = Math.max(0.1, Math.min(0.9, params.criticalFieldThreshold))
    }
    if (params.minFieldDistortion !== undefined) {
      this.config.minFieldDistortion = Math.max(0.05, Math.min(0.5, params.minFieldDistortion))
    }
    if (params.fieldRadius !== undefined) {
      this.config.fieldRadius = Math.max(0.01, Math.min(0.1, params.fieldRadius))
    }
    if (params.fieldDecayRate !== undefined) {
      this.config.fieldDecayRate = Math.max(1.0, Math.min(5.0, params.fieldDecayRate))
    }
    
    this.logger?.debug('Applied tuned parameters', {
      criticalFieldThreshold: this.config.criticalFieldThreshold,
      minFieldDistortion: this.config.minFieldDistortion
    })
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { currentPrice, candles } = context
    
    if (candles.length < 15) {
      return this.createSignal('hold', 0.3, 'Insufficient data for relativistic field analysis')
    }
    
    // 1. Calculate momentum field
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
    
    // Log performance metrics periodically
    if (this.fieldHistory.length % 50 === 0) {
      const metrics = this.getPerformanceMetrics()
      this.logger?.info('Relativistic Agent Performance', {
        winRate: `${(metrics.winRate * 100).toFixed(1)}%`,
        avgReturn: `${(metrics.avgReturn * 100).toFixed(2)}%`,
        signalCount: metrics.signalCount
      })
    }
    
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
  
  // Copy all the private methods from the original RelativisticFieldAgent
  // (I'll include the key ones here)
  
  private generateMomentumField(currentPrice: number, candles: any[]): MomentumField {
    const fieldPoints: FieldPoint[] = []
    const fieldRadius = currentPrice * this.config.fieldRadius
    const pointCount = this.config.fieldPointCount
    
    // Create field points above and below current price
    for (let i = 0; i < pointCount; i++) {
      const upPrice = currentPrice + (fieldRadius * (i + 1) / pointCount)
      const upPoint = this.createFieldPoint(upPrice, currentPrice, candles)
      fieldPoints.push(upPoint)
      
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
  
  private createFieldPoint(price: number, centerPrice: number, candles: any[]): FieldPoint {
    const priceDistance = Math.abs(price - centerPrice) / centerPrice
    const lorentzianDistance = Math.log(1 + priceDistance * priceDistance)
    const momentum = this.calculateMomentumAtPrice(price, candles)
    const fieldStrength = this.applyRelativisticDecay(momentum, lorentzianDistance)
    const momentumVector = {
      magnitude: Math.abs(momentum),
      direction: momentum > 0 ? 1 : -1
    }
    const relativistcWeight = 1 / (1 + lorentzianDistance * this.config.fieldDecayRate)
    
    return {
      price,
      distance: lorentzianDistance,
      momentumVector,
      fieldStrength,
      relativistcWeight
    }
  }
  
  private calculateMomentumAtPrice(targetPrice: number, candles: any[]): number {
    const cacheKey = `${targetPrice.toFixed(2)}-${candles.length}`
    if (this.momentumCache.has(cacheKey)) {
      return this.momentumCache.get(cacheKey)!
    }
    
    const recentCandles = candles.slice(-10)
    let weightedMomentum = 0
    let totalWeight = 0
    
    for (let i = 1; i < recentCandles.length; i++) {
      const price = recentCandles[i]!.close
      const prevPrice = recentCandles[i-1]!.close
      const priceChange = (price - prevPrice) / prevPrice * 100 // Scale to percentage
      
      const proximity = 1 / (1 + Math.abs(price - targetPrice) / targetPrice)
      const weight = proximity * proximity
      
      weightedMomentum += priceChange * weight
      totalWeight += weight
    }
    
    const momentum = totalWeight > 0 ? weightedMomentum / totalWeight : 0
    const limitedMomentum = Math.sign(momentum) * 
      Math.min(Math.abs(momentum), this.config.momentumSpeedLimit * 100)
    
    this.momentumCache.set(cacheKey, limitedMomentum)
    return limitedMomentum
  }
  
  private applyRelativisticDecay(momentum: number, distance: number): number {
    const decayFactor = 1 / (1 + distance * this.config.fieldDecayRate)
    return momentum * decayFactor
  }
  
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
  
  private calculateFieldDistortion(fieldPoints: FieldPoint[]): number {
    if (fieldPoints.length < 2) return 0
    
    let totalDistortion = 0
    let comparisons = 0
    
    const sortedPoints = [...fieldPoints].sort((a, b) => a.price - b.price)
    
    for (let i = 1; i < sortedPoints.length; i++) {
      const current = sortedPoints[i]!
      const previous = sortedPoints[i-1]!
      
      const momentumDiff = current.momentumVector.magnitude - previous.momentumVector.magnitude
      const directionDiff = current.momentumVector.direction - previous.momentumVector.direction
      const vectorDistance = Math.sqrt(momentumDiff * momentumDiff + directionDiff * directionDiff)
      const lorentzianDistortion = Math.log(1 + vectorDistance * vectorDistance)
      
      totalDistortion += lorentzianDistortion
      comparisons++
    }
    
    return comparisons > 0 ? totalDistortion / comparisons : 0
  }
  
  private calculateConvergenceRate(fieldPoints: FieldPoint[]): number {
    if (fieldPoints.length === 0) return 0
    
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
  
  private calculateRelativisticMomentum(fieldPoints: FieldPoint[]): number {
    if (fieldPoints.length === 0) return 0
    
    let totalMomentum = 0
    let validPoints = 0
    
    for (const point of fieldPoints) {
      const velocity = Math.abs(point.momentumVector.magnitude)
      const speedOfLight = this.config.momentumSpeedLimit * 100
      
      if (speedOfLight === 0 || !isFinite(velocity)) continue
      
      const velocityRatio = Math.min(0.99, velocity / speedOfLight)
      const lorentzFactor = 1 / Math.sqrt(1 - velocityRatio * velocityRatio)
      
      const relativistcMomentum = lorentzFactor * point.momentumVector.magnitude * point.relativistcWeight
      if (isFinite(relativistcMomentum)) {
        totalMomentum += relativistcMomentum * point.momentumVector.direction
        validPoints++
      }
    }
    
    return validPoints > 0 ? totalMomentum / validPoints : 0
  }
  
  private calculateFieldAlignment(fieldPoints: FieldPoint[]): number {
    if (fieldPoints.length === 0) return 0
    
    const directions = fieldPoints.map(p => p.momentumVector.direction * p.relativistcWeight)
    const avgDirection = directions.reduce((sum, dir) => sum + dir, 0) / directions.length
    
    const variance = directions.reduce((sum, dir) => 
      sum + Math.pow(dir - avgDirection, 2), 0) / directions.length
    
    return Math.max(0, 1 - variance)
  }
  
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
  
  private generateCriticalFieldSignal(
    fieldState: FieldState, 
    _currentPrice: number,
    context: MarketContext
  ): AgentSignal {
    const field = fieldState.field
    const momentum = field.relativistcMomentum
    const threshold = this.getTunableParameters().momentumThreshold || 0.005
    
    if (Math.abs(momentum) > threshold) {
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
  
  private detectFieldReversal(
    currentState: FieldState, 
    stateHistory: FieldState[]
  ): AgentSignal {
    if (stateHistory.length < 3) {
      return this.createSignal('hold', 0.4, 'Insufficient field reversal history')
    }
    
    const field = currentState.field
    const threshold = this.getTunableParameters().momentumThreshold || 0.005
    
    const convergingNegative = currentState.fieldTrend === 'converging' && 
                              field.relativistcMomentum < -threshold
    const divergingPositive = currentState.fieldTrend === 'diverging' && 
                             field.relativistcMomentum > threshold
    
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
  
  private matchHistoricalFieldPatterns(
    currentState: FieldState, 
    stateHistory: FieldState[]
  ): AgentSignal {
    if (stateHistory.length < 20) {
      return this.createSignal('hold', 0.4, 'Insufficient pattern history')
    }
    
    const current = currentState.field
    const matches: { distance: number, futureReturn: number }[] = []
    
    for (let i = 0; i < stateHistory.length - 5; i++) {
      const historical = stateHistory[i]!.field
      
      const alignmentDiff = current.fieldAlignment - historical.fieldAlignment
      const distortionDiff = current.fieldDistortion - historical.fieldDistortion
      const momentumDiff = current.relativistcMomentum - historical.relativistcMomentum
      
      const stateDiff = Math.sqrt(
        alignmentDiff * alignmentDiff + 
        distortionDiff * distortionDiff + 
        momentumDiff * momentumDiff
      )
      const lorentzianDistance = Math.log(1 + stateDiff * stateDiff)
      
      if (lorentzianDistance < 0.3) {
        const futureState = stateHistory[i + 5]
        if (futureState) {
          const futureReturn = (futureState.field.centerPrice - historical.centerPrice) / historical.centerPrice
          matches.push({ distance: lorentzianDistance, futureReturn })
        }
      }
    }
    
    if (matches.length >= 3) {
      const avgReturn = matches.reduce((sum, m) => sum + m.futureReturn, 0) / matches.length
      const confidence = Math.min(0.85, 0.5 + matches.length * 0.05)
      
      if (Math.abs(avgReturn) > 0.005) {
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
    await super.onReset()
    this.fieldHistory = []
    this.fieldStateHistory = []
    this.momentumCache.clear()
  }
}