import type {
  IPositionSizingStrategy,
  PositionSizingInput,
  PositionSizingOutput,
  PositionSizingConfig,
  MarketConditions,
  RiskParameters
} from './interfaces'
import { KellyCriterionStrategy } from './strategies/kelly-criterion'
import { FixedFractionalStrategy } from './strategies/fixed-fractional'
import { VolatilityAdjustedStrategy } from './strategies/volatility-adjusted'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

/**
 * Manages position sizing strategies with pluggable implementations.
 * 
 * Provides a unified interface for calculating position sizes using various
 * risk management strategies. Supports dynamic strategy selection based on
 * market conditions and performance metrics.
 */
export class PositionSizingManager {
  private readonly strategies = new Map<string, IPositionSizingStrategy>()
  private readonly config: PositionSizingConfig
  private readonly eventBus: EventBus
  private currentStrategy: string
  
  /**
   * Creates a new PositionSizingManager
   * @param config - Position sizing configuration
   * @param eventBus - Optional event bus for publishing sizing events
   */
  constructor(config: PositionSizingConfig, eventBus?: EventBus) {
    this.config = config
    this.eventBus = eventBus || EventBus.getInstance()
    this.currentStrategy = config.defaultStrategy
    
    // Initialize default strategies
    this.registerDefaultStrategies()
    this.registerEventTypes()
  }
  
  /**
   * Register event types used by position sizing
   */
  private registerEventTypes(): void {
    this.eventBus.registerEvent(EventTypes.SYSTEM_INFO)
  }
  
  /**
   * Register default position sizing strategies
   */
  private registerDefaultStrategies(): void {
    // Kelly Criterion with 25% safety factor
    this.registerStrategy('kelly', new KellyCriterionStrategy(0.25, 0.25))
    
    // Fixed Fractional with 1% risk
    this.registerStrategy('fixed', new FixedFractionalStrategy(0.01))
    
    // Volatility Adjusted with 1% base risk
    this.registerStrategy('volatility', new VolatilityAdjustedStrategy(0.01, 20, 3))
  }
  
  /**
   * Register a custom position sizing strategy
   * @param name - Strategy identifier
   * @param strategy - Strategy implementation
   */
  registerStrategy(name: string, strategy: IPositionSizingStrategy): void {
    this.strategies.set(name, strategy)
  }
  
  /**
   * Get available strategy names
   * @returns Array of registered strategy names
   */
  getAvailableStrategies(): string[] {
    return Array.from(this.strategies.keys())
  }
  
  /**
   * Set the active strategy
   * @param name - Strategy name to activate
   * @throws Error if strategy not found
   */
  setActiveStrategy(name: string): void {
    if (!this.strategies.has(name)) {
      throw new Error(`Strategy '${name}' not found`)
    }
    this.currentStrategy = name
  }
  
  /**
   * Get the current active strategy name
   * @returns Current strategy name
   */
  getActiveStrategy(): string {
    return this.currentStrategy
  }
  
  /**
   * Calculate position size using the active strategy
   * @param input - Position sizing parameters
   * @returns Position sizing output with recommendations
   */
  calculatePositionSize(input: PositionSizingInput): PositionSizingOutput {
    // Get the active strategy
    const strategy = this.strategies.get(this.currentStrategy)
    if (!strategy) {
      throw new Error(`Active strategy '${this.currentStrategy}' not found`)
    }
    
    // Validate input
    const validation = strategy.validate(input)
    if (!validation.valid) {
      return {
        positionSize: 0,
        positionValue: 0,
        riskAmount: 0,
        riskPercentage: 0,
        method: strategy.name,
        confidence: 0,
        reasoning: 'Validation failed',
        warnings: validation.errors,
        adjustments: []
      }
    }
    
    // Apply pre-processing adjustments
    const adjustedInput = this.applyPreProcessing(input)
    
    // Calculate size using strategy
    let output = strategy.calculateSize(adjustedInput)
    
    // Apply post-processing adjustments
    output = this.applyPostProcessing(output, adjustedInput)
    
    // Apply min/max constraints
    output = this.applyConstraints(output, adjustedInput)
    
    // Emit sizing event
    this.eventBus.emit(EventTypes.SYSTEM_INFO, {
      message: 'Position sizing calculated',
      context: 'position-sizing',
      details: {
        strategy: this.currentStrategy,
        input: adjustedInput,
        output
      },
      timestamp: new Date()
    })
    
    return output
  }
  
  /**
   * Apply pre-processing adjustments to input
   */
  private applyPreProcessing(input: PositionSizingInput): PositionSizingInput {
    // Clone input to avoid mutations
    const adjusted = { ...input }
    
    // Apply adaptive adjustments if enabled
    if (this.config.enableAdaptive && input.historicalMetrics) {
      // Reduce confidence during drawdowns
      if (input.historicalMetrics.currentConsecutiveLosses > 3) {
        adjusted.confidence *= 0.7
      }
      
      // Increase confidence during winning streaks
      if (input.historicalMetrics.profitFactor > 2 && input.historicalMetrics.sharpeRatio > 1.5) {
        adjusted.confidence = Math.min(adjusted.confidence * 1.2, 1.0)
      }
    }
    
    return adjusted
  }
  
  /**
   * Apply post-processing adjustments to output
   */
  private applyPostProcessing(
    output: PositionSizingOutput,
    input: PositionSizingInput
  ): PositionSizingOutput {
    const adjustedOutput = { 
      ...output,
      adjustments: [...output.adjustments],
      warnings: [...output.warnings]
    }
    
    // Apply market condition adjustments if enabled
    if (this.config.enableMarketAdjustments) {
      // Reduce size in low volume conditions
      if (input.marketConditions.relativeVolume < 0.5) {
        const volumeFactor = 0.5 + input.marketConditions.relativeVolume
        adjustedOutput.positionSize *= volumeFactor
        adjustedOutput.positionValue *= volumeFactor
        adjustedOutput.riskAmount *= volumeFactor
        adjustedOutput.adjustments.push({
          type: 'market_conditions',
          factor: volumeFactor,
          reason: 'Low volume adjustment'
        })
      }
      
      // Reduce size in high spread conditions
      if (input.marketConditions.spread > 0.002) { // 0.2%
        const spreadFactor = Math.max(0.5, 1 - input.marketConditions.spread * 50)
        adjustedOutput.positionSize *= spreadFactor
        adjustedOutput.positionValue *= spreadFactor
        adjustedOutput.riskAmount *= spreadFactor
        adjustedOutput.adjustments.push({
          type: 'market_conditions',
          factor: spreadFactor,
          reason: `High spread adjustment: ${(input.marketConditions.spread * 100).toFixed(3)}%`
        })
      }
    }
    
    return adjustedOutput
  }
  
  /**
   * Apply min/max constraints to output
   */
  private applyConstraints(
    output: PositionSizingOutput,
    _input: PositionSizingInput
  ): PositionSizingOutput {
    const constrained = { 
      ...output,
      adjustments: [...output.adjustments],
      warnings: [...output.warnings]
    }
    
    // Apply minimum position size
    if (constrained.positionSize < this.config.minPositionSize && constrained.positionSize > 0) {
      constrained.warnings.push(`Position size below minimum ${this.config.minPositionSize}`)
      constrained.positionSize = 0
      constrained.positionValue = 0
      constrained.riskAmount = 0
      constrained.riskPercentage = 0
      constrained.confidence = 0
      constrained.reasoning = 'Position too small - below minimum size'
    }
    
    // Apply maximum position size
    if (constrained.positionSize > this.config.maxPositionSize) {
      const scaleFactor = this.config.maxPositionSize / constrained.positionSize
      constrained.positionSize = this.config.maxPositionSize
      constrained.positionValue *= scaleFactor
      constrained.riskAmount *= scaleFactor
      constrained.riskPercentage *= scaleFactor
      constrained.warnings.push(`Position size capped at maximum ${this.config.maxPositionSize}`)
      constrained.adjustments.push({
        type: 'risk_limit',
        factor: scaleFactor,
        reason: 'Maximum position size limit'
      })
    }
    
    return constrained
  }
  
  /**
   * Get recommended strategy based on market conditions
   * @param conditions - Current market conditions
   * @returns Recommended strategy name and reasoning
   */
  getRecommendedStrategy(conditions: MarketConditions): {
    strategy: string
    confidence: number
    reasoning: string
  } {
    // High volatility - use volatility adjusted
    if (conditions.volatility > 0.7) {
      return {
        strategy: 'volatility',
        confidence: 0.9,
        reasoning: 'High volatility detected - volatility-adjusted sizing recommended'
      }
    }
    
    // Trending market with good liquidity - Kelly might work well
    if (conditions.regime === 'trending' && conditions.relativeVolume > 0.8 && conditions.spread < 0.001) {
      return {
        strategy: 'kelly',
        confidence: 0.8,
        reasoning: 'Strong trend with good liquidity - Kelly criterion may optimize returns'
      }
    }
    
    // Default to fixed fractional for stability
    return {
      strategy: 'fixed',
      confidence: 0.7,
      reasoning: 'Standard market conditions - fixed fractional provides consistent risk'
    }
  }
  
  /**
   * Create risk parameters from account state
   * @param accountBalance - Total account balance
   * @param currentExposure - Current total exposure
   * @param openPositions - Number of open positions
   * @returns Risk parameters object
   */
  static createRiskParameters(
    accountBalance: number,
    currentExposure = 0,
    openPositions = 0
  ): RiskParameters {
    return {
      accountBalance,
      maxRiskPerTrade: 0.02, // 2% default
      maxPortfolioRisk: 0.06, // 6% default
      currentExposure,
      openPositions,
      maxPositions: 5,
      riskFreeRate: 0.02 // 2% annual
    }
  }
  
  /**
   * Assess current market conditions
   * @param price - Current price
   * @param recentPrices - Recent price history
   * @param volume - Current volume
   * @param avgVolume - Average volume
   * @returns Market conditions assessment
   */
  static assessMarketConditions(
    price: number,
    recentPrices: number[],
    volume: number,
    avgVolume: number
  ): MarketConditions {
    // Calculate volatility (simplified - use standard deviation)
    const returns = recentPrices.slice(1).map((p, i) => {
      const prevPrice = recentPrices[i]
      return prevPrice ? (p - prevPrice) / prevPrice : 0
    })
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    const stdDev = Math.sqrt(variance)
    const annualizedVol = stdDev * Math.sqrt(252) // Assuming daily data
    
    // Normalize volatility to 0-1 scale (0.5 = 50% annual vol)
    const volatility = Math.min(1, annualizedVol / 0.5)
    
    // Calculate spread (simplified - would use order book in practice)
    const spread = 0.0005 // 0.05% placeholder
    
    // Detect trend
    const sma50 = recentPrices.slice(-50).reduce((a, b) => a + b, 0) / Math.min(50, recentPrices.length)
    const trendStrength = (price - sma50) / sma50
    
    // Determine regime
    let regime: 'trending' | 'ranging' | 'volatile'
    if (volatility > 0.7) {
      regime = 'volatile'
    } else if (Math.abs(trendStrength) > 0.1) {
      regime = 'trending'
    } else {
      regime = 'ranging'
    }
    
    // Time of day (simplified - crypto trades 24/7)
    const hour = new Date().getUTCHours()
    const timeOfDayFactor = (hour >= 13 && hour <= 21) ? 1.0 : 0.7 // Peak during US hours
    
    return {
      volatility,
      spread,
      trendStrength,
      relativeVolume: volume / avgVolume,
      regime,
      timeOfDayFactor
    }
  }
}