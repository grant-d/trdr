import { BaseAgent } from '@trdr/core'
import type { AgentSignal, MarketContext, AgentMetadata } from '@trdr/core/dist/agents/types'
import type { Logger } from '@trdr/types'
import { MarketRegimeDetector, type MarketRegime } from './market-regime-detector'

/**
 * Configuration options for adaptive agents
 */
export interface AdaptiveConfig {
  /** How quickly the agent adapts to new market conditions (0-1) */
  adaptationRate?: number
  /** Number of recent market regimes to remember for pattern detection */
  regimeMemory?: number
  /** Min/max bounds for each adaptive parameter */
  parameterBounds?: Record<string, { min: number, max: number }>
}

/**
 * Adaptive Base Agent Framework
 * 
 * Abstract base class that provides market-adaptive capabilities to trading agents.
 * This framework enables agents to dynamically adjust their parameters and strategies
 * based on detected market conditions.
 * 
 * Key Features:
 * - **Automatic Regime Detection**: Analyzes market conditions before each signal
 * - **Parameter Adaptation**: Adjusts agent-specific parameters based on regime
 * - **Confidence Adjustment**: Modifies signal confidence based on regime alignment
 * - **Performance Tracking**: Monitors agent performance for future adaptations
 * 
 * Usage:
 * 1. Extend this class instead of BaseAgent
 * 2. Implement `performAdaptiveAnalysis` for your agent logic
 * 3. Implement `adaptParameters` to adjust parameters based on regime
 * 4. Use `createAdaptiveSignal` to generate regime-aware signals
 * 
 * The agent will automatically:
 * - Detect market regime before analysis
 * - Call your adaptation logic
 * - Adjust signal confidence based on regime alignment
 * - Track performance metrics
 */
export abstract class AdaptiveBaseAgent<T = unknown> extends BaseAgent {
  protected regimeDetector = new MarketRegimeDetector()
  protected currentRegime: MarketRegime | null = null
  protected regimeHistory: MarketRegime[] = []
  protected adaptiveConfig: Required<AdaptiveConfig>
  
  // Performance tracking for adaptation
  protected recentPerformance: { correct: number, total: number } = { correct: 0, total: 0 }
  protected parameterPerformance: Map<string, { value: T, score: number }>
  
  constructor(metadata: AgentMetadata, logger?: Logger, adaptiveConfig?: AdaptiveConfig) {
    super(metadata, logger)
    this.parameterPerformance = new Map<string, { value: T, score: number }>()
    this.adaptiveConfig = {
      adaptationRate: adaptiveConfig?.adaptationRate ?? 0.1,
      regimeMemory: adaptiveConfig?.regimeMemory ?? 10,
      parameterBounds: adaptiveConfig?.parameterBounds ?? {}
    }
  }
  
  /**
   * Main analysis method that includes regime detection and adaptation
   * 
   * This method orchestrates the adaptive analysis process:
   * 1. Detects current market regime using multiple indicators
   * 2. Updates regime history for pattern recognition
   * 3. Calls agent-specific parameter adaptation
   * 4. Performs agent-specific analysis with adapted parameters
   * 5. Tracks performance metrics for future improvements
   * 
   * @param context - Current market context with price and volume data
   * @returns Trading signal with regime-adjusted confidence
   */
  async analyze(context: MarketContext): Promise<AgentSignal> {
    // Detect current market regime
    this.currentRegime = this.regimeDetector.detectRegime(context.candles)
    
    // Update regime history
    this.regimeHistory.push(this.currentRegime)
    if (this.regimeHistory.length > this.adaptiveConfig.regimeMemory) {
      this.regimeHistory.shift()
    }
    
    // Adapt parameters based on regime
    await this.adaptParameters(this.currentRegime)
    
    // Perform agent-specific analysis
    const signal = await this.performAdaptiveAnalysis(context, this.currentRegime)
    
    // Track performance for future adaptation
    this.trackSignalPerformance(signal, context)
    
    return signal
  }
  
  /**
   * Abstract method for agent-specific analysis that considers market regime
   */
  protected abstract performAdaptiveAnalysis(
    context: MarketContext, 
    regime: MarketRegime
  ): Promise<AgentSignal>
  
  /**
   * Abstract method for adapting agent-specific parameters
   */
  protected abstract adaptParameters(regime: MarketRegime): Promise<void>
  
  /**
   * Get parameter adjustment based on regime
   */
  protected getRegimeAdjustment(
    baseValue: number,
    parameterName: string,
    regime: MarketRegime
  ): number {
    let adjustment = 1.0
    
    // Adjust based on volatility
    if (regime.volatility === 'high') {
      adjustment *= parameterName.includes('period') ? 0.8 : 1.2 // Shorter periods in high volatility
    } else if (regime.volatility === 'low') {
      adjustment *= parameterName.includes('period') ? 1.2 : 0.8 // Longer periods in low volatility
    }
    
    // Adjust based on trend
    if (regime.trend !== 'neutral' && parameterName.includes('threshold')) {
      adjustment *= regime.momentum === 'strong' ? 1.1 : 0.9 // Relax thresholds in strong trends
    }
    
    // Adjust based on regime type
    switch (regime.regime) {
      case 'trending':
        if (parameterName.includes('momentum') || parameterName.includes('trend')) {
          adjustment *= 1.15 // Emphasize trend-following parameters
        }
        break
      case 'ranging':
        if (parameterName.includes('overbought') || parameterName.includes('oversold')) {
          adjustment *= 0.9 // Tighten mean-reversion parameters
        }
        break
      case 'breakout':
        if (parameterName.includes('volume') || parameterName.includes('volatility')) {
          adjustment *= 1.2 // Emphasize breakout parameters
        }
        break
      case 'reversal':
        if (parameterName.includes('divergence')) {
          adjustment *= 1.25 // Emphasize reversal parameters
        }
        break
    }
    
    // Apply bounds if specified
    const bounds = this.adaptiveConfig.parameterBounds[parameterName]
    if (bounds) {
      const adjustedValue = baseValue * adjustment
      return Math.max(bounds.min, Math.min(bounds.max, adjustedValue))
    }
    
    return baseValue * adjustment
  }
  
  /**
   * Adjust signal confidence based on regime alignment
   */
  protected adjustConfidenceForRegime(
    baseConfidence: number,
    signalType: 'buy' | 'sell' | 'hold',
    regime: MarketRegime
  ): number {
    let adjustment = 0
    
    // Trend alignment
    if (regime.trend === 'bullish' && signalType === 'buy') adjustment += 0.1
    else if (regime.trend === 'bearish' && signalType === 'sell') adjustment += 0.1
    else if (regime.trend !== 'neutral' && signalType === 'hold') adjustment -= 0.05
    
    // Momentum alignment
    if (regime.momentum === 'strong' && signalType !== 'hold') adjustment += 0.05
    else if (regime.momentum === 'weak' && signalType === 'hold') adjustment += 0.05
    
    // Regime-specific adjustments
    switch (regime.regime) {
      case 'trending':
        if ((regime.trend === 'bullish' && signalType === 'buy') ||
            (regime.trend === 'bearish' && signalType === 'sell')) {
          adjustment += 0.15
        }
        break
      case 'ranging':
        if (signalType === 'hold') adjustment += 0.1
        break
      case 'breakout':
        if (signalType !== 'hold' && regime.volume === 'increasing') adjustment += 0.1
        break
      case 'reversal':
        // Be more cautious during reversals
        adjustment -= 0.1
        break
    }
    
    // Apply regime confidence
    const regimeAdjustedConfidence = baseConfidence + adjustment
    const finalConfidence = regimeAdjustedConfidence * (0.7 + regime.confidence * 0.3)
    
    return Math.max(0.1, Math.min(0.95, finalConfidence))
  }
  
  /**
   * Track signal performance for adaptation
   */
  private trackSignalPerformance(_signal: AgentSignal, _context: MarketContext): void {
    // This would be called later to evaluate if the signal was correct
    // For now, we just track that a signal was made
    this.recentPerformance.total++
    
    // Limit performance history
    if (this.recentPerformance.total > 100) {
      this.recentPerformance.correct = Math.floor(this.recentPerformance.correct * 0.9)
      this.recentPerformance.total = Math.floor(this.recentPerformance.total * 0.9)
    }
  }
  
  /**
   * Get performance score for adaptation decisions
   */
  protected getPerformanceScore(): number {
    if (this.recentPerformance.total === 0) return 0.5
    return this.recentPerformance.correct / this.recentPerformance.total
  }
  
  /**
   * Create enhanced signal with regime information
   */
  protected createAdaptiveSignal(
    action: 'buy' | 'sell' | 'hold',
    baseConfidence: number,
    reason: string
  ): AgentSignal {
    const adjustedConfidence = this.adjustConfidenceForRegime(baseConfidence, action, this.currentRegime!)
    
    const enhancedReason = `${reason} [${this.currentRegime!.regime} market, ${this.currentRegime!.volatility} volatility]`
    
    return this.createSignal(action, adjustedConfidence, enhancedReason)
  }
}