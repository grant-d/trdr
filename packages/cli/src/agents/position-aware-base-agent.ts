import { BaseAgent } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { enforceNoShorting } from './helpers/position-aware'

/**
 * Base agent that enforces no shorting rule
 * All agents extending this class will automatically prevent short selling
 */
export abstract class PositionAwareBaseAgent extends BaseAgent {
  /**
   * Minimum position required to allow selling
   */
  protected minPositionForSell = 0.0001
  
  /**
   * Override analyze to enforce no shorting
   */
  async analyze(context: MarketContext): Promise<AgentSignal> {
    // Get original signal from parent
    const signal = await super.analyze(context)
    
    // Enforce no shorting rule
    return enforceNoShorting(signal, context, this.minPositionForSell)
  }
  
  /**
   * Create a position-aware signal that prevents shorting
   * This overrides the parent's createSignal method
   */
  protected createSignal(
    action: AgentSignal['action'],
    confidence: number,
    reason: string,
    analysis?: string,
    priceTarget?: number,
    stopLoss?: number,
    positionSize?: number
  ): AgentSignal {
    // Get the current context if available
    const context = this.currentContext
    
    const baseSignal = super.createSignal(
      action,
      confidence,
      reason,
      analysis,
      priceTarget,
      stopLoss,
      positionSize
    )
    
    // If we have context, enforce no shorting
    if (context) {
      return enforceNoShorting(baseSignal, context, this.minPositionForSell)
    }
    
    // No context available, return base signal
    // The analyze method will still enforce the rule
    return baseSignal
  }
  
  /**
   * Store current context for use in createSignal
   */
  private currentContext?: MarketContext
  
  /**
   * Override performAnalysis to capture context
   */
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    // Store context for createSignal
    this.currentContext = context
    
    try {
      // Call child's implementation
      const signal = await this.performAnalysisInternal(context)
      return signal
    } finally {
      // Clear context after analysis
      this.currentContext = undefined
    }
  }
  
  /**
   * Child classes should implement this instead of performAnalysis
   */
  protected abstract performAnalysisInternal(context: MarketContext): Promise<AgentSignal>
}