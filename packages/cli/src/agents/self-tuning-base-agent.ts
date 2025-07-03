import { BaseAgent } from '@trdr/core'
import type { AgentMetadata, AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import type { Logger } from '@trdr/types'

export interface SelfTuningConfig {
  /** Enable self-tuning */
  enableSelfTuning?: boolean
  /** Performance evaluation window (number of signals) */
  evaluationWindow?: number
  /** Minimum signals before tuning starts */
  minSignalsForTuning?: number
  /** Learning rate for parameter updates */
  learningRate?: number
  /** Performance threshold to trigger tuning */
  performanceThreshold?: number
}

interface PerformanceMetrics {
  signalCount: number
  correctDirections: number
  avgConfidence: number
  avgReturn: number
  winRate: number
  profitFactor: number
  lastTuned: number
}

interface SignalOutcome {
  signal: AgentSignal
  entryPrice: number
  exitPrice?: number
  return?: number
  correct?: boolean
  timestamp: number
}

/**
 * Self-Tuning Base Agent
 * 
 * Extends BaseAgent with self-tuning capabilities that automatically
 * adjust agent parameters based on performance metrics.
 * 
 * Features:
 * - Tracks signal outcomes and performance metrics
 * - Automatically adjusts thresholds and parameters
 * - Learns from both successes and failures
 * - Adapts to changing market conditions
 * - Provides performance reporting
 */
export abstract class SelfTuningBaseAgent extends BaseAgent {
  protected selfTuningConfig: Required<SelfTuningConfig>
  private signalHistory: SignalOutcome[] = []
  private performanceMetrics: PerformanceMetrics = {
    signalCount: 0,
    correctDirections: 0,
    avgConfidence: 0,
    avgReturn: 0,
    winRate: 0,
    profitFactor: 1,
    lastTuned: Date.now()
  }
  
  constructor(metadata: AgentMetadata, logger?: Logger, selfTuningConfig?: SelfTuningConfig) {
    super(metadata, logger)
    
    this.selfTuningConfig = {
      enableSelfTuning: selfTuningConfig?.enableSelfTuning ?? true,
      evaluationWindow: selfTuningConfig?.evaluationWindow ?? 20,
      minSignalsForTuning: selfTuningConfig?.minSignalsForTuning ?? 10,
      learningRate: selfTuningConfig?.learningRate ?? 0.1,
      performanceThreshold: selfTuningConfig?.performanceThreshold ?? 0.6
    }
  }
  
  /**
   * Override analyze to track signal outcomes
   */
  async analyze(context: MarketContext): Promise<AgentSignal> {
    // Update outcomes for previous signals
    this.updateSignalOutcomes(context.currentPrice)
    
    // Perform the actual analysis
    const signal = await this.performAnalysis(context)
    
    // Track this signal
    if (signal.action !== 'hold') {
      this.signalHistory.push({
        signal,
        entryPrice: context.currentPrice,
        timestamp: Date.now()
      })
    }
    
    // Perform self-tuning if enabled
    if (this.selfTuningConfig.enableSelfTuning) {
      this.performSelfTuning()
    }
    
    return signal
  }
  
  /**
   * Update outcomes for tracked signals
   */
  private updateSignalOutcomes(currentPrice: number): void {
    const oneHourAgo = Date.now() - 60 * 60 * 1000
    
    for (const outcome of this.signalHistory) {
      // Skip if already evaluated or too recent
      if (outcome.exitPrice || Date.now() - outcome.timestamp < 5 * 60 * 1000) {
        continue
      }
      
      // Evaluate after 5 minutes or 1 hour max
      if (Date.now() - outcome.timestamp >= 5 * 60 * 1000) {
        outcome.exitPrice = currentPrice
        outcome.return = (currentPrice - outcome.entryPrice) / outcome.entryPrice
        
        // Check if direction was correct
        if (outcome.signal.action === 'buy') {
          outcome.correct = outcome.return > 0
        } else if (outcome.signal.action === 'sell') {
          outcome.correct = outcome.return < 0
        }
      }
    }
    
    // Clean up old signals
    this.signalHistory = this.signalHistory.filter(s => s.timestamp > oneHourAgo)
  }
  
  /**
   * Perform self-tuning based on performance
   */
  private performSelfTuning(): void {
    const evaluatedSignals = this.signalHistory.filter(s => s.exitPrice !== undefined)
    
    if (evaluatedSignals.length < this.selfTuningConfig.minSignalsForTuning) {
      return
    }
    
    // Calculate recent performance
    const recentSignals = evaluatedSignals.slice(-this.selfTuningConfig.evaluationWindow)
    const metrics = this.calculatePerformanceMetrics(recentSignals)
    
    // Update stored metrics
    this.performanceMetrics = metrics
    
    // Check if tuning is needed
    if (metrics.winRate < this.selfTuningConfig.performanceThreshold) {
      this.logger?.info(`Self-tuning triggered: Win rate ${(metrics.winRate * 100).toFixed(1)}%`)
      this.tuneParameters(metrics)
      this.performanceMetrics.lastTuned = Date.now()
    }
  }
  
  /**
   * Calculate performance metrics from signal outcomes
   */
  private calculatePerformanceMetrics(signals: SignalOutcome[]): PerformanceMetrics {
    if (signals.length === 0) {
      return this.performanceMetrics
    }
    
    const correctSignals = signals.filter(s => s.correct).length
    const totalReturns = signals.reduce((sum, s) => sum + (s.return || 0), 0)
    const avgConfidence = signals.reduce((sum, s) => sum + s.signal.confidence, 0) / signals.length
    
    // Calculate profit factor
    const profits = signals.filter(s => (s.return || 0) > 0).reduce((sum, s) => sum + (s.return || 0), 0)
    const losses = Math.abs(signals.filter(s => (s.return || 0) < 0).reduce((sum, s) => sum + (s.return || 0), 0))
    const profitFactor = losses > 0 ? profits / losses : profits > 0 ? 999 : 1
    
    return {
      signalCount: signals.length,
      correctDirections: correctSignals,
      avgConfidence,
      avgReturn: totalReturns / signals.length,
      winRate: correctSignals / signals.length,
      profitFactor,
      lastTuned: this.performanceMetrics.lastTuned
    }
  }
  
  /**
   * Get current performance metrics
   */
  getPerformanceMetrics(): PerformanceMetrics {
    return { ...this.performanceMetrics }
  }
  
  /**
   * Get current tunable parameters
   */
  protected abstract getTunableParameters(): Record<string, number>
  
  /**
   * Apply tuned parameters
   */
  protected abstract applyTunedParameters(params: Record<string, number>): void
  
  /**
   * Tune parameters based on performance
   */
  private tuneParameters(metrics: PerformanceMetrics): void {
    const currentParams = this.getTunableParameters()
    const tunedParams: Record<string, number> = {}
    
    // Learning rate adjusted by performance
    const adaptiveLearningRate = this.selfTuningConfig.learningRate * (1 - metrics.winRate)
    
    for (const [key, value] of Object.entries(currentParams)) {
      // Different tuning strategies based on parameter type
      if (key.includes('threshold') || key.includes('Threshold')) {
        // For thresholds: increase if too many signals, decrease if too few
        const signalRate = metrics.signalCount / Math.max(1, this.signalHistory.length)
        if (signalRate > 0.3) {
          // Too many signals, increase threshold
          tunedParams[key] = value * (1 + adaptiveLearningRate)
        } else if (signalRate < 0.1) {
          // Too few signals, decrease threshold
          tunedParams[key] = value * (1 - adaptiveLearningRate)
        } else {
          tunedParams[key] = value
        }
      } else if (key.includes('sensitivity') || key.includes('Sensitivity')) {
        // For sensitivity: increase if missing opportunities, decrease if too noisy
        if (metrics.winRate < 0.4) {
          // Poor performance, adjust sensitivity
          tunedParams[key] = value * (1 - adaptiveLearningRate * 0.5)
        } else {
          tunedParams[key] = value
        }
      } else if (key.includes('period') || key.includes('Period')) {
        // For periods: adjust based on market volatility
        // Shorter periods for volatile markets, longer for stable
        tunedParams[key] = Math.round(value) // Keep as integer
      } else {
        // Default: small random adjustments
        const adjustment = (Math.random() - 0.5) * adaptiveLearningRate
        tunedParams[key] = value * (1 + adjustment)
      }
    }
    
    // Apply the tuned parameters
    this.applyTunedParameters(tunedParams)
    
    this.logger?.debug('Parameters tuned', {
      oldParams: currentParams,
      newParams: tunedParams,
      metrics
    })
  }
  
  /**
   * Reset performance tracking
   */
  protected async onReset(): Promise<void> {
    this.signalHistory = []
    this.performanceMetrics = {
      signalCount: 0,
      correctDirections: 0,
      avgConfidence: 0,
      avgReturn: 0,
      winRate: 0,
      profitFactor: 1,
      lastTuned: Date.now()
    }
  }
}