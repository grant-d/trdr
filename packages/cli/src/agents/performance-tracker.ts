import type { AgentSignal } from '@trdr/core/dist/agents/types'

export interface AgentPerformance {
  agentId: string
  totalSignals: number
  buySignals: number
  sellSignals: number
  holdSignals: number
  correctDirections: number
  wrongDirections: number
  totalReturn: number
  avgConfidence: number
  profitFactor: number
  sharpeRatio: number
  usefulness: number
}

export interface SignalEvaluation {
  signal: AgentSignal
  entryPrice: number
  exitPrice?: number
  outcome?: 'correct' | 'wrong' | 'neutral'
  return?: number
  evaluatedAt?: number
}

/**
 * Enhanced Performance Tracker for Trading Agents
 * 
 * Provides more nuanced usefulness calculations that:
 * - Give partial credit for reasonable decisions
 * - Consider confidence levels
 * - Track risk-adjusted returns
 * - Adapt to market conditions
 */
export class AgentPerformanceTracker {
  private readonly signalHistory = new Map<string, SignalEvaluation[]>()
  private readonly performanceCache = new Map<string, AgentPerformance>()
  private marketVolatility = 0.02 // Default 2% volatility
  
  constructor(private readonly lookbackPeriod = 20) {}
  
  /**
   * Record a new signal from an agent
   */
  recordSignal(agentId: string, signal: AgentSignal, entryPrice: number): void {
    const history = this.signalHistory.get(agentId) || []
    history.push({
      signal,
      entryPrice,
      evaluatedAt: Date.now()
    })
    this.signalHistory.set(agentId, history)
  }
  
  /**
   * Update market volatility for adaptive scoring
   */
  updateMarketVolatility(volatility: number): void {
    this.marketVolatility = Math.max(0.001, volatility)
  }
  
  /**
   * Evaluate signals with current price
   */
  evaluateSignals(currentPrice: number): void {
    const evaluationDelay = 5 * 60 * 1000 // 5 minutes
    const now = Date.now()
    
    for (const [agentId, history] of this.signalHistory) {
      for (const evaluation of history) {
        // Skip if already evaluated or too recent
        if (evaluation.exitPrice || now - evaluation.evaluatedAt! < evaluationDelay) {
          continue
        }
        
        evaluation.exitPrice = currentPrice
        evaluation.return = (currentPrice - evaluation.entryPrice) / evaluation.entryPrice
        evaluation.outcome = this.evaluateOutcome(evaluation)
      }
    }
  }
  
  /**
   * Evaluate signal outcome with nuanced scoring
   */
  private evaluateOutcome(evaluation: SignalEvaluation): 'correct' | 'wrong' | 'neutral' {
    const { signal, return: returnPct } = evaluation
    if (!returnPct) return 'neutral'
    
    const absReturn = Math.abs(returnPct)
    const significantMove = absReturn > this.marketVolatility * 0.5
    
    switch (signal.action) {
      case 'buy':
        if (returnPct > this.marketVolatility * 0.3) return 'correct'
        if (returnPct < -this.marketVolatility * 0.5) return 'wrong'
        return 'neutral'
        
      case 'sell':
        if (returnPct < -this.marketVolatility * 0.3) return 'correct'
        if (returnPct > this.marketVolatility * 0.5) return 'wrong'
        return 'neutral'
        
      case 'hold':
        // Hold is correct if price doesn't move significantly
        return significantMove ? 'wrong' : 'correct'
    }
  }
  
  /**
   * Calculate enhanced usefulness score
   */
  calculateUsefulness(agentId: string, currentPrice: number): number {
    // First evaluate any pending signals
    this.evaluateSignals(currentPrice)
    
    const history = this.signalHistory.get(agentId) || []
    const recentSignals = history.slice(-this.lookbackPeriod)
    
    if (recentSignals.length === 0) return 0.5
    
    let score = 0
    let weightSum = 0
    
    for (const evaluation of recentSignals) {
      if (!evaluation.outcome) continue
      
      const { signal, outcome, return: returnPct } = evaluation
      const weight = signal.confidence
      
      // Base score based on outcome
      let signalScore = outcome === 'correct' ? 0.8 : 
                       outcome === 'wrong' ? 0.2 : 0.5
      
      // Confidence bonus/penalty
      if (outcome === 'correct' && signal.confidence > 0.7) {
        signalScore += 0.1
      } else if (outcome === 'wrong' && signal.confidence > 0.7) {
        signalScore -= 0.1
      }
      
      // Magnitude bonus for profitable trades
      if (returnPct && outcome === 'correct') {
        const magnitudeBonus = Math.min(0.1, Math.abs(returnPct) * 2)
        signalScore += magnitudeBonus
      }
      
      // Risk adjustment - penalize high-confidence wrong signals more
      if (outcome === 'wrong') {
        signalScore *= (1 - signal.confidence * 0.3)
      }
      
      score += signalScore * weight
      weightSum += weight
    }
    
    // Calculate final usefulness
    const baseUsefulness = weightSum > 0 ? score / weightSum : 0.5
    
    // Add small bonus for consistency
    const consistency = this.calculateConsistency(recentSignals)
    const finalUsefulness = baseUsefulness * 0.9 + consistency * 0.1
    
    return Math.max(0, Math.min(1, finalUsefulness))
  }
  
  /**
   * Calculate consistency bonus
   */
  private calculateConsistency(signals: SignalEvaluation[]): number {
    const outcomes = signals.filter(s => s.outcome).map(s => s.outcome === 'correct' ? 1 : 0)
    if (outcomes.length < 3) return 0.5
    
    // Calculate variance in performance
    const mean = outcomes.reduce((a, b) => a + b, 0) / outcomes.length
    const variance = outcomes.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / outcomes.length
    
    // Lower variance = more consistent
    return Math.max(0, 1 - Math.sqrt(variance))
  }
  
  /**
   * Get comprehensive performance metrics
   */
  getPerformanceMetrics(agentId: string): AgentPerformance {
    const cached = this.performanceCache.get(agentId)
    if (cached && Date.now() - cached.totalSignals < 60000) {
      return cached
    }
    
    const history = this.signalHistory.get(agentId) || []
    const evaluatedSignals = history.filter(s => s.outcome)
    
    if (evaluatedSignals.length === 0) {
      return {
        agentId,
        totalSignals: 0,
        buySignals: 0,
        sellSignals: 0,
        holdSignals: 0,
        correctDirections: 0,
        wrongDirections: 0,
        totalReturn: 0,
        avgConfidence: 0,
        profitFactor: 1,
        sharpeRatio: 0,
        usefulness: 0.5
      }
    }
    
    // Calculate metrics
    const buySignals = evaluatedSignals.filter(s => s.signal.action === 'buy').length
    const sellSignals = evaluatedSignals.filter(s => s.signal.action === 'sell').length
    const holdSignals = evaluatedSignals.filter(s => s.signal.action === 'hold').length
    const correctDirections = evaluatedSignals.filter(s => s.outcome === 'correct').length
    const wrongDirections = evaluatedSignals.filter(s => s.outcome === 'wrong').length
    
    const returns = evaluatedSignals.map(s => s.return || 0)
    const totalReturn = returns.reduce((a, b) => a + b, 0)
    const avgConfidence = evaluatedSignals.reduce((sum, s) => sum + s.signal.confidence, 0) / evaluatedSignals.length
    
    // Profit factor
    const profits = returns.filter(r => r > 0).reduce((a, b) => a + b, 0)
    const losses = Math.abs(returns.filter(r => r < 0).reduce((a, b) => a + b, 0))
    const profitFactor = losses > 0 ? profits / losses : profits > 0 ? 999 : 1
    
    // Sharpe ratio (simplified)
    const avgReturn = totalReturn / evaluatedSignals.length
    const returnStdDev = Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length)
    const sharpeRatio = returnStdDev > 0 ? avgReturn / returnStdDev : 0
    
    const performance: AgentPerformance = {
      agentId,
      totalSignals: evaluatedSignals.length,
      buySignals,
      sellSignals,
      holdSignals,
      correctDirections,
      wrongDirections,
      totalReturn,
      avgConfidence,
      profitFactor,
      sharpeRatio,
      usefulness: this.calculateUsefulness(agentId, 0) // Will be updated with current price
    }
    
    this.performanceCache.set(agentId, performance)
    return performance
  }
  
  /**
   * Clean up old data
   */
  cleanup(maxAge = 24 * 60 * 60 * 1000): void {
    const cutoff = Date.now() - maxAge
    
    for (const [agentId, history] of this.signalHistory) {
      const filtered = history.filter(s => s.evaluatedAt! > cutoff)
      if (filtered.length > 0) {
        this.signalHistory.set(agentId, filtered)
      } else {
        this.signalHistory.delete(agentId)
      }
    }
    
    this.performanceCache.clear()
  }
}