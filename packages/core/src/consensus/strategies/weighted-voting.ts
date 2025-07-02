import type {
  IConsensusStrategy,
  AgentSignal,
  ConsensusConfig,
  ConsensusResult
} from '../interfaces'
import type { OrderSide } from '@trdr/shared'

/**
 * Weighted voting consensus strategy.
 * 
 * Uses agent confidence and weights to determine consensus.
 * Supports both simple and weighted voting modes.
 */
export class WeightedVotingStrategy implements IConsensusStrategy {
  readonly name = 'Weighted Voting'

  evaluate(signals: AgentSignal[], config: ConsensusConfig): ConsensusResult {
    const startTime = Date.now()
    
    // Check if we have minimum required agents
    if (signals.length < config.minAgentsRequired) {
      return this.createFallbackResult(
        signals,
        config,
        `Insufficient agents: ${signals.length} < ${config.minAgentsRequired}`,
        startTime
      )
    }

    // Calculate votes for each action
    const votes = this.calculateVotes(signals, config)
    
    // Determine winning action
    const { action, confidence } = this.determineWinningAction(votes, signals, config)
    
    // Calculate agreement level
    const agreementPercentage = this.calculateAgreement(signals)
    
    // Check if consensus threshold is met
    if (agreementPercentage < config.minAgreementThreshold) {
      return this.createFallbackResult(
        signals,
        config,
        `Agreement below threshold: ${(agreementPercentage * 100).toFixed(1)}% < ${(config.minAgreementThreshold * 100).toFixed(1)}%`,
        startTime
      )
    }

    // Check confidence threshold
    if (confidence < config.minConfidenceThreshold) {
      return this.createFallbackResult(
        signals,
        config,
        `Confidence below threshold: ${confidence.toFixed(3)} < ${config.minConfidenceThreshold}`,
        startTime
      )
    }

    // Calculate aggregate metrics
    const metrics = this.calculateAggregateMetrics(signals, action)
    
    return {
      consensusReached: true,
      action,
      confidence,
      expectedWinRate: metrics.expectedWinRate,
      expectedRiskReward: metrics.expectedRiskReward,
      trailDistance: metrics.trailDistance,
      leadAgentId: metrics.leadAgentId,
      agentSignals: signals,
      agreementPercentage,
      dissentLevel: 1 - agreementPercentage,
      usedFallback: false,
      processingTimeMs: Date.now() - startTime,
      timestamp: new Date()
    }
  }

  calculateAgreement(signals: AgentSignal[]): number {
    if (signals.length === 0) return 0
    
    // Count signals by action
    const actionCounts = new Map<string, number>()
    let totalWeight = 0
    
    for (const signal of signals) {
      const weight = signal.weight || 1
      const currentCount = actionCounts.get(signal.signal) || 0
      actionCounts.set(signal.signal, currentCount + weight)
      totalWeight += weight
    }
    
    // Find the most popular action
    let maxWeight = 0
    for (const weight of actionCounts.values()) {
      maxWeight = Math.max(maxWeight, weight)
    }
    
    return maxWeight / totalWeight
  }

  /**
   * Calculate weighted votes for each action
   */
  private calculateVotes(
    signals: AgentSignal[],
    config: ConsensusConfig
  ): Map<string, { weight: number; confidence: number; count: number }> {
    const votes = new Map<string, { weight: number; confidence: number; count: number }>()
    
    for (const signal of signals) {
      const weight = config.useWeightedVoting 
        ? (signal.weight || config.defaultAgentWeight)
        : 1
      
      const current = votes.get(signal.signal) || { weight: 0, confidence: 0, count: 0 }
      votes.set(signal.signal, {
        weight: current.weight + weight * signal.confidence,
        confidence: current.confidence + signal.confidence,
        count: current.count + 1
      })
    }
    
    return votes
  }

  /**
   * Determine the winning action based on votes
   */
  private determineWinningAction(
    votes: Map<string, { weight: number; confidence: number; count: number }>,
    signals: AgentSignal[],
    config: ConsensusConfig
  ): { action: OrderSide | 'hold'; confidence: number } {
    let bestAction: OrderSide | 'hold' = 'hold'
    let bestScore = 0
    let totalWeight = 0
    
    // Calculate total weight for normalization
    for (const vote of votes.values()) {
      totalWeight += vote.weight
    }
    
    // Find action with highest weighted score
    for (const [action, vote] of votes.entries()) {
      const score = vote.weight / totalWeight
      if (score > bestScore) {
        bestScore = score
        bestAction = action as OrderSide | 'hold'
      }
    }
    
    // Calculate confidence as weighted average of agreeing agents
    const agreeingSignals = signals.filter(s => s.signal === bestAction)
    const totalConfidence = agreeingSignals.reduce((sum, s) => {
      const weight = config.useWeightedVoting ? (s.weight || config.defaultAgentWeight) : 1
      return sum + s.confidence * weight
    }, 0)
    const totalAgreeingWeight = agreeingSignals.reduce((sum, s) => {
      return sum + (config.useWeightedVoting ? (s.weight || config.defaultAgentWeight) : 1)
    }, 0)
    
    const confidence = totalAgreeingWeight > 0 ? totalConfidence / totalAgreeingWeight : 0
    
    return { action: bestAction, confidence }
  }

  /**
   * Calculate aggregate metrics for the winning action
   */
  private calculateAggregateMetrics(
    signals: AgentSignal[],
    action: OrderSide | 'hold'
  ): {
    expectedWinRate: number
    expectedRiskReward: number
    trailDistance: number
    leadAgentId: string
  } {
    const actionSignals = signals.filter(s => s.signal === action)
    
    // Find lead agent (highest confidence for this action)
    let leadAgent = actionSignals[0]
    if (!leadAgent) {
      // No agents agreed on this action
      return {
        expectedWinRate: 0.5,
        expectedRiskReward: 1.0,
        trailDistance: 2.0,
        leadAgentId: ''
      }
    }
    
    for (const signal of actionSignals) {
      if (signal.confidence > leadAgent.confidence) {
        leadAgent = signal
      }
    }
    
    // Calculate weighted averages
    let totalWeight = 0
    let weightedWinRate = 0
    let weightedRiskReward = 0
    let weightedTrailDistance = 0
    
    for (const signal of actionSignals) {
      const weight = signal.confidence * (signal.weight || 1)
      totalWeight += weight
      
      if (signal.expectedWinRate) {
        weightedWinRate += signal.expectedWinRate * weight
      }
      if (signal.expectedRiskReward) {
        weightedRiskReward += signal.expectedRiskReward * weight
      }
      if (signal.trailDistance) {
        weightedTrailDistance += signal.trailDistance * weight
      }
    }
    
    return {
      expectedWinRate: totalWeight > 0 ? weightedWinRate / totalWeight : 0.5,
      expectedRiskReward: totalWeight > 0 ? weightedRiskReward / totalWeight : 1.0,
      trailDistance: totalWeight > 0 ? weightedTrailDistance / totalWeight : 2.0,
      leadAgentId: leadAgent.agentId
    }
  }

  /**
   * Create a fallback result when consensus cannot be reached
   */
  private createFallbackResult(
    signals: AgentSignal[],
    config: ConsensusConfig,
    reason: string,
    startTime: number
  ): ConsensusResult {
    let action: OrderSide | 'hold' = 'hold'
    let confidence = 0
    let leadAgentId = ''
    
    if (signals.length > 0 && config.fallbackStrategy !== 'hold') {
      if (config.fallbackStrategy === 'use-best') {
        // Use signal with highest confidence
        const bestSignal = signals.reduce((best, signal) => 
          signal.confidence > best.confidence ? signal : best
        )
        action = bestSignal.signal
        confidence = bestSignal.confidence * 0.5 // Reduce confidence for fallback
        leadAgentId = bestSignal.agentId
      } else if (config.fallbackStrategy === 'use-majority') {
        // Use simple majority vote
        const votes = this.calculateVotes(signals, { ...config, useWeightedVoting: false })
        let maxVotes = 0
        for (const [act, vote] of votes.entries()) {
          if (vote.count > maxVotes) {
            maxVotes = vote.count
            action = act as OrderSide | 'hold'
          }
        }
        confidence = (maxVotes / signals.length) * 0.5 // Reduce confidence for fallback
        leadAgentId = signals.find(s => s.signal === action)?.agentId || ''
      }
    }
    
    const metrics = action !== 'hold' 
      ? this.calculateAggregateMetrics(signals, action)
      : { expectedWinRate: 0.5, expectedRiskReward: 1.0, trailDistance: 2.0, leadAgentId }
    
    return {
      consensusReached: false,
      action,
      confidence,
      expectedWinRate: metrics.expectedWinRate,
      expectedRiskReward: metrics.expectedRiskReward,
      trailDistance: metrics.trailDistance,
      leadAgentId: metrics.leadAgentId,
      agentSignals: signals,
      agreementPercentage: this.calculateAgreement(signals),
      dissentLevel: 1 - this.calculateAgreement(signals),
      usedFallback: true,
      fallbackReason: reason,
      processingTimeMs: Date.now() - startTime,
      timestamp: new Date()
    }
  }
}