import { epochDateNow } from '@trdr/shared'
import type { AgentSignal, ConsensusStrategy } from './types'
import { enhanceConsensusWithPriceLevels } from './statistical-consensus'

/**
 * Calculate confidence intervals for consensus results
 */
export function calculateConfidenceInterval(
  signals: Map<string, AgentSignal>,
  weights: Map<string, number>,
  consensus: number
): { lower: number; upper: number } {
  // Calculate weighted variance
  let totalWeight = 0
  let weightedSquaredDiff = 0
  
  for (const [agentId, signal] of signals) {
    const weight = weights.get(agentId) || 1.0
    totalWeight += weight
    weightedSquaredDiff += weight * Math.pow(signal.confidence - consensus, 2)
  }
  
  const variance = weightedSquaredDiff / totalWeight
  const stdDev = Math.sqrt(variance)
  
  // 95% confidence interval (1.96 * standard error)
  const standardError = stdDev / Math.sqrt(signals.size)
  const margin = 1.96 * standardError
  
  return {
    lower: Math.max(0, consensus - margin),
    upper: Math.min(1, consensus + margin)
  }
}

/**
 * Normalize heterogeneous agent outputs
 */
export function normalizeSignals(
  signals: Map<string, AgentSignal>
): Map<string, AgentSignal> {
  const normalized = new Map<string, AgentSignal>()
  
  // Group signals by action
  const actionGroups = new Map<string, Array<{ signal: AgentSignal; agentId: string }>>()
  for (const [agentId, signal] of signals) {
    const group = actionGroups.get(signal.action) || []
    group.push({ signal, agentId })
    actionGroups.set(signal.action, group)
  }
  
  // Normalize confidence within each action group
  for (const [_, group] of actionGroups) {
    const confidences = group.map(s => s.signal.confidence)
    const minConf = Math.min(...confidences)
    const maxConf = Math.max(...confidences)
    const range = maxConf - minConf
    
    for (const item of group) {
      const normalizedConfidence = range > 0 
        ? (item.signal.confidence - minConf) / range 
        : item.signal.confidence
      
      normalized.set(item.agentId, {
        action: item.signal.action,
        confidence: normalizedConfidence,
        reason: item.signal.reason,
        timestamp: item.signal.timestamp
      })
    }
  }
  
  return normalized
}

/**
 * Performance-based weighted consensus strategy
 */
export const performanceWeightedConsensus: ConsensusStrategy = {
  name: 'performance-weighted',
  calculateConsensus: (signals, weights, agents, _priors, currentPrice) => {
    const scores = new Map<string, number>()
    let totalWeight = 0
    
    // Calculate performance-adjusted weights
    const adjustedWeights = new Map<string, number>()
    for (const [agentId] of signals) {
      let weight = weights.get(agentId) || 1.0
      
      // Adjust weight based on agent performance if available
      if (agents) {
        const agent = agents.get(agentId)
        if (agent?.getPerformance) {
          const performance = agent.getPerformance()
          // Boost weight based on win rate and Sharpe ratio
          const performanceMultiplier = 1 + (performance.winRate * 0.5) + 
                                       (Math.max(0, performance.sharpeRatio) * 0.25)
          weight *= performanceMultiplier
        }
      }
      
      adjustedWeights.set(agentId, weight)
      totalWeight += weight
    }
    
    // Calculate weighted scores
    for (const [agentId, signal] of signals) {
      const weight = adjustedWeights.get(agentId) || 1.0
      const score = scores.get(signal.action) || 0
      scores.set(signal.action, score + signal.confidence * weight)
    }
    
    // Find winning action
    let bestAction = 'hold'
    let bestScore = 0
    for (const [action, score] of scores) {
      if (score > bestScore) {
        bestAction = action
        bestScore = score
      }
    }
    
    const agreement = bestScore / totalWeight
    
    // Calculate confidence interval
    const interval = calculateConfidenceInterval(signals, adjustedWeights, agreement)
    
    // Combine reasons with agent weights
    const reasons: string[] = []
    const agentSignalsMap: Record<string, AgentSignal> = {}
    for (const [agentId, signal] of signals) {
      if (signal.action === bestAction) {
        const weight = adjustedWeights.get(agentId) || 1.0
        reasons.push(`${agentId} (w=${weight.toFixed(2)}): ${signal.reason}`)
      }
      agentSignalsMap[agentId] = signal
    }
    
    const baseConsensus = {
      action: bestAction as AgentSignal['action'],
      confidence: agreement,
      reason: reasons.join('; '),
      agentSignals: agentSignalsMap,
      agreement,
      participatingAgents: signals.size,
      timestamp: epochDateNow(),
      confidenceInterval: interval
    }
    
    // Enhance with statistical price levels if current price is available
    return currentPrice 
      ? enhanceConsensusWithPriceLevels(baseConsensus, signals, adjustedWeights, currentPrice)
      : baseConsensus
  }
}

/**
 * Exponential weighted consensus (recent signals have more weight)
 */
export const exponentialWeightedConsensus: ConsensusStrategy = {
  name: 'exponential-weighted',
  calculateConsensus: (signals, weights, _agents, _priors, currentPrice) => {
    const scores = new Map<string, number>()
    let totalWeight = 0
    const now = epochDateNow()
    
    // Calculate time-adjusted weights
    const adjustedWeights = new Map<string, number>()
    for (const [agentId, signal] of signals) {
      let weight = weights.get(agentId) || 1.0
      
      // Apply exponential decay based on signal age (half-life of 5 minutes)
      const ageMs = now - signal.timestamp
      const halfLifeMs = 5 * 60 * 1000
      const decayFactor = Math.pow(0.5, ageMs / halfLifeMs)
      weight *= decayFactor
      
      adjustedWeights.set(agentId, weight)
      totalWeight += weight
    }
    
    // Calculate weighted scores
    for (const [agentId, signal] of signals) {
      const weight = adjustedWeights.get(agentId) || 1.0
      const score = scores.get(signal.action) || 0
      scores.set(signal.action, score + signal.confidence * weight)
    }
    
    // Find winning action
    let bestAction = 'hold'
    let bestScore = 0
    for (const [action, score] of scores) {
      if (score > bestScore) {
        bestAction = action
        bestScore = score
      }
    }
    
    const agreement = totalWeight > 0 ? bestScore / totalWeight : 0
    
    // Combine reasons
    const reasons: string[] = []
    const agentSignalsMap: Record<string, AgentSignal> = {}
    for (const [agentId, signal] of signals) {
      if (signal.action === bestAction) {
        reasons.push(`${agentId}: ${signal.reason}`)
      }
      agentSignalsMap[agentId] = signal
    }
    
    const baseConsensus = {
      action: bestAction as AgentSignal['action'],
      confidence: agreement,
      reason: reasons.join('; '),
      agentSignals: agentSignalsMap,
      agreement,
      participatingAgents: signals.size,
      timestamp: epochDateNow()
    }
    
    // Enhance with statistical price levels if current price is available
    return currentPrice 
      ? enhanceConsensusWithPriceLevels(baseConsensus, signals, adjustedWeights, currentPrice)
      : baseConsensus
  }
}

/**
 * Confidence-weighted consensus (higher confidence signals have more influence)
 */
export const confidenceWeightedConsensus: ConsensusStrategy = {
  name: 'confidence-weighted',
  calculateConsensus: (signals, weights, _agents, _priors, currentPrice) => {
    const scores = new Map<string, number>()
    let totalWeight = 0
    
    // Don't normalize - use raw confidence values
    // Calculate confidence-adjusted weights
    for (const [agentId, signal] of signals) {
      const baseWeight = weights.get(agentId) || 1.0
      // Square the confidence to give higher confidence signals more influence
      const confidenceWeight = Math.pow(signal.confidence, 2)
      const weight = baseWeight * confidenceWeight
      
      const score = scores.get(signal.action) || 0
      scores.set(signal.action, score + weight)
      totalWeight += weight
    }
    
    // Find winning action
    let bestAction = 'hold'
    let bestScore = 0
    for (const [action, score] of scores) {
      if (score > bestScore) {
        bestAction = action
        bestScore = score
      }
    }
    
    const agreement = totalWeight > 0 ? bestScore / totalWeight : 0
    
    // Combine reasons sorted by confidence
    const reasons: string[] = []
    const agentSignalsMap: Record<string, AgentSignal> = {}
    const sortedSignals = Array.from(signals.entries())
      .filter(([_, signal]) => signal.action === bestAction)
      .sort(([_, a], [__, b]) => b.confidence - a.confidence)
    
    for (const [agentId, signal] of sortedSignals) {
      reasons.push(`${agentId} (${(signal.confidence * 100).toFixed(0)}%): ${signal.reason}`)
    }
    
    for (const [agentId, signal] of signals) {
      agentSignalsMap[agentId] = signal
    }
    
    const baseConsensus = {
      action: bestAction as AgentSignal['action'],
      confidence: agreement,
      reason: reasons.join('; '),
      agentSignals: agentSignalsMap,
      agreement,
      participatingAgents: signals.size,
      timestamp: epochDateNow()
    }
    
    // Enhance with statistical price levels if current price is available
    return currentPrice 
      ? enhanceConsensusWithPriceLevels(baseConsensus, signals, weights, currentPrice)
      : baseConsensus
  }
}

/**
 * Bayesian consensus (combines prior beliefs with agent signals)
 */
export const bayesianConsensus: ConsensusStrategy = {
  name: 'bayesian',
  calculateConsensus: (signals, weights, _, priors = { buy: 0.33, sell: 0.33, hold: 0.34 }, currentPrice) => {
    // Calculate likelihood for each action
    const likelihoods = new Map<string, number>()
    
    for (const action of ['buy', 'sell', 'hold']) {
      let likelihood = 1.0
      
      for (const [agentId, signal] of signals) {
        const weight = weights.get(agentId) || 1.0
        
        // Probability of agent giving this signal given the true action
        const signalProb = signal.action === action 
          ? signal.confidence 
          : (1 - signal.confidence) / 2 // Split remaining probability
        
        // Weight the probability
        likelihood *= Math.pow(signalProb, weight)
      }
      
      likelihoods.set(action, likelihood)
    }
    
    // Calculate posterior probabilities
    const posteriors = new Map<string, number>()
    let totalPosterior = 0
    
    for (const action of ['buy', 'sell', 'hold']) {
      const prior = priors[action] || 0.33
      const likelihood = likelihoods.get(action) || 0
      const posterior = prior * likelihood
      posteriors.set(action, posterior)
      totalPosterior += posterior
    }
    
    // Normalize posteriors
    for (const [action, posterior] of posteriors) {
      posteriors.set(action, posterior / totalPosterior)
    }
    
    // Find action with highest posterior
    let bestAction = 'hold'
    let bestPosterior = 0
    for (const [action, posterior] of posteriors) {
      if (posterior > bestPosterior) {
        bestAction = action
        bestPosterior = posterior
      }
    }
    
    // Combine reasons
    const reasons: string[] = []
    const agentSignalsMap: Record<string, AgentSignal> = {}
    for (const [agentId, signal] of signals) {
      if (signal.action === bestAction) {
        reasons.push(`${agentId}: ${signal.reason}`)
      }
      agentSignalsMap[agentId] = signal
    }
    
    const baseConsensus = {
      action: bestAction as AgentSignal['action'],
      confidence: bestPosterior,
      reason: `Bayesian posterior: ${(bestPosterior * 100).toFixed(1)}%. ${reasons.join('; ')}`,
      agentSignals: agentSignalsMap,
      agreement: bestPosterior,
      participatingAgents: signals.size,
      timestamp: epochDateNow(),
      posteriorProbabilities: Object.fromEntries(posteriors)
    }
    
    // Enhance with statistical price levels if current price is available
    return currentPrice 
      ? enhanceConsensusWithPriceLevels(baseConsensus, signals, weights, currentPrice)
      : baseConsensus
  }
}

/**
 * Ensemble voting with veto power for high-confidence dissenting agents
 */
export const vetoConsensus: ConsensusStrategy = {
  name: 'veto-consensus',
  calculateConsensus: (signals, weights, _agents, _priors, currentPrice) => {
    // First check for any high-confidence vetos
    const vetoThreshold = 0.9
    const vetoSignals: Array<[string, AgentSignal]> = []
    
    for (const [agentId, signal] of signals) {
      if (signal.confidence >= vetoThreshold && signal.action !== 'buy') {
        vetoSignals.push([agentId, signal])
      }
    }
    
    // If there are vetos, use the highest confidence veto
    if (vetoSignals.length > 0) {
      vetoSignals.sort(([_, a], [__, b]) => b.confidence - a.confidence)
      const [vetoAgentId, vetoSignal] = vetoSignals[0]!
      
      const agentSignalsMap: Record<string, AgentSignal> = {}
      for (const [agentId, signal] of signals) {
        agentSignalsMap[agentId] = signal
      }
      
      const baseConsensus = {
        action: vetoSignal.action,
        confidence: vetoSignal.confidence,
        reason: `VETO by ${vetoAgentId}: ${vetoSignal.reason}`,
        agentSignals: agentSignalsMap,
        agreement: vetoSignal.confidence,
        participatingAgents: signals.size,
        timestamp: epochDateNow(),
        vetoApplied: true
      }
      
      // Enhance with statistical price levels if current price is available
      return currentPrice 
        ? enhanceConsensusWithPriceLevels(baseConsensus, signals, weights, currentPrice)
        : baseConsensus
    }
    
    // Otherwise, use weighted voting
    const scores = new Map<string, number>()
    let totalWeight = 0
    
    for (const [agentId, signal] of signals) {
      const weight = weights.get(agentId) || 1.0
      const score = scores.get(signal.action) || 0
      scores.set(signal.action, score + signal.confidence * weight)
      totalWeight += weight
    }
    
    let bestAction = 'hold'
    let bestScore = 0
    for (const [action, score] of scores) {
      if (score > bestScore) {
        bestAction = action
        bestScore = score
      }
    }
    
    const agreement = bestScore / totalWeight
    
    const reasons: string[] = []
    const agentSignalsMap: Record<string, AgentSignal> = {}
    for (const [agentId, signal] of signals) {
      if (signal.action === bestAction) {
        reasons.push(`${agentId}: ${signal.reason}`)
      }
      agentSignalsMap[agentId] = signal
    }
    
    const baseConsensus = {
      action: bestAction as AgentSignal['action'],
      confidence: agreement,
      reason: reasons.join('; '),
      agentSignals: agentSignalsMap,
      agreement,
      participatingAgents: signals.size,
      timestamp: epochDateNow(),
      vetoApplied: false
    }
    
    // Enhance with statistical price levels if current price is available
    return currentPrice 
      ? enhanceConsensusWithPriceLevels(baseConsensus, signals, weights, currentPrice)
      : baseConsensus
  }
}