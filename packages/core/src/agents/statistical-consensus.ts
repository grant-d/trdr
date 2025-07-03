import type { AgentSignal, ConsensusResult } from './types'

/**
 * Statistical methods for calculating consensus stop/limit prices
 */

/**
 * Calculate weighted mean of a numeric value from signals
 */
function calculateWeightedMean(
  signals: Map<string, AgentSignal>,
  weights: Map<string, number>,
  getValue: (signal: AgentSignal) => number | undefined
): { mean: number; count: number } | undefined {
  let weightedSum = 0
  let totalWeight = 0
  let count = 0
  
  for (const [agentId, signal] of signals) {
    const value = getValue(signal)
    if (value !== undefined && !isNaN(value)) {
      const weight = (weights.get(agentId) || 1.0) * signal.confidence
      weightedSum += value * weight
      totalWeight += weight
      count++
    }
  }
  
  if (totalWeight === 0 || count === 0) return undefined
  
  return { mean: weightedSum / totalWeight, count }
}

/**
 * Calculate weighted standard deviation
 */
function calculateWeightedStdDev(
  signals: Map<string, AgentSignal>,
  weights: Map<string, number>,
  getValue: (signal: AgentSignal) => number | undefined,
  mean: number
): number {
  let weightedSquaredDiff = 0
  let totalWeight = 0
  
  for (const [agentId, signal] of signals) {
    const value = getValue(signal)
    if (value !== undefined && !isNaN(value)) {
      const weight = (weights.get(agentId) || 1.0) * signal.confidence
      weightedSquaredDiff += weight * Math.pow(value - mean, 2)
      totalWeight += weight
    }
  }
  
  if (totalWeight === 0) return 0
  
  return Math.sqrt(weightedSquaredDiff / totalWeight)
}

/**
 * Calculate robust price levels using trimmed mean to handle outliers
 */
function calculateRobustPrice(
  values: Array<{ value: number; weight: number }>,
  trimPercent = 0.1
): number {
  if (values.length === 0) return 0
  
  // Sort by value
  values.sort((a, b) => a.value - b.value)
  
  // Trim top and bottom percentiles
  const trimCount = Math.floor(values.length * trimPercent)
  const trimmedValues = values.slice(trimCount, values.length - trimCount)
  
  if (trimmedValues.length === 0) {
    // If all values were trimmed, use median of original
    return values[Math.floor(values.length / 2)]!.value
  }
  
  // Calculate weighted mean of trimmed values
  let weightedSum = 0
  let totalWeight = 0
  
  for (const item of trimmedValues) {
    weightedSum += item.value * item.weight
    totalWeight += item.weight
  }
  
  return totalWeight > 0 ? weightedSum / totalWeight : trimmedValues[0]!.value
}

/**
 * Calculate statistical stop loss based on agent signals
 */
export function calculateStatisticalStopLoss(
  signals: Map<string, AgentSignal>,
  weights: Map<string, number>,
  currentPrice: number,
  action: 'buy' | 'sell'
): { stopLoss: number; confidence: number; stdDev: number } | undefined {
  const stopLossValues: Array<{ value: number; weight: number }> = []
  
  for (const [agentId, signal] of signals) {
    if (signal.stopLoss !== undefined && !isNaN(signal.stopLoss)) {
      const weight = (weights.get(agentId) || 1.0) * signal.confidence
      stopLossValues.push({ value: signal.stopLoss, weight })
    }
  }
  
  if (stopLossValues.length === 0) {
    // No agents provided stop loss, calculate default based on volatility
    return calculateDefaultStopLoss(signals, weights, currentPrice, action)
  }
  
  // Calculate robust stop loss
  const robustStopLoss = calculateRobustPrice(stopLossValues)
  
  // Calculate statistics
  const meanResult = calculateWeightedMean(signals, weights, s => s.stopLoss)
  if (!meanResult) return undefined
  
  const stdDev = calculateWeightedStdDev(signals, weights, s => s.stopLoss, meanResult.mean)
  
  // Confidence based on agreement and participation
  const participationRate = meanResult.count / signals.size
  const normalizedStdDev = stdDev / Math.abs(meanResult.mean)
  const confidence = participationRate * (1 - Math.min(normalizedStdDev, 1))
  
  // Apply safety margin
  const safetyMargin = action === 'buy' ? 0.98 : 1.02 // 2% safety margin
  const finalStopLoss = robustStopLoss * safetyMargin
  
  return {
    stopLoss: finalStopLoss,
    confidence,
    stdDev
  }
}

/**
 * Calculate default stop loss when agents don't provide one
 */
function calculateDefaultStopLoss(
  signals: Map<string, AgentSignal>,
  weights: Map<string, number>,
  currentPrice: number,
  action: 'buy' | 'sell'
): { stopLoss: number; confidence: number; stdDev: number } {
  // Use average confidence as a proxy for risk
  let totalConfidence = 0
  let totalWeight = 0
  
  for (const [agentId, signal] of signals) {
    const weight = weights.get(agentId) || 1.0
    totalConfidence += signal.confidence * weight
    totalWeight += weight
  }
  
  const avgConfidence = totalWeight > 0 ? totalConfidence / totalWeight : 0.5
  
  // Higher confidence = tighter stop loss
  const stopLossPercent = 0.02 + (1 - avgConfidence) * 0.03 // 2-5% based on confidence
  
  const stopLoss = action === 'buy' 
    ? currentPrice * (1 - stopLossPercent)
    : currentPrice * (1 + stopLossPercent)
  
  return {
    stopLoss,
    confidence: avgConfidence * 0.7, // Lower confidence for default
    stdDev: currentPrice * stopLossPercent * 0.5
  }
}

/**
 * Calculate statistical limit price based on agent signals
 */
export function calculateStatisticalLimitPrice(
  signals: Map<string, AgentSignal>,
  weights: Map<string, number>,
  currentPrice: number,
  action: 'buy' | 'sell'
): { limitPrice: number; confidence: number; stdDev: number } | undefined {
  const limitPriceValues: Array<{ value: number; weight: number }> = []
  
  for (const [agentId, signal] of signals) {
    if (signal.limitPrice !== undefined && !isNaN(signal.limitPrice)) {
      const weight = (weights.get(agentId) || 1.0) * signal.confidence
      limitPriceValues.push({ value: signal.limitPrice, weight })
    }
  }
  
  if (limitPriceValues.length === 0) {
    // No agents provided limit price, calculate default
    return calculateDefaultLimitPrice(signals, weights, currentPrice, action)
  }
  
  // Calculate robust limit price
  const robustLimitPrice = calculateRobustPrice(limitPriceValues)
  
  // Calculate statistics
  const meanResult = calculateWeightedMean(signals, weights, s => s.limitPrice)
  if (!meanResult) return undefined
  
  const stdDev = calculateWeightedStdDev(signals, weights, s => s.limitPrice, meanResult.mean)
  
  // Confidence based on agreement and participation
  const participationRate = meanResult.count / signals.size
  const normalizedStdDev = stdDev / Math.abs(meanResult.mean)
  const confidence = participationRate * (1 - Math.min(normalizedStdDev, 1))
  
  // Ensure limit price is favorable
  let finalLimitPrice = robustLimitPrice
  if (action === 'buy' && finalLimitPrice > currentPrice) {
    // For buy orders, limit should be at or below current price
    finalLimitPrice = Math.min(currentPrice, robustLimitPrice)
  } else if (action === 'sell' && finalLimitPrice < currentPrice) {
    // For sell orders, limit should be at or above current price
    finalLimitPrice = Math.max(currentPrice, robustLimitPrice)
  }
  
  return {
    limitPrice: finalLimitPrice,
    confidence,
    stdDev
  }
}

/**
 * Calculate default limit price when agents don't provide one
 */
function calculateDefaultLimitPrice(
  _signals: Map<string, AgentSignal>,
  _weights: Map<string, number>,
  currentPrice: number,
  action: 'buy' | 'sell'
): { limitPrice: number; confidence: number; stdDev: number } {
  // Use market price with small adjustment for immediate execution
  const slippageTolerance = 0.001 // 0.1% slippage tolerance
  
  const limitPrice = action === 'buy'
    ? currentPrice * (1 + slippageTolerance) // Slightly above for buy
    : currentPrice * (1 - slippageTolerance) // Slightly below for sell
  
  return {
    limitPrice,
    confidence: 0.9, // High confidence for market-like execution
    stdDev: currentPrice * slippageTolerance
  }
}

/**
 * Calculate consensus position size
 */
export function calculateConsensusPositionSize(
  signals: Map<string, AgentSignal>,
  weights: Map<string, number>,
  defaultSize = 0.05 // 5% default
): number {
  const positionSizes: Array<{ value: number; weight: number }> = []
  
  for (const [agentId, signal] of signals) {
    if (signal.positionSize !== undefined && !isNaN(signal.positionSize)) {
      const weight = (weights.get(agentId) || 1.0) * signal.confidence
      positionSizes.push({ value: signal.positionSize, weight })
    }
  }
  
  if (positionSizes.length === 0) {
    // No position size suggestions, use default adjusted by average confidence
    let totalConfidence = 0
    let totalWeight = 0
    
    for (const [agentId, signal] of signals) {
      const weight = weights.get(agentId) || 1.0
      totalConfidence += signal.confidence * weight
      totalWeight += weight
    }
    
    const avgConfidence = totalWeight > 0 ? totalConfidence / totalWeight : 0.5
    return defaultSize * avgConfidence
  }
  
  // Use robust mean for position size
  return Math.min(0.25, Math.max(0.01, calculateRobustPrice(positionSizes))) // Clamp between 1% and 25%
}

/**
 * Enhance consensus result with statistical price levels
 */
export function enhanceConsensusWithPriceLevels(
  baseConsensus: ConsensusResult,
  signals: Map<string, AgentSignal>,
  weights: Map<string, number>,
  currentPrice: number
): ConsensusResult {
  // Only calculate price levels for actionable signals
  if (baseConsensus.action === 'hold') {
    return baseConsensus
  }
  
  // Calculate stop loss
  const stopLossResult = calculateStatisticalStopLoss(
    signals,
    weights,
    currentPrice,
    baseConsensus.action
  )
  
  // Calculate limit price
  const limitPriceResult = calculateStatisticalLimitPrice(
    signals,
    weights,
    currentPrice,
    baseConsensus.action
  )
  
  // Calculate position size
  const positionSize = calculateConsensusPositionSize(signals, weights)
  
  // Build price confidence object
  const priceConfidence: ConsensusResult['priceConfidence'] = {
    stopLoss: stopLossResult ? {
      mean: stopLossResult.stopLoss,
      stdDev: stopLossResult.stdDev,
      confidence: stopLossResult.confidence
    } : { mean: 0, stdDev: 0, confidence: 0 },
    limitPrice: limitPriceResult ? {
      mean: limitPriceResult.limitPrice,
      stdDev: limitPriceResult.stdDev,
      confidence: limitPriceResult.confidence
    } : { mean: currentPrice, stdDev: 0, confidence: 1 }
  }
  
  return {
    ...baseConsensus,
    stopLoss: stopLossResult?.stopLoss,
    limitPrice: limitPriceResult?.limitPrice,
    positionSize,
    priceConfidence
  }
}