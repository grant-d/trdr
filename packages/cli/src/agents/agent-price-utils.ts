/**
 * Utility functions for calculating stop loss and limit prices in agents
 */

export interface PriceCalculationParams {
  currentPrice: number
  direction: 'buy' | 'sell'
  confidence: number
  agentType?: 'momentum' | 'reversal' | 'volatility' | 'volume' | 'pattern'
  volatility?: number
  riskMultiplier?: number
}

/**
 * Calculate stop loss price based on agent type and market conditions
 */
export function calculateStopLoss(params: PriceCalculationParams): number {
  const {
    currentPrice,
    direction,
    confidence,
    agentType = 'momentum',
    volatility = 0.02,
    riskMultiplier = 1.0
  } = params

  // Base stop loss percentages by agent type
  const baseStopLoss = {
    momentum: 0.025,    // 2.5% - tighter stops for trend following
    reversal: 0.035,    // 3.5% - wider stops for mean reversion
    volatility: 0.03,   // 3% - adaptive to market conditions
    volume: 0.025,      // 2.5% - based on liquidity
    pattern: 0.04       // 4% - wider for pattern completion
  }

  // Get base percentage for agent type
  let stopLossPercent = baseStopLoss[agentType] || 0.03

  // Adjust based on confidence (lower confidence = wider stop)
  stopLossPercent = stopLossPercent * (2 - confidence) * riskMultiplier

  // Adjust based on volatility
  stopLossPercent = stopLossPercent * (1 + volatility)

  // Cap at reasonable limits
  stopLossPercent = Math.min(0.10, Math.max(0.01, stopLossPercent)) // 1-10%

  // Calculate actual stop price
  return direction === 'buy' 
    ? currentPrice * (1 - stopLossPercent)
    : currentPrice * (1 + stopLossPercent)
}

/**
 * Calculate limit price for order execution
 */
export function calculateLimitPrice(params: PriceCalculationParams): number {
  const {
    currentPrice,
    direction,
    confidence,
    agentType = 'momentum',
    volatility = 0.02
  } = params

  // Base slippage tolerance by agent type
  const baseSlippage = {
    momentum: 0.001,    // 0.1% - quick fills for momentum
    reversal: 0.0005,   // 0.05% - tighter for reversals
    volatility: 0.002,  // 0.2% - wider in volatile conditions
    volume: 0.0008,     // 0.08% - based on liquidity
    pattern: 0.0015     // 0.15% - allow some slippage for patterns
  }

  let slippageTolerance = baseSlippage[agentType] || 0.001

  // High confidence = more aggressive limit (closer to market)
  if (confidence > 0.8) {
    slippageTolerance = slippageTolerance * 0.5
  } else if (confidence < 0.6) {
    slippageTolerance = slippageTolerance * 1.5
  }

  // Adjust for volatility
  slippageTolerance = slippageTolerance * (1 + volatility * 0.5)

  // Calculate limit price
  return direction === 'buy'
    ? currentPrice * (1 + slippageTolerance)
    : currentPrice * (1 - slippageTolerance)
}

/**
 * Calculate position size based on confidence and risk
 */
export function calculatePositionSize(
  confidence: number,
  baseSize = 0.05,
  riskMultiplier = 1.0
): number {
  // Scale position size with confidence
  let positionSize = baseSize * confidence * riskMultiplier
  
  // Apply Kelly Criterion-inspired sizing
  // f = p - q/b where p = win probability, q = loss probability, b = win/loss ratio
  // Simplified: use confidence as win probability
  const kellyFraction = confidence - (1 - confidence) / 2 // Assume 2:1 reward/risk
  const kellySizing = Math.max(0, kellyFraction) * baseSize * 2
  
  // Blend standard and Kelly sizing
  positionSize = (positionSize + kellySizing) / 2
  
  // Cap position size
  return Math.min(0.25, Math.max(0.01, positionSize)) // 1-25% of capital
}

/**
 * Quick helper for momentum-based agents
 */
export function momentumPrices(currentPrice: number, direction: 'buy' | 'sell', confidence: number) {
  return {
    stopLoss: calculateStopLoss({
      currentPrice,
      direction,
      confidence,
      agentType: 'momentum'
    }),
    limitPrice: calculateLimitPrice({
      currentPrice,
      direction,
      confidence,
      agentType: 'momentum'
    }),
    positionSize: calculatePositionSize(confidence)
  }
}

/**
 * Quick helper for reversal-based agents
 */
export function reversalPrices(currentPrice: number, direction: 'buy' | 'sell', confidence: number) {
  return {
    stopLoss: calculateStopLoss({
      currentPrice,
      direction,
      confidence,
      agentType: 'reversal'
    }),
    limitPrice: calculateLimitPrice({
      currentPrice,
      direction,
      confidence,
      agentType: 'reversal'
    }),
    positionSize: calculatePositionSize(confidence, 0.04) // Slightly smaller for reversals
  }
}

/**
 * Quick helper for volume-based agents
 */
export function volumePrices(currentPrice: number, direction: 'buy' | 'sell', confidence: number) {
  return {
    stopLoss: calculateStopLoss({
      currentPrice,
      direction,
      confidence,
      agentType: 'volume'
    }),
    limitPrice: calculateLimitPrice({
      currentPrice,
      direction,
      confidence,
      agentType: 'volume'
    }),
    positionSize: calculatePositionSize(confidence, 0.06) // Larger for volume confirmation
  }
}