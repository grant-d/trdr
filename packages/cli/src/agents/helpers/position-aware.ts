import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'

/**
 * Ensure agents can only sell if they have inventory (no shorting)
 * 
 * @param signal The original agent signal
 * @param context The market context with position information
 * @param minPositionForSell Minimum position required to allow sell (default 0.0001)
 * @returns Modified signal that respects position constraints
 */
export function enforceNoShorting(
  signal: AgentSignal,
  context: MarketContext,
  minPositionForSell = 0.0001
): AgentSignal {
  // If not a sell signal, return as-is
  if (signal.action !== 'sell') {
    return signal
  }
  
  // Check current position
  const currentPosition = context.currentPosition ?? 0
  
  // If no position or position too small, convert to hold
  if (currentPosition < minPositionForSell) {
    return {
      ...signal,
      action: 'hold',
      confidence: signal.confidence * 0.5, // Reduce confidence
      reason: `${signal.reason} [No position to sell]`
    }
  }
  
  // Has position, allow sell
  return signal
}

/**
 * Create a position-aware signal that prevents shorting
 */
export function createPositionAwareSignal(
  action: AgentSignal['action'],
  confidence: number,
  reason: string,
  context: MarketContext,
  analysis?: string,
  priceTarget?: number,
  stopLoss?: number,
  positionSize?: number
): AgentSignal {
  const baseSignal: AgentSignal = {
    action,
    confidence: Math.max(0, Math.min(1, confidence)),
    reason,
    analysis,
    priceTarget,
    stopLoss,
    positionSize,
    timestamp: Date.now() as any // Will be converted to EpochDate
  }
  
  return enforceNoShorting(baseSignal, context)
}