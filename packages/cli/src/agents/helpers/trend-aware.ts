import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { epochDateNow } from '@trdr/shared'

/**
 * Simple trend detection based on moving averages
 */
export function detectTrend(context: MarketContext): {
  trend: 'bullish' | 'bearish' | 'neutral'
  strength: number // 0-1
} {
  const { candles } = context
  
  if (candles.length < 20) {
    return { trend: 'neutral', strength: 0 }
  }
  
  // Calculate short and long moving averages
  const shortPeriod = Math.min(10, Math.floor(candles.length / 2))
  const longPeriod = Math.min(20, candles.length)
  
  const recentCandles = candles.slice(-longPeriod)
  const shortMA = recentCandles.slice(-shortPeriod).reduce((sum, c) => sum + c.close, 0) / shortPeriod
  const longMA = recentCandles.reduce((sum, c) => sum + c.close, 0) / longPeriod
  
  // Calculate trend direction and strength
  const maRatio = shortMA / longMA
  const deviation = Math.abs(maRatio - 1)
  
  if (maRatio > 1.02) {
    return { trend: 'bullish', strength: Math.min(1, deviation * 10) }
  } else if (maRatio < 0.98) {
    return { trend: 'bearish', strength: Math.min(1, deviation * 10) }
  } else {
    return { trend: 'neutral', strength: deviation * 5 }
  }
}

/**
 * Adjust signal confidence based on trend alignment
 */
export function adjustSignalForTrend(
  signal: AgentSignal,
  context: MarketContext
): AgentSignal {
  const { trend, strength } = detectTrend(context)
  
  // If signal aligns with trend, boost confidence
  if (trend === 'bullish' && signal.action === 'buy') {
    return {
      ...signal,
      confidence: Math.min(0.95, signal.confidence + strength * 0.2),
      reason: `${signal.reason} [Trend aligned +${(strength * 20).toFixed(0)}%]`
    }
  } else if (trend === 'bearish' && signal.action === 'sell') {
    return {
      ...signal,
      confidence: Math.min(0.95, signal.confidence + strength * 0.2),
      reason: `${signal.reason} [Trend aligned +${(strength * 20).toFixed(0)}%]`
    }
  }
  
  // If signal goes against strong trend, reduce confidence significantly
  if (trend === 'bullish' && signal.action === 'sell' && strength > 0.5) {
    return {
      ...signal,
      confidence: signal.confidence * (1 - strength * 0.5),
      reason: `${signal.reason} [Against trend -${(strength * 50).toFixed(0)}%]`
    }
  } else if (trend === 'bearish' && signal.action === 'buy' && strength > 0.5) {
    return {
      ...signal,
      confidence: signal.confidence * (1 - strength * 0.5),
      reason: `${signal.reason} [Against trend -${(strength * 50).toFixed(0)}%]`
    }
  }
  
  return signal
}

/**
 * Generate trend-following signal when agent signal is hold
 */
export function generateTrendSignal(context: MarketContext): AgentSignal {
  const { trend, strength } = detectTrend(context)
  
  if (trend === 'bullish' && strength > 0.6) {
    return {
      action: 'buy',
      confidence: 0.5 + strength * 0.3,
      reason: `Strong bullish trend (${(strength * 100).toFixed(0)}% strength)`,
      timestamp: epochDateNow()
    }
  } else if (trend === 'bearish' && strength > 0.6) {
    return {
      action: 'sell',
      confidence: 0.5 + strength * 0.3,
      reason: `Strong bearish trend (${(strength * 100).toFixed(0)}% strength)`,
      timestamp: epochDateNow()
    }
  }
  
  return {
    action: 'hold',
    confidence: 0.4,
    reason: `No clear trend`,
    timestamp: epochDateNow()
  }
}