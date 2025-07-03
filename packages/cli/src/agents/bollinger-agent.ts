import { BaseAgent, IndicatorCalculator } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { calculateStopLoss, calculateLimitPrice, calculatePositionSize } from './agent-price-utils'
import { enforceNoShorting } from './helpers/position-aware'

export class BollingerBandsAgent extends BaseAgent {
  private readonly calculator = new IndicatorCalculator()
  private readonly period = 20
  private readonly stdDev = 2
  private readonly bandwidthHistory: number[] = []
  private readonly percentBHistory: number[] = []
  private readonly historyLength = 10 // Keep last 10 calculations for trend analysis
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Bollinger Bands Agent initialized', {
      period: this.period,
      stdDev: this.stdDev
    })
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const signal = await this.performBollingerAnalysis(context)
    return enforceNoShorting(signal, context)
  }

  private async performBollingerAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { currentPrice, candles } = context
    
    if (candles.length < this.period) {
      return this.createSignal('hold', 0.3, 'Insufficient data for Bollinger Bands calculation')
    }
    
    // Calculate Bollinger Bands
    const bands = this.calculator.bollingerBands(candles, {
      period: this.period,
      stdDevMultiplier: this.stdDev
    })
    
    if (!bands) {
      return this.createSignal('hold', 0.3, 'Bollinger Bands not available')
    }
    
    const { upper, middle, lower } = bands
    
    // Calculate enhanced metrics using original indicator capabilities
    const bandwidth = (upper - lower) / middle
    const percentB = (currentPrice - lower) / (upper - lower)
    
    // Update history
    this.bandwidthHistory.push(bandwidth)
    this.percentBHistory.push(percentB)
    
    // Keep history within limits
    if (this.bandwidthHistory.length > this.historyLength) {
      this.bandwidthHistory.shift()
    }
    if (this.percentBHistory.length > this.historyLength) {
      this.percentBHistory.shift()
    }
    
    // Enhanced squeeze detection using bandwidth history
    const isSqueeze = this.detectSqueeze()
    const squeezeRelease = this.detectSqueezeRelease()
    
    // Calculate %B trend for momentum
    const percentBTrend = this.calculatePercentBTrend()
    
    // Enhanced signal generation using historical analysis
    
    // High priority: Squeeze release (strong breakout signals)
    if (squeezeRelease) {
      const direction = percentBTrend === 'bullish' ? 'buy' : percentBTrend === 'bearish' ? 'sell' : (currentPrice > middle ? 'buy' : 'sell')
      const confidence = percentBTrend !== 'neutral' ? 0.85 : 0.7
      // Calculate prices for squeeze release (volatility breakout)
      const volatility = bandwidth // Use bandwidth as volatility proxy
      const stopLoss = calculateStopLoss({
        currentPrice,
        direction,
        confidence,
        agentType: 'volatility',
        volatility,
        riskMultiplier: 1.2 // Wider stops for breakouts
      })
      const limitPrice = calculateLimitPrice({
        currentPrice,
        direction,
        confidence,
        agentType: 'volatility',
        volatility
      })
      const positionSize = calculatePositionSize(confidence, 0.06) // Larger size for high conviction
      
      return this.createSignal(
        direction,
        confidence,
        `Bollinger squeeze release detected - ${direction === 'buy' ? 'upward' : 'downward'} breakout (bandwidth: ${(bandwidth * 100).toFixed(2)}%)`,
        undefined, // analysis
        undefined, // priceTarget
        stopLoss,
        positionSize,
        limitPrice
      )
    }
    
    // Band touch signals with %B trend confirmation
    if (percentB <= 0) { // Below lower band
      const penetration = Math.abs(percentB)
      let confidence = Math.min(0.9, 0.6 + penetration * 2)
      
      // Boost confidence if %B trend is turning bullish
      if (percentBTrend === 'bullish') confidence = Math.min(0.95, confidence + 0.1)
      
      // Calculate prices for band reversal
      const stopLoss = calculateStopLoss({
        currentPrice,
        direction: 'buy',
        confidence,
        agentType: 'reversal',
        volatility: bandwidth
      })
      const limitPrice = calculateLimitPrice({
        currentPrice,
        direction: 'buy',
        confidence,
        agentType: 'reversal',
        volatility: bandwidth
      })
      const positionSize = calculatePositionSize(confidence, 0.04)
      
      return this.createSignal(
        'buy',
        confidence,
        `Price below lower band (%B: ${(percentB * 100).toFixed(1)}%, trend: ${percentBTrend})`,
        undefined, // analysis
        undefined, // priceTarget
        stopLoss,
        positionSize,
        limitPrice
      )
    } else if (percentB >= 1) { // Above upper band
      const penetration = percentB - 1
      let confidence = Math.min(0.9, 0.6 + penetration * 2)
      
      // Boost confidence if %B trend is turning bearish
      if (percentBTrend === 'bearish') confidence = Math.min(0.95, confidence + 0.1)
      
      // Calculate prices for band reversal
      const stopLoss = calculateStopLoss({
        currentPrice,
        direction: 'sell',
        confidence,
        agentType: 'reversal',
        volatility: bandwidth
      })
      const limitPrice = calculateLimitPrice({
        currentPrice,
        direction: 'sell',
        confidence,
        agentType: 'reversal',
        volatility: bandwidth
      })
      const positionSize = calculatePositionSize(confidence, 0.04)
      
      return this.createSignal(
        'sell',
        confidence,
        `Price above upper band (%B: ${(percentB * 100).toFixed(1)}%, trend: ${percentBTrend})`,
        undefined, // analysis
        undefined, // priceTarget
        stopLoss,
        positionSize,
        limitPrice
      )
    }
    
    // Squeeze preparation (low volatility, high probability setup)
    if (isSqueeze) {
      const direction = percentB > 0.5 ? 'buy' : 'sell'
      const confidence = 0.65 + (Math.abs(percentB - 0.5) * 0.3) // Higher confidence when %B is more extreme
      
      // Calculate prices for squeeze setup (anticipating breakout)
      const stopLoss = calculateStopLoss({
        currentPrice,
        direction,
        confidence,
        agentType: 'volatility',
        volatility: bandwidth,
        riskMultiplier: 0.8 // Tighter stops in low volatility
      })
      const limitPrice = calculateLimitPrice({
        currentPrice,
        direction,
        confidence,
        agentType: 'volatility',
        volatility: bandwidth
      })
      const positionSize = calculatePositionSize(confidence, 0.03) // Smaller size for setup
      
      return this.createSignal(
        direction,
        confidence,
        `Bollinger squeeze - low volatility setup (%B: ${(percentB * 100).toFixed(1)}%, bandwidth: ${(bandwidth * 100).toFixed(2)}%)`,
        undefined, // analysis
        undefined, // priceTarget
        stopLoss,
        positionSize,
        limitPrice
      )
    }
    
    // Enhanced mean reversion with %B and trend consideration
    if (percentB < 0.2 || percentB > 0.8) { // Strong mean reversion zones
      const direction = percentB < 0.2 ? 'buy' : 'sell'
      let confidence = 0.6 + Math.abs(percentB - 0.5) * 0.8
      
      // Adjust confidence based on %B trend
      if ((direction === 'buy' && percentBTrend === 'bullish') || 
          (direction === 'sell' && percentBTrend === 'bearish')) {
        confidence = Math.min(0.85, confidence + 0.15)
      } else if ((direction === 'buy' && percentBTrend === 'bearish') || 
                 (direction === 'sell' && percentBTrend === 'bullish')) {
        confidence = Math.max(0.4, confidence - 0.15)
      }
      
      // Calculate prices for mean reversion
      const stopLoss = calculateStopLoss({
        currentPrice,
        direction,
        confidence,
        agentType: 'reversal',
        volatility: bandwidth
      })
      const limitPrice = calculateLimitPrice({
        currentPrice,
        direction,
        confidence,
        agentType: 'reversal',
        volatility: bandwidth
      })
      const positionSize = calculatePositionSize(confidence)
      
      return this.createSignal(
        direction,
        confidence,
        `Mean reversion signal (%B: ${(percentB * 100).toFixed(1)}%, trend: ${percentBTrend})`,
        undefined, // analysis
        undefined, // priceTarget
        stopLoss,
        positionSize,
        limitPrice
      )
    }
    
    // Neutral zone
    return this.createSignal(
      'hold',
      0.5,
      `Price in neutral zone (%B: ${(percentB * 100).toFixed(1)}%, bandwidth: ${(bandwidth * 100).toFixed(2)}%)`
    )
  }

  /**
   * Detect if bands are in a squeeze (low volatility period)
   * Squeeze occurs when bandwidth is in the lowest 20% of recent history
   */
  private detectSqueeze(): boolean {
    if (this.bandwidthHistory.length < 5) return false
    
    const currentBandwidth = this.bandwidthHistory[this.bandwidthHistory.length - 1]!
    const sortedBandwidths = [...this.bandwidthHistory].sort((a, b) => a - b)
    const percentile20 = sortedBandwidths[Math.floor(sortedBandwidths.length * 0.2)]!
    
    return currentBandwidth <= percentile20
  }

  /**
   * Detect if bands are releasing from a squeeze (volatility expansion)
   */
  private detectSqueezeRelease(): boolean {
    if (this.bandwidthHistory.length < 3) return false
    
    const current = this.bandwidthHistory[this.bandwidthHistory.length - 1]!
    const previous = this.bandwidthHistory[this.bandwidthHistory.length - 2]!
    const twoPrevious = this.bandwidthHistory[this.bandwidthHistory.length - 3]!
    
    // Bandwidth is expanding for 2 consecutive periods
    return current > previous && previous > twoPrevious
  }

  /**
   * Calculate %B trend to determine momentum direction
   */
  private calculatePercentBTrend(): 'bullish' | 'bearish' | 'neutral' {
    if (this.percentBHistory.length < 3) return 'neutral'
    
    const recent = this.percentBHistory.slice(-3)
    const current = recent[2]!
    const previous = recent[1]!
    const twoPrevious = recent[0]!
    
    // Check for consistent trend
    if (current > previous && previous > twoPrevious) {
      return 'bullish'
    } else if (current < previous && previous < twoPrevious) {
      return 'bearish'
    }
    
    return 'neutral'
  }
}