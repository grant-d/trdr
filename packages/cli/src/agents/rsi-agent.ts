import { BaseAgent, IndicatorCalculator } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'

export class RsiAgent extends BaseAgent {
  private readonly calculator = new IndicatorCalculator()
  private readonly rsiPeriod = 14
  private readonly oversoldLevel = 30
  private readonly overboughtLevel = 70
  private readonly rsiHistory: number[] = []
  private readonly priceHistory: number[] = []
  private readonly historyLength = 10 // Keep last 10 calculations for divergence analysis
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('RSI Agent initialized', {
      period: this.rsiPeriod,
      oversold: this.oversoldLevel,
      overbought: this.overboughtLevel
    })
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { candles, currentPrice } = context
    
    if (candles.length < this.rsiPeriod) {
      return this.createSignal('hold', 0.3, 'Insufficient data for RSI calculation')
    }
    
    // Calculate RSI
    const rsi = this.calculator.rsi(candles, this.rsiPeriod)
    
    if (!rsi || rsi.value === 0) {
      return this.createSignal('hold', 0.3, 'RSI not available')
    }
    
    const latestRSI = rsi.value
    
    // Update history for trend and divergence analysis
    this.rsiHistory.push(latestRSI)
    this.priceHistory.push(currentPrice)
    
    // Keep history within limits
    if (this.rsiHistory.length > this.historyLength) {
      this.rsiHistory.shift()
    }
    if (this.priceHistory.length > this.historyLength) {
      this.priceHistory.shift()
    }
    
    // Enhanced analysis
    const rsiTrend = this.calculateRsiTrend()
    const divergence = this.detectDivergence()
    const momentum = this.calculateMomentum()
    
    // Enhanced signal generation with divergence and momentum analysis
    
    // High priority: Divergence signals (most reliable)
    if (divergence.type !== 'none') {
      const direction = divergence.type === 'bullish' ? 'buy' : 'sell'
      let confidence = 0.8
      
      // Boost confidence if in oversold/overbought zone
      if ((divergence.type === 'bullish' && latestRSI < this.oversoldLevel + 10) ||
          (divergence.type === 'bearish' && latestRSI > this.overboughtLevel - 10)) {
        confidence = 0.9
      }
      
      return this.createSignal(
        direction,
        confidence,
        `${divergence.type} divergence detected (RSI: ${latestRSI.toFixed(1)}, momentum: ${momentum})`
      )
    }
    
    // Oversold with confirmation
    if (latestRSI < this.oversoldLevel) {
      let confidence = Math.min(0.9, (this.oversoldLevel - latestRSI) / this.oversoldLevel + 0.5)
      
      // Boost confidence if RSI trend is turning bullish or momentum is positive
      if (rsiTrend === 'bullish' || momentum === 'accelerating') {
        confidence = Math.min(0.95, confidence + 0.2)
      } else if (rsiTrend === 'bearish' && momentum === 'decelerating') {
        confidence = Math.max(0.5, confidence - 0.2)
      }
      
      return this.createSignal(
        'buy',
        confidence,
        `RSI oversold (${latestRSI.toFixed(1)}, trend: ${rsiTrend}, momentum: ${momentum})`
      )
    }
    
    // Overbought with confirmation
    if (latestRSI > this.overboughtLevel) {
      let confidence = Math.min(0.9, (latestRSI - this.overboughtLevel) / (100 - this.overboughtLevel) + 0.5)
      
      // Boost confidence if RSI trend is turning bearish or momentum is negative
      if (rsiTrend === 'bearish' || momentum === 'decelerating') {
        confidence = Math.min(0.95, confidence + 0.2)
      } else if (rsiTrend === 'bullish' && momentum === 'accelerating') {
        confidence = Math.max(0.5, confidence - 0.2)
      }
      
      return this.createSignal(
        'sell',
        confidence,
        `RSI overbought (${latestRSI.toFixed(1)}, trend: ${rsiTrend}, momentum: ${momentum})`
      )
    }
    
    // Momentum signals in neutral zone
    if (latestRSI > 45 && latestRSI < 55 && momentum === 'accelerating' && rsiTrend === 'bullish') {
      return this.createSignal(
        'buy',
        0.6,
        `RSI momentum building (${latestRSI.toFixed(1)}, trend: ${rsiTrend})`
      )
    } else if (latestRSI > 45 && latestRSI < 55 && momentum === 'decelerating' && rsiTrend === 'bearish') {
      return this.createSignal(
        'sell',
        0.6,
        `RSI momentum declining (${latestRSI.toFixed(1)}, trend: ${rsiTrend})`
      )
    }
    
    // Neutral zone
    return this.createSignal(
      'hold',
      0.5,
      `RSI neutral (${latestRSI.toFixed(1)}, trend: ${rsiTrend}, momentum: ${momentum})`
    )
  }

  /**
   * Calculate RSI trend direction based on recent values
   */
  private calculateRsiTrend(): 'bullish' | 'bearish' | 'neutral' {
    if (this.rsiHistory.length < 3) return 'neutral'
    
    const recent = this.rsiHistory.slice(-3)
    const current = recent[2]!
    const previous = recent[1]!
    const twoPrevious = recent[0]!
    
    // Check for consistent trend over 3 periods
    if (current > previous && previous > twoPrevious) {
      return 'bullish'
    } else if (current < previous && previous < twoPrevious) {
      return 'bearish'
    }
    
    return 'neutral'
  }

  /**
   * Detect bullish or bearish divergence between RSI and price
   */
  private detectDivergence(): { type: 'bullish' | 'bearish' | 'none', strength: number } {
    if (this.rsiHistory.length < 5 || this.priceHistory.length < 5) {
      return { type: 'none', strength: 0 }
    }
    
    // Get recent 5 periods for divergence analysis
    const recentRSI = this.rsiHistory.slice(-5)
    const recentPrices = this.priceHistory.slice(-5)
    
    // Find highs and lows in recent period
    const rsiHigh = Math.max(...recentRSI)
    const rsiLow = Math.min(...recentRSI)
    const priceHigh = Math.max(...recentPrices)
    const priceLow = Math.min(...recentPrices)
    
    const currentRSI = recentRSI[recentRSI.length - 1]!
    const currentPrice = recentPrices[recentPrices.length - 1]!
    
    // Bullish divergence: Price makes lower low, RSI makes higher low
    if (currentPrice === priceLow && currentRSI > rsiLow) {
      const rsiPrevLowIndex = recentRSI.indexOf(rsiLow)
      if (rsiPrevLowIndex < recentRSI.length - 1) { // Not the current period
        const strength = (currentRSI - rsiLow) / rsiLow
        return { type: 'bullish', strength }
      }
    }
    
    // Bearish divergence: Price makes higher high, RSI makes lower high
    if (currentPrice === priceHigh && currentRSI < rsiHigh) {
      const rsiPrevHighIndex = recentRSI.indexOf(rsiHigh)
      if (rsiPrevHighIndex < recentRSI.length - 1) { // Not the current period
        const strength = (rsiHigh - currentRSI) / rsiHigh
        return { type: 'bearish', strength }
      }
    }
    
    return { type: 'none', strength: 0 }
  }

  /**
   * Calculate momentum based on RSI rate of change
   */
  private calculateMomentum(): 'accelerating' | 'decelerating' | 'stable' {
    if (this.rsiHistory.length < 3) return 'stable'
    
    const recent = this.rsiHistory.slice(-3)
    const currentChange = recent[2]! - recent[1]!
    const previousChange = recent[1]! - recent[0]!
    
    // Momentum is accelerating if rate of change is increasing
    if (Math.abs(currentChange) > Math.abs(previousChange) * 1.2) {
      return 'accelerating'
    } else if (Math.abs(currentChange) < Math.abs(previousChange) * 0.8) {
      return 'decelerating'
    }
    
    return 'stable'
  }
}