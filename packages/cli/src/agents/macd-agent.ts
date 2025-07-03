import { BaseAgent, IndicatorCalculator } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'

export class MacdAgent extends BaseAgent {
  private readonly calculator = new IndicatorCalculator()
  private readonly fastPeriod = 12
  private readonly slowPeriod = 26
  private readonly signalPeriod = 9
  private readonly macdHistory: number[] = []
  private readonly signalHistory: number[] = []
  private readonly histogramHistory: number[] = []
  private readonly priceHistory: number[] = []
  private readonly historyLength = 10 // Keep last 10 calculations for crossover and divergence analysis
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('MACD Agent initialized', {
      fast: this.fastPeriod,
      slow: this.slowPeriod,
      signal: this.signalPeriod
    })
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    const { candles, currentPrice } = context
    
    if (candles.length < this.slowPeriod + this.signalPeriod) {
      return this.createSignal('hold', 0.3, 'Insufficient data for MACD calculation')
    }
    
    // Calculate MACD
    const macd = this.calculator.macd(candles, {
      fastPeriod: this.fastPeriod,
      slowPeriod: this.slowPeriod,
      signalPeriod: this.signalPeriod
    })
    
    if (!macd) {
      return this.createSignal('hold', 0.3, 'MACD not available')
    }
    
    const { macd: macdLine, signal, histogram } = macd
    
    // Update history for enhanced analysis
    this.macdHistory.push(macdLine)
    this.signalHistory.push(signal)
    this.histogramHistory.push(histogram)
    this.priceHistory.push(currentPrice)
    
    // Keep history within limits
    if (this.macdHistory.length > this.historyLength) {
      this.macdHistory.shift()
    }
    if (this.signalHistory.length > this.historyLength) {
      this.signalHistory.shift()
    }
    if (this.histogramHistory.length > this.historyLength) {
      this.histogramHistory.shift()
    }
    if (this.priceHistory.length > this.historyLength) {
      this.priceHistory.shift()
    }
    
    // Enhanced analysis
    const crossover = this.detectCrossover()
    const divergence = this.detectDivergence()
    const momentum = this.calculateMomentum()
    const zeroLineCross = this.detectZeroLineCross()
    
    // Enhanced signal generation using historical analysis
    
    // High priority: Divergence signals (most reliable)
    if (divergence.type !== 'none') {
      const direction = divergence.type === 'bullish' ? 'buy' : 'sell'
      let confidence = 0.85
      
      // Boost confidence if confirming other signals
      if ((divergence.type === 'bullish' && zeroLineCross === 'bullish') ||
          (divergence.type === 'bearish' && zeroLineCross === 'bearish')) {
        confidence = 0.92
      }
      
      // Calculate stop loss and limit price for divergence signals
      const stopLossPercent = 0.02 + (1 - confidence) * 0.015 // 2-3.5% based on confidence
      const stopLoss = direction === 'buy' 
        ? currentPrice * (1 - stopLossPercent)
        : currentPrice * (1 + stopLossPercent)
      
      // Slightly aggressive limit for divergence signals (high conviction)
      const limitSlippage = 0.0005 // 0.05% slippage tolerance
      const limitPrice = direction === 'buy'
        ? currentPrice * (1 + limitSlippage)
        : currentPrice * (1 - limitSlippage)
      
      return this.createSignal(
        direction,
        confidence,
        `MACD ${divergence.type} divergence detected (MACD: ${macdLine.toFixed(4)}, momentum: ${momentum})`,
        undefined, // analysis
        undefined, // priceTarget
        stopLoss,
        undefined, // positionSize
        limitPrice
      )
    }
    
    // Zero line crossovers (trend confirmation)
    if (zeroLineCross !== 'none') {
      const direction = zeroLineCross === 'bullish' ? 'buy' : 'sell'
      let confidence = 0.75
      
      // Boost confidence if momentum is strengthening
      if (momentum === 'strengthening') confidence = 0.85
      
      // Calculate stop loss and limit price for zero line crossovers
      const stopLossPercent = 0.025 + (1 - confidence) * 0.02 // 2.5-4.5% based on confidence
      const stopLoss = direction === 'buy' 
        ? currentPrice * (1 - stopLossPercent)
        : currentPrice * (1 + stopLossPercent)
      
      // Conservative limit for trend signals
      const limitPrice = direction === 'buy'
        ? currentPrice * 1.001 // 0.1% above for quick fill
        : currentPrice * 0.999 // 0.1% below for quick fill
      
      return this.createSignal(
        direction,
        confidence,
        `MACD zero line ${zeroLineCross} crossover (MACD: ${macdLine.toFixed(4)}, signal: ${signal.toFixed(4)})`,
        undefined, // analysis
        undefined, // priceTarget
        stopLoss,
        undefined, // positionSize
        limitPrice
      )
    }
    
    // Signal line crossovers (entry/exit signals)
    if (crossover.type !== 'none') {
      const direction = crossover.type === 'bullish' ? 'buy' : 'sell'
      let confidence = 0.7
      
      // Boost confidence based on histogram magnitude and zero line position
      const histogramMagnitude = Math.abs(histogram)
      if (histogramMagnitude > 0.01) confidence = Math.min(0.88, confidence + histogramMagnitude * 2)
      
      // Extra confidence if crossing above/below zero line
      if ((crossover.type === 'bullish' && macdLine > 0) || 
          (crossover.type === 'bearish' && macdLine < 0)) {
        confidence = Math.min(0.9, confidence + 0.1)
      }
      
      return this.createSignal(
        direction,
        confidence,
        `MACD ${crossover.type} crossover (histogram: ${histogram.toFixed(4)}, momentum: ${momentum})`
      )
    }
    
    // Momentum continuation signals
    if (momentum === 'strengthening' && Math.abs(histogram) > 0.005) {
      const direction = histogram > 0 ? 'buy' : 'sell'
      const confidence = 0.6 + Math.min(0.2, Math.abs(histogram) * 10)
      
      return this.createSignal(
        direction,
        confidence,
        `MACD momentum strengthening (histogram: ${histogram.toFixed(4)}, trend: ${macdLine > 0 ? 'bullish' : 'bearish'})`
      )
    }
    
    // Neutral zone
    return this.createSignal(
      'hold',
      0.5,
      `MACD neutral (MACD: ${macdLine.toFixed(4)}, signal: ${signal.toFixed(4)}, histogram: ${histogram.toFixed(4)})`
    )
  }

  /**
   * Detect MACD line crossing signal line (using histogram)
   */
  private detectCrossover(): { type: 'bullish' | 'bearish' | 'none', strength: number } {
    if (this.histogramHistory.length < 2) {
      return { type: 'none', strength: 0 }
    }
    
    const current = this.histogramHistory[this.histogramHistory.length - 1]!
    const previous = this.histogramHistory[this.histogramHistory.length - 2]!
    
    // Bullish crossover: histogram crosses above zero
    if (previous <= 0 && current > 0) {
      return { type: 'bullish', strength: Math.abs(current) }
    }
    
    // Bearish crossover: histogram crosses below zero  
    if (previous >= 0 && current < 0) {
      return { type: 'bearish', strength: Math.abs(current) }
    }
    
    return { type: 'none', strength: Math.abs(current) }
  }

  /**
   * Detect MACD line crossing zero line
   */
  private detectZeroLineCross(): 'bullish' | 'bearish' | 'none' {
    if (this.macdHistory.length < 2) return 'none'
    
    const current = this.macdHistory[this.macdHistory.length - 1]!
    const previous = this.macdHistory[this.macdHistory.length - 2]!
    
    // Bullish: MACD crosses above zero
    if (previous <= 0 && current > 0) {
      return 'bullish'
    }
    
    // Bearish: MACD crosses below zero
    if (previous >= 0 && current < 0) {
      return 'bearish'
    }
    
    return 'none'
  }

  /**
   * Detect divergence between MACD and price
   */
  private detectDivergence(): { type: 'bullish' | 'bearish' | 'none', strength: number } {
    if (this.macdHistory.length < 5 || this.priceHistory.length < 5) {
      return { type: 'none', strength: 0 }
    }
    
    // Get recent 5 periods for divergence analysis
    const recentMacd = this.macdHistory.slice(-5)
    const recentPrices = this.priceHistory.slice(-5)
    
    // Find highs and lows
    const macdHigh = Math.max(...recentMacd)
    const macdLow = Math.min(...recentMacd)
    const priceHigh = Math.max(...recentPrices)
    const priceLow = Math.min(...recentPrices)
    
    const currentMacd = recentMacd[recentMacd.length - 1]!
    const currentPrice = recentPrices[recentPrices.length - 1]!
    
    // Bullish divergence: Price makes lower low, MACD makes higher low
    if (currentPrice === priceLow && currentMacd > macdLow) {
      const macdPrevLowIndex = recentMacd.indexOf(macdLow)
      if (macdPrevLowIndex < recentMacd.length - 1) {
        const strength = Math.abs(currentMacd - macdLow)
        return { type: 'bullish', strength }
      }
    }
    
    // Bearish divergence: Price makes higher high, MACD makes lower high
    if (currentPrice === priceHigh && currentMacd < macdHigh) {
      const macdPrevHighIndex = recentMacd.indexOf(macdHigh)
      if (macdPrevHighIndex < recentMacd.length - 1) {
        const strength = Math.abs(macdHigh - currentMacd)
        return { type: 'bearish', strength }
      }
    }
    
    return { type: 'none', strength: 0 }
  }

  /**
   * Calculate momentum based on histogram changes
   */
  private calculateMomentum(): 'strengthening' | 'weakening' | 'stable' {
    if (this.histogramHistory.length < 3) return 'stable'
    
    const recent = this.histogramHistory.slice(-3)
    const current = Math.abs(recent[2]!)
    const previous = Math.abs(recent[1]!)
    const twoPrevious = Math.abs(recent[0]!)
    
    // Momentum is strengthening if histogram magnitude is increasing
    if (current > previous && previous > twoPrevious) {
      return 'strengthening'
    } else if (current < previous && previous < twoPrevious) {
      return 'weakening'
    }
    
    return 'stable'
  }
}