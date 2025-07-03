import { BaseAgent, IndicatorCalculator } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { momentumPrices } from './agent-price-utils'
import { enforceNoShorting } from './helpers/position-aware'

interface MomentumConfig {
  rsiPeriod?: number
  rsiOversold?: number
  rsiOverbought?: number
  macdFast?: number
  macdSlow?: number
  macdSignal?: number
  historyLength?: number
  divergenceThreshold?: number
  momentumThreshold?: number
}

interface DivergenceAnalysis {
  type: 'bullish' | 'bearish' | 'none'
  strength: number
  indicator: 'rsi' | 'macd' | 'both'
  confidence: number
}

interface MomentumCondition {
  type: 'overbought' | 'oversold' | 'neutral'
  severity: 'extreme' | 'moderate' | 'mild'
  confidence: number
}

interface MomentumSignal {
  direction: 'bullish' | 'bearish' | 'neutral'
  strength: 'strong' | 'moderate' | 'weak'
  confluence: number // 0-1, how many indicators agree
  confidence: number
}

export class MomentumAgent extends BaseAgent {
  private readonly calculator = new IndicatorCalculator()
  
  // Configuration
  protected readonly config: Required<MomentumConfig>
  
  // RSI tracking
  private rsiHistory: number[] = []
  
  // MACD tracking  
  private macdHistory: number[] = []
  private macdSignalHistory: number[] = []
  private macdHistogramHistory: number[] = []
  
  // Price tracking for divergence analysis
  private priceHistory: number[] = []
  private priceHighs: Array<{value: number, index: number}> = []
  private priceLows: Array<{value: number, index: number}> = []
  
  // Analysis cache
  private lastAnalysis: {
    timestamp: number
    rsi: number
    macd: {line: number, signal: number, histogram: number}
    divergence: DivergenceAnalysis
    momentum: MomentumSignal
  } | null = null
  
  constructor(metadata: any, logger?: any, config?: MomentumConfig) {
    super(metadata, logger)
    
    this.config = {
      rsiPeriod: config?.rsiPeriod ?? 14,
      rsiOversold: config?.rsiOversold ?? 30,
      rsiOverbought: config?.rsiOverbought ?? 70,
      macdFast: config?.macdFast ?? 12,
      macdSlow: config?.macdSlow ?? 26,
      macdSignal: config?.macdSignal ?? 9,
      historyLength: config?.historyLength ?? 20,
      divergenceThreshold: config?.divergenceThreshold ?? 0.15,
      momentumThreshold: config?.momentumThreshold ?? 0.1,
    }
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Momentum Agent initialized', {
      rsi: {
        period: this.config.rsiPeriod,
        oversold: this.config.rsiOversold,
        overbought: this.config.rsiOverbought
      },
      macd: {
        fast: this.config.macdFast,
        slow: this.config.macdSlow,
        signal: this.config.macdSignal
      },
      historyLength: this.config.historyLength
    })
  }
  
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    try {
      const { candles, currentPrice } = context
      
      // Check minimum data requirements
      const minCandles = Math.max(this.config.rsiPeriod, this.config.macdSlow + this.config.macdSignal)
      if (candles.length < minCandles) {
        return this.createSignal('hold', 0.3, 'Insufficient data for momentum analysis')
      }
      
      // Calculate indicators
      const rsiResult = this.calculateRsi(candles)
      const macdResult = this.calculateMacd(candles)
      
      if (!rsiResult.success || !macdResult.success) {
        return this.createSignal('hold', 0.3, 'Indicator calculation failed')
      }
      
      const rsi = rsiResult.value!
      const macd = macdResult.value!
      
      // Update history
      this.updateHistory(rsi, macd, currentPrice)
      
      // Perform comprehensive analysis
      const divergence = this.detectDivergences()
      const momentumCondition = this.assessOverboughtOversold(rsi)
      const momentumSignal = this.generateMomentumSignal(rsi, macd, divergence)
      
      // Cache analysis for logging
      this.lastAnalysis = {
        timestamp: Date.now(),
        rsi,
        macd,
        divergence,
        momentum: momentumSignal
      }
      
      // Generate final signal with confidence
      const signal = this.synthesizeSignal(rsi, macd, divergence, momentumCondition, momentumSignal, currentPrice)
      
      // Apply position constraints
      return enforceNoShorting(signal, context)
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.logger?.error('Momentum analysis failed', { error: errorMessage })
      return this.createSignal('hold', 0.2, `Analysis error: ${errorMessage}`)
    }
  }
  
  /**
   * Calculate RSI with error handling
   */
  private calculateRsi(candles: readonly any[]): {success: boolean, value?: number, error?: string} {
    try {
      const rsi = this.calculator.rsi([...candles], this.config.rsiPeriod)
      if (!rsi || typeof rsi.value !== 'number' || isNaN(rsi.value)) {
        return {success: false, error: 'Invalid RSI calculation'}
      }
      return {success: true, value: rsi.value}
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      return {success: false, error: `RSI calculation error: ${errorMessage}`}
    }
  }
  
  /**
   * Calculate MACD with error handling
   */
  private calculateMacd(candles: readonly any[]): {success: boolean, value?: {line: number, signal: number, histogram: number}, error?: string} {
    try {
      const macd = this.calculator.macd([...candles], {
        fastPeriod: this.config.macdFast,
        slowPeriod: this.config.macdSlow,
        signalPeriod: this.config.macdSignal
      })
      
      if (!macd || typeof macd.macd !== 'number' || typeof macd.signal !== 'number') {
        return {success: false, error: 'Invalid MACD calculation'}
      }
      
      const result = {
        line: macd.macd,
        signal: macd.signal,
        histogram: macd.histogram || (macd.macd - macd.signal)
      }
      
      if (Object.values(result).some(v => isNaN(v))) {
        return {success: false, error: 'NaN values in MACD calculation'}
      }
      
      return {success: true, value: result}
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      return {success: false, error: `MACD calculation error: ${errorMessage}`}
    }
  }
  
  /**
   * Update all historical data arrays
   */
  private updateHistory(rsi: number, macd: {line: number, signal: number, histogram: number}, price: number): void {
    // Update arrays
    this.rsiHistory.push(rsi)
    this.macdHistory.push(macd.line)
    this.macdSignalHistory.push(macd.signal)
    this.macdHistogramHistory.push(macd.histogram)
    this.priceHistory.push(price)
    
    // Maintain history length
    const arrays = [
      this.rsiHistory,
      this.macdHistory,
      this.macdSignalHistory,
      this.macdHistogramHistory,
      this.priceHistory
    ]
    
    arrays.forEach(arr => {
      while (arr.length > this.config.historyLength) {
        arr.shift()
      }
    })
    
    // Update pivot points for divergence analysis
    this.updatePivotPoints(price)
  }
  
  /**
   * Track price highs and lows for divergence analysis
   */
  private updatePivotPoints(_price: number): void {
    const lookback = 3 // periods to look back for pivot confirmation
    
    if (this.priceHistory.length < lookback * 2 + 1) return
    
    const centerIndex = this.priceHistory.length - lookback - 1
    const centerPrice = this.priceHistory[centerIndex]!
    
    // Check if center point is a high
    let isHigh = true
    let isLow = true
    
    for (let i = centerIndex - lookback; i <= centerIndex + lookback; i++) {
      if (i === centerIndex) continue
      const checkPrice = this.priceHistory[i]!
      
      if (checkPrice >= centerPrice) isHigh = false
      if (checkPrice <= centerPrice) isLow = false
    }
    
    // Add confirmed pivots
    if (isHigh) {
      this.priceHighs.push({value: centerPrice, index: centerIndex})
      // Keep only recent highs
      if (this.priceHighs.length > 5) this.priceHighs.shift()
    }
    
    if (isLow) {
      this.priceLows.push({value: centerPrice, index: centerIndex})
      // Keep only recent lows  
      if (this.priceLows.length > 5) this.priceLows.shift()
    }
  }
  
  /**
   * Detect bullish and bearish divergences across RSI and MACD
   */
  private detectDivergences(): DivergenceAnalysis {
    if (this.priceHighs.length < 2 && this.priceLows.length < 2) {
      return {type: 'none', strength: 0, indicator: 'rsi', confidence: 0}
    }
    
    const rsiDivergence = this.detectRsiDivergence()
    const macdDivergence = this.detectMacdDivergence()
    
    // Combine divergences for confluence
    if (rsiDivergence.type !== 'none' && macdDivergence.type !== 'none' && 
        rsiDivergence.type === macdDivergence.type) {
      return {
        type: rsiDivergence.type,
        strength: Math.max(rsiDivergence.strength, macdDivergence.strength),
        indicator: 'both',
        confidence: Math.min(0.95, (rsiDivergence.strength + macdDivergence.strength) / 2 + 0.2)
      }
    }
    
    // Return stronger individual divergence
    if (rsiDivergence.strength > macdDivergence.strength) {
      return {...rsiDivergence, indicator: 'rsi', confidence: rsiDivergence.strength * 0.8}
    } else if (macdDivergence.strength > 0) {
      return {...macdDivergence, indicator: 'macd', confidence: macdDivergence.strength * 0.8}
    }
    
    return {type: 'none', strength: 0, indicator: 'rsi', confidence: 0}
  }
  
  /**
   * Detect RSI divergences
   */
  private detectRsiDivergence(): {type: 'bullish' | 'bearish' | 'none', strength: number} {
    // Bullish divergence: lower price lows with higher RSI lows
    if (this.priceLows.length >= 2) {
      const recentLow = this.priceLows[this.priceLows.length - 1]!
      const prevLow = this.priceLows[this.priceLows.length - 2]!
      
      if (recentLow.value < prevLow.value) {
        const recentRsi = this.rsiHistory[recentLow.index] || this.rsiHistory[this.rsiHistory.length - 1]!
        const prevRsi = this.rsiHistory[prevLow.index] || this.rsiHistory[Math.max(0, this.rsiHistory.length - 5)]!
        
        if (recentRsi > prevRsi + this.config.divergenceThreshold) {
          const strength = (recentRsi - prevRsi) / 100
          return {type: 'bullish', strength: Math.min(1, strength * 5)}
        }
      }
    }
    
    // Bearish divergence: higher price highs with lower RSI highs
    if (this.priceHighs.length >= 2) {
      const recentHigh = this.priceHighs[this.priceHighs.length - 1]!
      const prevHigh = this.priceHighs[this.priceHighs.length - 2]!
      
      if (recentHigh.value > prevHigh.value) {
        const recentRsi = this.rsiHistory[recentHigh.index] || this.rsiHistory[this.rsiHistory.length - 1]!
        const prevRsi = this.rsiHistory[prevHigh.index] || this.rsiHistory[Math.max(0, this.rsiHistory.length - 5)]!
        
        if (recentRsi < prevRsi - this.config.divergenceThreshold) {
          const strength = (prevRsi - recentRsi) / 100
          return {type: 'bearish', strength: Math.min(1, strength * 5)}
        }
      }
    }
    
    return {type: 'none', strength: 0}
  }
  
  /**
   * Detect MACD divergences
   */
  private detectMacdDivergence(): {type: 'bullish' | 'bearish' | 'none', strength: number} {
    // Similar logic to RSI but using MACD histogram
    if (this.priceLows.length >= 2) {
      const recentLow = this.priceLows[this.priceLows.length - 1]!
      const prevLow = this.priceLows[this.priceLows.length - 2]!
      
      if (recentLow.value < prevLow.value) {
        const recentMacd = this.macdHistogramHistory[recentLow.index] || 
                          this.macdHistogramHistory[this.macdHistogramHistory.length - 1]!
        const prevMacd = this.macdHistogramHistory[prevLow.index] || 
                        this.macdHistogramHistory[Math.max(0, this.macdHistogramHistory.length - 5)]!
        
        if (recentMacd > prevMacd + this.config.momentumThreshold) {
          const strength = Math.abs(recentMacd - prevMacd)
          return {type: 'bullish', strength: Math.min(1, strength * 10)}
        }
      }
    }
    
    if (this.priceHighs.length >= 2) {
      const recentHigh = this.priceHighs[this.priceHighs.length - 1]!
      const prevHigh = this.priceHighs[this.priceHighs.length - 2]!
      
      if (recentHigh.value > prevHigh.value) {
        const recentMacd = this.macdHistogramHistory[recentHigh.index] || 
                          this.macdHistogramHistory[this.macdHistogramHistory.length - 1]!
        const prevMacd = this.macdHistogramHistory[prevHigh.index] || 
                        this.macdHistogramHistory[Math.max(0, this.macdHistogramHistory.length - 5)]!
        
        if (recentMacd < prevMacd - this.config.momentumThreshold) {
          const strength = Math.abs(prevMacd - recentMacd)
          return {type: 'bearish', strength: Math.min(1, strength * 10)}
        }
      }
    }
    
    return {type: 'none', strength: 0}
  }
  
  /**
   * Assess overbought/oversold conditions
   */
  private assessOverboughtOversold(rsi: number): MomentumCondition {
    if (rsi <= 20) {
      return {type: 'oversold', severity: 'extreme', confidence: 0.9}
    } else if (rsi <= this.config.rsiOversold) {
      return {type: 'oversold', severity: 'moderate', confidence: 0.7}
    } else if (rsi <= 40) {
      return {type: 'oversold', severity: 'mild', confidence: 0.5}
    } else if (rsi >= 80) {
      return {type: 'overbought', severity: 'extreme', confidence: 0.9}
    } else if (rsi >= this.config.rsiOverbought) {
      return {type: 'overbought', severity: 'moderate', confidence: 0.7}
    } else if (rsi >= 60) {
      return {type: 'overbought', severity: 'mild', confidence: 0.5}
    }
    
    return {type: 'neutral', severity: 'mild', confidence: 0.4}
  }
  
  /**
   * Generate comprehensive momentum signal
   */
  private generateMomentumSignal(
    rsi: number, 
    macd: {line: number, signal: number, histogram: number}, 
    divergence: DivergenceAnalysis
  ): MomentumSignal {
    let bullishSignals = 0
    let bearishSignals = 0
    let totalSignals = 0
    
    // RSI momentum
    if (rsi < this.config.rsiOversold) bullishSignals++
    else if (rsi > this.config.rsiOverbought) bearishSignals++
    totalSignals++
    
    // MACD momentum
    if (macd.histogram > 0 && macd.line > macd.signal) bullishSignals++
    else if (macd.histogram < 0 && macd.line < macd.signal) bearishSignals++
    totalSignals++
    
    // MACD zero line
    if (macd.line > 0) bullishSignals++
    else if (macd.line < 0) bearishSignals++
    totalSignals++
    
    // Divergence
    if (divergence.type === 'bullish') bullishSignals += 2
    else if (divergence.type === 'bearish') bearishSignals += 2
    totalSignals += 2
    
    const confluence = Math.max(bullishSignals, bearishSignals) / totalSignals
    const direction = bullishSignals > bearishSignals ? 'bullish' : 
                     bearishSignals > bullishSignals ? 'bearish' : 'neutral'
    
    let strength: 'strong' | 'moderate' | 'weak'
    if (confluence >= 0.8) strength = 'strong'
    else if (confluence >= 0.6) strength = 'moderate'
    else strength = 'weak'
    
    return {
      direction,
      strength,
      confluence,
      confidence: confluence
    }
  }
  
  /**
   * Synthesize final trading signal
   */
  private synthesizeSignal(
    rsi: number,
    macd: {line: number, signal: number, histogram: number},
    divergence: DivergenceAnalysis,
    condition: MomentumCondition,
    momentum: MomentumSignal,
    currentPrice: number
  ): AgentSignal {
    // High priority: Strong divergences
    if (divergence.type !== 'none' && divergence.confidence > 0.8) {
      const action = divergence.type === 'bullish' ? 'buy' : 'sell'
      const prices = momentumPrices(currentPrice, action, divergence.confidence)
      return this.createSignal(
        action,
        divergence.confidence,
        `Strong ${divergence.type} divergence detected (${divergence.indicator.toUpperCase()}, RSI: ${rsi.toFixed(1)}, MACD: ${macd.histogram.toFixed(4)})`,
        undefined, // analysis
        undefined, // priceTarget
        prices.stopLoss,
        prices.positionSize,
        prices.limitPrice
      )
    }
    
    // Extreme overbought/oversold with momentum confirmation
    if (condition.severity === 'extreme') {
      const action = condition.type === 'oversold' ? 'buy' : 'sell'
      let confidence = condition.confidence
      
      // Boost if momentum agrees
      if ((condition.type === 'oversold' && momentum.direction === 'bullish') ||
          (condition.type === 'overbought' && momentum.direction === 'bearish')) {
        confidence = Math.min(0.95, confidence + 0.15)
      }
      
      const prices = momentumPrices(currentPrice, action, confidence)
      return this.createSignal(
        action,
        confidence,
        `Extreme ${condition.type} (RSI: ${rsi.toFixed(1)}, momentum: ${momentum.direction})`,
        undefined, // analysis
        undefined, // priceTarget
        prices.stopLoss,
        prices.positionSize,
        prices.limitPrice
      )
    }
    
    // Strong momentum with confluence
    if (momentum.strength === 'strong' && momentum.confluence > 0.75) {
      const action = momentum.direction === 'bullish' ? 'buy' : 'sell'
      const prices = momentumPrices(currentPrice, action, momentum.confidence)
      return this.createSignal(
        action,
        momentum.confidence,
        `Strong ${momentum.direction} momentum (confluence: ${(momentum.confluence * 100).toFixed(0)}%, RSI: ${rsi.toFixed(1)})`,
        undefined, // analysis
        undefined, // priceTarget
        prices.stopLoss,
        prices.positionSize,
        prices.limitPrice
      )
    }
    
    // Moderate signals
    if (momentum.strength === 'moderate' && momentum.confluence > 0.6) {
      const action = momentum.direction === 'bullish' ? 'buy' : 'sell'
      return this.createSignal(
        action,
        Math.min(0.7, momentum.confidence),
        `Moderate ${momentum.direction} momentum (RSI: ${rsi.toFixed(1)}, MACD hist: ${macd.histogram.toFixed(4)})`
      )
    }
    
    // Weak divergence signals
    if (divergence.type !== 'none' && divergence.confidence > 0.5) {
      const action = divergence.type === 'bullish' ? 'buy' : 'sell'
      return this.createSignal(
        action,
        Math.min(0.6, divergence.confidence),
        `${divergence.type} divergence (${divergence.indicator}, strength: ${(divergence.strength * 100).toFixed(0)}%)`
      )
    }
    
    // Default to neutral
    return this.createSignal(
      'hold',
      0.5,
      `Momentum neutral (RSI: ${rsi.toFixed(1)}, MACD: ${macd.line.toFixed(4)}, confluence: ${(momentum.confluence * 100).toFixed(0)}%)`
    )
  }
  
  /**
   * Get detailed analysis for logging/debugging
   */
  public getLastAnalysis() {
    return this.lastAnalysis
  }
  
  /**
   * Reset agent state for backtesting
   */
  protected async onReset(): Promise<void> {
    this.rsiHistory = []
    this.macdHistory = []
    this.macdSignalHistory = []
    this.macdHistogramHistory = []
    this.priceHistory = []
    this.priceHighs = []
    this.priceLows = []
    this.lastAnalysis = null
  }
}