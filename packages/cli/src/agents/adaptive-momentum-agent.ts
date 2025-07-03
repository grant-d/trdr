import { IndicatorCalculator } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { AdaptiveBaseAgent } from './adaptive-base-agent'
import type { MarketRegime } from './market-regime-detector'

interface AdaptiveMomentumConfig {
  baseRsiPeriod?: number
  baseMacdFast?: number
  baseMacdSlow?: number
  baseMacdSignal?: number
  baseDivergenceThreshold?: number
}

/**
 * Adaptive Momentum Agent
 * 
 * A sophisticated momentum analysis agent that combines adaptive RSI and MACD
 * with advanced divergence detection. This agent provides comprehensive momentum
 * signals by synthesizing multiple indicators and adapting to market conditions.
 * 
 * ## Core Features:
 * - Combines RSI and MACD for robust momentum analysis
 * - Advanced divergence detection using pivot points
 * - Dynamic parameter adjustment based on market regime
 * - Confluence-based signal generation
 * 
 * ## Adaptive Parameters:
 * 
 * ### By Market Regime
 * 
 * **Trending Markets**:
 * - RSI Period: 85% of base (faster response)
 * - MACD: Fast 90%, Slow 90% of base
 * - RSI Levels: Bullish (40/80), Bearish (20/60)
 * - Divergence Threshold: 80% of base (more sensitive)
 * 
 * **Ranging Markets**:
 * - All parameters at base values
 * - Standard RSI levels (30/70)
 * - Normal divergence sensitivity
 * 
 * **Breakout Markets**:
 * - RSI Period: 70% of base (very responsive)
 * - MACD: Fast 80%, Slow 85% of base
 * - RSI Levels: 25/75 (moderate)
 * - Divergence Threshold: 130% of base (less sensitive)
 * 
 * **Reversal Markets**:
 * - Standard periods for balance
 * - Normal RSI levels (30/70)
 * - Divergence Threshold: 70% of base (more sensitive)
 * 
 * ### Volatility Adjustments
 * - **High**: MACD Signal period -2 (min: 6)
 * - **Low**: MACD Signal period +1 (max: 12)
 * 
 * ## Signal Generation:
 * 
 * ### Priority 1: Strong Divergences (>70% confidence)
 * - Both RSI and MACD divergence = 95% max confidence
 * - Single indicator divergence = 80% confidence
 * - Boosted in reversal regimes (+10%)
 * - Reduced in strong trends (×0.8)
 * 
 * ### Priority 2: Regime-Specific Signals
 * 
 * **Trending**:
 * - Aligned momentum signals (85% max confidence)
 * - Requires direction match with trend
 * 
 * **Ranging**:
 * - Overbought/oversold conditions
 * - Classic mean reversion approach
 * 
 * **Breakout**:
 * - Strong momentum with high confluence
 * - Requires >70% indicator agreement
 * 
 * **Reversal**:
 * - Divergences with >50% confidence
 * - Focus on momentum shifts
 * 
 * ## Confluence Scoring:
 * - RSI momentum signal
 * - MACD histogram direction
 * - MACD zero line position
 * - Divergence signals (weighted 2x)
 * - Final confidence based on agreement percentage
 */
export class AdaptiveMomentumAgent extends AdaptiveBaseAgent {
  private readonly calculator = new IndicatorCalculator()
  
  // Base configuration
  private readonly baseConfig: Required<AdaptiveMomentumConfig>
  
  // Adaptive parameters
  private rsiPeriod: number
  private rsiOversold: number
  private rsiOverbought: number
  private macdFast: number
  private macdSlow: number
  private macdSignal: number
  private divergenceThreshold: number
  
  // Historical tracking
  private rsiHistory: number[] = []
  private macdHistory: number[] = []
  private macdSignalHistory: number[] = []
  private macdHistogramHistory: number[] = []
  private priceHistory: number[] = []
  private priceHighs: Array<{value: number, index: number}> = []
  private priceLows: Array<{value: number, index: number}> = []
  private readonly historyLength = 20
  
  constructor(metadata: any, logger?: any, config?: AdaptiveMomentumConfig) {
    super(metadata, logger, {
      adaptationRate: 0.15,
      regimeMemory: 10,
      parameterBounds: {
        rsiPeriod: { min: 7, max: 28 },
        rsiOversold: { min: 20, max: 40 },
        rsiOverbought: { min: 60, max: 80 },
        macdFast: { min: 8, max: 18 },
        macdSlow: { min: 20, max: 35 },
        divergenceThreshold: { min: 0.1, max: 0.3 }
      }
    })
    
    this.baseConfig = {
      baseRsiPeriod: config?.baseRsiPeriod ?? 14,
      baseMacdFast: config?.baseMacdFast ?? 12,
      baseMacdSlow: config?.baseMacdSlow ?? 26,
      baseMacdSignal: config?.baseMacdSignal ?? 9,
      baseDivergenceThreshold: config?.baseDivergenceThreshold ?? 0.15
    }
    
    // Initialize with base values
    this.rsiPeriod = this.baseConfig.baseRsiPeriod
    this.rsiOversold = 30
    this.rsiOverbought = 70
    this.macdFast = this.baseConfig.baseMacdFast
    this.macdSlow = this.baseConfig.baseMacdSlow
    this.macdSignal = this.baseConfig.baseMacdSignal
    this.divergenceThreshold = this.baseConfig.baseDivergenceThreshold
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Adaptive Momentum Agent initialized', {
      baseRsiPeriod: this.baseConfig.baseRsiPeriod,
      baseMacdFast: this.baseConfig.baseMacdFast,
      baseMacdSlow: this.baseConfig.baseMacdSlow
    })
  }
  
  protected async adaptParameters(regime: MarketRegime): Promise<void> {
    // Adapt based on market regime
    switch (regime.regime) {
      case 'trending':
        // In trending markets, use faster settings and wider bands
        this.rsiPeriod = Math.floor(this.baseConfig.baseRsiPeriod * 0.85)
        this.macdFast = Math.floor(this.baseConfig.baseMacdFast * 0.9)
        this.macdSlow = Math.floor(this.baseConfig.baseMacdSlow * 0.9)
        
        // Adjust RSI levels for trend following
        if (regime.trend === 'bullish') {
          this.rsiOversold = 40
          this.rsiOverbought = 80
        } else if (regime.trend === 'bearish') {
          this.rsiOversold = 20
          this.rsiOverbought = 60
        }
        
        // Lower divergence threshold in strong trends
        this.divergenceThreshold = this.baseConfig.baseDivergenceThreshold * 0.8
        break
        
      case 'ranging':
        // In ranging markets, use standard settings
        this.rsiPeriod = this.baseConfig.baseRsiPeriod
        this.macdFast = this.baseConfig.baseMacdFast
        this.macdSlow = this.baseConfig.baseMacdSlow
        this.rsiOversold = 30
        this.rsiOverbought = 70
        
        // Standard divergence threshold
        this.divergenceThreshold = this.baseConfig.baseDivergenceThreshold
        break
        
      case 'breakout':
        // During breakouts, use very responsive settings
        this.rsiPeriod = Math.floor(this.baseConfig.baseRsiPeriod * 0.7)
        this.macdFast = Math.floor(this.baseConfig.baseMacdFast * 0.8)
        this.macdSlow = Math.floor(this.baseConfig.baseMacdSlow * 0.85)
        
        // Extreme levels for breakouts
        this.rsiOversold = 25
        this.rsiOverbought = 75
        
        // Higher divergence threshold to avoid false signals
        this.divergenceThreshold = this.baseConfig.baseDivergenceThreshold * 1.3
        break
        
      case 'reversal':
        // During reversals, emphasize divergences
        this.rsiPeriod = this.baseConfig.baseRsiPeriod
        this.macdFast = this.baseConfig.baseMacdFast
        this.macdSlow = this.baseConfig.baseMacdSlow
        this.rsiOversold = 30
        this.rsiOverbought = 70
        
        // Lower divergence threshold for sensitivity
        this.divergenceThreshold = this.baseConfig.baseDivergenceThreshold * 0.7
        break
    }
    
    // Volatility adjustments
    if (regime.volatility === 'high') {
      this.macdSignal = Math.max(6, this.baseConfig.baseMacdSignal - 2)
    } else if (regime.volatility === 'low') {
      this.macdSignal = Math.min(12, this.baseConfig.baseMacdSignal + 1)
    } else {
      this.macdSignal = this.baseConfig.baseMacdSignal
    }
    
    // Apply bounds
    this.rsiPeriod = Math.max(7, Math.min(28, this.rsiPeriod))
    this.macdFast = Math.max(8, Math.min(18, this.macdFast))
    this.macdSlow = Math.max(20, Math.min(35, this.macdSlow))
    this.divergenceThreshold = Math.max(0.1, Math.min(0.3, this.divergenceThreshold))
    
    this.logger?.debug('Momentum parameters adapted', {
      regime: regime.regime,
      volatility: regime.volatility,
      rsiPeriod: this.rsiPeriod,
      rsiLevels: `${this.rsiOversold}/${this.rsiOverbought}`,
      macdPeriods: `${this.macdFast}/${this.macdSlow}/${this.macdSignal}`,
      divergenceThreshold: this.divergenceThreshold
    })
  }
  
  protected async performAdaptiveAnalysis(
    context: MarketContext,
    regime: MarketRegime
  ): Promise<AgentSignal> {
    const { candles, currentPrice } = context
    
    // Check minimum data requirements
    const minCandles = Math.max(this.rsiPeriod, this.macdSlow + this.macdSignal)
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
    
    // Generate regime-aware signal
    return this.synthesizeRegimeAwareSignal(
      rsi,
      macd,
      divergence,
      momentumCondition,
      momentumSignal,
      regime
    )
  }
  
  private synthesizeRegimeAwareSignal(
    rsi: number,
    macd: {line: number, signal: number, histogram: number},
    divergence: {type: 'bullish' | 'bearish' | 'none', strength: number, indicator: string, confidence: number},
    condition: {type: string, severity: string, confidence: number},
    momentum: {direction: string, strength: string, confluence: number, confidence: number},
    regime: MarketRegime
  ): AgentSignal {
    // High priority: Strong divergences with regime alignment
    if (divergence.type !== 'none' && divergence.confidence > 0.7) {
      let confidence = divergence.confidence
      
      // Boost confidence if aligned with reversal regime
      if (regime.regime === 'reversal') {
        confidence = Math.min(0.95, confidence + 0.1)
      }
      
      // Reduce confidence if against strong trend
      if (regime.regime === 'trending' && regime.momentum === 'strong') {
        confidence *= 0.8
      }
      
      const action = divergence.type === 'bullish' ? 'buy' : 'sell'
      return this.createAdaptiveSignal(
        action,
        confidence,
        `Strong ${divergence.type} divergence (${divergence.indicator})`
      )
    }
    
    // Regime-specific logic
    switch (regime.regime) {
      case 'trending':
        // In trends, focus on continuation signals
        if (momentum.direction === 'bullish' && regime.trend === 'bullish' && momentum.strength !== 'weak') {
          return this.createAdaptiveSignal(
            'buy',
            Math.min(0.85, momentum.confidence + 0.1),
            `Bullish momentum in uptrend (RSI: ${rsi.toFixed(1)})`
          )
        }
        if (momentum.direction === 'bearish' && regime.trend === 'bearish' && momentum.strength !== 'weak') {
          return this.createAdaptiveSignal(
            'sell',
            Math.min(0.85, momentum.confidence + 0.1),
            `Bearish momentum in downtrend (RSI: ${rsi.toFixed(1)})`
          )
        }
        break
        
      case 'ranging':
        // In ranges, use overbought/oversold
        if (condition.type === 'oversold' && condition.severity !== 'mild') {
          return this.createAdaptiveSignal(
            'buy',
            condition.confidence,
            `Oversold in range (RSI: ${rsi.toFixed(1)})`
          )
        }
        if (condition.type === 'overbought' && condition.severity !== 'mild') {
          return this.createAdaptiveSignal(
            'sell',
            condition.confidence,
            `Overbought in range (RSI: ${rsi.toFixed(1)})`
          )
        }
        break
        
      case 'breakout':
        // During breakouts, follow strong momentum
        if (momentum.strength === 'strong' && momentum.confluence > 0.7) {
          const action = momentum.direction === 'bullish' ? 'buy' : 'sell'
          return this.createAdaptiveSignal(
            action,
            momentum.confidence,
            `Strong ${momentum.direction} breakout momentum`
          )
        }
        break
        
      case 'reversal':
        // Look for divergences and extreme conditions
        if (divergence.type !== 'none' && divergence.confidence > 0.5) {
          const action = divergence.type === 'bullish' ? 'buy' : 'sell'
          return this.createAdaptiveSignal(
            action,
            divergence.confidence,
            `${divergence.type} reversal signal`
          )
        }
        break
    }
    
    // Default signal
    return this.createAdaptiveSignal(
      'hold',
      0.5,
      `Momentum neutral (RSI: ${rsi.toFixed(1)}, regime: ${regime.regime})`
    )
  }
  
  private calculateRsi(candles: readonly any[]): {success: boolean, value?: number, error?: string} {
    try {
      const rsi = this.calculator.rsi([...candles], this.rsiPeriod)
      if (!rsi || typeof rsi.value !== 'number' || isNaN(rsi.value)) {
        return {success: false, error: 'Invalid RSI calculation'}
      }
      return {success: true, value: rsi.value}
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      return {success: false, error: `RSI calculation error: ${errorMessage}`}
    }
  }
  
  private calculateMacd(candles: readonly any[]): {success: boolean, value?: {line: number, signal: number, histogram: number}, error?: string} {
    try {
      const macd = this.calculator.macd([...candles], {
        fastPeriod: this.macdFast,
        slowPeriod: this.macdSlow,
        signalPeriod: this.macdSignal
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
  
  private updateHistory(rsi: number, macd: {line: number, signal: number, histogram: number}, price: number): void {
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
      while (arr.length > this.historyLength) {
        arr.shift()
      }
    })
    
    // Update pivot points
    this.updatePivotPoints(price)
  }
  
  private updatePivotPoints(price: number): void {
    const lookback = 3
    
    if (this.priceHistory.length < lookback * 2 + 1) return
    
    const centerIndex = this.priceHistory.length - lookback - 1
    const centerPrice = this.priceHistory[centerIndex]!
    
    let isHigh = true
    let isLow = true
    
    for (let i = centerIndex - lookback; i <= centerIndex + lookback; i++) {
      if (i === centerIndex) continue
      const checkPrice = this.priceHistory[i]!
      
      if (checkPrice >= centerPrice) isHigh = false
      if (checkPrice <= centerPrice) isLow = false
    }
    
    if (isHigh) {
      this.priceHighs.push({value: centerPrice, index: centerIndex})
      if (this.priceHighs.length > 5) this.priceHighs.shift()
    }
    
    if (isLow) {
      this.priceLows.push({value: centerPrice, index: centerIndex})
      if (this.priceLows.length > 5) this.priceLows.shift()
    }
  }
  
  private detectDivergences(): {type: 'bullish' | 'bearish' | 'none', strength: number, indicator: string, confidence: number} {
    if (this.priceHighs.length < 2 && this.priceLows.length < 2) {
      return {type: 'none', strength: 0, indicator: 'none', confidence: 0}
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
    
    return {type: 'none', strength: 0, indicator: 'none', confidence: 0}
  }
  
  private detectRsiDivergence(): {type: 'bullish' | 'bearish' | 'none', strength: number} {
    if (this.priceLows.length >= 2) {
      const recentLow = this.priceLows[this.priceLows.length - 1]!
      const prevLow = this.priceLows[this.priceLows.length - 2]!
      
      if (recentLow.value < prevLow.value) {
        const recentRsi = this.rsiHistory[recentLow.index] || this.rsiHistory[this.rsiHistory.length - 1]!
        const prevRsi = this.rsiHistory[prevLow.index] || this.rsiHistory[Math.max(0, this.rsiHistory.length - 5)]!
        
        if (recentRsi > prevRsi + this.divergenceThreshold) {
          const strength = (recentRsi - prevRsi) / 100
          return {type: 'bullish', strength: Math.min(1, strength * 5)}
        }
      }
    }
    
    if (this.priceHighs.length >= 2) {
      const recentHigh = this.priceHighs[this.priceHighs.length - 1]!
      const prevHigh = this.priceHighs[this.priceHighs.length - 2]!
      
      if (recentHigh.value > prevHigh.value) {
        const recentRsi = this.rsiHistory[recentHigh.index] || this.rsiHistory[this.rsiHistory.length - 1]!
        const prevRsi = this.rsiHistory[prevHigh.index] || this.rsiHistory[Math.max(0, this.rsiHistory.length - 5)]!
        
        if (recentRsi < prevRsi - this.divergenceThreshold) {
          const strength = (prevRsi - recentRsi) / 100
          return {type: 'bearish', strength: Math.min(1, strength * 5)}
        }
      }
    }
    
    return {type: 'none', strength: 0}
  }
  
  private detectMacdDivergence(): {type: 'bullish' | 'bearish' | 'none', strength: number} {
    if (this.priceLows.length >= 2) {
      const recentLow = this.priceLows[this.priceLows.length - 1]!
      const prevLow = this.priceLows[this.priceLows.length - 2]!
      
      if (recentLow.value < prevLow.value) {
        const recentMacd = this.macdHistogramHistory[recentLow.index] || 
                          this.macdHistogramHistory[this.macdHistogramHistory.length - 1]!
        const prevMacd = this.macdHistogramHistory[prevLow.index] || 
                        this.macdHistogramHistory[Math.max(0, this.macdHistogramHistory.length - 5)]!
        
        if (recentMacd > prevMacd + this.divergenceThreshold * 0.1) {
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
        
        if (recentMacd < prevMacd - this.divergenceThreshold * 0.1) {
          const strength = Math.abs(prevMacd - recentMacd)
          return {type: 'bearish', strength: Math.min(1, strength * 10)}
        }
      }
    }
    
    return {type: 'none', strength: 0}
  }
  
  private assessOverboughtOversold(rsi: number): {type: 'overbought' | 'oversold' | 'neutral', severity: 'extreme' | 'moderate' | 'mild', confidence: number} {
    if (rsi <= 20) {
      return {type: 'oversold', severity: 'extreme', confidence: 0.9}
    } else if (rsi <= this.rsiOversold) {
      return {type: 'oversold', severity: 'moderate', confidence: 0.7}
    } else if (rsi <= this.rsiOversold + 10) {
      return {type: 'oversold', severity: 'mild', confidence: 0.5}
    } else if (rsi >= 80) {
      return {type: 'overbought', severity: 'extreme', confidence: 0.9}
    } else if (rsi >= this.rsiOverbought) {
      return {type: 'overbought', severity: 'moderate', confidence: 0.7}
    } else if (rsi >= this.rsiOverbought - 10) {
      return {type: 'overbought', severity: 'mild', confidence: 0.5}
    }
    
    return {type: 'neutral', severity: 'mild', confidence: 0.4}
  }
  
  private generateMomentumSignal(
    rsi: number, 
    macd: {line: number, signal: number, histogram: number}, 
    divergence: {type: 'bullish' | 'bearish' | 'none', strength: number, indicator: string, confidence: number}
  ): {direction: 'bullish' | 'bearish' | 'neutral', strength: 'strong' | 'moderate' | 'weak', confluence: number, confidence: number} {
    let bullishSignals = 0
    let bearishSignals = 0
    let totalSignals = 0
    
    // RSI momentum
    if (rsi < this.rsiOversold) bullishSignals++
    else if (rsi > this.rsiOverbought) bearishSignals++
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
  
  protected async onReset(): Promise<void> {
    this.rsiHistory = []
    this.macdHistory = []
    this.macdSignalHistory = []
    this.macdHistogramHistory = []
    this.priceHistory = []
    this.priceHighs = []
    this.priceLows = []
    this.currentRegime = null
    this.regimeHistory = []
    
    // Reset to base parameters
    this.rsiPeriod = this.baseConfig.baseRsiPeriod
    this.rsiOversold = 30
    this.rsiOverbought = 70
    this.macdFast = this.baseConfig.baseMacdFast
    this.macdSlow = this.baseConfig.baseMacdSlow
    this.macdSignal = this.baseConfig.baseMacdSignal
    this.divergenceThreshold = this.baseConfig.baseDivergenceThreshold
  }
  
  // Override the base performAnalysis to use adaptive analysis
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    return this.analyze(context)
  }
}