import { IndicatorCalculator } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { AdaptiveBaseAgent } from './adaptive-base-agent'
import type { MarketRegime } from './market-regime-detector'

interface AdaptiveMacdConfig {
  baseFastPeriod?: number
  baseSlowPeriod?: number
  baseSignalPeriod?: number
}

/**
 * Adaptive MACD Agent
 * 
 * An intelligent MACD-based trading agent that dynamically adjusts its parameters
 * based on market conditions. This agent optimizes MACD responsiveness for different
 * market regimes while maintaining signal quality.
 * 
 * ## Adaptive Parameters:
 * 
 * ### Period Adjustments by Regime
 * - **Trending Markets**: 
 *   - All periods reduced to 90% of base for faster trend following
 *   - Helps capture trend continuations more quickly
 * - **Ranging Markets**:
 *   - Fast period increased by 20%, slow by 10%, signal by 10%
 *   - Reduces whipsaws in sideways markets
 * - **Breakout Markets**:
 *   - Very responsive: Fast at 80%, slow at 85%, signal at 80%
 *   - Captures explosive moves early
 * - **Reversal Markets**:
 *   - Standard settings for balanced sensitivity
 * 
 * ### Volatility Adjustments
 * - **High Volatility**: Signal period reduced by 1 (min: 6)
 * - **Low Volatility**: Signal period increased by 1 (max: 12)
 * 
 * ## Signal Generation by Regime:
 * 
 * ### Trending Markets
 * - Zero line crossovers in trend direction (85% confidence)
 * - Histogram expansion confirming trend (75% confidence)
 * - Focuses on continuation patterns
 * 
 * ### Ranging Markets
 * - Signal line crossovers with histogram confirmation
 * - Confidence based on crossover strength
 * - Avoids signals near zero line
 * 
 * ### Breakout Markets
 * - Strong histogram readings (>0.02) with volume
 * - Quick reaction to momentum shifts
 * - 80% base confidence for aligned signals
 * 
 * ### Reversal Markets
 * - Combines crossovers with weakening momentum
 * - More cautious approach (65% base confidence)
 * - Looks for divergence patterns
 */
export class AdaptiveMacdAgent extends AdaptiveBaseAgent {
  private readonly calculator = new IndicatorCalculator()
  
  // Base configuration
  private readonly baseConfig: Required<AdaptiveMacdConfig>
  
  // Adaptive parameters
  private fastPeriod: number
  private slowPeriod: number
  private signalPeriod: number
  
  // Historical data
  private macdHistory: number[] = []
  private signalHistory: number[] = []
  private histogramHistory: number[] = []
  private priceHistory: number[] = []
  private readonly historyLength = 15
  
  constructor(metadata: any, logger?: any, config?: AdaptiveMacdConfig) {
    super(metadata, logger, {
      adaptationRate: 0.15,
      regimeMemory: 10,
      parameterBounds: {
        fastPeriod: { min: 8, max: 18 },
        slowPeriod: { min: 20, max: 35 },
        signalPeriod: { min: 6, max: 12 }
      }
    })
    
    this.baseConfig = {
      baseFastPeriod: config?.baseFastPeriod ?? 12,
      baseSlowPeriod: config?.baseSlowPeriod ?? 26,
      baseSignalPeriod: config?.baseSignalPeriod ?? 9
    }
    
    // Initialize with base values
    this.fastPeriod = this.baseConfig.baseFastPeriod
    this.slowPeriod = this.baseConfig.baseSlowPeriod
    this.signalPeriod = this.baseConfig.baseSignalPeriod
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Adaptive MACD Agent initialized', {
      baseFast: this.baseConfig.baseFastPeriod,
      baseSlow: this.baseConfig.baseSlowPeriod,
      baseSignal: this.baseConfig.baseSignalPeriod
    })
  }
  
  protected async adaptParameters(regime: MarketRegime): Promise<void> {
    // Adapt periods based on market conditions
    switch (regime.regime) {
      case 'trending':
        // In trending markets, use slightly faster settings
        this.fastPeriod = Math.floor(this.baseConfig.baseFastPeriod * 0.9)
        this.slowPeriod = Math.floor(this.baseConfig.baseSlowPeriod * 0.9)
        this.signalPeriod = Math.floor(this.baseConfig.baseSignalPeriod * 0.9)
        break
        
      case 'ranging':
        // In ranging markets, use slower settings for noise reduction
        this.fastPeriod = Math.floor(this.baseConfig.baseFastPeriod * 1.2)
        this.slowPeriod = Math.floor(this.baseConfig.baseSlowPeriod * 1.1)
        this.signalPeriod = Math.floor(this.baseConfig.baseSignalPeriod * 1.1)
        break
        
      case 'breakout':
        // During breakouts, use very responsive settings
        this.fastPeriod = Math.floor(this.baseConfig.baseFastPeriod * 0.8)
        this.slowPeriod = Math.floor(this.baseConfig.baseSlowPeriod * 0.85)
        this.signalPeriod = Math.floor(this.baseConfig.baseSignalPeriod * 0.8)
        break
        
      case 'reversal':
        // During reversals, use standard settings
        this.fastPeriod = this.baseConfig.baseFastPeriod
        this.slowPeriod = this.baseConfig.baseSlowPeriod
        this.signalPeriod = this.baseConfig.baseSignalPeriod
        break
    }
    
    // Adjust for volatility
    if (regime.volatility === 'high') {
      this.signalPeriod = Math.max(6, this.signalPeriod - 1)
    } else if (regime.volatility === 'low') {
      this.signalPeriod = Math.min(12, this.signalPeriod + 1)
    }
    
    // Apply bounds
    this.fastPeriod = Math.max(8, Math.min(18, this.fastPeriod))
    this.slowPeriod = Math.max(20, Math.min(35, this.slowPeriod))
    this.signalPeriod = Math.max(6, Math.min(12, this.signalPeriod))
    
    this.logger?.debug('MACD parameters adapted', {
      regime: regime.regime,
      volatility: regime.volatility,
      fastPeriod: this.fastPeriod,
      slowPeriod: this.slowPeriod,
      signalPeriod: this.signalPeriod
    })
  }
  
  protected async performAdaptiveAnalysis(
    context: MarketContext,
    regime: MarketRegime
  ): Promise<AgentSignal> {
    const { candles, currentPrice } = context
    
    if (candles.length < this.slowPeriod + this.signalPeriod) {
      return this.createSignal('hold', 0.3, 'Insufficient data for MACD calculation')
    }
    
    // Calculate MACD with adaptive periods
    const macd = this.calculator.macd([...candles], {
      fastPeriod: this.fastPeriod,
      slowPeriod: this.slowPeriod,
      signalPeriod: this.signalPeriod
    })
    
    if (!macd) {
      return this.createSignal('hold', 0.3, 'MACD not available')
    }
    
    const { macd: macdLine, signal, histogram } = macd
    
    // Update history
    this.macdHistory.push(macdLine)
    this.signalHistory.push(signal)
    this.histogramHistory.push(histogram)
    this.priceHistory.push(currentPrice)
    
    // Keep history within limits
    if (this.macdHistory.length > this.historyLength) {
      this.macdHistory.shift()
      this.signalHistory.shift()
      this.histogramHistory.shift()
      this.priceHistory.shift()
    }
    
    // Enhanced analysis
    const crossover = this.detectCrossover()
    const divergence = this.detectDivergence()
    const momentum = this.calculateMomentum()
    const zeroLineCross = this.detectZeroLineCross()
    
    // Generate regime-aware signal
    return this.generateRegimeAwareSignal(
      macdLine,
      signal,
      histogram,
      crossover,
      divergence,
      momentum,
      zeroLineCross,
      regime
    )
  }
  
  private generateRegimeAwareSignal(
    macdLine: number,
    signal: number,
    histogram: number,
    crossover: { type: string, strength: number },
    divergence: { type: string, strength: number },
    momentum: string,
    zeroLineCross: string,
    regime: MarketRegime
  ): AgentSignal {
    // High priority: Divergence signals
    if (divergence.type !== 'none' && divergence.strength > 0.2) {
      const direction = divergence.type === 'bullish' ? 'buy' : 'sell'
      let confidence = 0.75 + divergence.strength * 0.15
      
      // Boost confidence if aligned with regime
      if (regime.regime === 'reversal') confidence += 0.1
      if (regime.momentum === 'weak') confidence += 0.05
      
      return this.createAdaptiveSignal(
        direction,
        confidence,
        `MACD ${divergence.type} divergence (${this.fastPeriod}/${this.slowPeriod}/${this.signalPeriod})`
      )
    }
    
    // Regime-specific signal generation
    switch (regime.regime) {
      case 'trending':
        return this.generateTrendingSignal(macdLine, signal, histogram, crossover, zeroLineCross, momentum, regime)
      case 'ranging':
        return this.generateRangingSignal(macdLine, signal, histogram, crossover, momentum)
      case 'breakout':
        return this.generateBreakoutSignal(macdLine, signal, histogram, crossover, zeroLineCross, regime)
      case 'reversal':
        return this.generateReversalSignal(macdLine, signal, histogram, crossover, momentum)
      default:
        return this.generateDefaultSignal(macdLine, signal, histogram, crossover, momentum)
    }
  }
  
  private generateTrendingSignal(
    macdLine: number,
    signal: number,
    histogram: number,
    crossover: { type: string, strength: number },
    zeroLineCross: string,
    momentum: string,
    regime: MarketRegime
  ): AgentSignal {
    // In trending markets, focus on continuation signals
    if (regime.trend === 'bullish') {
      // Zero line crossovers in trend direction
      if (zeroLineCross === 'bullish') {
        return this.createAdaptiveSignal(
          'buy',
          0.85,
          `MACD bullish trend confirmation (above zero)`
        )
      }
      // Histogram expansion in trend direction
      if (histogram > 0 && momentum === 'strengthening') {
        return this.createAdaptiveSignal(
          'buy',
          0.75,
          `MACD bullish momentum strengthening`
        )
      }
    } else if (regime.trend === 'bearish') {
      if (zeroLineCross === 'bearish') {
        return this.createAdaptiveSignal(
          'sell',
          0.85,
          `MACD bearish trend confirmation (below zero)`
        )
      }
      if (histogram < 0 && momentum === 'strengthening') {
        return this.createAdaptiveSignal(
          'sell',
          0.75,
          `MACD bearish momentum strengthening`
        )
      }
    }
    
    return this.generateDefaultSignal(macdLine, signal, histogram, crossover, momentum)
  }
  
  private generateRangingSignal(
    macdLine: number,
    signal: number,
    histogram: number,
    crossover: { type: string, strength: number },
    momentum: string
  ): AgentSignal {
    // In ranging markets, focus on crossovers
    if (crossover.type === 'bullish' && crossover.strength > 0.01) {
      return this.createAdaptiveSignal(
        'buy',
        Math.min(0.8, 0.65 + crossover.strength * 10),
        `MACD bullish crossover in range`
      )
    }
    
    if (crossover.type === 'bearish' && crossover.strength > 0.01) {
      return this.createAdaptiveSignal(
        'sell',
        Math.min(0.8, 0.65 + crossover.strength * 10),
        `MACD bearish crossover in range`
      )
    }
    
    return this.createAdaptiveSignal('hold', 0.5, `MACD neutral in range`)
  }
  
  private generateBreakoutSignal(
    macdLine: number,
    signal: number,
    histogram: number,
    crossover: { type: string, strength: number },
    zeroLineCross: string,
    regime: MarketRegime
  ): AgentSignal {
    // During breakouts, look for strong momentum
    if (regime.volume === 'increasing') {
      if (histogram > 0 && Math.abs(histogram) > 0.02) {
        return this.createAdaptiveSignal(
          'buy',
          0.8,
          `MACD bullish breakout signal`
        )
      }
      if (histogram < 0 && Math.abs(histogram) > 0.02) {
        return this.createAdaptiveSignal(
          'sell',
          0.8,
          `MACD bearish breakout signal`
        )
      }
    }
    
    return this.createAdaptiveSignal('hold', 0.4, `MACD breakout developing`)
  }
  
  private generateReversalSignal(
    macdLine: number,
    signal: number,
    histogram: number,
    crossover: { type: string, strength: number },
    momentum: string
  ): AgentSignal {
    // During reversals, be cautious but look for early signals
    if (crossover.type !== 'none' && momentum === 'weakening') {
      const direction = crossover.type === 'bullish' ? 'buy' : 'sell'
      return this.createAdaptiveSignal(
        direction,
        0.65,
        `MACD potential ${crossover.type} reversal`
      )
    }
    
    return this.createAdaptiveSignal('hold', 0.6, `MACD reversal monitoring`)
  }
  
  private generateDefaultSignal(
    macdLine: number,
    signal: number,
    histogram: number,
    crossover: { type: string, strength: number },
    momentum: string
  ): AgentSignal {
    // Standard MACD logic
    if (crossover.type === 'bullish') {
      return this.createAdaptiveSignal('buy', 0.7, `MACD bullish crossover`)
    }
    
    if (crossover.type === 'bearish') {
      return this.createAdaptiveSignal('sell', 0.7, `MACD bearish crossover`)
    }
    
    return this.createAdaptiveSignal('hold', 0.5, `MACD neutral`)
  }
  
  private detectCrossover(): { type: 'bullish' | 'bearish' | 'none', strength: number } {
    if (this.histogramHistory.length < 2) {
      return { type: 'none', strength: 0 }
    }
    
    const current = this.histogramHistory[this.histogramHistory.length - 1]!
    const previous = this.histogramHistory[this.histogramHistory.length - 2]!
    
    if (previous <= 0 && current > 0) {
      return { type: 'bullish', strength: Math.abs(current) }
    }
    
    if (previous >= 0 && current < 0) {
      return { type: 'bearish', strength: Math.abs(current) }
    }
    
    return { type: 'none', strength: Math.abs(current) }
  }
  
  private detectZeroLineCross(): 'bullish' | 'bearish' | 'none' {
    if (this.macdHistory.length < 2) return 'none'
    
    const current = this.macdHistory[this.macdHistory.length - 1]!
    const previous = this.macdHistory[this.macdHistory.length - 2]!
    
    if (previous <= 0 && current > 0) return 'bullish'
    if (previous >= 0 && current < 0) return 'bearish'
    
    return 'none'
  }
  
  private detectDivergence(): { type: 'bullish' | 'bearish' | 'none', strength: number } {
    if (this.macdHistory.length < 5 || this.priceHistory.length < 5) {
      return { type: 'none', strength: 0 }
    }
    
    const recentMacd = this.macdHistory.slice(-5)
    const recentPrices = this.priceHistory.slice(-5)
    
    const macdHigh = Math.max(...recentMacd)
    const macdLow = Math.min(...recentMacd)
    const priceHigh = Math.max(...recentPrices)
    const priceLow = Math.min(...recentPrices)
    
    const currentMacd = recentMacd[recentMacd.length - 1]!
    const currentPrice = recentPrices[recentPrices.length - 1]!
    
    if (currentPrice === priceLow && currentMacd > macdLow) {
      const macdPrevLowIndex = recentMacd.indexOf(macdLow)
      if (macdPrevLowIndex < recentMacd.length - 1) {
        const strength = Math.abs(currentMacd - macdLow)
        return { type: 'bullish', strength }
      }
    }
    
    if (currentPrice === priceHigh && currentMacd < macdHigh) {
      const macdPrevHighIndex = recentMacd.indexOf(macdHigh)
      if (macdPrevHighIndex < recentMacd.length - 1) {
        const strength = Math.abs(macdHigh - currentMacd)
        return { type: 'bearish', strength }
      }
    }
    
    return { type: 'none', strength: 0 }
  }
  
  private calculateMomentum(): 'strengthening' | 'weakening' | 'stable' {
    if (this.histogramHistory.length < 3) return 'stable'
    
    const recent = this.histogramHistory.slice(-3)
    const current = Math.abs(recent[2]!)
    const previous = Math.abs(recent[1]!)
    const twoPrevious = Math.abs(recent[0]!)
    
    if (current > previous && previous > twoPrevious) {
      return 'strengthening'
    } else if (current < previous && previous < twoPrevious) {
      return 'weakening'
    }
    
    return 'stable'
  }
  
  protected async onReset(): Promise<void> {
    this.macdHistory = []
    this.signalHistory = []
    this.histogramHistory = []
    this.priceHistory = []
    this.currentRegime = null
    this.regimeHistory = []
    
    // Reset to base parameters
    this.fastPeriod = this.baseConfig.baseFastPeriod
    this.slowPeriod = this.baseConfig.baseSlowPeriod
    this.signalPeriod = this.baseConfig.baseSignalPeriod
  }
  
  // Override the base performAnalysis to use adaptive analysis
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    return this.analyze(context)
  }
}