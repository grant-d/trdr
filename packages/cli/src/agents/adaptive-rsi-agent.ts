import { IndicatorCalculator } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { AdaptiveBaseAgent } from './adaptive-base-agent'
import type { MarketRegime } from './market-regime-detector'

interface AdaptiveRsiConfig {
  basePeriod?: number
  baseOversold?: number
  baseOverbought?: number
}

/**
 * Adaptive RSI Agent
 * 
 * An intelligent RSI-based trading agent that dynamically adjusts its parameters
 * based on detected market conditions. This agent goes beyond traditional RSI
 * by adapting to different market regimes.
 * 
 * ## Adaptive Parameters:
 * 
 * ### RSI Period
 * - **High Volatility**: Shortened to 80% of base (more responsive)
 * - **Low Volatility**: Extended to 130% of base (smoother signals)
 * - **Normal**: Uses base period (default: 14)
 * 
 * ### Overbought/Oversold Levels
 * - **Trending Markets**:
 *   - Bullish: Levels raised by 5 points (35/75) to avoid premature sells
 *   - Bearish: Levels lowered by 5 points (25/65) to avoid premature buys
 * - **Ranging Markets**: Tightened by 5 points (25/75) for mean reversion
 * - **Breakout Markets**: Widened by 10 points (20/80) to avoid false signals
 * - **Reversal Markets**: Standard levels (30/70) with selective signals
 * 
 * ## Signal Generation by Regime:
 * 
 * ### Trending Markets
 * - Looks for pullbacks in the trend direction
 * - RSI 30-50 in uptrends = buy opportunity
 * - RSI 50-70 in downtrends = sell opportunity
 * 
 * ### Ranging Markets
 * - Classic overbought/oversold strategy
 * - Higher confidence at extreme levels
 * 
 * ### Breakout Markets
 * - Allows RSI to remain extreme during strong moves
 * - Combines with volume and momentum confirmation
 * 
 * ### Reversal Markets
 * - Focuses on momentum loss at extremes
 * - Lower base confidence due to uncertainty
 */
export class AdaptiveRsiAgent extends AdaptiveBaseAgent {
  private readonly calculator = new IndicatorCalculator()
  
  // Base configuration
  private readonly baseConfig: Required<AdaptiveRsiConfig>
  
  // Adaptive parameters
  private rsiPeriod: number
  private oversoldLevel: number
  private overboughtLevel: number
  
  // Historical data for analysis
  private rsiHistory: number[] = []
  private priceHistory: number[] = []
  private readonly historyLength = 20
  
  constructor(metadata: any, logger?: any, config?: AdaptiveRsiConfig) {
    super(metadata, logger, {
      adaptationRate: 0.15,
      regimeMemory: 10,
      parameterBounds: {
        rsiPeriod: { min: 7, max: 28 },
        oversoldLevel: { min: 20, max: 40 },
        overboughtLevel: { min: 60, max: 80 }
      }
    })
    
    this.baseConfig = {
      basePeriod: config?.basePeriod ?? 14,
      baseOversold: config?.baseOversold ?? 30,
      baseOverbought: config?.baseOverbought ?? 70
    }
    
    // Initialize with base values
    this.rsiPeriod = this.baseConfig.basePeriod
    this.oversoldLevel = this.baseConfig.baseOversold
    this.overboughtLevel = this.baseConfig.baseOverbought
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Adaptive RSI Agent initialized', {
      basePeriod: this.baseConfig.basePeriod,
      baseOversold: this.baseConfig.baseOversold,
      baseOverbought: this.baseConfig.baseOverbought
    })
  }
  
  protected async adaptParameters(regime: MarketRegime): Promise<void> {
    // Adapt RSI period based on volatility
    if (regime.volatility === 'high') {
      this.rsiPeriod = Math.max(7, Math.floor(this.baseConfig.basePeriod * 0.8))
    } else if (regime.volatility === 'low') {
      this.rsiPeriod = Math.min(28, Math.floor(this.baseConfig.basePeriod * 1.3))
    } else {
      this.rsiPeriod = this.baseConfig.basePeriod
    }
    
    // Adapt overbought/oversold levels based on trend and regime
    switch (regime.regime) {
      case 'trending':
        if (regime.trend === 'bullish') {
          // In bull trends, raise overbought and oversold levels
          this.oversoldLevel = this.baseConfig.baseOversold + 5
          this.overboughtLevel = this.baseConfig.baseOverbought + 5
        } else if (regime.trend === 'bearish') {
          // In bear trends, lower overbought and oversold levels
          this.oversoldLevel = this.baseConfig.baseOversold - 5
          this.overboughtLevel = this.baseConfig.baseOverbought - 5
        }
        break
        
      case 'ranging':
        // Tighten levels for mean reversion in ranging markets
        this.oversoldLevel = this.baseConfig.baseOversold - 5
        this.overboughtLevel = this.baseConfig.baseOverbought + 5
        break
        
      case 'breakout':
        // Widen levels during breakouts to avoid false signals
        this.oversoldLevel = this.baseConfig.baseOversold - 10
        this.overboughtLevel = this.baseConfig.baseOverbought + 10
        break
        
      case 'reversal':
        // Use standard levels but be more selective
        this.oversoldLevel = this.baseConfig.baseOversold
        this.overboughtLevel = this.baseConfig.baseOverbought
        break
    }
    
    // Apply bounds
    this.oversoldLevel = Math.max(20, Math.min(40, this.oversoldLevel))
    this.overboughtLevel = Math.max(60, Math.min(80, this.overboughtLevel))
    
    this.logger?.debug('RSI parameters adapted', {
      regime: regime.regime,
      trend: regime.trend,
      volatility: regime.volatility,
      rsiPeriod: this.rsiPeriod,
      oversoldLevel: this.oversoldLevel,
      overboughtLevel: this.overboughtLevel
    })
  }
  
  protected async performAdaptiveAnalysis(
    context: MarketContext,
    regime: MarketRegime
  ): Promise<AgentSignal> {
    const { candles, currentPrice } = context
    
    if (candles.length < this.rsiPeriod) {
      return this.createSignal('hold', 0.3, 'Insufficient data for RSI calculation')
    }
    
    // Calculate RSI with adaptive period
    const rsi = this.calculator.rsi([...candles], this.rsiPeriod)
    
    if (!rsi || rsi.value === 0) {
      return this.createSignal('hold', 0.3, 'RSI not available')
    }
    
    const latestRSI = rsi.value
    
    // Update history
    this.rsiHistory.push(latestRSI)
    this.priceHistory.push(currentPrice)
    
    if (this.rsiHistory.length > this.historyLength) {
      this.rsiHistory.shift()
      this.priceHistory.shift()
    }
    
    // Enhanced analysis
    const rsiTrend = this.calculateRsiTrend()
    const divergence = this.detectDivergence()
    const momentum = this.calculateMomentum()
    
    // Generate signal based on regime
    return this.generateRegimeAwareSignal(
      latestRSI,
      rsiTrend,
      divergence,
      momentum,
      regime
    )
  }
  
  private generateRegimeAwareSignal(
    rsi: number,
    rsiTrend: string,
    divergence: { type: string, strength: number },
    momentum: string,
    regime: MarketRegime
  ): AgentSignal {
    // High priority: Divergence signals (most reliable)
    if (divergence.type !== 'none' && divergence.strength > 0.3) {
      const action = divergence.type === 'bullish' ? 'buy' : 'sell'
      let confidence = 0.7 + divergence.strength * 0.2
      
      // Boost confidence if aligned with regime
      if (regime.regime === 'reversal') confidence += 0.1
      
      return this.createAdaptiveSignal(
        action,
        confidence,
        `${divergence.type} RSI divergence (RSI: ${rsi.toFixed(1)}, ${this.oversoldLevel}/${this.overboughtLevel})`
      )
    }
    
    // Regime-specific signal generation
    switch (regime.regime) {
      case 'trending':
        return this.generateTrendingSignal(rsi, rsiTrend, momentum, regime)
      case 'ranging':
        return this.generateRangingSignal(rsi, rsiTrend, momentum)
      case 'breakout':
        return this.generateBreakoutSignal(rsi, rsiTrend, momentum, regime)
      case 'reversal':
        return this.generateReversalSignal(rsi, rsiTrend, momentum)
      default:
        return this.generateDefaultSignal(rsi, rsiTrend, momentum)
    }
  }
  
  private generateTrendingSignal(
    rsi: number,
    rsiTrend: string,
    momentum: string,
    regime: MarketRegime
  ): AgentSignal {
    // In trending markets, use RSI differently
    if (regime.trend === 'bullish') {
      // Look for pullbacks in uptrends
      if (rsi < 50 && rsi > 30 && rsiTrend === 'bullish') {
        return this.createAdaptiveSignal(
          'buy',
          0.75,
          `Bullish trend pullback (RSI: ${rsi.toFixed(1)})`
        )
      }
    } else if (regime.trend === 'bearish') {
      // Look for rallies in downtrends
      if (rsi > 50 && rsi < 70 && rsiTrend === 'bearish') {
        return this.createAdaptiveSignal(
          'sell',
          0.75,
          `Bearish trend rally (RSI: ${rsi.toFixed(1)})`
        )
      }
    }
    
    return this.generateDefaultSignal(rsi, rsiTrend, momentum)
  }
  
  private generateRangingSignal(
    rsi: number,
    rsiTrend: string,
    momentum: string
  ): AgentSignal {
    // Classic overbought/oversold in ranging markets
    if (rsi < this.oversoldLevel) {
      const confidence = Math.min(0.85, (this.oversoldLevel - rsi) / this.oversoldLevel + 0.5)
      return this.createAdaptiveSignal(
        'buy',
        confidence,
        `RSI oversold in range (${rsi.toFixed(1)})`
      )
    }
    
    if (rsi > this.overboughtLevel) {
      const confidence = Math.min(0.85, (rsi - this.overboughtLevel) / (100 - this.overboughtLevel) + 0.5)
      return this.createAdaptiveSignal(
        'sell',
        confidence,
        `RSI overbought in range (${rsi.toFixed(1)})`
      )
    }
    
    return this.createAdaptiveSignal('hold', 0.5, `RSI neutral in range (${rsi.toFixed(1)})`)
  }
  
  private generateBreakoutSignal(
    rsi: number,
    rsiTrend: string,
    momentum: string,
    regime: MarketRegime
  ): AgentSignal {
    // During breakouts, extreme RSI can continue
    if (rsi > this.overboughtLevel && regime.momentum === 'strong' && regime.volume === 'increasing') {
      return this.createAdaptiveSignal(
        'buy',
        0.7,
        `Bullish breakout momentum (RSI: ${rsi.toFixed(1)})`
      )
    }
    
    if (rsi < this.oversoldLevel && regime.momentum === 'strong' && regime.volume === 'increasing') {
      return this.createAdaptiveSignal(
        'sell',
        0.7,
        `Bearish breakout momentum (RSI: ${rsi.toFixed(1)})`
      )
    }
    
    return this.createAdaptiveSignal('hold', 0.4, `Breakout uncertain (RSI: ${rsi.toFixed(1)})`)
  }
  
  private generateReversalSignal(
    rsi: number,
    rsiTrend: string,
    momentum: string
  ): AgentSignal {
    // Look for extreme readings with momentum loss
    if (rsi < this.oversoldLevel && momentum !== 'decelerating') {
      return this.createAdaptiveSignal(
        'buy',
        0.65,
        `Potential bullish reversal (RSI: ${rsi.toFixed(1)})`
      )
    }
    
    if (rsi > this.overboughtLevel && momentum !== 'accelerating') {
      return this.createAdaptiveSignal(
        'sell',
        0.65,
        `Potential bearish reversal (RSI: ${rsi.toFixed(1)})`
      )
    }
    
    return this.createAdaptiveSignal('hold', 0.6, `Reversal developing (RSI: ${rsi.toFixed(1)})`)
  }
  
  private generateDefaultSignal(
    rsi: number,
    rsiTrend: string,
    momentum: string
  ): AgentSignal {
    // Standard RSI logic
    if (rsi < this.oversoldLevel) {
      return this.createAdaptiveSignal('buy', 0.7, `RSI oversold (${rsi.toFixed(1)})`)
    }
    
    if (rsi > this.overboughtLevel) {
      return this.createAdaptiveSignal('sell', 0.7, `RSI overbought (${rsi.toFixed(1)})`)
    }
    
    return this.createAdaptiveSignal('hold', 0.5, `RSI neutral (${rsi.toFixed(1)})`)
  }
  
  private calculateRsiTrend(): 'bullish' | 'bearish' | 'neutral' {
    if (this.rsiHistory.length < 3) return 'neutral'
    
    const recent = this.rsiHistory.slice(-3)
    const current = recent[2]!
    const previous = recent[1]!
    const twoPrevious = recent[0]!
    
    if (current > previous && previous > twoPrevious) {
      return 'bullish'
    } else if (current < previous && previous < twoPrevious) {
      return 'bearish'
    }
    
    return 'neutral'
  }
  
  private detectDivergence(): { type: 'bullish' | 'bearish' | 'none', strength: number } {
    if (this.rsiHistory.length < 5 || this.priceHistory.length < 5) {
      return { type: 'none', strength: 0 }
    }
    
    const recentRSI = this.rsiHistory.slice(-5)
    const recentPrices = this.priceHistory.slice(-5)
    
    const rsiHigh = Math.max(...recentRSI)
    const rsiLow = Math.min(...recentRSI)
    const priceHigh = Math.max(...recentPrices)
    const priceLow = Math.min(...recentPrices)
    
    const currentRSI = recentRSI[recentRSI.length - 1]!
    const currentPrice = recentPrices[recentPrices.length - 1]!
    
    // Bullish divergence
    if (currentPrice === priceLow && currentRSI > rsiLow) {
      const rsiPrevLowIndex = recentRSI.indexOf(rsiLow)
      if (rsiPrevLowIndex < recentRSI.length - 1) {
        const strength = (currentRSI - rsiLow) / rsiLow
        return { type: 'bullish', strength }
      }
    }
    
    // Bearish divergence
    if (currentPrice === priceHigh && currentRSI < rsiHigh) {
      const rsiPrevHighIndex = recentRSI.indexOf(rsiHigh)
      if (rsiPrevHighIndex < recentRSI.length - 1) {
        const strength = (rsiHigh - currentRSI) / rsiHigh
        return { type: 'bearish', strength }
      }
    }
    
    return { type: 'none', strength: 0 }
  }
  
  private calculateMomentum(): 'accelerating' | 'decelerating' | 'stable' {
    if (this.rsiHistory.length < 3) return 'stable'
    
    const recent = this.rsiHistory.slice(-3)
    const currentChange = recent[2]! - recent[1]!
    const previousChange = recent[1]! - recent[0]!
    
    if (Math.abs(currentChange) > Math.abs(previousChange) * 1.2) {
      return 'accelerating'
    } else if (Math.abs(currentChange) < Math.abs(previousChange) * 0.8) {
      return 'decelerating'
    }
    
    return 'stable'
  }
  
  protected async onReset(): Promise<void> {
    this.rsiHistory = []
    this.priceHistory = []
    this.currentRegime = null
    this.regimeHistory = []
    
    // Reset to base parameters
    this.rsiPeriod = this.baseConfig.basePeriod
    this.oversoldLevel = this.baseConfig.baseOversold
    this.overboughtLevel = this.baseConfig.baseOverbought
  }
  
  // Override the base performAnalysis to use adaptive analysis
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    return this.analyze(context)
  }
}