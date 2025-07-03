import { IndicatorCalculator } from '@trdr/core'
import type { AgentSignal, MarketContext } from '@trdr/core/dist/agents/types'
import { AdaptiveBaseAgent } from './adaptive-base-agent'
import type { MarketRegime } from './market-regime-detector'

interface AdaptiveBollingerConfig {
  basePeriod?: number
  baseStdDev?: number
}

/**
 * Adaptive Bollinger Bands Agent
 * 
 * An intelligent Bollinger Bands trading agent that dynamically adjusts band width
 * and period based on market conditions. This agent optimizes for different volatility
 * environments and market regimes.
 * 
 * ## Adaptive Parameters:
 * 
 * ### Period Adjustments
 * - **High Volatility**: Reduced to 80% of base (more responsive)
 * - **Low Volatility**: Increased to 120% of base (smoother bands)
 * - **Normal**: Uses base period (default: 20)
 * 
 * ### Standard Deviation Adjustments by Regime
 * - **Trending Markets**:
 *   - Base multiplier increased by 20% (wider bands)
 *   - Additional 10% if momentum is strong
 *   - Prevents premature exits in trends
 * - **Ranging Markets**:
 *   - Multiplier reduced by 10% (tighter bands)
 *   - Better for mean reversion trades
 * - **Breakout Markets**:
 *   - Standard multiplier for balanced approach
 * - **Reversal Markets**:
 *   - Multiplier increased by 10% for safety
 * 
 * ### Volume-Based Adjustments
 * - High volume + High volatility: Additional 10% wider bands
 * 
 * ## Signal Generation by Regime:
 * 
 * ### Trending Markets
 * - **Bullish Trends**: 
 *   - Buy on middle band touches (80% confidence)
 *   - Walk the upper band continuation (75% confidence)
 * - **Bearish Trends**:
 *   - Sell on middle band touches (80% confidence)
 *   - Walk the lower band continuation (75% confidence)
 * 
 * ### Ranging Markets
 * - Classic mean reversion at bands (85% confidence)
 * - Near-band signals at 90%/10% levels (70% confidence)
 * 
 * ### Breakout Markets
 * - Squeeze breakouts with volume (80% confidence)
 * - Band breakouts with increasing volume (75% confidence)
 * 
 * ### Reversal Markets
 * - Band touches with expansion (70% confidence)
 * - More conservative approach
 * 
 * ## Special Features:
 * - **Squeeze Detection**: Identifies low volatility compression
 * - **Band Trend Analysis**: Tracks expansion/contraction
 * - **Price Position Tracking**: Monitors location within bands
 */
export class AdaptiveBollingerBandsAgent extends AdaptiveBaseAgent {
  private readonly calculator = new IndicatorCalculator()
  
  // Base configuration
  private readonly baseConfig: Required<AdaptiveBollingerConfig>
  
  // Adaptive parameters
  private period: number
  private stdDev: number
  
  // Historical data
  private squeezeHistory: number[] = []
  private percentBHistory: number[] = []
  private bandwidthHistory: number[] = []
  private priceHistory: number[] = []
  private readonly historyLength = 20
  
  constructor(metadata: any, logger?: any, config?: AdaptiveBollingerConfig) {
    super(metadata, logger, {
      adaptationRate: 0.15,
      regimeMemory: 10,
      parameterBounds: {
        period: { min: 10, max: 30 },
        stdDev: { min: 1.5, max: 3.0 }
      }
    })
    
    this.baseConfig = {
      basePeriod: config?.basePeriod ?? 20,
      baseStdDev: config?.baseStdDev ?? 2.0
    }
    
    // Initialize with base values
    this.period = this.baseConfig.basePeriod
    this.stdDev = this.baseConfig.baseStdDev
  }
  
  protected async onInitialize(): Promise<void> {
    this.logger?.info('Adaptive Bollinger Bands Agent initialized', {
      basePeriod: this.baseConfig.basePeriod,
      baseStdDev: this.baseConfig.baseStdDev
    })
  }
  
  protected async adaptParameters(regime: MarketRegime): Promise<void> {
    // Adapt period based on volatility and regime
    switch (regime.volatility) {
      case 'high':
        // Shorter period in high volatility for responsiveness
        this.period = Math.floor(this.baseConfig.basePeriod * 0.8)
        break
      case 'low':
        // Longer period in low volatility to reduce whipsaws
        this.period = Math.floor(this.baseConfig.basePeriod * 1.2)
        break
      default:
        this.period = this.baseConfig.basePeriod
    }
    
    // Adapt standard deviation based on regime
    switch (regime.regime) {
      case 'trending':
        // Wider bands in trending markets
        this.stdDev = this.baseConfig.baseStdDev * 1.2
        if (regime.momentum === 'strong') {
          this.stdDev *= 1.1
        }
        break
        
      case 'ranging':
        // Tighter bands in ranging markets for mean reversion
        this.stdDev = this.baseConfig.baseStdDev * 0.9
        break
        
      case 'breakout':
        // Standard bands during breakouts
        this.stdDev = this.baseConfig.baseStdDev
        break
        
      case 'reversal':
        // Slightly wider bands during reversals
        this.stdDev = this.baseConfig.baseStdDev * 1.1
        break
    }
    
    // Further adjust based on volume
    if (regime.volume === 'increasing' && regime.volatility === 'high') {
      // Significant market activity - widen bands
      this.stdDev *= 1.1
    }
    
    // Apply bounds
    this.period = Math.max(10, Math.min(30, this.period))
    this.stdDev = Math.max(1.5, Math.min(3.0, this.stdDev))
    
    this.logger?.debug('Bollinger parameters adapted', {
      regime: regime.regime,
      volatility: regime.volatility,
      period: this.period,
      stdDev: this.stdDev
    })
  }
  
  protected async performAdaptiveAnalysis(
    context: MarketContext,
    regime: MarketRegime
  ): Promise<AgentSignal> {
    const { candles, currentPrice } = context
    
    if (candles.length < this.period) {
      return this.createSignal('hold', 0.3, 'Insufficient data for Bollinger Bands calculation')
    }
    
    // Calculate Bollinger Bands with adaptive parameters
    const bollinger = this.calculator.bollingerBands([...candles], { period: this.period, stdDev: this.stdDev })

    if (!bollinger) {
      return this.createSignal('hold', 0.3, 'Bollinger Bands not available')
    }
    
    const { upper, middle, lower } = bollinger
    
    // Calculate key metrics
    const bandwidth = (upper - lower) / middle
    const percentB = (currentPrice - lower) / (upper - lower)
    
    // Detect squeeze (low volatility)
    const squeeze = this.detectSqueeze(bandwidth)
    
    // Update history
    this.bandwidthHistory.push(bandwidth)
    this.percentBHistory.push(percentB)
    this.squeezeHistory.push(squeeze ? 1 : 0)
    this.priceHistory.push(currentPrice)
    
    // Maintain history length
    if (this.bandwidthHistory.length > this.historyLength) {
      this.bandwidthHistory.shift()
      this.percentBHistory.shift()
      this.squeezeHistory.shift()
      this.priceHistory.shift()
    }
    
    // Enhanced analysis
    const bandTrend = this.analyzeBandTrend()
    const pricePosition = this.analyzePricePosition(percentB, currentPrice, upper, middle, lower)
    const squeezeSignal = this.analyzeSqueezePattern()
    
    // Generate regime-aware signal
    return this.generateRegimeAwareSignal(
      currentPrice,
      upper,
      middle,
      lower,
      percentB,
      bandwidth,
      squeeze,
      bandTrend,
      pricePosition,
      squeezeSignal,
      regime
    )
  }
  
  private generateRegimeAwareSignal(
    price: number,
    upper: number,
    middle: number,
    lower: number,
    percentB: number,
    bandwidth: number,
    squeeze: boolean,
    bandTrend: string,
    pricePosition: string,
    squeezeSignal: string,
    regime: MarketRegime
  ): AgentSignal {
    // Regime-specific signal generation
    switch (regime.regime) {
      case 'trending':
        return this.generateTrendingSignal(price, upper, middle, lower, percentB, bandwidth, regime)
      case 'ranging':
        return this.generateRangingSignal(price, upper, middle, lower, percentB, squeeze)
      case 'breakout':
        return this.generateBreakoutSignal(price, upper, middle, lower, percentB, squeeze, squeezeSignal, regime)
      case 'reversal':
        return this.generateReversalSignal(price, upper, middle, lower, percentB, bandTrend)
      default:
        return this.generateDefaultSignal(price, upper, middle, lower, percentB)
    }
  }
  
  private generateTrendingSignal(
    price: number,
    upper: number,
    middle: number,
    lower: number,
    percentB: number,
    bandwidth: number,
    regime: MarketRegime
  ): AgentSignal {
    // In trending markets, use bands differently
    if (regime.trend === 'bullish') {
      // Walk the upper band in uptrends
      if (price > middle && price < upper && percentB > 0.5 && percentB < 0.8) {
        return this.createAdaptiveSignal(
          'buy',
          0.75,
          `Bullish trend continuation (BB walk)`
        )
      }
      // Bounce off middle band
      if (price > lower && price <= middle * 1.01 && regime.momentum !== 'weak') {
        return this.createAdaptiveSignal(
          'buy',
          0.8,
          `Bullish trend pullback to MA`
        )
      }
    } else if (regime.trend === 'bearish') {
      // Walk the lower band in downtrends
      if (price < middle && price > lower && percentB < 0.5 && percentB > 0.2) {
        return this.createAdaptiveSignal(
          'sell',
          0.75,
          `Bearish trend continuation (BB walk)`
        )
      }
      // Bounce off middle band
      if (price < upper && price >= middle * 0.99 && regime.momentum !== 'weak') {
        return this.createAdaptiveSignal(
          'sell',
          0.8,
          `Bearish trend pullback to MA`
        )
      }
    }
    
    return this.generateDefaultSignal(price, upper, middle, lower, percentB)
  }
  
  private generateRangingSignal(
    price: number,
    upper: number,
    middle: number,
    lower: number,
    percentB: number,
    squeeze: boolean
  ): AgentSignal {
    // Classic mean reversion in ranging markets
    if (price > upper) {
      return this.createAdaptiveSignal(
        'sell',
        0.85,
        `BB overbought in range (above upper band)`
      )
    }
    
    if (price < lower) {
      return this.createAdaptiveSignal(
        'buy',
        0.85,
        `BB oversold in range (below lower band)`
      )
    }
    
    // Near bands
    if (percentB > 0.9) {
      return this.createAdaptiveSignal(
        'sell',
        0.7,
        `Near upper band in range`
      )
    }
    
    if (percentB < 0.1) {
      return this.createAdaptiveSignal(
        'buy',
        0.7,
        `Near lower band in range`
      )
    }
    
    return this.createAdaptiveSignal('hold', 0.5, `BB neutral in range`)
  }
  
  private generateBreakoutSignal(
    price: number,
    upper: number,
    middle: number,
    lower: number,
    percentB: number,
    squeeze: boolean,
    squeezeSignal: string,
    regime: MarketRegime
  ): AgentSignal {
    // Squeeze breakout signals
    if (squeezeSignal === 'breakout' && regime.volume === 'increasing') {
      if (price > middle && percentB > 0.5) {
        return this.createAdaptiveSignal(
          'buy',
          0.8,
          `BB squeeze breakout (bullish)`
        )
      }
      if (price < middle && percentB < 0.5) {
        return this.createAdaptiveSignal(
          'sell',
          0.8,
          `BB squeeze breakout (bearish)`
        )
      }
    }
    
    // Band breakouts with volume
    if (price > upper && regime.volume === 'increasing') {
      return this.createAdaptiveSignal(
        'buy',
        0.75,
        `BB upper band breakout`
      )
    }
    
    if (price < lower && regime.volume === 'increasing') {
      return this.createAdaptiveSignal(
        'sell',
        0.75,
        `BB lower band breakout`
      )
    }
    
    return this.createAdaptiveSignal('hold', 0.4, `BB breakout developing`)
  }
  
  private generateReversalSignal(
    price: number,
    upper: number,
    middle: number,
    lower: number,
    percentB: number,
    bandTrend: string
  ): AgentSignal {
    // Look for band touches with expansion
    if (price > upper && bandTrend === 'expanding') {
      return this.createAdaptiveSignal(
        'sell',
        0.7,
        `Potential bearish reversal (BB expansion)`
      )
    }
    
    if (price < lower && bandTrend === 'expanding') {
      return this.createAdaptiveSignal(
        'buy',
        0.7,
        `Potential bullish reversal (BB expansion)`
      )
    }
    
    return this.createAdaptiveSignal('hold', 0.6, `BB reversal monitoring`)
  }
  
  private generateDefaultSignal(
    price: number,
    upper: number,
    middle: number,
    lower: number,
    percentB: number
  ): AgentSignal {
    // Standard Bollinger logic
    if (price > upper) {
      return this.createAdaptiveSignal('sell', 0.7, `Price above upper band`)
    }
    
    if (price < lower) {
      return this.createAdaptiveSignal('buy', 0.7, `Price below lower band`)
    }
    
    return this.createAdaptiveSignal('hold', 0.5, `Price within bands`)
  }
  
  private detectSqueeze(bandwidth: number): boolean {
    if (this.bandwidthHistory.length < 20) return false
    
    const recentBandwidths = this.bandwidthHistory.slice(-20)
    const avgBandwidth = recentBandwidths.reduce((sum, bw) => sum + bw, 0) / recentBandwidths.length
    
    return bandwidth < avgBandwidth * 0.75
  }
  
  private analyzeBandTrend(): 'expanding' | 'contracting' | 'stable' {
    if (this.bandwidthHistory.length < 5) return 'stable'
    
    const recent = this.bandwidthHistory.slice(-5)
    const trend = recent[4]! - recent[0]!
    
    if (trend > recent[0]! * 0.1) return 'expanding'
    if (trend < -recent[0]! * 0.1) return 'contracting'
    
    return 'stable'
  }
  
  private analyzePricePosition(
    percentB: number,
    price: number,
    upper: number,
    middle: number,
    lower: number
  ): string {
    if (price > upper) return 'above_upper'
    if (price < lower) return 'below_lower'
    if (percentB > 0.8) return 'near_upper'
    if (percentB < 0.2) return 'near_lower'
    if (Math.abs(price - middle) / middle < 0.01) return 'at_middle'
    
    return 'neutral'
  }
  
  private analyzeSqueezePattern(): 'pre_squeeze' | 'squeeze' | 'breakout' | 'none' {
    if (this.squeezeHistory.length < 10) return 'none'
    
    const recent = this.squeezeHistory.slice(-10)
    const squeezeCount = recent.filter(s => s === 1).length
    
    // Detect squeeze breakout
    if (recent[recent.length - 1] === 0 && recent[recent.length - 2] === 1 && squeezeCount >= 3) {
      return 'breakout'
    }
    
    // In squeeze
    if (squeezeCount >= 5) return 'squeeze'
    
    // Pre-squeeze (bandwidth contracting)
    const bandTrend = this.analyzeBandTrend()
    if (bandTrend === 'contracting' && squeezeCount >= 2) return 'pre_squeeze'
    
    return 'none'
  }
  
  protected async onReset(): Promise<void> {
    this.squeezeHistory = []
    this.percentBHistory = []
    this.bandwidthHistory = []
    this.priceHistory = []
    this.currentRegime = null
    this.regimeHistory = []
    
    // Reset to base parameters
    this.period = this.baseConfig.basePeriod
    this.stdDev = this.baseConfig.baseStdDev
  }
  
  // Override the base performAnalysis to use adaptive analysis
  protected async performAnalysis(context: MarketContext): Promise<AgentSignal> {
    return this.analyze(context)
  }
}