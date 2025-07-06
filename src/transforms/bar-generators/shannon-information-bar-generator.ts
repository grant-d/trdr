import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'
import type { Transform } from '../../interfaces'

export interface ShannonInformationBarParams extends BarGeneratorParams {
  /** 
   * Lookback period for building return distribution
   * @example 20 // Use 20 periods for return distribution
   * @minimum 10
   */
  lookback: number
  
  /** 
   * Information content threshold (bits)
   * @example 5.0 // Complete bar when cumulative info exceeds 5 bits
   * @minimum 1.0
   */
  threshold: number
  
  /** 
   * Exponential decay rate for cumulative information
   * @example 0.90 // 90% retention rate per period
   * @minimum 0.8
   * @maximum 0.99
   */
  decayRate: number
}

interface ShannonInformationState extends BarState {
  /** Historical returns for probability estimation */
  returnHistory: number[]
  /** Cumulative information content */
  cumulativeInformation: number
  /** Previous price for return calculation */
  previousPrice: number | null
}

/**
 * Shannon Information Bar Generator
 * 
 * Uses Shannon information theory to measure the "surprise" content
 * of price movements. Higher information = more unexpected moves.
 * 
 * - Builds probability distribution of price changes over rolling window
 * - Calculates entropy and surprise of each new price update
 * - Information = -log2(probability of observed move)
 * - Cumulates information content with exponential decay
 * - More responsive during news/events, quieter during drift
 * - New bar when cumulative information exceeds threshold
 *
 * Key insight: Rare price moves carry more information than common ones
 * A 5% move when volatility is low = high information
 * A 5% move when volatility is high = low information
 * 
 * ## Algorithm
 * 1. Maintain rolling window of price returns
 * 2. Calculate current return and its statistical rarity
 * 3. Convert rarity to information content (bits)
 * 4. Apply exponential decay to cumulative information
 * 5. Add new information content
 * 6. Create new bar when cumulative information exceeds threshold
 * 
 * ## Use Cases
 * - News-driven trading strategies
 * - Event detection and reaction
 * - Adaptive position sizing based on information flow
 * - Market regime identification
 * 
 * ## Advantages
 * - Quantifies information content of price moves
 * - Adaptive to volatility regimes
 * - Captures surprise/unexpectedness
 * - Self-adjusting sensitivity
 * 
 * @example
 * ```typescript
 * const generator = new ShannonInformationBarGenerator({
 *   lookback: 20,        // 20-period return distribution
 *   threshold: 5.0,      // 5 bits information threshold
 *   decayRate: 0.90      // 90% information retention
 * })
 * ```
 */
export class ShannonInformationBarGenerator extends BarGeneratorTransform<ShannonInformationBarParams> {
  constructor(params: ShannonInformationBarParams) {
    super(params, 'shannonInformation', 'Shannon Information Bar Generator')
    this.validate()
  }

  public validate(): void {
    super.validate()
    
    if (this.params.lookback < 10) {
      throw new Error('Shannon Information Bar Generator: lookback must be at least 10')
    }
    
    if (this.params.threshold < 1.0) {
      throw new Error('Shannon Information Bar Generator: threshold must be at least 1.0')
    }
    
    if (this.params.decayRate < 0.8 || this.params.decayRate >= 1.0) {
      throw new Error('Shannon Information Bar Generator: decayRate must be between 0.8 and 0.99')
    }
  }

  isBarComplete(_symbol: string, tick: OhlcvDto, state: ShannonInformationState): boolean {
    // Apply decay and calculate new information
    this.updateInformation(tick, state)
    
    // Check if cumulative information exceeds threshold
    return state.cumulativeInformation >= this.params.threshold
  }

  createNewBar(symbol: string, tick: OhlcvDto): ShannonInformationState {
    return {
      currentBar: {
        timestamp: tick.timestamp,
        symbol,
        exchange: tick.exchange,
        open: tick.close,
        high: tick.high,
        low: tick.low,
        close: tick.close,
        volume: tick.volume
      },
      complete: false,
      returnHistory: [],
      cumulativeInformation: 0.0,
      previousPrice: tick.close
    }
  }

  updateBar(_symbol: string, currentBar: OhlcvDto, tick: OhlcvDto, state: ShannonInformationState): OhlcvDto {
    // Update previous price for next iteration
    state.previousPrice = tick.close
    
    // Update bar OHLCV
    return {
      timestamp: currentBar.timestamp,
      symbol: currentBar.symbol,
      exchange: currentBar.exchange,
      open: currentBar.open,
      high: Math.max(currentBar.high, tick.high),
      low: Math.min(currentBar.low, tick.low),
      close: tick.close,
      volume: currentBar.volume + tick.volume
    }
  }

  resetState(state: ShannonInformationState): void {
    // Reset cumulative information but keep some recent return history
    state.cumulativeInformation = 0.0
    
    // Keep partial history for continuity
    const keepHistory = Math.floor(this.params.lookback / 2)
    if (state.returnHistory.length > keepHistory) {
      state.returnHistory = state.returnHistory.slice(-keepHistory)
    }
    
    state.complete = false
  }

  private updateInformation(tick: OhlcvDto, state: ShannonInformationState): void {
    // Apply exponential decay to existing information
    state.cumulativeInformation *= this.params.decayRate
    
    // Calculate current return if we have a previous price
    if (state.previousPrice !== null && state.previousPrice > 0) {
      const currentReturn = (tick.close - state.previousPrice) / state.previousPrice
      
      // Update return history
      state.returnHistory.push(currentReturn)
      
      // Keep only lookback periods
      if (state.returnHistory.length > this.params.lookback) {
        state.returnHistory = state.returnHistory.slice(-this.params.lookback)
      }
      
      // Calculate information content if we have enough history
      if (state.returnHistory.length >= 10) {
        const informationContent = this.calculateInformationContent(currentReturn, state.returnHistory)
        state.cumulativeInformation += informationContent
      }
    }
  }

  private calculateInformationContent(currentReturn: number, returnHistory: number[]): number {
    if (returnHistory.length < 2) return 0
    
    // Calculate standard deviation of return history
    const mean = returnHistory.reduce((sum, ret) => sum + ret, 0) / returnHistory.length
    const variance = returnHistory.reduce((sum, ret) => sum + (ret - mean) * (ret - mean), 0) / (returnHistory.length - 1)
    const stdev = Math.sqrt(variance)
    
    if (stdev <= 0) return 0
    
    // Calculate Z-score of current return
    const zScore = Math.abs(currentReturn / stdev)
    
    // Information content based on rarity (squared Z-score scaled)
    // Higher Z-scores (rarer events) contribute more information
    const informationContent = Math.pow(zScore, 2) * 0.5
    
    return informationContent
  }


  public withParams(params: Partial<ShannonInformationBarParams>): Transform<ShannonInformationBarParams> {
    return new ShannonInformationBarGenerator({ ...this.params, ...params })
  }
}