import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'

export interface TickImbalanceBarParams extends BarGeneratorParams {
  /** 
   * Imbalance threshold that triggers bar completion
   * @example 10 // Complete bars when imbalance reaches ±10 ticks or volume units
   * @minimum 0 (exclusive)
   */
  imbalanceThreshold: number
  
  /** 
   * Use volume imbalance instead of tick count imbalance
   * @default false
   * - false: Count buy vs sell ticks
   * - true: Sum buy vs sell volume
   */
  useVolume?: boolean
}

interface TickImbalanceState extends BarState {
  buyTicks: number
  sellTicks: number
  buyVolume: number
  sellVolume: number
  previousClose?: number
}

/**
 * Tick Imbalance Bar Generator
 * 
 * Generates bars based on tick imbalance - the difference between buy-side and sell-side
 * activity. This advanced bar type is particularly useful for detecting order flow imbalances
 * and market microstructure patterns.
 * 
 * ## Algorithm
 * 
 * 1. **Tick Classification**: Classify each tick as buy, sell, or neutral based on price movement
 *    - Buy tick: current price > previous price  
 *    - Sell tick: current price < previous price
 *    - Neutral: current price = previous price
 * 
 * 2. **Imbalance Calculation**: Track imbalance using either:
 *    - Tick Count: |buyTicks - sellTicks|
 *    - Volume: |buyVolume - sellVolume|
 * 
 * 3. **Bar Completion**: When imbalance >= threshold, complete the bar
 * 4. **Bar Reset**: Start new bar with fresh imbalance tracking
 * 
 * ## Use Cases
 * 
 * - **Order Flow Analysis**: Detect buying and selling pressure
 * - **Market Microstructure**: Analyze tick-by-tick market dynamics
 * - **Momentum Detection**: Identify periods of directional pressure
 * - **Liquidity Studies**: Understand market maker vs. taker activity
 * - **High-Frequency Trading**: Detect short-term imbalances for arbitrage
 * 
 * ## Advantages over Time-Based Bars
 * 
 * - **Directional Sensitivity**: Captures periods of strong directional flow
 * - **Market Pressure**: Highlights buying/selling pressure imbalances
 * - **Natural Breakpoints**: Bars form at natural inflection points
 * - **Microstructure Insights**: Reveals hidden order flow patterns
 * 
 * ## Considerations
 * 
 * - **Direction Dependent**: Bars form based on directional imbalance strength
 * - **Market Regime Sensitivity**: Threshold may need adjustment for different assets
 * - **Neutral Ticks**: Price-unchanged ticks don't contribute to imbalance
 * - **Volatility Impact**: High volatility may cause frequent direction changes
 * 
 * @example
 * ```typescript
 * // Detect imbalances of 20 ticks in either direction
 * const tickImbalanceBars = new TickImbalanceBarGenerator({
 *   imbalanceThreshold: 20,
 *   useVolume: false // Count ticks, not volume
 * })
 * 
 * // Use volume-weighted imbalance for institutional flow detection
 * const volumeImbalanceBars = new TickImbalanceBarGenerator({
 *   imbalanceThreshold: 100000, // 100K volume imbalance
 *   useVolume: true
 * })
 * 
 * // High-frequency micro-imbalance detection
 * const microImbalanceBars = new TickImbalanceBarGenerator({
 *   imbalanceThreshold: 5, // Very sensitive
 *   useVolume: false
 * })
 * ```
 */
export class TickImbalanceBarGenerator extends BarGeneratorTransform<TickImbalanceBarParams> {
  constructor(params: TickImbalanceBarParams) {
    super(params, 'tickImbalanceBars' as any, 'Tick Imbalance Bar Generator')
    
    if (!params.imbalanceThreshold || params.imbalanceThreshold <= 0) {
      throw new Error('imbalanceThreshold must be greater than 0')
    }
  }

  private classifyTick(currentPrice: number, previousPrice: number | undefined): 'buy' | 'sell' | 'neutral' {
    if (previousPrice === undefined) {
      return 'neutral'
    }
    
    if (currentPrice > previousPrice) {
      return 'buy'
    } else if (currentPrice < previousPrice) {
      return 'sell'
    } else {
      return 'neutral'
    }
  }

  isBarComplete(_symbol: string, _tick: OhlcvDto, state: TickImbalanceState): boolean {
    // Check current imbalance (already updated in updateBar)
    const imbalance = this.params.useVolume 
      ? Math.abs((state.buyVolume || 0) - (state.sellVolume || 0))
      : Math.abs((state.buyTicks || 0) - (state.sellTicks || 0))
    
    return imbalance >= this.params.imbalanceThreshold
  }

  createNewBar(symbol: string, tick: OhlcvDto): TickImbalanceState {
    const currentState = this.symbolState.get(symbol) as TickImbalanceState
    const previousClose = currentState?.currentBar?.close
    const tickType = this.classifyTick(tick.close, previousClose)
    
    const newState: TickImbalanceState = {
      currentBar: {
        ...tick,
        timestamp: tick.timestamp
      },
      buyTicks: tickType === 'buy' ? 1 : 0,
      sellTicks: tickType === 'sell' ? 1 : 0,
      buyVolume: tickType === 'buy' ? tick.volume : 0,
      sellVolume: tickType === 'sell' ? tick.volume : 0,
      previousClose: tick.close,
      tickImbalance: 0,
      complete: false
    }
    
    return newState
  }

  updateBar(_symbol: string, currentBar: OhlcvDto, tick: OhlcvDto, state: TickImbalanceState): OhlcvDto {
    const tickType = this.classifyTick(tick.close, state.previousClose)
    
    // Update tick counts and volumes
    if (tickType === 'buy') {
      state.buyTicks = (state.buyTicks || 0) + 1
      state.buyVolume = (state.buyVolume || 0) + tick.volume
    } else if (tickType === 'sell') {
      state.sellTicks = (state.sellTicks || 0) + 1
      state.sellVolume = (state.sellVolume || 0) + tick.volume
    }
    
    // Update imbalance
    state.tickImbalance = this.params.useVolume
      ? state.buyVolume - state.sellVolume
      : state.buyTicks - state.sellTicks
    
    // Update previous close
    state.previousClose = tick.close

    // Update OHLCV values
    return {
      ...currentBar,
      high: Math.max(currentBar.high, tick.high),
      low: Math.min(currentBar.low, tick.low),
      close: tick.close,
      volume: currentBar.volume + tick.volume,
      // Keep the timestamp of the last tick
      timestamp: tick.timestamp
    }
  }

  resetState(state: TickImbalanceState): void {
    state.buyTicks = 0
    state.sellTicks = 0
    state.buyVolume = 0
    state.sellVolume = 0
    state.tickImbalance = 0
    state.complete = false
    // Keep previousClose for continuity
  }

  withParams(params: Partial<TickImbalanceBarParams>): TickImbalanceBarGenerator {
    return new TickImbalanceBarGenerator({ ...this.params, ...params })
  }

  validate(): void {
    // Skip base validation that checks for input columns
    // Bar generators use standard OHLCV fields instead
    
    if (this.params.imbalanceThreshold <= 0) {
      throw new Error('imbalanceThreshold must be greater than 0')
    }
  }

  /**
   * Override getState to include imbalance-specific state
   */
  getState(): Record<string, BarState> {
    const state: Record<string, BarState> = {}
    for (const [symbol, barState] of this.symbolState.entries()) {
      const imbalanceState = barState as TickImbalanceState
      state[symbol] = {
        ...barState,
        buyTicks: imbalanceState.buyTicks,
        sellTicks: imbalanceState.sellTicks,
        buyVolume: imbalanceState.buyVolume,
        sellVolume: imbalanceState.sellVolume,
        previousClose: imbalanceState.previousClose,
        tickImbalance: imbalanceState.tickImbalance
      } as any
    }
    return state
  }
}