import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'

export interface TickRunBarParams extends BarGeneratorParams {
  /** 
   * Number of consecutive ticks in same direction to trigger a new bar
   * @example 5 // Complete bars after 5 consecutive up or down ticks
   * @minimum 0 (exclusive)
   */
  runLength: number
  
  /** 
   * Use volume runs instead of tick count runs
   * @default false
   * - false: Count consecutive ticks in same direction
   * - true: Accumulate volume units in same direction runs
   */
  useVolume?: boolean
}

interface TickRunState extends BarState {
  currentRunLength: number
  currentRunDirection: 'up' | 'down' | 'neutral'
  previousClose?: number
  volumeInRun: number
}

/**
 * Tick Run Bar Generator
 * 
 * Generates bars based on tick runs - consecutive ticks moving in the same direction.
 * This advanced bar type is particularly effective for capturing momentum periods
 * and directional market moves.
 * 
 * ## Algorithm
 * 
 * 1. **Direction Classification**: For each tick, determine movement direction:
 *    - Up: current price > previous price
 *    - Down: current price < previous price  
 *    - Neutral: current price = previous price
 * 
 * 2. **Run Tracking**: Track consecutive ticks in the same direction:
 *    - Same direction: increment run length
 *    - Direction change: reset run length to 1
 *    - Neutral: maintain current direction (doesn't break run)
 * 
 * 3. **Bar Completion**: When run length >= runLength threshold, complete the bar
 * 4. **Bar Reset**: Start new bar with fresh run tracking
 * 
 * ## Use Cases
 * 
 * - **Momentum Analysis**: Capture periods of sustained directional movement
 * - **Trend Detection**: Identify the start and end of trending moves
 * - **Breakout Analysis**: Detect when price starts running in one direction
 * - **Market Microstructure**: Analyze tick-level momentum patterns
 * - **Reversal Studies**: Identify when momentum exhausts and reverses
 * 
 * ## Advantages over Time-Based Bars
 * 
 * - **Momentum Sensitive**: Bars form during periods of sustained movement
 * - **Natural Breakpoints**: Bars end when directional momentum changes
 * - **Trend Alignment**: Each bar represents a cohesive directional move
 * - **Volatility Adaptive**: More bars during volatile (trending) periods
 * 
 * ## Considerations
 * 
 * - **Momentum Dependent**: Requires sustained directional movement to form bars
 * - **Neutral Ticks**: Price-unchanged ticks don't break runs but don't extend them
 * - **Market Regime Sensitivity**: Run length may need adjustment for different assets
 * - **Choppy Markets**: May produce few bars in sideways/choppy conditions
 * 
 * @example
 * ```typescript
 * // Capture runs of 10 consecutive directional ticks
 * const tickRunBars = new TickRunBarGenerator({
 *   runLength: 10,
 *   useVolume: false // Count ticks, not volume
 * })
 * 
 * // Use volume-weighted runs for institutional momentum
 * const volumeRunBars = new TickRunBarGenerator({
 *   runLength: 50000, // 50K volume in same direction
 *   useVolume: true
 * })
 * 
 * // High-sensitivity momentum detection
 * const microRunBars = new TickRunBarGenerator({
 *   runLength: 3, // Very short runs
 *   useVolume: false
 * })
 * ```
 */
export class TickRunBarGenerator extends BarGeneratorTransform<TickRunBarParams> {
  constructor(params: TickRunBarParams) {
    super(params, 'tickRunBars' as any, 'Tick Run Bar Generator')
    
    if (!params.runLength || params.runLength <= 0) {
      throw new Error('runLength must be greater than 0')
    }
  }

  private getTickDirection(currentPrice: number, previousPrice: number | undefined): 'up' | 'down' | 'neutral' {
    if (previousPrice === undefined) {
      return 'neutral'
    }
    
    if (currentPrice > previousPrice) {
      return 'up'
    } else if (currentPrice < previousPrice) {
      return 'down'
    } else {
      return 'neutral'
    }
  }

  isBarComplete(_symbol: string, _tick: OhlcvDto, state: TickRunState): boolean {
    // Check if we have enough ticks in the current run
    // This is called AFTER updateBar, so currentRunLength has been updated
    if (this.params.useVolume) {
      // For volume runs, check volume units
      const avgVolumePerTick = state.currentBar.volume / (state.tickCount || 1)
      const volumeRunUnits = state.volumeInRun / avgVolumePerTick
      return volumeRunUnits >= this.params.runLength
    } else {
      // For tick count runs, check if we've reached the threshold
      return state.currentRunLength >= this.params.runLength
    }
  }

  createNewBar(symbol: string, tick: OhlcvDto): TickRunState {
    const currentState = this.symbolState.get(symbol) as TickRunState
    const previousClose = currentState?.currentBar?.close
    const tickDirection = this.getTickDirection(tick.close, previousClose)
    
    // Mark the previous bar as complete if it exists
    if (currentState) {
      currentState.complete = true
    }
    
    const newState: TickRunState = {
      currentBar: {
        ...tick,
        timestamp: tick.timestamp
      },
      // Always start fresh run count for new bar
      currentRunLength: tickDirection === 'neutral' ? 0 : 1,
      currentRunDirection: tickDirection === 'neutral' ? 
        (currentState?.currentRunDirection || 'neutral') : tickDirection,
      previousClose: tick.close,
      volumeInRun: tick.volume,
      tickCount: 1,
      complete: false
    }
    
    return newState
  }

  updateBar(_symbol: string, currentBar: OhlcvDto, tick: OhlcvDto, state: TickRunState): OhlcvDto {
    // Update OHLCV values first
    const updatedBar = {
      ...currentBar,
      high: Math.max(currentBar.high, tick.high),
      low: Math.min(currentBar.low, tick.low),
      close: tick.close,
      volume: currentBar.volume + tick.volume,
      // Keep the timestamp of the last tick
      timestamp: tick.timestamp
    }
    
    // Then update state for next isBarComplete check
    const tickDirection = this.getTickDirection(tick.close, state.previousClose)
    
    // Update tick count
    state.tickCount = (state.tickCount || 0) + 1
    
    // Update run tracking
    if (tickDirection !== 'neutral') {
      if (tickDirection === state.currentRunDirection || state.currentRunDirection === 'neutral') {
        // Same direction or first direction, increment run
        if (state.currentRunDirection === 'neutral') {
          state.currentRunDirection = tickDirection
        }
        state.currentRunLength++
        state.volumeInRun += tick.volume
      } else {
        // Direction changed, reset run
        state.currentRunDirection = tickDirection
        state.currentRunLength = 1
        state.volumeInRun = tick.volume
      }
    } else {
      // Neutral tick, just add volume
      state.volumeInRun += tick.volume
    }
    
    // Update previous close
    state.previousClose = tick.close

    return updatedBar
  }

  resetState(state: TickRunState): void {
    state.currentRunLength = 0
    state.volumeInRun = 0
    state.tickCount = 0
    state.complete = false
    // Keep previousClose and currentRunDirection for continuity
  }

  withParams(params: Partial<TickRunBarParams>): TickRunBarGenerator {
    return new TickRunBarGenerator({ ...this.params, ...params })
  }

  validate(): void {
    // Skip base validation that checks for input columns
    // Bar generators use standard OHLCV fields instead
    
    if (this.params.runLength <= 0) {
      throw new Error('runLength must be greater than 0')
    }
  }

  /**
   * Override getState to include run-specific state
   */
  getState(): Record<string, BarState> {
    const state: Record<string, BarState> = {}
    this.symbolState.forEach((barState, symbol) => {
      const runState = barState as TickRunState
      state[symbol] = {
        ...barState,
        currentRunLength: runState.currentRunLength,
        currentRunDirection: runState.currentRunDirection,
        previousClose: runState.previousClose,
        volumeInRun: runState.volumeInRun
      } as any
    })
    return state
  }
}