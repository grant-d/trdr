import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'

export interface TickBarParams extends BarGeneratorParams {
  /** 
   * Number of ticks/trades that trigger bar completion
   * @example 100 // Complete bars after every 100 ticks
   * @minimum 1
   */
  ticksPerBar: number
}

/**
 * Tick Bar Generator
 * 
 * Generates bars based on a fixed number of ticks or trades. This is one of the most basic
 * alternative bar types that provides equal representation of market activity rather than time.
 * 
 * ## Algorithm
 * 
 * 1. **Tick Counting**: Count each incoming tick/trade
 * 2. **Bar Completion**: When tick count reaches `ticksPerBar`, complete the bar
 * 3. **Bar Reset**: Start a new bar with count reset to 1 for the next tick
 * 
 * ## Use Cases
 * 
 * - **Market Activity Analysis**: Each bar represents equal market activity in terms of trades
 * - **Noise Reduction**: Filters out time-based noise in low-activity periods
 * - **Fair Comparison**: Compare bars with equal number of market events
 * - **Algorithmic Trading**: Consistent data points for statistical analysis
 * 
 * ## Advantages over Time-Based Bars
 * 
 * - **Activity-Based**: Bars form based on actual market activity, not arbitrary time
 * - **Consistent Data Points**: Each bar contains exactly the same number of market events
 * - **Natural Pace**: Follows the natural rhythm of the market
 * - **No Empty Periods**: No bars with zero activity during market closures
 * 
 * ## Considerations
 * 
 * - **Variable Time**: Bars can span different time periods depending on market activity
 * - **Volume Variation**: Each bar may contain vastly different volumes
 * - **Market Hours**: More bars during active periods, fewer during quiet times
 * 
 * @example
 * ```typescript
 * // Create bars every 50 ticks for active stocks
 * const tickBars = new TickBarGenerator({
 *   ticksPerBar: 50
 * })
 * 
 * // For less active stocks, use fewer ticks per bar
 * const quietStockBars = new TickBarGenerator({
 *   ticksPerBar: 10
 * })
 * 
 * // High-frequency analysis with very small bars
 * const microBars = new TickBarGenerator({
 *   ticksPerBar: 5
 * })
 * ```
 */
export class TickBarGenerator extends BarGeneratorTransform<TickBarParams> {
  constructor(params: TickBarParams) {
    super(params, 'tickBars' as any, 'Tick Bar Generator')
    
    if (!params.ticksPerBar || params.ticksPerBar < 1) {
      throw new Error('ticksPerBar must be at least 1')
    }
  }

  isBarComplete(_symbol: string, _tick: OhlcvDto, state: BarState): boolean {
    return (state.tickCount || 0) >= this.params.ticksPerBar
  }

  createNewBar(_symbol: string, tick: OhlcvDto): BarState {
    return {
      currentBar: {
        ...tick,
        timestamp: tick.timestamp
      },
      tickCount: 1,
      complete: false
    }
  }

  updateBar(_symbol: string, currentBar: OhlcvDto, tick: OhlcvDto, state: BarState): OhlcvDto {
    // Increment tick count
    state.tickCount = (state.tickCount || 0) + 1

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

  resetState(state: BarState): void {
    state.tickCount = 0
    state.complete = false
  }

  withParams(params: Partial<TickBarParams>): TickBarGenerator {
    return new TickBarGenerator({ ...this.params, ...params })
  }

  validate(): void {
    // Skip base validation that checks for input columns
    // Bar generators use standard OHLCV fields instead
    
    if (this.params.ticksPerBar < 1) {
      throw new Error('ticksPerBar must be at least 1')
    }
    
    if (!Number.isInteger(this.params.ticksPerBar)) {
      throw new Error('ticksPerBar must be an integer')
    }
  }
}