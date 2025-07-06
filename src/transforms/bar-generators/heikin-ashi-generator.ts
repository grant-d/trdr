import { BaseTransform } from '../base-transform'
import type { OhlcvDto } from '../../models'
import type { TransformType } from '../../interfaces'
import type { BaseTransformParams } from '../transform-params'

export interface HeikinAshiParams extends BaseTransformParams {
  /** 
   * Prefix for output fields
   * @default 'ha_'
   * @example 'ha_' results in ha_open, ha_high, ha_low, ha_close fields
   */
  outputPrefix?: string
}

interface HeikinAshiState {
  previousHAOpen: number
  previousHAClose: number
}

/**
 * Heikin-Ashi Generator
 * 
 * Converts regular OHLC bars to Heikin-Ashi (HA) bars for smoother trend visualization
 * and noise reduction. Heikin-Ashi is a Japanese candlestick technique that creates
 * more readable charts by filtering out market noise.
 * 
 * ## Algorithm
 * 
 * Heikin-Ashi formulas:
 * 1. **HA_Close = (Open + High + Low + Close) / 4** - Average price of the period
 * 2. **HA_Open = (Previous HA_Open + Previous HA_Close) / 2** - Midpoint of previous HA bar
 * 3. **HA_High = Max(High, HA_Open, HA_Close)** - Highest of period high and HA values
 * 4. **HA_Low = Min(Low, HA_Open, HA_Close)** - Lowest of period low and HA values
 * 
 * ## Special Case: First Bar
 * For the first bar where no previous HA values exist:
 * - HA_Open = (Open + Close) / 2
 * - Other formulas remain the same
 * 
 * ## Use Cases
 * 
 * - **Trend Analysis**: Smoother visualization of price trends
 * - **Noise Reduction**: Filters out small price fluctuations and whipsaws
 * - **Signal Generation**: Clearer buy/sell signals with reduced false positives
 * - **Momentum Detection**: Easier identification of trend strength
 * - **Support/Resistance**: More reliable levels due to smoothing effect
 * 
 * ## Advantages over Regular OHLC
 * 
 * - **Trend Clarity**: Smoother representation of price action
 * - **Noise Filtering**: Reduces market noise and false signals
 * - **Visual Appeal**: More readable and interpretable charts
 * - **Momentum Visualization**: Stronger trends show as longer, more consistent candles
 * 
 * ## Considerations
 * 
 * - **Lagging Indicator**: HA values lag actual prices due to averaging
 * - **Gap Handling**: Gaps are naturally smoothed out, which may obscure important information
 * - **Real Price**: HA prices don't represent actual tradeable prices
 * - **Historical Only**: Best used for analysis, not real-time entry/exit decisions
 * 
 * ## Visual Characteristics
 * 
 * - **Uptrend**: Long green/white candles with small lower wicks
 * - **Downtrend**: Long red/black candles with small upper wicks  
 * - **Consolidation**: Candles with both upper and lower wicks
 * - **Trend Change**: Color changes and wick patterns shift
 * 
 * @example
 * ```typescript
 * // Basic Heikin-Ashi conversion with default prefix
 * const heikinAshi = new HeikinAshiGenerator({
 *   in: ['open', 'high', 'low', 'close'],
 *   out: ['ha_open', 'ha_high', 'ha_low', 'ha_close']
 * })
 * 
 * // Custom prefix for multiple timeframes
 * const ha5min = new HeikinAshiGenerator({
 *   outputPrefix: 'ha5_' // Creates ha5_open, ha5_high, etc.
 * })
 * 
 * // For trend analysis systems
 * const smoothedBars = new HeikinAshiGenerator({
 *   outputPrefix: 'smooth_'
 * })
 * ```
 */
export class HeikinAshiGenerator extends BaseTransform<HeikinAshiParams> {
  private readonly symbolState = new Map<string, HeikinAshiState>()

  constructor(params: HeikinAshiParams = {}) {
    super('heikinAshi' as TransformType, 'Heikin-Ashi Generator', 'Converts OHLC to Heikin-Ashi bars', params)
  }

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let result = await data.next()
    while (!result.done) {
      const bar = result.value
      const symbol = bar.symbol
      const prefix = this.params.outputPrefix || 'ha_'

      // Calculate Heikin-Ashi Close (average price)
      const haClose = (bar.open + bar.high + bar.low + bar.close) / 4

      let haOpen: number
      let state = this.symbolState.get(symbol)

      if (!state) {
        // First bar - use regular open/close
        haOpen = (bar.open + bar.close) / 2
        state = {
          previousHAOpen: haOpen,
          previousHAClose: haClose
        }
        this.symbolState.set(symbol, state)
      } else {
        // Subsequent bars - use previous HA values
        haOpen = (state.previousHAOpen + state.previousHAClose) / 2
        
        // Update state for next bar
        state.previousHAOpen = haOpen
        state.previousHAClose = haClose
      }

      // Calculate Heikin-Ashi High and Low
      const haHigh = Math.max(bar.high, haOpen, haClose)
      const haLow = Math.min(bar.low, haOpen, haClose)

      // Create new bar with Heikin-Ashi values
      const haBar = {
        ...bar,
        [`${prefix}open`]: haOpen,
        [`${prefix}high`]: haHigh,
        [`${prefix}low`]: haLow,
        [`${prefix}close`]: haClose
      }

      yield haBar
      result = await data.next()
    }
  }

  getRequiredFields(): string[] {
    return ['open', 'high', 'low', 'close']
  }

  getOutputFields(): string[] {
    const prefix = this.params.outputPrefix || 'ha_'
    return [
      `${prefix}open`,
      `${prefix}high`,
      `${prefix}low`,
      `${prefix}close`
    ]
  }

  withParams(params: Partial<HeikinAshiParams>): HeikinAshiGenerator {
    return new HeikinAshiGenerator({ ...this.params, ...params })
  }

  validate(): void {
    // Skip base validation that checks for input columns
    // Bar generators use standard OHLCV fields instead
    
    if (this.params.outputPrefix !== undefined && typeof this.params.outputPrefix !== 'string') {
      throw new Error('outputPrefix must be a string')
    }
  }

  /**
   * Get current state for serialization/monitoring
   */
  getState(): Record<string, HeikinAshiState> {
    const state: Record<string, HeikinAshiState> = {}
    for (const [symbol, haState] of this.symbolState.entries()) {
      state[symbol] = { ...haState }
    }
    return state
  }

  /**
   * Restore state from serialized data
   */
  restoreState(state: Record<string, HeikinAshiState>): void {
    this.symbolState.clear()
    for (const [symbol, haState] of Object.entries(state)) {
      this.symbolState.set(symbol, haState)
    }
  }
}