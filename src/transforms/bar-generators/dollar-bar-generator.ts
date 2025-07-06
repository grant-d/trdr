import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'

export interface DollarBarParams extends BarGeneratorParams {
  /** 
   * Threshold dollar value that triggers bar completion
   * @example 10000 // Complete bars when $10,000 worth of volume is reached
   */
  dollarValuePerBar: number
  
  /** 
   * Price field to use for dollar value calculation 
   * @default 'close'
   * - 'close': Use closing price
   * - 'vwap': Use volume-weighted average price if available, fallback to typical price
   * - 'typical': Use typical price (high + low + close) / 3
   */
  priceField?: 'close' | 'vwap' | 'typical'
}

/**
 * Dollar Bar Generator
 * 
 * Generates bars based on accumulated dollar value (price * volume). This technique is used
 * in high-frequency trading and market microstructure analysis to create more uniform bars
 * in terms of economic significance rather than just time or tick count.
 * 
 * ## Algorithm
 * 
 * 1. **Dollar Value Calculation**: For each tick, calculate dollar value = price × volume
 * 2. **Accumulation**: Add the dollar value to the current bar's accumulated total
 * 3. **Bar Completion**: When accumulated value >= dollarValuePerBar threshold, complete the bar
 * 4. **Bar Reset**: Start a new bar with the next tick
 * 
 * ## Use Cases
 * 
 * - **Volume-Weighted Analysis**: Bars represent equal economic activity rather than time
 * - **Market Impact Studies**: Each bar represents similar market impact in dollar terms
 * - **Liquidity Analysis**: Bars are normalized by actual capital flow
 * - **High-Frequency Trading**: More consistent bar sizes for algorithmic strategies
 * 
 * ## Advantages over Time-Based Bars
 * 
 * - **Economic Significance**: Each bar represents similar economic activity
 * - **Market Microstructure**: Better captures true market activity patterns
 * - **Volatility Normalization**: High volatility periods get more bars naturally
 * - **Liquidity Awareness**: Accounts for both price movement and volume
 * 
 * @example
 * ```typescript
 * // Create bars when $50,000 of volume is traded
 * const dollarBars = new DollarBarGenerator({
 *   dollarValuePerBar: 50000,
 *   priceField: 'close' // Use closing price for calculations
 * })
 * 
 * // For high-priced stocks, you might need higher thresholds
 * const expensiveStockBars = new DollarBarGenerator({
 *   dollarValuePerBar: 1000000, // $1M threshold
 *   priceField: 'vwap' // Use VWAP if available
 * })
 * ```
 */
export class DollarBarGenerator extends BarGeneratorTransform<DollarBarParams> {
  constructor(params: DollarBarParams) {
    super(params, 'dollarBars' as any, 'Dollar Bar Generator')
    
    if (!params.dollarValuePerBar || params.dollarValuePerBar <= 0) {
      throw new Error('dollarValuePerBar must be greater than 0')
    }
  }

  private getPrice(tick: OhlcvDto): number {
    switch (this.params.priceField || 'close') {
      case 'vwap':
        // If VWAP is available, use it; otherwise calculate typical price
        return (tick as any).vwap || (tick.high + tick.low + tick.close) / 3
      case 'typical':
        return (tick.high + tick.low + tick.close) / 3
      case 'close':
      default:
        return tick.close
    }
  }

  isBarComplete(_symbol: string, _tick: OhlcvDto, state: BarState): boolean {
    // Check if accumulated value has reached threshold
    // Note: updateBar has already added the current tick's value to state.accumulatedValue
    return (state.accumulatedValue || 0) >= this.params.dollarValuePerBar
  }

  createNewBar(_symbol: string, tick: OhlcvDto): BarState {
    const price = this.getPrice(tick)
    const tickDollarValue = price * tick.volume

    return {
      currentBar: {
        ...tick,
        timestamp: tick.timestamp
      },
      accumulatedValue: tickDollarValue,
      complete: false
    }
  }

  updateBar(_symbol: string, currentBar: OhlcvDto, tick: OhlcvDto, state: BarState): OhlcvDto {
    const price = this.getPrice(tick)
    const tickDollarValue = price * tick.volume
    
    // Update accumulated dollar value
    state.accumulatedValue = (state.accumulatedValue || 0) + tickDollarValue

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
    // Reset accumulated value for next bar
    state.accumulatedValue = 0
    state.complete = false
  }

  withParams(params: Partial<DollarBarParams>): DollarBarGenerator {
    return new DollarBarGenerator({ ...this.params, ...params })
  }

  validate(): void {
    // Skip base validation that checks for input columns
    // Bar generators use standard OHLCV fields instead
    
    if (this.params.dollarValuePerBar <= 0) {
      throw new Error('dollarValuePerBar must be greater than 0')
    }
    
    const validPriceFields = ['close', 'vwap', 'typical']
    if (this.params.priceField && !validPriceFields.includes(this.params.priceField)) {
      throw new Error(`priceField must be one of: ${validPriceFields.join(', ')}`)
    }
  }
}