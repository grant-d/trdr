import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'

export interface DollarBarParams extends BarGeneratorParams {
  dollarValuePerBar: number
  priceField?: 'close' | 'vwap' | 'typical'
}

/**
 * Generates bars based on accumulated dollar value (price * volume)
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

  isBarComplete(_symbol: string, tick: OhlcvDto, state: BarState): boolean {
    const price = this.getPrice(tick)
    const tickDollarValue = price * tick.volume
    const totalDollarValue = (state.accumulatedValue || 0) + tickDollarValue
    return totalDollarValue >= this.params.dollarValuePerBar
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