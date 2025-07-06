import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'

export interface TickBarParams extends BarGeneratorParams {
  ticksPerBar: number
}

/**
 * Generates bars based on a fixed number of ticks/trades
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