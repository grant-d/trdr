import { BaseTransform } from '../base-transform'
import type { OhlcvDto } from '../../models'
import type { TransformType } from '../../interfaces'
import type { BaseTransformParams } from '../transform-params'

export interface HeikinAshiParams extends BaseTransformParams {
  outputPrefix?: string // Prefix for output fields (e.g., 'ha_' results in ha_open, ha_high, etc.)
}

interface HeikinAshiState {
  previousHAOpen: number
  previousHAClose: number
}

/**
 * Converts regular OHLC bars to Heikin-Ashi bars for smoother trend visualization
 * 
 * Heikin-Ashi formulas:
 * - HA_Close = (Open + High + Low + Close) / 4
 * - HA_Open = (Previous HA_Open + Previous HA_Close) / 2
 * - HA_High = Max(High, HA_Open, HA_Close)
 * - HA_Low = Min(Low, HA_Open, HA_Close)
 */
export class HeikinAshiGenerator extends BaseTransform<HeikinAshiParams> {
  private symbolState: Map<string, HeikinAshiState> = new Map()

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