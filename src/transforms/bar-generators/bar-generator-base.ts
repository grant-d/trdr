import { BaseTransform } from '../base-transform'
import type { OhlcvDto } from '../../models'
import type { TransformType } from '../../interfaces'
import type { BaseTransformParams } from '../transform-params'

export interface BarGeneratorParams extends BaseTransformParams {
  resetDaily?: boolean
}

export interface BarState {
  currentBar: OhlcvDto
  tickCount?: number
  accumulatedVolume?: number
  accumulatedValue?: number
  tickImbalance?: number
  complete: boolean
}

/**
 * Base class for bar generators that aggregate tick/trade data into bars
 */
export abstract class BarGeneratorTransform<T extends BarGeneratorParams = BarGeneratorParams> extends BaseTransform<T> {
  protected symbolState = new Map<string, BarState>()
  protected lastResetTime = 0

  constructor(params: T, type: TransformType, name: string) {
    super(type, name, `${name} bar generator`, params)
  }

  /**
   * Check if the current bar is complete based on generator-specific logic
   */
  abstract isBarComplete(_symbol: string, tick: OhlcvDto, state: BarState): boolean

  /**
   * Create a new bar from the first tick
   */
  abstract createNewBar(symbol: string, tick: OhlcvDto): BarState

  /**
   * Update the current bar with new tick data
   */
  abstract updateBar(symbol: string, currentBar: OhlcvDto, tick: OhlcvDto, state: BarState): OhlcvDto

  /**
   * Reset any generator-specific state when starting a new bar
   */
  abstract resetState(state: BarState): void

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    const pendingNewBars = new Map<string, boolean>()
    
    let result = await data.next()
    while (!result.done) {
      const tick = result.value
      const symbol = tick.symbol

      // Check if we need to reset state (daily reset)
      if (this.params.resetDaily && this.shouldResetDaily(tick.timestamp)) {
        // Emit any incomplete bars before resetting
        const entries = Array.from(this.symbolState.entries())
        for (const [, state] of entries) {
          if (!state.complete) {
            yield state.currentBar
          }
        }
        
        this.symbolState.clear()
        pendingNewBars.clear()
        this.lastResetTime = tick.timestamp
      }

      // Check if we need to start a new bar for this symbol
      if (pendingNewBars.get(symbol)) {
        this.symbolState.set(symbol, this.createNewBar(symbol, tick))
        pendingNewBars.delete(symbol)
      } else if (!this.symbolState.has(symbol)) {
        // First tick for this symbol - create new bar
        this.symbolState.set(symbol, this.createNewBar(symbol, tick))
      } else {
        const state = this.symbolState.get(symbol)!
        
        // Update current bar with new tick
        state.currentBar = this.updateBar(symbol, state.currentBar, tick, state)

        // Check if bar is complete
        if (this.isBarComplete(symbol, tick, state)) {
          // Mark bar as complete
          state.complete = true
          
          // Emit completed bar
          yield state.currentBar

          // Mark that we need a new bar on the next tick
          pendingNewBars.set(symbol, true)
        }
      }

      result = await data.next()
    }

    // Emit any incomplete bars when stream ends
    const entries = Array.from(this.symbolState.entries())
    for (const [, state] of entries) {
      if (!state.complete) {
        yield state.currentBar
      }
    }
  }

  private shouldResetDaily(timestamp: number): boolean {
    const currentDate = new Date(timestamp)
    const lastResetDate = new Date(this.lastResetTime)
    
    return currentDate.getUTCDate() !== lastResetDate.getUTCDate() ||
           currentDate.getUTCMonth() !== lastResetDate.getUTCMonth() ||
           currentDate.getUTCFullYear() !== lastResetDate.getUTCFullYear()
  }

  getRequiredFields(): string[] {
    return ['open', 'high', 'low', 'close', 'volume']
  }

  getOutputFields(): string[] {
    return [] // Bar generators don't add new fields, they aggregate existing ones
  }

  /**
   * Get current state for serialization/monitoring
   */
  getState(): Record<string, BarState> {
    const state: Record<string, BarState> = {}
    const entries = Array.from(this.symbolState.entries())
    for (const [symbol, barState] of entries) {
      state[symbol] = { ...barState }
    }
    return state
  }

  /**
   * Restore state from serialized data
   */
  restoreState(state: Record<string, BarState>): void {
    this.symbolState.clear()
    for (const [symbol, barState] of Object.entries(state)) {
      this.symbolState.set(symbol, barState)
    }
  }
}