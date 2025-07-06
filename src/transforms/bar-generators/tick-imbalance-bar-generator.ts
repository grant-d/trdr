import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'

export interface TickImbalanceBarParams extends BarGeneratorParams {
  imbalanceThreshold: number
  useVolume?: boolean // Use volume imbalance instead of tick count imbalance
}

interface TickImbalanceState extends BarState {
  buyTicks: number
  sellTicks: number
  buyVolume: number
  sellVolume: number
  previousClose?: number
}

/**
 * Generates bars based on tick imbalance (buy ticks - sell ticks)
 * A tick is classified as buy if price > previous price, sell if price < previous price
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