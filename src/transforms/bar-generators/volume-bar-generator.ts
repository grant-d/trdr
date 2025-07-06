import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'

export interface VolumeBarParams extends BarGeneratorParams {
  volumePerBar: number
}

/**
 * Generates bars based on accumulated volume thresholds
 */
export class VolumeBarGenerator extends BarGeneratorTransform<VolumeBarParams> {
  constructor(params: VolumeBarParams) {
    super(params, 'volumeBars' as any, 'Volume Bar Generator')
    
    if (!params.volumePerBar || params.volumePerBar <= 0) {
      throw new Error('volumePerBar must be greater than 0')
    }
  }

  isBarComplete(_symbol: string, _tick: OhlcvDto, state: BarState): boolean {
    // Check accumulated volume (already updated in updateBar)
    return (state.accumulatedVolume || 0) >= this.params.volumePerBar
  }

  createNewBar(_symbol: string, tick: OhlcvDto): BarState {
    return {
      currentBar: {
        ...tick,
        timestamp: tick.timestamp
      },
      accumulatedVolume: tick.volume,
      complete: false
    }
  }

  updateBar(_symbol: string, currentBar: OhlcvDto, tick: OhlcvDto, state: BarState): OhlcvDto {
    // Update accumulated volume
    state.accumulatedVolume = (state.accumulatedVolume || 0) + tick.volume

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
    state.accumulatedVolume = 0
    state.complete = false
  }

  withParams(params: Partial<VolumeBarParams>): VolumeBarGenerator {
    return new VolumeBarGenerator({ ...this.params, ...params })
  }

  validate(): void {
    // Skip base validation that checks for input columns
    // Bar generators use standard OHLCV fields instead
    
    if (this.params.volumePerBar <= 0) {
      throw new Error('volumePerBar must be greater than 0')
    }
  }
}