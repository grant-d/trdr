import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'

export interface VolumeBarParams extends BarGeneratorParams {
  /** 
   * Volume threshold that triggers bar completion
   * @example 10000 // Complete bars when 10,000 shares/units have been traded
   * @minimum 0 (exclusive)
   */
  volumePerBar: number
}

/**
 * Volume Bar Generator
 * 
 * Generates bars based on accumulated volume rather than time or tick count. This creates
 * bars that represent equal liquidity consumption, making them particularly useful for
 * analyzing market microstructure and institutional trading patterns.
 * 
 * ## Algorithm
 * 
 * 1. **Volume Accumulation**: Sum the volume of each incoming tick
 * 2. **Threshold Check**: When accumulated volume >= volumePerBar, complete the bar
 * 3. **Bar Reset**: Start a new bar with volume accumulation reset
 * 
 * ## Use Cases
 * 
 * - **Liquidity Analysis**: Each bar represents equal liquidity consumption
 * - **Institutional Trading**: Detect large block trading patterns
 * - **Market Impact Studies**: Analyze price impact per unit of volume
 * - **Volume Profile Analysis**: Understand volume distribution patterns
 * - **Execution Algorithms**: TWAP/VWAP strategy development
 * 
 * ## Advantages over Time-Based Bars
 * 
 * - **Liquidity-Normalized**: Each bar represents similar liquidity impact
 * - **Market Structure**: Better reflects actual market participation
 * - **Volume Clustering**: Naturally groups periods of similar volume activity
 * - **Institutional Focus**: Highlights periods of large institutional activity
 * 
 * ## Considerations
 * 
 * - **Variable Time Spans**: High-volume periods create faster bars
 * - **Tick Count Variation**: Each bar may contain different numbers of trades
 * - **Price Range Variation**: Volume doesn't guarantee price movement
 * - **Market Regime Sensitivity**: May need adjustment for different market conditions
 * 
 * @example
 * ```typescript
 * // Create bars when 50,000 shares are traded
 * const volumeBars = new VolumeBarGenerator({
 *   volumePerBar: 50000
 * })
 * 
 * // For crypto or high-volume assets
 * const cryptoBars = new VolumeBarGenerator({
 *   volumePerBar: 1000000 // 1M units
 * })
 * 
 * // For low-volume assets  
 * const smallCapBars = new VolumeBarGenerator({
 *   volumePerBar: 5000 // 5K shares
 * })
 * ```
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