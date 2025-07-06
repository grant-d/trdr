import { BarGeneratorTransform, type BarGeneratorParams, type BarState } from './bar-generator-base'
import type { OhlcvDto } from '../../models'
import type { Transform } from '../../interfaces'

export interface LorentzianDistanceBarParams extends BarGeneratorParams {
  /** 
   * Scaling factor for time component in Lorentzian distance
   * @example 1.0 // Standard scaling factor
   * @minimum 0.1
   */
  cFactor: number
  
  /** 
   * Distance threshold for bar completion
   * @example 50.0 // Complete bar when distance exceeds 50
   * @minimum 10.0
   */
  threshold: number
}

interface LorentzianDistanceState extends BarState {
  /** Anchor point price for distance calculation */
  anchorPrice: number
  /** Anchor point volume for distance calculation */
  anchorVolume: number
  /** Anchor point time (bar index) for distance calculation */
  anchorTime: number
  /** Current time index */
  currentTime: number
}

/**
 * Lorentzian Distance Bar Generator
 * 
 * Calculates Lorentzian distance between current state and anchor point
 * in price-time-volume space using relativistic geometry.
 * 
 * Formula: d = √(c²Δt² - Δp² - Δv²) where c is a scaling factor
 * 
 * - Captures relativistic "warping" of market spacetime during high volatility
 * - More sensitive to rapid moves than Euclidean distance
 * - When distance is "space-like" (negative under square root), uses 
 *   Euclidean distance in price-volume space
 * - New bar forms when distance from anchor exceeds threshold
 * - Excels at capturing acceleration/deceleration dynamics
 * 
 * ## Algorithm
 * 1. Set anchor point at bar start (price, volume, time)
 * 2. For each new tick, calculate deltas from anchor
 * 3. Compute Lorentzian distance using relativistic formula
 * 4. If space-like interval (negative), use Euclidean fallback
 * 5. Create new bar when distance exceeds threshold
 * 
 * ## Use Cases
 * - High-frequency trading algorithms
 * - Volatility regime detection
 * - Acceleration/deceleration analysis
 * - Market microstructure studies
 * 
 * ## Advantages
 * - Captures temporal dynamics missed by traditional bars
 * - Sensitive to velocity and acceleration changes
 * - Relativistic approach handles extreme market conditions
 * - Adaptive to different timeframes and volatility regimes
 * 
 * @example
 * ```typescript
 * const generator = new LorentzianDistanceBarGenerator({
 *   cFactor: 1.0,        // Time scaling factor
 *   threshold: 50.0      // Distance threshold
 * })
 * ```
 */
export class LorentzianDistanceBarGenerator extends BarGeneratorTransform<LorentzianDistanceBarParams> {
  constructor(params: LorentzianDistanceBarParams) {
    super(params, 'lorentzianDistance', 'Lorentzian Distance Bar Generator')
    this.validate()
  }

  public validate(): void {
    super.validate()
    
    if (this.params.cFactor < 0.1) {
      throw new Error('Lorentzian Distance Bar Generator: cFactor must be at least 0.1')
    }
    
    if (this.params.threshold < 10.0) {
      throw new Error('Lorentzian Distance Bar Generator: threshold must be at least 10.0')
    }
  }

  isBarComplete(_symbol: string, tick: OhlcvDto, state: LorentzianDistanceState): boolean {
    // Calculate Lorentzian distance from anchor point
    const distance = this.calculateLorentzianDistance(tick, state)
    return distance >= this.params.threshold
  }

  createNewBar(symbol: string, tick: OhlcvDto): LorentzianDistanceState {
    return {
      currentBar: {
        timestamp: tick.timestamp,
        symbol,
        exchange: tick.exchange,
        open: tick.close,
        high: tick.high,
        low: tick.low,
        close: tick.close,
        volume: tick.volume
      },
      complete: false,
      anchorPrice: tick.close,
      anchorVolume: tick.volume,
      anchorTime: 0, // Will be set by the calling code
      currentTime: 0
    }
  }

  updateBar(_symbol: string, currentBar: OhlcvDto, tick: OhlcvDto, state: LorentzianDistanceState): OhlcvDto {
    // Increment time index (simulating bar_index progression)
    state.currentTime++
    
    // Update bar OHLCV
    return {
      timestamp: currentBar.timestamp,
      symbol: currentBar.symbol,
      exchange: currentBar.exchange,
      open: currentBar.open,
      high: Math.max(currentBar.high, tick.high),
      low: Math.min(currentBar.low, tick.low),
      close: tick.close,
      volume: currentBar.volume + tick.volume
    }
  }

  resetState(state: LorentzianDistanceState): void {
    // Reset anchor point to current state
    const currentBar = state.currentBar
    state.anchorPrice = currentBar.close
    state.anchorVolume = currentBar.volume
    state.anchorTime = state.currentTime
    state.complete = false
  }

  private calculateLorentzianDistance(tick: OhlcvDto, state: LorentzianDistanceState): number {
    // Calculate deltas from anchor point
    const deltaTime = state.currentTime - state.anchorTime
    
    // Normalize price delta as percentage
    const deltaPrice = state.anchorPrice !== 0 
      ? (tick.close - state.anchorPrice) / state.anchorPrice * 100 
      : 0
    
    // Log-normalize volume delta to handle large variations
    const deltaVolume = state.anchorVolume > 0 
      ? Math.log(tick.volume / state.anchorVolume + 1) * 10
      : 0
    
    // Calculate Lorentzian distance: d = √(c²Δt² - Δp² - Δv²)
    const lorentzianComponent = 
      this.params.cFactor * this.params.cFactor * deltaTime * deltaTime 
      - deltaPrice * deltaPrice 
      - deltaVolume * deltaVolume
    
    // If component is negative (space-like interval), use Euclidean distance in price-volume space
    if (lorentzianComponent > 0) {
      return Math.sqrt(lorentzianComponent)
    } else {
      // Euclidean fallback
      return Math.sqrt(deltaPrice * deltaPrice + deltaVolume * deltaVolume)
    }
  }

  public withParams(params: Partial<LorentzianDistanceBarParams>): Transform<LorentzianDistanceBarParams> {
    return new LorentzianDistanceBarGenerator({ ...this.params, ...params })
  }
}