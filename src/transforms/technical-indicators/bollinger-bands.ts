import { BaseTechnicalIndicator, type TechnicalIndicatorParams } from './base-indicator'
import type { OhlcvDto } from '../../models'

export interface BollingerBandsParams extends TechnicalIndicatorParams {
  period?: number
  stdDev?: number
}

/**
 * Bollinger Bands indicator
 * Creates upper and lower bands based on standard deviation from a moving average
 */
export class BollingerBands extends BaseTechnicalIndicator {
  private readonly buffers = new Map<string, number[]>()
  private readonly stdDev: number

  constructor(params: BollingerBandsParams) {
    super(params, 'bollinger' as any, 'Bollinger Bands')
    this.stdDev = params.stdDev || 2
  }

  validate(): void {
    if (!this.params.in || this.params.in.length === 0) {
      throw new Error('Input fields (in) are required')
    }
    if (!this.params.out || this.params.out.length !== 3) {
      throw new Error('Bollinger Bands requires exactly 3 output fields: [middle, upper, lower]')
    }
    if (this.period < 2) {
      throw new Error('Period must be at least 2')
    }
    if (this.stdDev <= 0) {
      throw new Error('Standard deviation multiplier must be positive')
    }
  }

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    // Initialize buffer for the input field
    const inField = this.params.in[0]!
    this.buffers.set(inField, [])

    let result = await data.next()
    while (!result.done) {
      const item = result.value
      const transformedItem = { ...item }
      const buffer = this.buffers.get(inField)!

      // Add current value to buffer
      const value = this.getValue(item, inField)
      buffer.push(value)

      // Keep buffer at max size
      if (buffer.length > this.period) {
        buffer.shift()
      }

      // Calculate bands when we have enough data
      if (buffer.length === this.period) {
        // Calculate SMA (middle band)
        const sum = buffer.reduce((a, b) => a + b, 0)
        const sma = sum / this.period

        // Calculate standard deviation
        const squaredDiffs = buffer.map(v => Math.pow(v - sma, 2))
        const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / this.period
        const stdDeviation = Math.sqrt(avgSquaredDiff)

        // Calculate bands
        const upperBand = sma + (this.stdDev * stdDeviation)
        const lowerBand = sma - (this.stdDev * stdDeviation)

        // Set output values
        this.setValue(transformedItem, this.params.out[0]!, sma)        // Middle band
        this.setValue(transformedItem, this.params.out[1]!, upperBand)  // Upper band
        this.setValue(transformedItem, this.params.out[2]!, lowerBand)  // Lower band
      }

      yield transformedItem
      result = await data.next()
    }
  }

  withParams(params: Partial<BollingerBandsParams>): BollingerBands {
    return new BollingerBands({ ...this.params, ...params })
  }
}