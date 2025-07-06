import { BaseTechnicalIndicator, type TechnicalIndicatorParams } from './base-indicator'
import type { OhlcvDto } from '../../models'

export interface MovingAverageParams extends TechnicalIndicatorParams {
  period: number
}

/**
 * Simple Moving Average (SMA) indicator
 */
export class SimpleMovingAverage extends BaseTechnicalIndicator {
  private readonly buffers = new Map<string, number[]>()

  constructor(params: MovingAverageParams) {
    super(params, 'sma' as any, 'Simple Moving Average')
  }

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    // Initialize buffers for each input field
    for (const field of this.params.in) {
      this.buffers.set(field, [])
    }

    let result = await data.next()
    while (!result.done) {
      const item = result.value
      const transformedItem = { ...item }

      // Process each input/output field pair
      for (let i = 0; i < this.params.in.length; i++) {
        const inField = this.params.in[i]!
        const outField = this.params.out[i]!
        const buffer = this.buffers.get(inField)!

        // Add current value to buffer
        const value = this.getValue(item, inField)
        buffer.push(value)

        // Keep buffer at max size
        if (buffer.length > this.period) {
          buffer.shift()
        }

        // Calculate SMA
        if (buffer.length === this.period) {
          const sum = buffer.reduce((a, b) => a + b, 0)
          const sma = sum / this.period
          this.setValue(transformedItem, outField, sma)
        }
      }

      yield transformedItem
      result = await data.next()
    }
  }

  withParams(params: Partial<MovingAverageParams>): SimpleMovingAverage {
    return new SimpleMovingAverage({ ...this.params, ...params, period: params.period ?? this.period })
  }
}

/**
 * Exponential Moving Average (EMA) indicator
 */
export class ExponentialMovingAverage extends BaseTechnicalIndicator {
  private readonly emas = new Map<string, number | undefined>()
  private readonly multiplier: number

  constructor(params: MovingAverageParams) {
    super(params, 'ema' as any, 'Exponential Moving Average')
    this.multiplier = 2 / (this.period + 1)
  }

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    // Initialize EMAs for each field
    for (const field of this.params.in) {
      this.emas.set(field, undefined)
    }

    let count = 0
    const smaBuffers = new Map<string, number[]>()
    for (const field of this.params.in) {
      smaBuffers.set(field, [])
    }

    let result = await data.next()
    while (!result.done) {
      const item = result.value
      const transformedItem = { ...item }
      count++

      // Process each input/output field pair
      for (let i = 0; i < this.params.in.length; i++) {
        const inField = this.params.in[i]!
        const outField = this.params.out[i]!
        const value = this.getValue(item, inField)

        let ema = this.emas.get(inField)

        if (ema === undefined) {
          // Use SMA for initial EMA value
          const buffer = smaBuffers.get(inField)!
          buffer.push(value)

          if (buffer.length === this.period) {
            const sum = buffer.reduce((a, b) => a + b, 0)
            ema = sum / this.period
            this.emas.set(inField, ema)
            this.setValue(transformedItem, outField, ema)
          }
        } else {
          // Calculate EMA: EMA = (Close - EMA(previous)) * multiplier + EMA(previous)
          ema = (value - ema) * this.multiplier + ema
          this.emas.set(inField, ema)
          this.setValue(transformedItem, outField, ema)
        }
      }

      yield transformedItem
      result = await data.next()
    }
  }

  withParams(params: Partial<MovingAverageParams>): ExponentialMovingAverage {
    return new ExponentialMovingAverage({ ...this.params, ...params, period: params.period ?? this.period })
  }
}