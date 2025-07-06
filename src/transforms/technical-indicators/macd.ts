import { BaseTechnicalIndicator, type TechnicalIndicatorParams } from './base-indicator'
import type { OhlcvDto } from '../../models'

export interface MacdParams extends TechnicalIndicatorParams {
  fastPeriod?: number
  slowPeriod?: number
  signalPeriod?: number
}

/**
 * MACD (Moving Average Convergence Divergence) indicator
 * Shows the relationship between two moving averages of prices
 */
export class Macd extends BaseTechnicalIndicator {
  private readonly fastPeriod: number
  private readonly slowPeriod: number
  private readonly signalPeriod: number
  private readonly fastEmas = new Map<string, number | undefined>()
  private readonly slowEmas = new Map<string, number | undefined>()
  private readonly signalEmas = new Map<string, number | undefined>()
  private readonly fastMultiplier: number
  private readonly slowMultiplier: number
  private readonly signalMultiplier: number
  private readonly fastBuffers = new Map<string, number[]>()
  private readonly slowBuffers = new Map<string, number[]>()
  private readonly macdBuffers = new Map<string, number[]>()

  constructor(params: MacdParams) {
    super(params, 'macd' as any, 'MACD')
    this.fastPeriod = params.fastPeriod || 12
    this.slowPeriod = params.slowPeriod || 26
    this.signalPeriod = params.signalPeriod || 9
    this.fastMultiplier = 2 / (this.fastPeriod + 1)
    this.slowMultiplier = 2 / (this.slowPeriod + 1)
    this.signalMultiplier = 2 / (this.signalPeriod + 1)
  }

  validate(): void {
    if (!this.params.in || this.params.in.length === 0) {
      throw new Error('Input fields (in) are required')
    }
    if (!this.params.out || this.params.out.length !== 3) {
      throw new Error('MACD requires exactly 3 output fields: [macd, signal, histogram]')
    }
    if (this.fastPeriod >= this.slowPeriod) {
      throw new Error('Fast period must be less than slow period')
    }
    if (this.fastPeriod < 1 || this.slowPeriod < 1 || this.signalPeriod < 1) {
      throw new Error('All periods must be at least 1')
    }
  }

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    // Initialize for each input field
    const inField = this.params.in[0]!
    this.fastEmas.set(inField, undefined)
    this.slowEmas.set(inField, undefined)
    this.signalEmas.set(inField, undefined)
    this.fastBuffers.set(inField, [])
    this.slowBuffers.set(inField, [])
    this.macdBuffers.set(inField, [])

    let result = await data.next()
    while (!result.done) {
      const item = result.value
      const transformedItem = { ...item }
      const value = this.getValue(item, inField)

      // Update fast EMA
      let fastEma = this.fastEmas.get(inField)
      const fastBuffer = this.fastBuffers.get(inField)!
      
      if (fastEma === undefined) {
        fastBuffer.push(value)
        if (fastBuffer.length === this.fastPeriod) {
          const sum = fastBuffer.reduce((a, b) => a + b, 0)
          fastEma = sum / this.fastPeriod
          this.fastEmas.set(inField, fastEma)
        }
      } else {
        fastEma = (value - fastEma) * this.fastMultiplier + fastEma
        this.fastEmas.set(inField, fastEma)
      }

      // Update slow EMA
      let slowEma = this.slowEmas.get(inField)
      const slowBuffer = this.slowBuffers.get(inField)!
      
      if (slowEma === undefined) {
        slowBuffer.push(value)
        if (slowBuffer.length === this.slowPeriod) {
          const sum = slowBuffer.reduce((a, b) => a + b, 0)
          slowEma = sum / this.slowPeriod
          this.slowEmas.set(inField, slowEma)
        }
      } else {
        slowEma = (value - slowEma) * this.slowMultiplier + slowEma
        this.slowEmas.set(inField, slowEma)
      }

      // Calculate MACD line when both EMAs are ready
      if (fastEma !== undefined && slowEma !== undefined) {
        const macdValue = fastEma - slowEma
        this.setValue(transformedItem, this.params.out[0]!, macdValue)
        
        // Update signal line
        const macdBuffer = this.macdBuffers.get(inField)!
        macdBuffer.push(macdValue)
        
        let signalEma = this.signalEmas.get(inField)
        if (signalEma === undefined && macdBuffer.length >= this.signalPeriod) {
          // Initialize signal EMA with SMA
          const sum = macdBuffer.slice(0, this.signalPeriod).reduce((a, b) => a + b, 0)
          signalEma = sum / this.signalPeriod
          this.signalEmas.set(inField, signalEma)
        } else if (signalEma !== undefined) {
          // Update signal EMA
          signalEma = (macdValue - signalEma) * this.signalMultiplier + signalEma
          this.signalEmas.set(inField, signalEma)
        }
        
        if (signalEma !== undefined) {
          this.setValue(transformedItem, this.params.out[1]!, signalEma)
          
          // Calculate histogram (MACD - Signal)
          const histogram = macdValue - signalEma
          this.setValue(transformedItem, this.params.out[2]!, histogram)
        }
      }

      yield transformedItem
      result = await data.next()
    }
  }

  withParams(params: Partial<MacdParams>): Macd {
    return new Macd({ ...this.params, ...params })
  }
}