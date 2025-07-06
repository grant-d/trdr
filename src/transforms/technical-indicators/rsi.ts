import { BaseTechnicalIndicator, type TechnicalIndicatorParams } from './base-indicator'
import type { OhlcvDto } from '../../models'

export interface RsiParams extends TechnicalIndicatorParams {
  period?: number
}

/**
 * Relative Strength Index (RSI) indicator
 * Measures momentum by comparing magnitude of recent gains to recent losses
 */
export class RelativeStrengthIndex extends BaseTechnicalIndicator {
  private readonly previousValues = new Map<string, number | undefined>()
  private readonly avgGains = new Map<string, number>()
  private readonly avgLosses = new Map<string, number>()
  private readonly initialCounts = new Map<string, number>()

  constructor(params: RsiParams) {
    super(params, 'rsi' as any, 'Relative Strength Index')
  }

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    // Initialize tracking for each field
    for (const field of this.params.in) {
      this.previousValues.set(field, undefined)
      this.avgGains.set(field, 0)
      this.avgLosses.set(field, 0)
      this.initialCounts.set(field, 0)
    }

    let result = await data.next()
    while (!result.done) {
      const item = result.value
      const transformedItem = { ...item }

      // Process each input/output field pair
      for (let i = 0; i < this.params.in.length; i++) {
        const inField = this.params.in[i]!
        const outField = this.params.out[i]!
        const value = this.getValue(item, inField)
        const previous = this.previousValues.get(inField)

        if (previous !== undefined) {
          const change = value - previous
          const gain = change > 0 ? change : 0
          const loss = change < 0 ? -change : 0
          
          const count = this.initialCounts.get(inField)!
          
          if (count < this.period) {
            // Initial period - accumulate gains and losses
            this.avgGains.set(inField, this.avgGains.get(inField)! + gain)
            this.avgLosses.set(inField, this.avgLosses.get(inField)! + loss)
            this.initialCounts.set(inField, count + 1)
            
            if (count + 1 === this.period) {
              // Calculate initial averages
              this.avgGains.set(inField, this.avgGains.get(inField)! / this.period)
              this.avgLosses.set(inField, this.avgLosses.get(inField)! / this.period)
            }
          } else {
            // Smooth the averages using Wilder's smoothing method
            const avgGain = this.avgGains.get(inField)!
            const avgLoss = this.avgLosses.get(inField)!
            
            this.avgGains.set(inField, (avgGain * (this.period - 1) + gain) / this.period)
            this.avgLosses.set(inField, (avgLoss * (this.period - 1) + loss) / this.period)
          }
          
          // Calculate RSI when we have enough data
          if (count >= this.period - 1) {
            const currentAvgGain = this.avgGains.get(inField)!
            const currentAvgLoss = this.avgLosses.get(inField)!
            
            let rsi: number
            if (currentAvgLoss === 0) {
              rsi = 100
            } else {
              const rs = currentAvgGain / currentAvgLoss
              rsi = 100 - (100 / (1 + rs))
            }
            
            this.setValue(transformedItem, outField, rsi)
          }
        }

        this.previousValues.set(inField, value)
      }

      yield transformedItem
      result = await data.next()
    }
  }

  withParams(params: Partial<RsiParams>): RelativeStrengthIndex {
    return new RelativeStrengthIndex({ ...this.params, ...params })
  }
}