import { BaseTechnicalIndicator, type TechnicalIndicatorParams } from './base-indicator'
import type { OhlcvDto } from '../../models'

export interface AtrParams extends TechnicalIndicatorParams {
  period?: number
}

/**
 * Average True Range (ATR) indicator
 * Measures market volatility by analyzing the entire range of an asset price for a period
 */
export class AverageTrueRange extends BaseTechnicalIndicator {
  private previousClose: number | undefined
  private atr: number | undefined
  private readonly trValues: number[] = []

  constructor(params: AtrParams) {
    super({ ...params, in: [], out: params.out }, 'atr' as any, 'Average True Range')
  }

  validate(): void {
    if (!this.params.out || this.params.out.length !== 1) {
      throw new Error('ATR requires exactly 1 output field')
    }
    if (this.period < 1) {
      throw new Error('Period must be at least 1')
    }
  }

  getRequiredFields(): string[] {
    return ['high', 'low', 'close']
  }

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let result = await data.next()
    while (!result.done) {
      const item = result.value
      const transformedItem = { ...item }

      // Calculate True Range
      let tr: number
      if (this.previousClose === undefined) {
        // First data point: TR = High - Low
        tr = item.high - item.low
      } else {
        // TR = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
        const highLow = item.high - item.low
        const highPrevClose = Math.abs(item.high - this.previousClose)
        const lowPrevClose = Math.abs(item.low - this.previousClose)
        tr = Math.max(highLow, highPrevClose, lowPrevClose)
      }

      this.trValues.push(tr)

      // Calculate ATR
      if (this.trValues.length < this.period) {
        // Not enough data yet
      } else if (this.trValues.length === this.period) {
        // First ATR: Simple average of TR values
        const sum = this.trValues.reduce((a, b) => a + b, 0)
        this.atr = sum / this.period
        this.setValue(transformedItem, this.params.out[0]!, this.atr)
      } else {
        // Subsequent ATR: Use Wilder's smoothing method
        // ATR = ((Previous ATR * (period - 1)) + Current TR) / period
        this.atr = ((this.atr! * (this.period - 1)) + tr) / this.period
        this.setValue(transformedItem, this.params.out[0]!, this.atr)
        // Remove oldest TR value
        this.trValues.shift()
      }

      this.previousClose = item.close

      yield transformedItem
      result = await data.next()
    }
  }

  withParams(params: Partial<AtrParams>): AverageTrueRange {
    return new AverageTrueRange({ ...this.params, ...params })
  }
}