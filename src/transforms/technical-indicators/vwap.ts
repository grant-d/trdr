import { BaseTechnicalIndicator, type TechnicalIndicatorParams } from './base-indicator'
import type { OhlcvDto } from '../../models'

export interface VwapParams extends TechnicalIndicatorParams {
  anchorPeriod?: 'session' | 'day' | 'week' | 'month' | 'rolling'
  rollingPeriod?: number // Number of milliseconds for rolling VWAP
}

/**
 * Volume Weighted Average Price (VWAP) indicator
 * Shows the average price weighted by volume
 */
export class VolumeWeightedAveragePrice extends BaseTechnicalIndicator {
  private cumulativeTypicalPriceVolume = 0
  private cumulativeVolume = 0
  private lastResetTimestamp = 0
  private readonly anchorPeriod: 'session' | 'day' | 'week' | 'month' | 'rolling'
  private readonly rollingPeriod: number

  constructor(params: VwapParams) {
    super({ ...params, in: [], out: params.out }, 'vwap', 'Volume Weighted Average Price')
    this.anchorPeriod = params.anchorPeriod || 'day'
    this.rollingPeriod = params.rollingPeriod || 3600000 // Default to 1 hour
  }

  validate(): void {
    if (!this.params.out || this.params.out.length !== 1) {
      throw new Error('VWAP requires exactly 1 output field')
    }
    const validPeriods = ['session', 'day', 'week', 'month', 'rolling']
    if (!validPeriods.includes(this.anchorPeriod)) {
      throw new Error(`Anchor period must be one of: ${validPeriods.join(', ')}`)
    }
    const vwapParams = this.params as VwapParams
    if (this.anchorPeriod === 'rolling' && vwapParams.rollingPeriod !== undefined && vwapParams.rollingPeriod < 1) {
      throw new Error('Rolling period must be positive')
    }
  }

  getRequiredFields(): string[] {
    return ['high', 'low', 'close', 'volume']
  }

  private shouldReset(currentTimestamp: number): boolean {
    if (this.lastResetTimestamp === 0) {
      return false
    }

    const current = new Date(currentTimestamp)
    const lastReset = new Date(this.lastResetTimestamp)

    switch (this.anchorPeriod) {
      case 'session':
        // Reset every trading session (using a simple 24-hour period)
        return currentTimestamp - this.lastResetTimestamp >= 24 * 60 * 60 * 1000
      
      case 'day':
        // Reset when day changes
        return current.getDate() !== lastReset.getDate() ||
               current.getMonth() !== lastReset.getMonth() ||
               current.getFullYear() !== lastReset.getFullYear()
      
      case 'week':
        // Reset when week changes (Monday is start of week)
        const currentWeek = this.getWeekNumber(current)
        const lastWeek = this.getWeekNumber(lastReset)
        return currentWeek !== lastWeek || current.getFullYear() !== lastReset.getFullYear()
      
      case 'month':
        // Reset when month changes
        return current.getMonth() !== lastReset.getMonth() ||
               current.getFullYear() !== lastReset.getFullYear()
      
      case 'rolling':
        // Reset when we've exceeded the rolling period
        return currentTimestamp - this.lastResetTimestamp > this.rollingPeriod
      
      default:
        return false
    }
  }

  private getWeekNumber(date: Date): number {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()))
    const dayNum = d.getUTCDay() || 7
    d.setUTCDate(d.getUTCDate() + 4 - dayNum)
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1))
    return Math.ceil((((d.getTime() - yearStart.getTime()) / 86400000) + 1) / 7)
  }

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let result = await data.next()
    while (!result.done) {
      const item = result.value
      const transformedItem = { ...item }

      // Check if we need to reset accumulation
      if (this.shouldReset(item.timestamp)) {
        this.cumulativeTypicalPriceVolume = 0
        this.cumulativeVolume = 0
        this.lastResetTimestamp = item.timestamp
      }

      if (this.lastResetTimestamp === 0) {
        this.lastResetTimestamp = item.timestamp
      }

      // Calculate typical price (HLC/3)
      const typicalPrice = (item.high + item.low + item.close) / 3

      // Update cumulative values
      this.cumulativeTypicalPriceVolume += typicalPrice * item.volume
      this.cumulativeVolume += item.volume

      // Calculate VWAP
      if (this.cumulativeVolume > 0) {
        const vwap = this.cumulativeTypicalPriceVolume / this.cumulativeVolume
        this.setValue(transformedItem, this.params.out[0]!, vwap)
      }

      yield transformedItem
      result = await data.next()
    }
  }

  withParams(params: Partial<VwapParams>): VolumeWeightedAveragePrice {
    return new VolumeWeightedAveragePrice({ ...this.params, ...params })
  }
}