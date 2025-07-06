import { BaseTransform } from '../base-transform'
import type { OhlcvDto } from '../../models'
import type { TransformType } from '../../interfaces'

export interface TechnicalIndicatorParams {
  in: string[]
  out: string[]
  period?: number
}

/**
 * Base class for technical indicators
 */
export abstract class BaseTechnicalIndicator extends BaseTransform<TechnicalIndicatorParams> {
  protected period: number

  constructor(params: TechnicalIndicatorParams, type: TransformType, name: string) {
    super(type as TransformType, name, `${name} with period ${params.period || 14}`, params)
    this.period = params.period || 14
  }

  getRequiredFields(): string[] {
    return this.params.in
  }

  getOutputFields(): string[] {
    return this.params.out
  }

  validate(): void {
    if (!this.params.in || this.params.in.length === 0) {
      throw new Error('Input fields (in) are required')
    }
    if (!this.params.out || this.params.out.length === 0) {
      throw new Error('Output fields (out) are required')
    }
    if (this.params.in.length !== this.params.out.length) {
      throw new Error('Number of input fields must match number of output fields')
    }
    if (this.period < 1) {
      throw new Error('Period must be at least 1')
    }
  }

  /**
   * Helper to get numeric value from data
   */
  protected getValue(data: OhlcvDto, field: string): number {
    if (!(field in data)) {
      throw new Error(`Field ${field} not found in data`)
    }
    const value = (data as any)[field]
    if (typeof value !== 'number') {
      throw new Error(`Field ${field} is not a number`)
    }
    return value
  }

  /**
   * Helper to set numeric value in data
   */
  protected setValue(data: OhlcvDto, field: string, value: number): void {
    (data as any)[field] = value
  }
}