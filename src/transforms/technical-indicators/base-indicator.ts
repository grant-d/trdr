import type { BaseTransformParams, TransformType } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

export interface TechnicalIndicatorParams extends BaseTransformParams {
  // in: string[]
  // out: string[]
  // period?: number
}

/**
 * Base class for technical indicators
 */
export abstract class BaseTechnicalIndicator extends BaseTransform<BaseTransformParams> {
  // protected period: number

  protected constructor(
    params: BaseTransformParams,
    type: TransformType,
    name: string,
    description: string,
    inputSlice: DataSlice
  ) {
    super(type, name, description, params, inputSlice)
    // this.period = params.period ?? 14
  }

  // validate(): void {
  //   super.validate()

  //   if (!this.params.in || this.params.in.length === 0) {
  //     throw new Error('Input fields (in) are required')
  //   }
  //   if (!this.params.out || this.params.out.length === 0) {
  //     throw new Error('Output fields (out) are required')
  //   }
  //   if (this.params.in.length !== this.params.out.length) {
  //     throw new Error('Number of input fields must match number of output fields')
  //   }
  //   if (this.period < 1) {
  //     throw new Error('Period must be at least 1')
  //   }
  // }
}
