import type { BaseTransformParams, Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import { BaseTransform } from './base-transform'

/**
 * Parameters for the mapping transform
 */
export interface MappingTransformParams extends BaseTransformParams {
  /** Input field names to copy from */
  in: string[]
  /** Output field names to copy to */
  out: string[]
}

/**
 * A simple mapping transform that copies values from input fields to output fields
 * Useful for creating column aliases or duplicating data without any processing
 * 
 * Example usage:
 * - Copy close to adj-close: { in: ['close'], out: ['adj-close'] }
 * - Multiple mappings: { in: ['close', 'volume'], out: ['adj-close', 'adj-volume'] }
 */
export class MappingTransform extends BaseTransform<MappingTransformParams> {
  constructor(params: MappingTransformParams) {
    super(
      'map',
      'Field Mapping Transform',
      `Maps ${params.in?.length || 0} field(s) to new names`,
      params
    )
  }

  public validate(): void {
    super.validate()

    if (!this.params.in || this.params.in.length === 0) {
      throw new Error('Input fields (in) are required')
    }

    if (!this.params.out || this.params.out.length === 0) {
      throw new Error('Output fields (out) are required')
    }

    if (this.params.in.length !== this.params.out.length) {
      throw new Error('Number of input fields must match number of output fields')
    }
  }

  public getOutputFields(): string[] {
    return this.getOutputColumns()
  }

  public getRequiredFields(): string[] {
    return this.getInputColumns()
  }

  public isReady(): boolean {
    // Mapping transform is always ready - no aggregation or windowing
    return true
  }

  protected async* transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let item = await data.next()
    
    while (!item.done) {
      const current = item.value
      const result = { ...current }

      // Map each input field to its corresponding output field
      for (let i = 0; i < this.params.in.length; i++) {
        const inputField = this.params.in[i]!
        const outputField = this.params.out[i]!
        
        // Copy the value from input field to output field
        if (inputField in current) {
          (result as any)[outputField] = (current as any)[inputField]
        }
      }

      yield result
      item = await data.next()
    }
  }

  public withParams(params: Partial<MappingTransformParams>): Transform<MappingTransformParams> {
    return new MappingTransform({ ...this.params, ...params })
  }
}