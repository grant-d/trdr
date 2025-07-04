import type { BaseTransformParams, Transform, TransformCoefficients, TransformResult, TransformType } from '../interfaces'
import type { OhlcvDto } from '../models'

/**
 * Abstract base class for all transforms
 * Provides common functionality for transform type tracking and coefficient storage
 */
export abstract class BaseTransform<T extends BaseTransformParams = BaseTransformParams>
  implements Transform<T> {

  public readonly type: TransformType
  public readonly name: string
  public readonly description: string
  public readonly isReversible: boolean
  public readonly params: T

  protected coefficients: TransformCoefficients | null = null

  protected constructor(
    type: TransformType,
    name: string,
    description: string,
    params: T,
    isReversible = false,
  ) {
    this.type = type
    this.name = name
    this.description = description
    this.params = params
    this.isReversible = isReversible
  }

  /**
   * Apply the transformation to a stream of OHLCV data
   * Child classes must implement the actual transformation logic
   */
  public async apply(data: AsyncIterator<OhlcvDto>): Promise<TransformResult> {
    // Validate before applying
    this.validate()

    // Create the transformed data stream
    const transformedData = this.transform(data)

    // Return a result that provides live access to coefficients
    const self = this
    const result: TransformResult = {
      data: transformedData as AsyncIterator<OhlcvDto>,
      get coefficients() {
        return self.coefficients || undefined
      }
    }

    return result
  }

  /**
   * Abstract method that child classes must implement to perform the actual transformation
   */
  protected abstract transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto>

  /**
   * Default validation - can be overridden by child classes
   */
  public validate(): void {
    // Base validation - child classes can add more specific validation
    if (!this.params) {
      throw new Error(`Transform ${this.name} requires parameters`)
    }
  }

  /**
   * Get the list of fields this transform will add to the data
   * Must be implemented by child classes
   */
  public abstract getOutputFields(): string[]

  /**
   * Get the list of fields this transform requires to be present
   * Must be implemented by child classes
   */
  public abstract getRequiredFields(): string[]

  /**
   * Reverse the transformation using stored coefficients
   * Only implemented by reversible transforms
   */
  public reverse?(
    _data: AsyncIterator<OhlcvDto>,
    _coefficients: TransformCoefficients
  ): AsyncGenerator<OhlcvDto> {
    if (!this.isReversible) {
      throw new Error(`Transform ${this.name} is not reversible`)
    }
    throw new Error(`Reverse method not implemented for ${this.name}`)
  }

  /**
   * Create a copy of this transform with new parameters
   */
  public abstract withParams(params: Partial<T>): Transform<T>

  /**
   * Store coefficients for this transform
   */
  protected setCoefficients(symbol: string, values: Record<string, number>): void {
    this.coefficients = {
      type: this.type,
      timestamp: Date.now(),
      symbol,
      values
    }
  }

  /**
   * Get stored coefficients
   */
  protected getCoefficients(): TransformCoefficients | null {
    return this.coefficients
  }

  /**
   * Helper method to create an async generator from an array
   * Useful for testing
   */
  protected async* arrayToAsyncIterator<T>(array: T[]): AsyncGenerator<T> {
    for (const item of array) {
      yield item
    }
  }

  /**
   * Helper method to collect all items from an async iterator into an array
   * Should be used with caution on large datasets
   */
  protected async collectAsyncIterator<T>(iterator: AsyncIterator<T>): Promise<T[]> {
    const result: T[] = []
    let item = await iterator.next()

    while (!item.done) {
      result.push(item.value)
      item = await iterator.next()
    }

    return result
  }
}
