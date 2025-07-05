import type { BaseTransformParams, Transform, TransformResult, TransformType } from '../interfaces'
import type { OhlcvDto } from '../models'

/**
 * Abstract base class for all transforms
 * Provides common functionality for transform type tracking
 */
export abstract class BaseTransform<T extends BaseTransformParams = BaseTransformParams>
  implements Transform<T> {

  public readonly type: TransformType
  public readonly name: string
  public readonly description: string
  public readonly params: T

  protected constructor(
    type: TransformType,
    name: string,
    description: string,
    params: T
  ) {
    this.type = type
    this.name = name
    this.description = description
    this.params = params
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

    const result: TransformResult = {
      data: transformedData as AsyncIterator<OhlcvDto>
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
    
    // Validate input/output columns if specified
    if (this.params.in !== undefined || this.params.out !== undefined) {
      if (!this.params.in || !this.params.out) {
        throw new Error(`Transform ${this.name}: Both inputColumns and outputColumns must be specified together`)
      }
      
      if (this.params.in.length !== this.params.out.length) {
        throw new Error(`Transform ${this.name}: inputColumns and outputColumns must have the same length`)
      }
      
      if (this.params.in.length === 0) {
        throw new Error(`Transform ${this.name}: inputColumns cannot be empty`)
      }
      
      // Check for duplicate output columns (excluding null values)
      // Input columns can have duplicates (e.g., transforming same column multiple ways)
      const outputColumnsNonNull = this.params.out.filter((col): col is string => col !== null)
      const uniqueOutputColumns = new Set(outputColumnsNonNull)
      if (uniqueOutputColumns.size !== outputColumnsNonNull.length) {
        throw new Error(`Transform ${this.name}: Output columns must be unique. Found duplicates in: ${outputColumnsNonNull.join(', ')}`)
      }
      
      // Validate column names (null is allowed for dropping columns)
      const columnRegex = /^[a-zA-Z0-9_]{1,20}$/
      for (let i = 0; i < this.params.in.length; i++) {
        const inputCol = this.params.in[i]!
        const outputCol = this.params.out[i]
        
        if (!columnRegex.test(inputCol)) {
          throw new Error(`Transform ${this.name}: Invalid input column name '${inputCol}'. Column names must be alphanumeric with underscores and 1-20 characters long`)
        }
        
        // Output column can be null (to drop column) or a valid column name
        if (outputCol !== null && outputCol !== undefined && !columnRegex.test(outputCol)) {
          throw new Error(`Transform ${this.name}: Invalid output column name '${outputCol}'. Column names must be alphanumeric with underscores and 1-20 characters long, or null to drop the column`)
        }
      }
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
   * Create a copy of this transform with new parameters
   */
  public abstract withParams(params: Partial<T>): Transform<T>

  /**
   * Default implementation: most transforms are ready immediately
   * Override in subclasses that need buffer periods (like SMA)
   */
  public isReady(): boolean {
    return true
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

  /**
   * Get the input columns for this transform
   * @returns Array of input column names
   */
  protected getInputColumns(): string[] {
    if (this.params.in && this.params.in.length > 0) {
      return this.params.in
    }
    // Default to standard OHLCV columns
    return ['open', 'high', 'low', 'close', 'volume']
  }
  
  /**
   * Get the output columns for this transform
   * @returns Array of output column names (filtering out null values)
   */
  protected getOutputColumns(): string[] {
    if (this.params.out && this.params.out.length > 0) {
      // Filter out null values (columns marked for dropping)
      return this.params.out.filter((col): col is string => col !== null)
    } else {
      // Default behavior: overwrite the input columns
      return this.getInputColumns()
    }
  }
  
  /**
   * Get columns that should be dropped (set to null in outputColumns)
   * @returns Array of column names to drop
   */
  protected getDroppedColumns(): string[] {
    if (!this.params.out || !this.params.in) {
      return []
    }
    
    const droppedColumns: string[] = []
    for (let i = 0; i < this.params.out.length; i++) {
      if (this.params.out[i] === null) {
        droppedColumns.push(this.params.in[i]!)
      }
    }
    return droppedColumns
  }
  
}
