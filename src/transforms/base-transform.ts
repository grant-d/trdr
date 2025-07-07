import type { BaseTransformParams, Transform, TransformType } from '../interfaces'
import type { DataBuffer } from '../utils'
import { DataSlice } from '../utils'

/**
 * Abstract base class for all transforms
 * Provides common functionality for transform type tracking
 */
export abstract class BaseTransform<T extends BaseTransformParams>
  implements Transform<T> {
  protected readonly inputSlice: DataSlice
  protected _outputBuffer?: DataBuffer
  protected _isReady: boolean
  private _batchNumber = 0

  public readonly type: TransformType
  public readonly name: string
  public readonly description: string
  public readonly params: T

  public get batchNumber(): number {
    return this._batchNumber
  }

  /**
   * Default implementation: most transforms are ready immediately
   * Override in subclasses that need buffer periods (like SMA)
   */
  public get isReady(): boolean {
    return this._isReady
  }

  /**
   * Get the buffer that should be passed to the next transform.
   * By default, returns the input buffer (in-place modification).
   * Override in reshaping transforms to return a separate output buffer.
   */
  public get outputBuffer(): DataBuffer {
    return this._outputBuffer || this.inputSlice.underlyingBuffer
  }

  protected constructor(
    type: TransformType,
    name: string,
    description: string,
    params: T,
    inputSlice: DataSlice
  ) {
    this.type = type
    this.name = name
    this.description = description
    this.params = params
    this.inputSlice = inputSlice
    this._isReady = false
  }

  /**
   * Called for each batch of rows to process
   * Concrete classes can use this to maintain their own output buffer or perform other actions
   * @param from Start index in the internal buffer to process
   * @param to End index in the internal buffer to process
   * @returns The range of rows that were actually updated { from, to }
   */
  protected abstract processBatch(from: number, to: number): { from: number; to: number };

  next(from: number, to: number): DataSlice {
    this._batchNumber++
    const result = this.processBatch(from, to)

    // Return a new slice with the range of rows that were actually updated
    return new DataSlice(this.outputBuffer, result.from, result.to)
  }
}
