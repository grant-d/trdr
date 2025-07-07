import type { BaseTransformParams, Transform } from '../interfaces'
import type { DataBuffer } from '../utils'
import { DataSlice } from '../utils'

/**
 * Parameters for the transform pipeline
 */
export interface TransformPipelineParams extends BaseTransformParams {
  /** Transforms to apply in sequence */
  transforms: Transform[];
}

/**
 * A pipeline that chains multiple transforms together
 * Applies transforms sequentially
 */
export class TransformPipeline implements Transform<TransformPipelineParams> {
  public readonly type = 'pipeline' as const
  public readonly name: string
  public readonly description: string
  public readonly params: TransformPipelineParams
  public readonly batchNumber: number = 0

  private readonly transforms: Transform[]

  constructor(params: TransformPipelineParams) {
    this.params = params
    this.transforms = [...params.transforms]
    this.name = 'Transform Pipeline'
    this.description =
      params.description || `Pipeline of ${this.transforms.length} transforms`
  }

  /**
   * Get the output buffer from the last transform in the pipeline
   */
  public get outputBuffer(): DataBuffer {
    if (this.transforms.length === 0) {
      throw new Error('Pipeline has no transforms')
    }
    return this.transforms[this.transforms.length - 1]!.outputBuffer
  }

  /**
   * Process a batch through all transforms in the pipeline
   */
  public next(from: number, to: number): DataSlice {
    let currentSlice: DataSlice | undefined
    for (const transform of this.transforms) {
      currentSlice = transform.next(from, to)
    }
    // Return the last transform's output slice or empty slice
    return currentSlice || new DataSlice(this.outputBuffer, from, to)
  }

  /**
   * Add a transform to the end of the pipeline
   */
  public add(transform: Transform): TransformPipeline {
    const newTransforms = [...this.transforms, transform]
    return new TransformPipeline({
      ...this.params,
      transforms: newTransforms
    })
  }

  /**
   * Remove a transform at the specified index
   */
  public remove(index: number): TransformPipeline {
    if (index < 0 || index >= this.transforms.length) {
      throw new Error(
        `Invalid index ${index}. Pipeline has ${this.transforms.length} transforms`
      )
    }

    const newTransforms = [...this.transforms]
    newTransforms.splice(index, 1)

    return new TransformPipeline({
      ...this.params,
      transforms: newTransforms
    })
  }

  /**
   * Insert a transform at the specified index
   */
  public insert(index: number, transform: Transform): TransformPipeline {
    if (index < 0 || index > this.transforms.length) {
      throw new Error(
        `Invalid index ${index}. Valid range is 0 to ${this.transforms.length}`
      )
    }

    const newTransforms = [...this.transforms]
    newTransforms.splice(index, 0, transform)

    return new TransformPipeline({
      ...this.params,
      transforms: newTransforms
    })
  }

  /**
   * Move a transform from one position to another
   */
  public move(fromIndex: number, toIndex: number): TransformPipeline {
    if (fromIndex < 0 || fromIndex >= this.transforms.length) {
      throw new Error(`Invalid fromIndex ${fromIndex}`)
    }
    if (toIndex < 0 || toIndex >= this.transforms.length) {
      throw new Error(`Invalid toIndex ${toIndex}`)
    }

    const newTransforms = [...this.transforms]
    const [transform] = newTransforms.splice(fromIndex, 1)
    newTransforms.splice(toIndex, 0, transform!)

    return new TransformPipeline({
      ...this.params,
      transforms: newTransforms
    })
  }

  /**
   * Get the transforms in the pipeline
   */
  public getTransforms(): readonly Transform[] {
    return [...this.transforms]
  }

  /**
   * Clear all transforms from the pipeline
   */
  public clear(): TransformPipeline {
    return new TransformPipeline({
      ...this.params,
      transforms: []
    })
  }

  /**
   * Pipeline is ready when all transforms in it are ready
   */
  public get isReady(): boolean {
    return this.transforms.every((transform) => transform.isReady)
  }

  /*
  // These methods are from the old streaming interface and need to be removed/updated
  
  public validate(): void {
    if (this.transforms.length === 0) {
      return // Empty pipeline is valid
    }

    // Validate each transform
    for (let i = 0; i < this.transforms.length; i++) {
      const transform = this.transforms[i]!
      try {
        transform.validate()
      } catch (error) {
        throw new Error(
          `Transform ${i} (${transform.name}) validation failed: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`
        )
      }
    }

    // Validate field dependencies between transforms
    for (let i = 1; i < this.transforms.length; i++) {
      const currentTransform = this.transforms[i]!
      const requiredFields = currentTransform.getRequiredFields()
      
      if (requiredFields.length > 0) {
        // For now, we'll skip this validation as it would require
        // tracking all available fields through the pipeline
        // This could be enhanced in the future
      }
    }
  }

  public getOutputFields(): string[] {
    // Aggregate all output fields from all transforms
    const allFields = new Set<string>()
    
    for (const transform of this.transforms) {
      const fields = transform.getOutputFields()
      for (const field of fields) {
        allFields.add(field)
      }
    }
    
    return Array.from(allFields)
  }

  public getRequiredFields(): string[] {
    if (this.transforms.length === 0) {
      return []
    }
    
    // The required fields are those needed by the first transform
    return this.transforms[0]!.getRequiredFields()
  }

  public withParams(params: Partial<TransformPipelineParams>): Transform<TransformPipelineParams> {
    return new TransformPipeline({ ...this.params, ...params })
  }
  */
}

/**
 * Helper function to create a pipeline from an array of transforms
 */
export function createPipeline(
  transforms: Transform[],
  description?: string
): TransformPipeline {
  return new TransformPipeline({ transforms, description })
}

// TODO: TransformPipeline needs to be updated to implement the Transform interface properly
// For now, it's just a container for transforms
