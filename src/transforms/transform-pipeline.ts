import type { BaseTransformParams, Transform, TransformResult } from '../interfaces'
import type { OhlcvDto } from '../models'

/**
 * Parameters for the transform pipeline
 */
export interface TransformPipelineParams extends BaseTransformParams {
  /** Transforms to apply in sequence */
  transforms: Transform[]
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

  private readonly transforms: Transform[]

  constructor(params: TransformPipelineParams) {
    this.params = params
    this.transforms = [...params.transforms]
    this.name = 'Transform Pipeline'
    this.description = params.description || `Pipeline of ${this.transforms.length} transforms`
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
      throw new Error(`Invalid index ${index}. Pipeline has ${this.transforms.length} transforms`)
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
      throw new Error(`Invalid index ${index}. Valid range is 0 to ${this.transforms.length}`)
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
  public isReady(): boolean {
    return this.transforms.every(transform => transform.isReady())
  }

  public async apply(data: AsyncIterator<OhlcvDto>): Promise<TransformResult> {
    if (this.transforms.length === 0) {
      return { data }
    }

    return {
      data: this.createReadinessAwareStream(data)
    }
  }

  /**
   * Creates a streaming pipeline that chains transforms together
   * Each transform only processes data when it and all its ancestors are ready
   */
  private async* createReadinessAwareStream(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    // Create a chain of transform streams with readiness awareness
    let currentStream: AsyncGenerator<OhlcvDto> = this.streamFromIterator(data)
    
    for (let i = 0; i < this.transforms.length; i++) {
      const transform = this.transforms[i]!
      const previousTransforms = this.transforms.slice(0, i)
      
      // Create a readiness-aware wrapper for this transform
      currentStream = this.createReadinessWrapper(
        currentStream, 
        transform, 
        previousTransforms
      )
    }
    
    yield* currentStream
  }

  /**
   * Wraps a transform to only yield data when it and all ancestors are ready
   */
  private async* createReadinessWrapper(
    inputStream: AsyncGenerator<OhlcvDto>,
    transform: Transform,
    ancestors: Transform[]
  ): AsyncGenerator<OhlcvDto> {
    const result = await transform.apply(this.iteratorFromStream(inputStream))
    
    for await (const item of this.streamFromIterator(result.data)) {
      // Check if all ancestors are ready
      const ancestorsReady = ancestors.every(t => t.isReady())
      // Check if this transform is ready
      const selfReady = transform.isReady()
      
      if (ancestorsReady && selfReady) {
        yield item
      }
      // If not ready, consume but don't yield
      // This allows transforms to build up their internal state
    }
  }


  /**
   * Convert AsyncIterator to AsyncGenerator
   */
  private async* streamFromIterator(iterator: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let item = await iterator.next()
    while (!item.done) {
      yield item.value
      item = await iterator.next()
    }
  }

  /**
   * Convert AsyncGenerator to AsyncIterator
   */
  private iteratorFromStream(stream: AsyncGenerator<OhlcvDto>): AsyncIterator<OhlcvDto> {
    return stream
  }


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
}

/**
 * Helper function to create a pipeline from an array of transforms
 */
export function createPipeline(transforms: Transform[], description?: string): TransformPipeline {
  return new TransformPipeline({ transforms, description })
}