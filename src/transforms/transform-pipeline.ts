import type { Transform, TransformResult, TransformCoefficients, BaseTransformParams } from '../interfaces'
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
 * Applies transforms sequentially and aggregates coefficients
 */
export class TransformPipeline implements Transform<TransformPipelineParams> {
  public readonly type = 'pipeline' as const
  public readonly name: string
  public readonly description: string
  public readonly isReversible: boolean
  public readonly params: TransformPipelineParams

  private readonly transforms: Transform[]

  constructor(params: TransformPipelineParams) {
    this.params = params
    this.transforms = [...params.transforms]
    this.name = params.name || 'Transform Pipeline'
    this.description = `Pipeline of ${this.transforms.length} transforms`
    
    // Pipeline is reversible only if all transforms are reversible
    this.isReversible = this.transforms.every(t => t.isReversible)
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

  public async apply(data: AsyncIterator<OhlcvDto>): Promise<TransformResult> {
    if (this.transforms.length === 0) {
      return { data }
    }

    const coefficients: TransformCoefficients[] = []
    let currentDataArray: OhlcvDto[] = []
    
    // Convert initial data to array so we can process it multiple times
    let item = await data.next()
    while (!item.done) {
      currentDataArray.push(item.value)
      item = await data.next()
    }

    // Apply each transform in sequence
    for (let i = 0; i < this.transforms.length; i++) {
      const transform = this.transforms[i]!
      
      // Convert array back to async iterator for each transform
      const result = await transform.apply(this.arrayToAsyncIterator(currentDataArray))
      
      // Consume the transform's data stream to get coefficients
      currentDataArray = []
      let transformItem = await result.data.next()
      while (!transformItem.done) {
        currentDataArray.push(transformItem.value)
        transformItem = await result.data.next()
      }
      
      // Collect coefficients if available, with transform index
      if (result.coefficients) {
        // Add the transform index to the coefficient for proper aggregation
        const coeffWithIndex = {
          ...result.coefficients,
          transformIndex: i
        }
        coefficients.push(coeffWithIndex as any)
      }
    }

    // Aggregate coefficients if any were collected
    const aggregatedCoefficients = coefficients.length > 0
      ? this.aggregateCoefficients(coefficients)
      : undefined

    return {
      data: this.arrayToAsyncIterator(currentDataArray),
      coefficients: aggregatedCoefficients
    }
  }

  /**
   * Helper to convert array to async iterator
   */
  private async *arrayToAsyncIterator(array: OhlcvDto[]): AsyncGenerator<OhlcvDto> {
    for (const item of array) {
      yield item
    }
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

  public async *reverse(
    data: AsyncIterator<OhlcvDto>, 
    coefficients: TransformCoefficients
  ): AsyncGenerator<OhlcvDto> {
    if (!this.isReversible) {
      throw new Error('Pipeline contains non-reversible transforms')
    }

    // Extract individual transform coefficients from aggregated coefficients
    const transformCoefficients = this.extractTransformCoefficients(coefficients)
    
    // Apply transforms in reverse order
    let currentData = data
    for (let i = this.transforms.length - 1; i >= 0; i--) {
      const transform = this.transforms[i]!
      const coeff = transformCoefficients[i]
      
      if (!transform.reverse || !coeff) {
        throw new Error(`Transform ${i} (${transform.name}) cannot be reversed`)
      }
      
      // Create an async generator from the reverse method
      const reversedData = transform.reverse(currentData, coeff)
      currentData = reversedData
    }

    // Yield all data from the final reversed stream
    for await (const item of currentData as AsyncGenerator<OhlcvDto>) {
      yield item
    }
  }

  public withParams(params: Partial<TransformPipelineParams>): Transform<TransformPipelineParams> {
    return new TransformPipeline({ ...this.params, ...params })
  }

  /**
   * Aggregate multiple transform coefficients into a single coefficient object
   */
  private aggregateCoefficients(coefficients: any[]): TransformCoefficients {
    if (coefficients.length === 0) {
      throw new Error('No coefficients to aggregate')
    }

    // Use the symbol from the first coefficient (should be the same for all)
    const symbol = coefficients[0]!.symbol
    
    // Aggregate all coefficient values with transform index prefix
    const aggregatedValues: Record<string, number> = {}
    
    for (const coeff of coefficients) {
      const transformIndex = coeff.transformIndex
      for (const [key, value] of Object.entries(coeff.values)) {
        // Prefix with transform index to avoid collisions
        aggregatedValues[`t${transformIndex}_${key}`] = value as number
      }
    }

    return {
      type: 'pipeline' as any, // Pipeline type for aggregated coefficients
      timestamp: Date.now(),
      symbol,
      values: aggregatedValues
    }
  }

  /**
   * Extract individual transform coefficients from aggregated pipeline coefficients
   */
  private extractTransformCoefficients(
    aggregated: TransformCoefficients
  ): (TransformCoefficients | undefined)[] {
    const coefficients: (TransformCoefficients | undefined)[] = []
    
    // Group values by transform index
    const groupedValues: Record<number, Record<string, number>> = {}
    
    for (const [key, value] of Object.entries(aggregated.values || {})) {
      const match = /^t(\d+)_(.+)$/.exec(key)
      if (match) {
        const index = parseInt(match[1]!, 10)
        const fieldKey = match[2]!
        
        if (!groupedValues[index]) {
          groupedValues[index] = {}
        }
        groupedValues[index][fieldKey] = value
      }
    }
    
    // Create coefficient objects for each transform
    for (let i = 0; i < this.transforms.length; i++) {
      const transform = this.transforms[i]!
      const values = groupedValues[i]
      
      if (values && Object.keys(values).length > 0) {
        coefficients.push({
          type: transform.type,
          timestamp: aggregated.timestamp,
          symbol: aggregated.symbol,
          values
        })
      } else {
        coefficients.push(undefined)
      }
    }
    
    return coefficients
  }
}

/**
 * Helper function to create a pipeline from an array of transforms
 */
export function createPipeline(transforms: Transform[], name?: string): TransformPipeline {
  return new TransformPipeline({ transforms, name })
}