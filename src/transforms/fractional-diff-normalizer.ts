import type { Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import { BaseTransform } from './base-transform'
import type { FractionalDiffParams } from './transform-params'

/**
 * Transform that applies fractional differentiation to time series data
 * Fractional differentiation makes time series stationary while preserving memory
 * Based on "Advances in Financial Machine Learning" by Marcos López de Prado
 */
export class FractionalDiffNormalizer extends BaseTransform<FractionalDiffParams> {
  private readonly weights: number[]
  private readonly maxLookback: number
  private dataHistory: Map<string, Record<string, number[]>>
  
  constructor(params: FractionalDiffParams) {
    super(
      'fractionalDiff',
      'Fractional Differentiation Normalizer',
      'Applies fractional differentiation to make time series stationary while preserving memory',
      params
    )
    
    // Validate d parameter
    if (params.d < 0 || params.d > 2) {
      throw new Error('Differencing parameter d must be between 0 and 2')
    }
    
    // Calculate weights using the expanding window method
    this.weights = this.calculateWeights(
      params.d,
      params.maxWeights || 100,
      params.minWeight || 1e-5
    )
    
    this.maxLookback = this.weights.length
    this.dataHistory = new Map()
  }
  
  /**
   * Calculate fractional differentiation weights
   * Uses the binomial series expansion for (1-L)^d
   */
  private calculateWeights(d: number, maxWeights: number, minWeight: number): number[] {
    const weights: number[] = [1]
    
    for (let k = 1; k < maxWeights; k++) {
      // Calculate weight using the recursive formula
      // w_k = -w_{k-1} * (d - k + 1) / k
      const weight = -weights[k - 1]! * (d - k + 1) / k
      
      // Stop if weight is too small
      if (Math.abs(weight) < minWeight) {
        break
      }
      
      weights.push(weight)
    }
    
    return weights
  }
  
  public validate(): void {
    super.validate()
    
    if (this.params.d === undefined) {
      throw new Error('Differencing parameter d is required')
    }
    
    if (this.params.maxWeights !== undefined && this.params.maxWeights <= 0) {
      throw new Error('maxWeights must be positive')
    }
    
    if (this.params.minWeight !== undefined && this.params.minWeight <= 0) {
      throw new Error('minWeight must be positive')
    }
  }
  
  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let item = await data.next()
    
    while (!item.done) {
      const current = item.value
      const symbolKey = `${current.symbol}-${current.exchange}`
      const inputColumns = this.getInputColumns()
      const outputColumns = this.getOutputColumns()
      const droppedColumns = this.getDroppedColumns()
      
      // Initialize history for this symbol if needed
      if (!this.dataHistory.has(symbolKey)) {
        const history: Record<string, number[]> = {}
        for (const col of inputColumns) {
          history[col] = []
        }
        this.dataHistory.set(symbolKey, history)
      }
      
      const history = this.dataHistory.get(symbolKey)!
      const result = { ...current } as any
      
      // Apply fractional differentiation to each column
      for (let i = 0; i < inputColumns.length; i++) {
        const inputCol = inputColumns[i]!
        const outputCol = outputColumns[i]
        
        // Skip if this column is marked for dropping
        if (this.params.out && this.params.out[i] === null) {
          continue
        }
        
        const actualOutputCol = outputCol || inputCol
        const currentValue = current[inputCol]
        
        if (currentValue !== undefined && typeof currentValue === 'number') {
          // Add current value to history
          history[inputCol]!.push(currentValue)
          
          // Keep only the necessary history
          if (history[inputCol]!.length > this.maxLookback) {
            history[inputCol] = history[inputCol]!.slice(-this.maxLookback)
          }
          
          // Calculate fractionally differentiated value
          const diffValue = this.calculateFractionalDiff(history[inputCol]!)
          result[actualOutputCol] = diffValue
        }
      }
      
      // Drop columns marked for dropping
      for (const colToDrop of droppedColumns) {
        delete result[colToDrop]
      }
      
      yield result as OhlcvDto
      
      item = await data.next()
    }
  }
  
  /**
   * Calculate fractionally differentiated value using the weighted sum
   */
  private calculateFractionalDiff(history: number[]): number {
    if (history.length === 0) {
      return 0
    }
    
    let diffValue = 0
    const n = Math.min(history.length, this.weights.length)
    
    // Apply weights in reverse order (most recent data first)
    for (let i = 0; i < n; i++) {
      const dataIndex = history.length - 1 - i
      const value = history[dataIndex]!
      const weight = this.weights[i]!
      diffValue += weight * value
    }
    
    return diffValue
  }
  
  public getOutputFields(): string[] {
    return this.getOutputColumns()
  }
  
  public getRequiredFields(): string[] {
    return this.getInputColumns()
  }
  
  public isReady(): boolean {
    // Need at least one data point to start producing output
    // The quality improves as more data is collected
    return true
  }
  
  public withParams(params: Partial<FractionalDiffParams>): Transform<FractionalDiffParams> {
    return new FractionalDiffNormalizer({ ...this.params, ...params })
  }
}