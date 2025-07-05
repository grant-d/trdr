import type { Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import { BaseTransform } from './base-transform'
import type { ZScoreParams } from './transform-params'

/**
 * Transform that normalizes data to z-scores (mean=0, std=1)
 * Uses rolling window normalization
 */
export class ZScoreNormalizer extends BaseTransform<ZScoreParams> {
  private readonly windowSize: number

  // For rolling window normalization
  private readonly windowBuffers = new Map<string, OhlcvDto[]>()
  
  // Track total data points processed for readiness
  private dataPointsProcessed = 0

  constructor(params: ZScoreParams) {
    super(
      'zScore',
      'Z-Score Normalizer',
      `Normalizes data to z-scores using ${params.windowSize || 20} period rolling window`,
      params
    )

    this.windowSize = params.windowSize || 20
  }

  public validate(): void {
    super.validate()

    if (this.windowSize < 2) {
      throw new Error('Window size must be at least 2 for z-score calculation')
    }
  }

  public getOutputFields(): string[] {
    return this.getOutputColumns()
  }

  public getRequiredFields(): string[] {
    return this.getInputColumns()
  }

  public isReady(): boolean {
    // Rolling window normalization is ready after windowSize data points
    return this.dataPointsProcessed >= this.windowSize
  }

  protected async* transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    // Only rolling window normalization
    yield* this.rollingNormalization(data)
  }


  private async* rollingNormalization(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let item = await data.next()
    let isFirstBar = true

    while (!item.done) {
      const current = item.value
      const symbolKey = `${current.symbol}-${current.exchange}`

      // Update readiness tracking on first item
      if (isFirstBar) {
        isFirstBar = false
      }

      // Initialize buffer for this symbol if needed
      if (!this.windowBuffers.has(symbolKey)) {
        this.windowBuffers.set(symbolKey, [])
      }

      const buffer = this.windowBuffers.get(symbolKey)!
      buffer.push(current)

      // Keep only the window size
      if (buffer.length > this.windowSize) {
        buffer.shift()
      }

      // Increment data points processed for readiness tracking
      this.dataPointsProcessed++

      // Calculate z-scores if we have enough data
      if (buffer.length >= this.windowSize) {
        yield this.normalizeItemWithWindow(current, buffer)
      } else {
        // Not enough data yet, output with zero z-scores for new fields
        const inputColumns = this.getInputColumns()
        const outputColumns = this.getOutputColumns()
        const droppedColumns = this.getDroppedColumns()
        const result = { ...current } as any
        
        // Add zero values for specified transforms
        if (this.params.in && this.params.out) {
          for (let i = 0; i < inputColumns.length; i++) {
            const outputCol: string | null | undefined = this.params.out[i]
            
            // Skip if output column is null (will be dropped later)
            if (outputCol === null) {
              continue
            }
            
            const actualOutputCol = outputCol || inputColumns[i]!
            result[actualOutputCol] = 0
          }
        } else {
          // Default behavior: zero out all OHLCV columns
          for (let i = 0; i < inputColumns.length; i++) {
            result[outputColumns[i]!] = 0
          }
        }
        
        // Drop columns marked for dropping
        for (const colToDrop of droppedColumns) {
          delete result[colToDrop]
        }
        
        yield result as OhlcvDto
      }

      item = await data.next()
    }
  }


  private normalizeItemWithWindow(item: OhlcvDto, window: OhlcvDto[]): OhlcvDto {
    const inputColumns = this.getInputColumns()
    const outputColumns = this.getOutputColumns()
    const droppedColumns = this.getDroppedColumns()
    const result = { ...item } as any

    // Apply z-score normalization to specified input/output pairs
    if (this.params.in && this.params.out) {
      for (let i = 0; i < inputColumns.length; i++) {
        const inputCol = inputColumns[i]!
        const outputCol: string | null | undefined = this.params.out[i]
        
        // Skip if output column is null (will be dropped later)
        if (outputCol === null) {
          continue
        }
        
        const actualOutputCol = outputCol || inputCol
        
        // Calculate mean and std for this column in the window
        let sum = 0
        let sumSquares = 0
        let count = 0

        for (const windowItem of window) {
          const value = windowItem[inputCol]
          if (value !== undefined && typeof value === 'number') {
            sum += value
            sumSquares += value * value
            count += 1
          }
        }

        if (count >= 2) {
          const mean = sum / count
          const variance = (sumSquares / count) - (mean * mean)
          const std = Math.sqrt(Math.max(0, variance))

          const value = result[inputCol]
          if (value !== undefined && typeof value === 'number') {
            result[actualOutputCol] = std > 0 ? (value - mean) / std : 0
          }
        } else {
          // Not enough data for normalization
          result[actualOutputCol] = 0
        }
      }
    } else {
      // Default behavior: transform all OHLCV columns
      for (let i = 0; i < inputColumns.length; i++) {
        const inputCol = inputColumns[i]!
        const outputCol = outputColumns[i]!
        
        // Calculate mean and std for this column in the window
        let sum = 0
        let sumSquares = 0
        let count = 0

        for (const windowItem of window) {
          const value = windowItem[inputCol]
          if (value !== undefined && typeof value === 'number') {
            sum += value
            sumSquares += value * value
            count += 1
          }
        }

        if (count >= 2) {
          const mean = sum / count
          const variance = (sumSquares / count) - (mean * mean)
          const std = Math.sqrt(Math.max(0, variance))

          const value = result[inputCol]
          if (value !== undefined && typeof value === 'number') {
            result[outputCol] = std > 0 ? (value - mean) / std : 0
          }
        } else {
          // Not enough data for normalization
          result[outputCol] = 0
        }
      }
    }

    // Drop columns marked for dropping
    for (const colToDrop of droppedColumns) {
      delete result[colToDrop]
    }

    return result as OhlcvDto
  }


  public withParams(params: Partial<ZScoreParams>): Transform<ZScoreParams> {
    return new ZScoreNormalizer({ ...this.params, ...params })
  }
}
