import type { Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import { BaseTransform } from './base-transform'
import type { MinMaxParams } from './transform-params'

/**
 * Transform that normalizes data to a specific range using min-max scaling
 * Uses rolling window normalization
 */
export class MinMaxNormalizer extends BaseTransform<MinMaxParams> {
  private readonly windowSize: number
  private readonly targetMin: number
  private readonly targetMax: number
  private readonly targetRange: number

  // For rolling window normalization
  private readonly windowBuffers = new Map<string, OhlcvDto[]>()
  
  // Track total data points processed for readiness
  private dataPointsProcessed = 0

  constructor(params: MinMaxParams) {
    super(
      'minMax',
      'Min-Max Normalizer',
      `Normalizes data to range [${params.min || 0}, ${params.max || 1}] using ${params.windowSize || 20} period rolling window`,
      params,
    )

    this.windowSize = params.windowSize || 20
    this.targetMin = params.min || 0
    this.targetMax = params.max || 1
    this.targetRange = this.targetMax - this.targetMin
  }

  public validate(): void {
    super.validate()

    if (this.windowSize < 2) {
      throw new Error('Window size must be at least 2 for min-max calculation')
    }

    if (this.targetMin >= this.targetMax) {
      throw new Error('targetMax must be greater than targetMin')
    }
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

      // Update readiness tracking on first item
      if (isFirstBar) {
        isFirstBar = false
      }

      // Normalize if we have enough data
      // Only yield when we have enough data (when ready)
      if (buffer.length >= this.windowSize) {
        yield this.normalizeItemWithWindow(current, buffer)
      }
      // Don't yield anything until ready - just continue processing

      item = await data.next()
    }
  }


  private normalizeItemWithWindow(item: OhlcvDto, window: OhlcvDto[]): OhlcvDto {
    const inputColumns = this.getInputColumns()
    const outputColumns = this.getOutputColumns()
    const droppedColumns = this.getDroppedColumns()
    const result = { ...item } as any

    // Apply min-max normalization to specified input/output pairs
    if (this.params.in && this.params.out) {
      for (let i = 0; i < inputColumns.length; i++) {
        const inputCol = inputColumns[i]!
        const outputCol: string | null | undefined = this.params.out[i]
        
        // Skip if output column is null (will be dropped later)
        if (outputCol === null) {
          continue
        }
        
        const actualOutputCol = outputCol || inputCol
        
        // Find min and max in the window
        let min = Number.POSITIVE_INFINITY
        let max = Number.NEGATIVE_INFINITY

        for (const windowItem of window) {
          const value = windowItem[inputCol]
          if (value !== undefined && typeof value === 'number') {
            min = Math.min(min, value)
            max = Math.max(max, value)
          }
        }

        const range = max - min
        const value = result[inputCol]
        
        if (value !== undefined && typeof value === 'number') {
          if (range > 0) {
            // Scale to [0, 1] then to target range
            const scaled01 = (value - min) / range
            result[actualOutputCol] = this.targetMin + (scaled01 * this.targetRange)
          } else {
            // All values in window are the same
            result[actualOutputCol] = (this.targetMin + this.targetMax) / 2
          }
        }
      }
    } else {
      // Default behavior: transform all OHLCV columns
      for (let i = 0; i < inputColumns.length; i++) {
        const inputCol = inputColumns[i]!
        const outputCol = outputColumns[i]!
        
        // Find min and max in the window
        let min = Number.POSITIVE_INFINITY
        let max = Number.NEGATIVE_INFINITY

        for (const windowItem of window) {
          const value = windowItem[inputCol]
          if (value !== undefined && typeof value === 'number') {
            min = Math.min(min, value)
            max = Math.max(max, value)
          }
        }

        const range = max - min
        const value = result[inputCol]
        
        if (value !== undefined && typeof value === 'number') {
          if (range > 0) {
            // Scale to [0, 1] then to target range
            const scaled01 = (value - min) / range
            result[outputCol] = this.targetMin + (scaled01 * this.targetRange)
          } else {
            // All values in window are the same
            result[outputCol] = (this.targetMin + this.targetMax) / 2
          }
        }
      }
    }

    // Drop columns marked for dropping
    for (const colToDrop of droppedColumns) {
      delete result[colToDrop]
    }

    return result as OhlcvDto
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

  public withParams(params: Partial<MinMaxParams>): Transform<MinMaxParams> {
    return new MinMaxNormalizer({ ...this.params, ...params })
  }
}
