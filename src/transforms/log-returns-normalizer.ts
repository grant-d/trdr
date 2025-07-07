import type { Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import { BaseTransform } from './base-transform'
import type { LogReturnsParams } from './transform-params'

/**
 * Transform that calculates log returns from price data
 * Log returns are additive and normally distributed, making them ideal for many analyses
 */
export class LogReturnsNormalizer extends BaseTransform<LogReturnsParams> {
  private readonly logFunction: (x: number) => number
  private readonly symbolDataPoints = new Map<string, number>()

  constructor(params: LogReturnsParams) {
    super(
      'logReturns',
      'Log Returns Normalizer',
      'Calculates logarithmic returns from price data',
      params,
    )

    this.logFunction = params.base === 'log10' ? Math.log10 : Math.log
  }

  public validate(): void {
    super.validate()
    // No additional validation needed - we now apply to all fields
  }

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let item = await data.next()
    let isFirstBar = true
    const lastValues = new Map<string, Record<string, number>>()

    while (!item.done) {
      const current = item.value
      const symbolKey = `${current.symbol}-${current.exchange}`
      const inputColumns = this.getInputColumns()

      // Update readiness tracking on first item
      if (isFirstBar) {
        isFirstBar = false
      }

      const result = { ...current } as any
      const lastRecord = lastValues.get(symbolKey)
      const droppedColumns = this.getDroppedColumns()

      // Increment data points processed for readiness tracking per symbol
      const currentCount = this.symbolDataPoints.get(symbolKey) || 0
      this.symbolDataPoints.set(symbolKey, currentCount + 1)

      // Calculate log returns for specified input/output pairs
      if (this.params.in && this.params.out) {
        for (let i = 0; i < inputColumns.length; i++) {
          const inputCol = inputColumns[i]!
          const outputCol: string | null | undefined = this.params.out[i]
          
          // Skip if output column is null (will be dropped later)
          if (outputCol === null) {
            continue
          }
          
          const actualOutputCol = outputCol || inputCol
          const currentValue = result[inputCol]
          
          if (currentValue !== undefined && typeof currentValue === 'number') {
            if (lastRecord?.[inputCol] !== undefined) {
              result[actualOutputCol] = this.calculateLogReturn(currentValue, lastRecord[inputCol])
            } else {
              // First bar - no previous data, so returns are 0
              result[actualOutputCol] = 0
            }
          }
        }
      } else {
        // Default behavior: transform all OHLCV columns
        const outputCols = this.getOutputColumns()
        for (let i = 0; i < inputColumns.length; i++) {
          const inputCol = inputColumns[i]!
          const outputCol = outputCols[i]!
          const currentValue = result[inputCol]
          
          if (currentValue !== undefined && typeof currentValue === 'number') {
            if (lastRecord?.[inputCol] !== undefined) {
              result[outputCol] = this.calculateLogReturn(currentValue, lastRecord[inputCol])
            } else {
              // First bar - no previous data, so returns are 0
              result[outputCol] = 0
            }
          }
        }
      }

      // Update last values for this symbol
      if (!lastValues.has(symbolKey)) {
        lastValues.set(symbolKey, {})
      }
      const recordValues = lastValues.get(symbolKey)!
      for (const col of inputColumns) {
        const value = current[col]
        if (value !== undefined && typeof value === 'number') {
          recordValues[col] = value
        }
      }

      // Drop columns marked for dropping
      for (const colToDrop of droppedColumns) {
        delete result[colToDrop]
      }

      // Only yield after we have at least 2 data points for this symbol (when ready)
      if ((this.symbolDataPoints.get(symbolKey) || 0) >= 2) {
        yield result as OhlcvDto
      }

      item = await data.next()
    }
  }

  private calculateLogReturn(currentValue: number, previousValue: number): number {
    if (previousValue > 0 && currentValue > 0) {
      return this.logFunction(currentValue / previousValue)
    }
    return 0
  }

  public getOutputFields(): string[] {
    return this.getOutputColumns()
  }

  public getRequiredFields(): string[] {
    return this.getInputColumns()
  }

  public isReady(): boolean {
    // Log returns need at least 2 data points (current and previous)
    // Consider ready if ALL symbols have at least 2 data points
    const counts = Array.from(this.symbolDataPoints.values())
    return counts.length > 0 && counts.every(count => count >= 2)
  }

  public withParams(params: Partial<LogReturnsParams>): Transform<LogReturnsParams> {
    return new LogReturnsNormalizer({ ...this.params, ...params })
  }
}
