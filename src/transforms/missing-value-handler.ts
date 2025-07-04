import type { Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import logger from '../utils/logger'
import { BaseTransform } from './base-transform'
import type { MissingValueParams } from './transform-params'

/**
 * Transform that handles missing values in OHLCV data streams
 * Supports multiple strategies: forward fill, backward fill, interpolation, and custom value
 */
export class MissingValueHandler extends BaseTransform<MissingValueParams> {
  private readonly numericFields = ['open', 'high', 'low', 'close', 'volume']
  private readonly lastValidValues = new Map<string, Partial<OhlcvDto>>()

  constructor(params: MissingValueParams) {
    super(
      'missingValues',
      'Missing Value Handler',
      'Handles missing or invalid values in the data stream',
      params,
      false, // Not reversible - we lose information about what was missing
    )
  }

  public validate(): void {
    super.validate()

    if (this.params.strategy === 'value' && this.params.fillValue === undefined) {
      throw new Error('fillValue must be provided when using "value" strategy')
    }

    if (this.params.maxFillGap !== undefined && this.params.maxFillGap < 1) {
      throw new Error('maxFillGap must be at least 1')
    }
  }

  protected async* transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    // For backward fill, we need to buffer data
    if (this.params.strategy === 'backward') {
      yield* this.backwardFillTransform(data)
      return
    }

    // For other strategies, we can process streaming
    let item = await data.next()
    while (!item.done) {
      const processed = this.processItem(item.value)
      if (processed) {
        yield processed
      }
      item = await data.next()
    }
  }

  private async* backwardFillTransform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    // Buffer all data first for backward fill
    const allData: OhlcvDto[] = []
    let item = await data.next()
    while (!item.done) {
      allData.push(item.value)
      item = await data.next()
    }

    // Process backward
    for (let i = allData.length - 1; i >= 0; i--) {
      const current = allData[i]!
      const symbolKey = `${current.symbol}-${current.exchange}`

      // Check each field for missing values
      const fieldsToCheck = this.params.fields || this.numericFields
      const processed = { ...current }

      for (const field of fieldsToCheck) {
        if (field in current && this.isMissingValue(current[field as keyof OhlcvDto])) {
          const lastValid = this.lastValidValues.get(symbolKey)

          if (lastValid && field in lastValid) {
            processed[field as keyof OhlcvDto] = lastValid[field as keyof OhlcvDto] as any // TODO: Remove any cast
          } else if (this.params.strategy === 'value') {
            processed[field as keyof OhlcvDto] = this.params.fillValue as any // TODO: Remove any cast
          }
        }
      }

      // Update last valid values
      this.updateLastValid(symbolKey, processed)

      // Store for forward output
      allData[i] = processed
    }

    // Yield in forward order
    for (const item of allData) {
      yield item
    }
  }

  private processItem(item: OhlcvDto): OhlcvDto | null {
    const symbolKey = `${item.symbol}-${item.exchange}`
    const fieldsToCheck = this.params.fields || this.numericFields
    let hasChanges = false
    const processed = { ...item }

    for (const field of fieldsToCheck) {
      if (field in item && this.isMissingValue(item[field as keyof OhlcvDto])) {
        hasChanges = true

        switch (this.params.strategy) {
          case 'forward': {
            const lastValid = this.lastValidValues.get(symbolKey)
            if (lastValid && field in lastValid) {
              processed[field as keyof OhlcvDto] = lastValid[field as keyof OhlcvDto] as any // TODO: Remove any cast
            } else if (this.params.fillValue !== undefined) {
              processed[field as keyof OhlcvDto] = this.params.fillValue as any // TODO: Remove any cast
            }
            break
          }

          case 'value': {
            processed[field as keyof OhlcvDto] = this.params.fillValue as any // TODO: Remove any cast
            break
          }

          case 'interpolate': {
            // For interpolation, we'd need to buffer and look ahead
            // For now, fall back to forward fill
            const lastValid = this.lastValidValues.get(symbolKey)
            if (lastValid && field in lastValid) {
              processed[field as keyof OhlcvDto] = lastValid[field as keyof OhlcvDto] as any // TODO: Remove any cast
            }
            break
          }
        }
      }
    }

    // Update last valid values
    this.updateLastValid(symbolKey, processed)

    if (hasChanges) {
      logger.debug('Filled missing values', {
        symbol: item.symbol,
        exchange: item.exchange,
        timestamp: item.timestamp,
        strategy: this.params.strategy,
      })
    }

    return processed
  }

  private isMissingValue(value: any): boolean {
    return value === null ||
      value === undefined ||
      (typeof value === 'number' && isNaN(value))
  }

  private updateLastValid(symbolKey: string, item: OhlcvDto): void {
    const fieldsToCheck = this.params.fields || this.numericFields
    const lastValid = this.lastValidValues.get(symbolKey) || {}

    for (const field of fieldsToCheck) {
      if (field in item && !this.isMissingValue(item[field as keyof OhlcvDto])) {
        lastValid[field as keyof OhlcvDto] = item[field as keyof OhlcvDto]
      }
    }

    this.lastValidValues.set(symbolKey, lastValid)
  }

  public getOutputFields(): string[] {
    return [] // Doesn't add new fields, only modifies existing ones
  }

  public getRequiredFields(): string[] {
    return this.params.fields || []
  }

  public withParams(params: Partial<MissingValueParams>): Transform<MissingValueParams> {
    return new MissingValueHandler({ ...this.params, ...params })
  }
}
