import type { Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import { BaseTransform } from './base-transform'
import type { ZScoreParams } from './transform-params'

/**
 * Transform that normalizes data to z-scores (mean=0, std=1)
 * Can operate in global mode (entire dataset) or rolling window mode
 */
export class ZScoreNormalizer extends BaseTransform<ZScoreParams> {
  private readonly fields: string[]
  private readonly suffix: string
  private readonly windowSize: number | null

  // For global normalization
  private readonly globalStats = new Map<string, { sum: number; sumSquares: number; count: number }>()
  private readonly globalMeans = new Map<string, number>()
  private readonly globalStds = new Map<string, number>()
  private readonly dataBuffer: OhlcvDto[] = []

  // For rolling window normalization
  private readonly windowBuffers = new Map<string, OhlcvDto[]>()

  constructor(params: ZScoreParams) {
    super(
      'zScore',
      'Z-Score Normalizer',
      'Normalizes data to standard scores (mean=0, std=1)',
      params,
      true, // Reversible with coefficients
    )

    this.fields = params.fields || ['open', 'high', 'low', 'close', 'volume']
    this.suffix = params.addSuffix !== false ? (params.suffix || '_zscore') : ''
    this.windowSize = params.windowSize || null
  }

  public validate(): void {
    super.validate()

    if (this.windowSize !== null && this.windowSize < 2) {
      throw new Error('Window size must be at least 2 for z-score calculation')
    }
  }

  protected async* transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    if (this.windowSize === null) {
      // Global normalization - need two passes
      yield* this.globalNormalization(data)
    } else {
      // Rolling window normalization
      yield* this.rollingNormalization(data)
    }
  }

  private async* globalNormalization(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    // First pass: collect statistics
    let item = await data.next()
    while (!item.done) {
      const current = item.value
      this.dataBuffer.push(current)

      for (const field of this.fields) {
        if (field in current && typeof current[field as keyof OhlcvDto] === 'number') {
          const value = current[field as keyof OhlcvDto] as number
          const key = `${field}-${current.symbol}-${current.exchange}`

          if (!this.globalStats.has(key)) {
            this.globalStats.set(key, { sum: 0, sumSquares: 0, count: 0 })
          }

          const stats = this.globalStats.get(key)!
          stats.sum += value
          stats.sumSquares += value * value
          stats.count += 1
        }
      }

      item = await data.next()
    }

    // Calculate means and standard deviations
    for (const [key, stats] of this.globalStats) {
      const mean = stats.sum / stats.count
      const variance = (stats.sumSquares / stats.count) - (mean * mean)
      const std = Math.sqrt(Math.max(0, variance))

      this.globalMeans.set(key, mean)
      this.globalStds.set(key, std)
    }

    // Store coefficients for the first symbol we encounter
    if (this.dataBuffer.length > 0 && !this.getCoefficients()) {
      const firstItem = this.dataBuffer[0]!
      const coefficients: Record<string, number> = {}

      for (const field of this.fields) {
        const key = `${field}-${firstItem.symbol}-${firstItem.exchange}`
        coefficients[`${field}_mean`] = this.globalMeans.get(key) || 0
        coefficients[`${field}_std`] = this.globalStds.get(key) || 1
      }

      this.setCoefficients(firstItem.symbol, coefficients)
    }

    // Second pass: normalize the data
    for (const item of this.dataBuffer) {
      yield this.normalizeItem(item)
    }
  }

  private async* rollingNormalization(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let item = await data.next()

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
      if (buffer.length > this.windowSize!) {
        buffer.shift()
      }

      // Calculate z-scores if we have enough data
      if (buffer.length >= Math.min(2, this.windowSize!)) {
        yield this.normalizeItemWithWindow(current, buffer)
      } else {
        // Not enough data yet, output with zero z-scores
        const normalized = { ...current }
        for (const field of this.fields) {
          if (field in current && typeof current[field as keyof OhlcvDto] === 'number') {
            const outputField = field + this.suffix
            normalized[outputField] = 0
          }
        }
        yield normalized
      }

      item = await data.next()
    }
  }

  private normalizeItem(item: OhlcvDto): OhlcvDto {
    const normalized = { ...item }

    for (const field of this.fields) {
      if (field in item && typeof item[field as keyof OhlcvDto] === 'number') {
        const value = item[field as keyof OhlcvDto] as number
        const key = `${field}-${item.symbol}-${item.exchange}`
        const mean = this.globalMeans.get(key) || 0
        const std = this.globalStds.get(key) || 1

        const outputField = field + this.suffix
        normalized[outputField] = std > 0 ? (value - mean) / std : 0
      }
    }

    return normalized
  }

  private normalizeItemWithWindow(item: OhlcvDto, window: OhlcvDto[]): OhlcvDto {
    const normalized = { ...item }

    for (const field of this.fields) {
      if (field in item && typeof item[field as keyof OhlcvDto] === 'number') {
        // Calculate mean and std for this field in the window
        let sum = 0
        let sumSquares = 0
        let count = 0

        for (const windowItem of window) {
          if (field in windowItem && typeof windowItem[field as keyof OhlcvDto] === 'number') {
            const value = windowItem[field as keyof OhlcvDto] as number
            sum += value
            sumSquares += value * value
            count += 1
          }
        }

        if (count >= 2) {
          const mean = sum / count
          const variance = (sumSquares / count) - (mean * mean)
          const std = Math.sqrt(Math.max(0, variance))

          const value = item[field as keyof OhlcvDto] as number
          const outputField = field + this.suffix
          normalized[outputField] = std > 0 ? (value - mean) / std : 0
        } else {
          const outputField = field + this.suffix
          normalized[outputField] = 0
        }
      }
    }

    return normalized
  }

  public getOutputFields(): string[] {
    return this.fields.map(field => field + this.suffix)
  }

  public getRequiredFields(): string[] {
    return this.fields
  }

  public withParams(params: Partial<ZScoreParams>): Transform<ZScoreParams> {
    return new ZScoreNormalizer({ ...this.params, ...params })
  }

  public async* reverse(
    data: AsyncIterator<OhlcvDto>,
    coefficients: any
  ): AsyncGenerator<OhlcvDto> {
    // Reverse z-score normalization: x = z * std + mean
    let item = await data.next()

    while (!item.done) {
      const current = item.value
      const reversed = { ...current }

      for (const field of this.fields) {
        const outputField = field + this.suffix
        if (outputField in current && typeof current[outputField] === 'number') {
          const zScore = current[outputField]
          const mean = coefficients.values[`${field}_mean`] || 0
          const std = coefficients.values[`${field}_std`] || 1

          // Reverse the normalization
          reversed[field as keyof OhlcvDto] = (zScore * std + mean)

          // Remove the z-score field
          delete reversed[outputField]
        }
      }

      yield reversed
      item = await data.next()
    }
  }
}
