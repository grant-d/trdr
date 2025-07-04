import type { Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import { BaseTransform } from './base-transform'
import type { MinMaxParams } from './transform-params'

/**
 * Transform that normalizes data to a specific range using min-max scaling
 * Can operate in global mode (entire dataset) or rolling window mode
 */
export class MinMaxNormalizer extends BaseTransform<MinMaxParams> {
  private readonly fields: string[]
  private readonly suffix: string
  private readonly windowSize: number | null
  private readonly targetMin: number
  private readonly targetMax: number
  private readonly targetRange: number

  // For global normalization
  private readonly globalMins = new Map<string, number>()
  private readonly globalMaxs = new Map<string, number>()
  private readonly dataBuffer: OhlcvDto[] = []

  // For rolling window normalization
  private readonly windowBuffers = new Map<string, OhlcvDto[]>()

  constructor(params: MinMaxParams) {
    super(
      'minMax',
      'Min-Max Normalizer',
      `Normalizes data to range [${params.targetMin || 0}, ${params.targetMax || 1}]`,
      params,
      true, // Reversible with coefficients
    )

    this.fields = params.fields || ['open', 'high', 'low', 'close', 'volume']
    this.suffix = params.addSuffix !== false ? (params.suffix || '_norm') : ''
    this.windowSize = params.windowSize || null
    this.targetMin = params.targetMin || 0
    this.targetMax = params.targetMax || 1
    this.targetRange = this.targetMax - this.targetMin
  }

  public validate(): void {
    super.validate()

    if (this.windowSize !== null && this.windowSize < 2) {
      throw new Error('Window size must be at least 2 for min-max calculation')
    }

    if (this.targetMin >= this.targetMax) {
      throw new Error('targetMax must be greater than targetMin')
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
    // First pass: find min and max values
    let item = await data.next()
    while (!item.done) {
      const current = item.value
      this.dataBuffer.push(current)

      for (const field of this.fields) {
        if (field in current && typeof current[field as keyof OhlcvDto] === 'number') {
          const value = current[field as keyof OhlcvDto] as number
          const key = `${field}-${current.symbol}-${current.exchange}`

          if (!this.globalMins.has(key) || value < this.globalMins.get(key)!) {
            this.globalMins.set(key, value)
          }

          if (!this.globalMaxs.has(key) || value > this.globalMaxs.get(key)!) {
            this.globalMaxs.set(key, value)
          }
        }
      }

      item = await data.next()
    }

    // Store coefficients for the first symbol we encounter
    if (this.dataBuffer.length > 0 && !this.getCoefficients()) {
      const firstItem = this.dataBuffer[0]!
      const coefficients: Record<string, number> = {}

      for (const field of this.fields) {
        const key = `${field}-${firstItem.symbol}-${firstItem.exchange}`
        coefficients[`${field}_min`] = this.globalMins.get(key) || 0
        coefficients[`${field}_max`] = this.globalMaxs.get(key) || 1
      }

      coefficients.target_min = this.targetMin
      coefficients.target_max = this.targetMax

      this.setCoefficients(firstItem.symbol, coefficients)
    }

    // Second pass: normalize the data
    for (const item of this.dataBuffer) {
      yield this.normalizeItem(item)
    }
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
      if (buffer.length > this.windowSize!) {
        buffer.shift()
      }

      // Store coefficients on first item (for rolling, we just store target range)
      if (isFirstBar && !this.getCoefficients()) {
        this.setCoefficients(current.symbol, {
          target_min: this.targetMin,
          target_max: this.targetMax,
          window_size: this.windowSize!,
        })
        isFirstBar = false
      }

      // Normalize if we have enough data
      if (buffer.length >= Math.min(2, this.windowSize!)) {
        yield this.normalizeItemWithWindow(current, buffer)
      } else {
        // Not enough data yet, output with target min values
        const normalized = { ...current }
        for (const field of this.fields) {
          if (field in current && typeof current[field as keyof OhlcvDto] === 'number') {
            const outputField = field + this.suffix
            normalized[outputField] = this.targetMin
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
        const min = this.globalMins.get(key) ?? 0
        const max = this.globalMaxs.get(key) ?? 0
        const range = max - min

        const outputField = field + this.suffix
        if (range > 0) {
          // Scale to [0, 1] then to target range
          const scaled01 = (value - min) / range
          normalized[outputField] = this.targetMin + (scaled01 * this.targetRange)
        } else {
          // All values are the same, use middle of target range
          normalized[outputField] = (this.targetMin + this.targetMax) / 2
        }
      }
    }

    return normalized
  }

  private normalizeItemWithWindow(item: OhlcvDto, window: OhlcvDto[]): OhlcvDto {
    const normalized = { ...item }

    for (const field of this.fields) {
      if (field in item && typeof item[field as keyof OhlcvDto] === 'number') {
        // Find min and max in the window
        let min = Number.POSITIVE_INFINITY
        let max = Number.NEGATIVE_INFINITY

        for (const windowItem of window) {
          if (field in windowItem && typeof windowItem[field as keyof OhlcvDto] === 'number') {
            const value = windowItem[field as keyof OhlcvDto] as number
            min = Math.min(min, value)
            max = Math.max(max, value)
          }
        }

        const range = max - min
        const value = item[field as keyof OhlcvDto] as number
        const outputField = field + this.suffix

        if (range > 0) {
          // Scale to [0, 1] then to target range
          const scaled01 = (value - min) / range
          normalized[outputField] = this.targetMin + (scaled01 * this.targetRange)
        } else {
          // All values in window are the same
          normalized[outputField] = (this.targetMin + this.targetMax) / 2
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

  public withParams(params: Partial<MinMaxParams>): Transform<MinMaxParams> {
    return new MinMaxNormalizer({ ...this.params, ...params })
  }

  public async* reverse(
    data: AsyncIterator<OhlcvDto>,
    coefficients: any
  ): AsyncGenerator<OhlcvDto> {
    // Reverse min-max normalization: x = (norm - target_min) / target_range * (max - min) + min
    const targetMin = coefficients.values.target_min || 0
    const targetMax = coefficients.values.target_max || 1
    const targetRange = targetMax - targetMin

    let item = await data.next()

    while (!item.done) {
      const current = item.value
      const reversed = { ...current }

      for (const field of this.fields) {
        const outputField = field + this.suffix
        if (outputField in current && typeof current[outputField] === 'number') {
          const normalized = current[outputField]
          const min = coefficients.values[`${field}_min`] || 0
          const max = coefficients.values[`${field}_max`] || 1
          const range = max - min

          // Reverse the normalization
          if (targetRange > 0) {
            const scaled01 = (normalized - targetMin) / targetRange
            reversed[field as keyof OhlcvDto] = (scaled01 * range + min)
          } else {
            reversed[field as keyof OhlcvDto] = min
          }

          // Remove the normalized field
          delete reversed[outputField]
        }
      }

      yield reversed
      item = await data.next()
    }
  }
}
