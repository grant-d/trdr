import type { Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import { BaseTransform } from './base-transform'
import type { LogReturnsParams } from './transform-params'

/**
 * Transform that calculates log returns from price data
 * Log returns are additive and normally distributed, making them ideal for many analyses
 */
export class LogReturnsNormalizer extends BaseTransform<LogReturnsParams> {
  private readonly priceField: keyof Pick<OhlcvDto, 'open' | 'high' | 'low' | 'close'>
  private readonly outputField: string
  private readonly logFunction: (x: number) => number
  private readonly lastPrices = new Map<string, number>()

  constructor(params: LogReturnsParams) {
    super(
      'logReturns',
      'Log Returns Normalizer',
      'Calculates logarithmic returns from price data',
      params,
      true, // Reversible with coefficients
    )

    this.priceField = params.priceField || 'close'
    this.outputField = params.outputField || `${this.priceField}_log_return`
    this.logFunction = params.base === 'log10' ? Math.log10 : Math.log
  }

  public validate(): void {
    super.validate()

    const validFields = ['open', 'high', 'low', 'close']
    if (this.params.priceField && !validFields.includes(this.params.priceField)) {
      throw new Error(`Invalid price field: ${this.params.priceField}`)
    }
  }

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let item = await data.next()
    let isFirstBar = true

    while (!item.done) {
      const current = item.value
      const symbolKey = `${current.symbol}-${current.exchange}`
      const currentPrice = current[this.priceField]

      // Store coefficients on first item
      if (isFirstBar && !this.getCoefficients()) {
        this.setCoefficients(current.symbol, {
          priceField: this.priceField === 'close' ? 0 : 
                      this.priceField === 'open' ? 1 :
                      this.priceField === 'high' ? 2 : 3,
          base: this.params.base === 'log10' ? 10 : Math.E,
        })
        isFirstBar = false
      }

      const lastPrice = this.lastPrices.get(symbolKey)
      let logReturn = 0

      if (lastPrice !== undefined && lastPrice > 0 && currentPrice > 0) {
        // Calculate log return: log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
        logReturn = this.logFunction(currentPrice / lastPrice)
      }

      // Update last price for this symbol
      this.lastPrices.set(symbolKey, currentPrice)

      // Add the log return to the output
      yield {
        ...current,
        [this.outputField]: logReturn,
      }

      item = await data.next()
    }
  }

  public getOutputFields(): string[] {
    return [this.outputField]
  }

  public getRequiredFields(): string[] {
    return [this.priceField]
  }

  public withParams(params: Partial<LogReturnsParams>): Transform<LogReturnsParams> {
    return new LogReturnsNormalizer({ ...this.params, ...params })
  }

  public async *reverse(
    _data: AsyncIterator<OhlcvDto>,
    _coefficients: any
  ): AsyncGenerator<OhlcvDto> {
    // To reverse log returns, we need the initial price
    // This would require storing the initial price as a coefficient
    // For now, we'll throw an error indicating this limitation
    throw new Error(
      'Reversing log returns requires initial price information. ' +
      'Consider using a different transform or storing initial prices separately.'
    )
  }
}
