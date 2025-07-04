import type { Transform } from '../interfaces'
import type { OhlcvDto } from '../models'
import { BaseTransform } from './base-transform'
import type { PriceCalcParams } from './transform-params'

/**
 * Transform that calculates derived price values from OHLC data
 * Supports common calculations like HLC3, OHLC4, typical price, weighted close, etc.
 */
export class PriceCalculations extends BaseTransform<PriceCalcParams> {
  private readonly calculation: PriceCalcParams['calculation']
  private readonly outputField: string
  private readonly keepOriginal: boolean
  private readonly customFormula?: string

  constructor(params: PriceCalcParams) {
    super(
      'priceCalc',
      'Price Calculations',
      'Calculates derived price values from OHLC data',
      params,
      false // Not reversible
    )

    this.calculation = params.calculation
    this.outputField = params.outputField || this.getDefaultOutputField()
    this.keepOriginal = params.keepOriginal ?? true
    this.customFormula = params.customFormula
  }

  public validate(): void {
    super.validate()

    if (this.calculation === 'custom' && !this.customFormula) {
      throw new Error('Custom formula is required when calculation type is "custom"')
    }

    if (this.customFormula && this.calculation !== 'custom') {
      throw new Error('Custom formula should only be provided when calculation type is "custom"')
    }
  }

  protected async *transform(data: AsyncIterator<OhlcvDto>): AsyncGenerator<OhlcvDto> {
    let item = await data.next()

    while (!item.done) {
      const current = item.value
      const calculatedPrice = this.calculatePrice(current)

      // Add the calculated field
      const transformed = {
        ...current,
        [this.outputField]: calculatedPrice
      }

      // Remove original OHLC fields if requested
      if (!this.keepOriginal) {
        delete (transformed as any).open
        delete (transformed as any).high
        delete (transformed as any).low
        delete (transformed as any).close
      }

      yield transformed
      item = await data.next()
    }
  }

  private calculatePrice(data: OhlcvDto): number {
    const { open, high, low, close } = data

    switch (this.calculation) {
      case 'hlc3':
        // High + Low + Close / 3
        return (high + low + close) / 3

      case 'ohlc4':
        // Open + High + Low + Close / 4
        return (open + high + low + close) / 4

      case 'typical':
        // Same as HLC3 - typical price
        return (high + low + close) / 3

      case 'weighted':
        // Weighted Close: (High + Low + Close + Close) / 4
        // Close is weighted double
        return (high + low + close + close) / 4

      case 'median':
        // Median of High and Low
        return (high + low) / 2

      case 'custom':
        // Evaluate custom formula
        return this.evaluateCustomFormula(data)

      default:
        throw new Error(`Unknown calculation type: ${this.calculation}`)
    }
  }

  private evaluateCustomFormula(data: OhlcvDto): number {
    if (!this.customFormula) {
      throw new Error('Custom formula not provided')
    }

    // Simple formula evaluation - supports basic arithmetic with OHLCV fields
    // This is a basic implementation - could be enhanced with a proper expression parser
    try {
      // Replace field names with values
      let formula = this.customFormula.toLowerCase()
      formula = formula.replace(/\bopen\b/g, data.open.toString())
      formula = formula.replace(/\bhigh\b/g, data.high.toString())
      formula = formula.replace(/\blow\b/g, data.low.toString())
      formula = formula.replace(/\bclose\b/g, data.close.toString())
      formula = formula.replace(/\bvolume\b/g, data.volume.toString())

      // Basic security check - only allow numbers, operators, and parentheses
      if (!/^[0-9+\-*/().\s]+$/.test(formula)) {
        throw new Error('Invalid characters in formula')
      }

      // Evaluate the formula
      // Note: Using Function constructor for evaluation - in production, consider a safer expression parser
      const result = new Function('return ' + formula)()

      if (typeof result !== 'number' || !isFinite(result)) {
        throw new Error('Formula did not produce a valid number')
      }

      return result
    } catch (error) {
      throw new Error(`Failed to evaluate custom formula: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  private getDefaultOutputField(): string {
    switch (this.calculation) {
      case 'hlc3':
        return 'hlc3'
      case 'ohlc4':
        return 'ohlc4'
      case 'typical':
        return 'typical_price'
      case 'weighted':
        return 'weighted_close'
      case 'median':
        return 'median_price'
      case 'custom':
        return 'custom_price'
      default:
        return 'calculated_price'
    }
  }

  public getOutputFields(): string[] {
    const fields = [this.outputField]
    
    // If not keeping original fields, we're replacing them
    if (!this.keepOriginal) {
      return fields
    }
    
    return fields
  }

  public getRequiredFields(): string[] {
    // Always need OHLC fields for price calculations
    return ['open', 'high', 'low', 'close']
  }

  public withParams(params: Partial<PriceCalcParams>): Transform<PriceCalcParams> {
    return new PriceCalculations({ ...this.params, ...params })
  }
}