import { z } from 'zod/v4'
import type { BaseTransformParams } from '../interfaces'
import type { DataSlice } from '../utils'
import { BaseTransform } from './base-transform'

/**
 * Schema for individual price calculation configuration
 * @property {string} out - Output column name for the calculated price
 * @property {string} calc - Calculation type: 'hlc3', 'ohlc4', 'typical', 'weighted', 'median', 'hl2', 'custom'
 * @property {string} [formula] - Custom formula when calc is 'custom' (e.g., '(high + low + close * 2) / 4')
 */
const txSchema = z
  .object({
    out: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
    calc: z
      .enum(['hlc3', 'ohlc4', 'typical', 'weighted', 'median', 'hl2', 'custom'])
      .default('hlc3'),
    formula: z.string().optional()
  })
  .refine(
    (data) => {
      if (data.calc === 'custom' && !data.formula) {
        return false
      }
      return true
    },
    {
      message: 'Formula is required when calc type is "custom"'
    }
  )

/**
 * Main schema for PriceCalculations transform
 * @property {string} [description] - Optional description of the transform
 * @property {object} in - Input column names (all optional, required based on calculation type)
 * @property {string} [in.open] - Open price column name
 * @property {string} [in.high] - High price column name
 * @property {string} [in.low] - Low price column name
 * @property {string} [in.close] - Close price column name
 * @property {object|array} tx - Single calculation config or array of configs
 *
 * @example
 * // Single calculation
 * { tx: { out: 'typical', calc: 'hlc3' } }
 *
 * @example
 * // Multiple calculations
 * {
 *   tx: [
 *     { out: 'typical', calc: 'hlc3' },
 *     { out: 'weighted', calc: 'weighted' }
 *   ]
 * }
 */
const schema = z
  .object({
    description: z.string().optional(),
    in: z
      .object({
        open: z
          .string()
          .regex(/^[a-zA-Z0-9_]{1,20}$/)
          .default('open'),
        high: z
          .string()
          .regex(/^[a-zA-Z0-9_]{1,20}$/)
          .default('high'),
        low: z
          .string()
          .regex(/^[a-zA-Z0-9_]{1,20}$/)
          .default('low'),
        close: z
          .string()
          .regex(/^[a-zA-Z0-9_]{1,20}$/)
          .default('close')
      })
      .partial(),
    tx: z.union([txSchema, z.array(txSchema)])
  })
  .refine(
    (data) => {
      // Ensure no duplicate output names
      const outNames = Array.isArray(data.tx)
        ? data.tx.map((o) => o.out)
        : [data.tx.out]
      return outNames.length === new Set(outNames).size
    },
    {
      message: 'Output names must be unique'
    }
  )

interface PriceCalcParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface State {
  readonly calc: string;
  readonly formula?: string;
}

/**
 * Price Calculations transform
 *
 * Calculates derived price values from OHLC data using various formulas:
 * - **hlc3/typical**: (High + Low + Close) / 3 - Standard typical price
 * - **ohlc4**: (Open + High + Low + Close) / 4 - Average of all OHLC values
 * - **weighted**: (High + Low + Close + Close) / 4 - Close-weighted price
 * - **median/hl2**: (High + Low) / 2 - Midpoint price
 * - **custom**: User-defined formula using open, high, low, close variables
 *
 * @example
 * ```typescript
 * const priceCalc = new PriceCalculations({
 *   in: { high: 'h', low: 'l', close: 'c' },
 *   tx: [
 *     { out: 'typical', calc: 'hlc3' },
 *     { out: 'midpoint', calc: 'hl2' },
 *     { out: 'custom_weighted', calc: 'custom', formula: '(high * 2 + low + close) / 4' }
 *   ]
 * }, inputSlice)
 * ```
 */
export class PriceCalculations extends BaseTransform<PriceCalcParams> {
  // Input column indices (shared across all outputs)
  private readonly _inputs: {
    readonly open?: number;
    readonly high?: number;
    readonly low?: number;
    readonly close?: number;
  }

  // State for each output
  private readonly _state = new Map<number, State>()
  private _totalRowsProcessed = 0

  constructor(config: PriceCalcParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'priceCalc',
      'PriceCalc',
      config.description || 'Price Calculations',
      parsed,
      inputSlice
    )

    // Get input column indices
    this._inputs = {
      open: parsed.in.open
        ? inputSlice.getColumn(parsed.in.open)?.index
        : undefined,
      high: parsed.in.high
        ? inputSlice.getColumn(parsed.in.high)?.index
        : undefined,
      low: parsed.in.low
        ? inputSlice.getColumn(parsed.in.low)?.index
        : undefined,
      close: parsed.in.close
        ? inputSlice.getColumn(parsed.in.close)?.index
        : undefined
    }

    // Validate required inputs based on calculation types
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    for (const params of tx) {
      const calc = params.calc
      if (calc !== 'custom') {
        // Check required columns based on calculation type
        if (
          (calc === 'hlc3' || calc === 'typical') &&
          (this._inputs.high === undefined ||
            this._inputs.low === undefined ||
            this._inputs.close === undefined)
        ) {
          throw new Error(
            `Calculation '${calc}' requires high, low, and close columns`
          )
        }
        if (
          calc === 'ohlc4' &&
          (this._inputs.open === undefined ||
            this._inputs.high === undefined ||
            this._inputs.low === undefined ||
            this._inputs.close === undefined)
        ) {
          throw new Error(
            `Calculation '${calc}' requires open, high, low, and close columns`
          )
        }
        if (
          calc === 'weighted' &&
          (this._inputs.high === undefined ||
            this._inputs.low === undefined ||
            this._inputs.close === undefined)
        ) {
          throw new Error(
            `Calculation '${calc}' requires high, low, and close columns`
          )
        }
        if (
          (calc === 'median' || calc === 'hl2') &&
          (this._inputs.high === undefined || this._inputs.low === undefined)
        ) {
          throw new Error(
            `Calculation '${calc}' requires high and low columns`
          )
        }
      }

      // Ensure output column exists
      const outIndex = this.outputBuffer.ensureColumn(params.out)

      this._state.set(outIndex, {
        calc: params.calc,
        formula: params.formula
      })
    }
  }

  protected processBatch(): { from: number; to: number } {
    let firstValidRow = -1
    const rowCount = this.inputSlice.length()

    for (let rid = 0; rid < rowCount; rid++) {
      this._totalRowsProcessed += 1

      // Get input values (may be undefined)
      const open =
        this._inputs.open !== undefined
          ? this.inputSlice.getValue(rid, this._inputs.open) || 0.0
          : 0.0
      const high =
        this._inputs.high !== undefined
          ? this.inputSlice.getValue(rid, this._inputs.high) || 0.0
          : 0.0
      const low =
        this._inputs.low !== undefined
          ? this.inputSlice.getValue(rid, this._inputs.low) || 0.0
          : 0.0
      const close =
        this._inputs.close !== undefined
          ? this.inputSlice.getValue(rid, this._inputs.close) || 0.0
          : 0.0

      for (const [outIndex, state] of this._state) {
        let calculatedPrice: number

        switch (state.calc) {
          case 'hlc3':
          case 'typical':
            // High + Low + Close / 3
            calculatedPrice = (high + low + close) / 3
            break

          case 'ohlc4':
            // Open + High + Low + Close / 4
            calculatedPrice = (open + high + low + close) / 4
            break

          case 'weighted':
            // Weighted Close: (High + Low + Close + Close) / 4
            calculatedPrice = (high + low + close + close) / 4
            break

          case 'median':
          case 'hl2':
            // Median of High and Low
            calculatedPrice = (high + low) / 2
            break

          case 'custom':
            // Evaluate custom formula
            calculatedPrice = this.evaluateCustomFormula(
              open,
              high,
              low,
              close,
              state.formula || ''
            )
            break

          default:
            calculatedPrice = close
        }

        this.inputSlice.updateValue(rid, outIndex, calculatedPrice)

        // Track first row with valid output (in absolute buffer coordinates)
        if (firstValidRow === -1) {
          firstValidRow = this.inputSlice.from + rid
        }
      }
    }

    this._isReady ||= this._totalRowsProcessed >= 1

    // Return the range of rows that have valid price calculations (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? this.inputSlice.to : firstValidRow,
      to: this.inputSlice.to
    }
  }

  private evaluateCustomFormula(
    open: number,
    high: number,
    low: number,
    close: number,
    formula: string
  ): number {
    if (!formula) {
      return close
    }

    try {
      // Replace field names with values
      let processedFormula = formula.toLowerCase()
      processedFormula = processedFormula.replace(/\bopen\b/g, open.toString())
      processedFormula = processedFormula.replace(/\bhigh\b/g, high.toString())
      processedFormula = processedFormula.replace(/\blow\b/g, low.toString())
      processedFormula = processedFormula.replace(
        /\bclose\b/g,
        close.toString()
      )

      // Basic security check - only allow numbers, operators, and parentheses
      if (!/^[0-9+\-*/().\s]+$/.test(processedFormula)) {
        throw new Error('Invalid characters in formula')
      }

      // Evaluate the formula
      const result = new Function('return ' + processedFormula)()

      if (typeof result !== 'number' || !isFinite(result)) {
        return close
      }

      return result
    } catch {
      return close
    }
  }
}
