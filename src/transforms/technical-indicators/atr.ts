import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Schema for individual ATR configuration
 * @property {string} out - Output column name for the ATR values
 * @property {number} window - Window size (period) for the ATR calculation
 */
const txSchema = z.object({
  out: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
  window: z.number().min(2).max(9_999).default(14)
})

/**
 * Main schema for AverageTrueRange transform
 * @property {string} [description] - Optional description of the transform
 * @property {object} in - Input column names for OHLC data
 * @property {string} in.high - High price column name
 * @property {string} in.low - Low price column name
 * @property {string} in.close - Close price column name
 * @property {object|array} tx - Single ATR config or array of configs
 *
 * @example
 * // Single ATR
 * {
 *   in: { high: 'h', low: 'l', close: 'c' },
 *   tx: { out: 'atr_14', window: 14 }
 * }
 *
 * @example
 * // Multiple ATRs with different periods
 * {
 *   tx: [
 *     { out: 'atr_14', window: 14 },
 *     { out: 'atr_20', window: 20 },
 *     { out: 'atr_50', window: 50 }
 *   ]
 * }
 */
const schema = z
  .object({
    description: z.string().optional(),
    in: z.object({
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
    }),
    tx: z.union([txSchema, z.array(txSchema)])
  })
  .refine(
    (data) => {
      // Ensure no duplicate output names
      const atrNames = Array.isArray(data.tx)
        ? data.tx.map((o) => o.out)
        : [data.tx.out]
      return atrNames.length === new Set(atrNames).size
    },
    {
      message: 'Output names (atr) must be unique'
    }
  )

export interface AtrParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface State {
  readonly window: number;
  readonly mult: number;
  close: number;
  atr: number;
}

/**
 * Average True Range (ATR) indicator
 *
 * Measures market volatility by calculating the average of true ranges over a specified period.
 * Uses Wilder's smoothing method (equivalent to EMA with multiplier = 1/period).
 *
 * **True Range (TR)** = Max of:
 * - Current High - Current Low
 * - |Current High - Previous Close|
 * - |Current Low - Previous Close|
 *
 * **ATR Calculation**:
 * - First N periods: Simple average of TR values
 * - Subsequent periods: (Previous ATR × (N-1) + Current TR) / N
 *
 * @example
 * ```typescript
 * const atr = new AverageTrueRange({
 *   in: { high: 'high', low: 'low', close: 'close' },
 *   tx: [
 *     { out: 'atr_14', window: 14 },
 *     { out: 'atr_20', window: 20 }
 *   ]
 * }, inputBuffer)
 * ```
 */
export class AverageTrueRange extends BaseTransform<AtrParams> {
  // Input column indices (shared across transforms)
  private readonly _inputs: {
    readonly hi: number;
    readonly lo: number;
    readonly cl: number;
  }

  // State for each output index: [Out index, state]
  private readonly _state = new Map<number, State>() // [Out index, multiplier]

  private _totalRowsProcessed = 0.0
  private readonly _maxWindowSize: number = 0

  constructor(config: AtrParams, inputSlice: DataSlice) {
    // Validate config
    const parsed: AtrParams = schema.parse(config)

    // Base class constructor
    super(
      'atr',
      'ATR',
      config.description || 'Average True Range',
      parsed,
      inputSlice
    )

    // All transforms share same input columns
    const hi = inputSlice.getColumn(parsed.in.high)?.index
    if (typeof hi !== 'number')
      throw new Error(
        `Input column '${parsed.in.high}' not found in input buffer.`
      )
    const lo = inputSlice.getColumn(parsed.in.low)?.index
    if (typeof lo !== 'number')
      throw new Error(
        `Input column '${parsed.in.low}' not found in input buffer.`
      )
    const cl = inputSlice.getColumn(parsed.in.close)?.index
    if (typeof cl !== 'number')
      throw new Error(
        `Input column '${parsed.in.close}' not found in input buffer.`
      )

    this._inputs = { hi, lo, cl }

    // Initialize each output column
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    for (const params of tx) {
      // Ensure output column exists (whether new or existing)
      const outIndex: number = this.outputBuffer.ensureColumn(params.out)

      // ATR uses Wilder's smoothing (same as EMA with multiplier = 1/period)
      this._state.set(outIndex, {
        window: params.window,
        mult: 1.0 / params.window,
        close: 0.0,
        atr: 0.0
      })

      this._maxWindowSize = Math.max(this._maxWindowSize, params.window)
    }
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    let firstValidRow = -1

    // Process rows in the buffer range [from, to)
    for (let bufferIndex = from; bufferIndex < to; bufferIndex++) {
      this._totalRowsProcessed += 1

      // Transforms share same input column (indices)
      const hi = this.outputBuffer.getValue(bufferIndex, this._inputs.hi) || 0.0
      const lo = this.outputBuffer.getValue(bufferIndex, this._inputs.lo) || 0.0
      const cl = this.outputBuffer.getValue(bufferIndex, this._inputs.cl) || 0.0
      const hi_lo = hi - lo

      // Update state for this output index
      for (const [outIndex, state] of this._state.entries()) {
        // First row: TR = High - Low
        let tr: number = hi_lo

        // Next rows: TR = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
        if (this._totalRowsProcessed > 1) {
          tr = Math.max(
            hi_lo, // High - Low
            Math.abs(hi - state.close),
            Math.abs(lo - state.close)
          )
        }

        // Store current close for next iteration
        state.close = cl

        // Calculate initial ATR using SMA
        let atr: number
        if (this._totalRowsProcessed <= state.window) {
          // Store SUM for first windowSize rows
          state.atr += tr

          // But return average for first windowSize rows
          atr = state.atr / this._totalRowsProcessed

          // Store the average once we have enough data
          if (this._totalRowsProcessed == state.window) {
            state.atr /= state.window
            atr = state.atr
          }
        }
          // ATR = (previousATR * (period - 1) + currentTR) / period
        // Which is equivalent to: ATR = currentTR * multiplier + previousATR * (1 - multiplier)
        else {
          state.atr = tr * state.mult + state.atr * (1 - state.mult)
          atr = state.atr
        }

        // Update state for this output index
        this._state.set(outIndex, state)

        // Set output value
        this.outputBuffer.updateValue(bufferIndex, outIndex, atr)

        // Track first valid row (in absolute buffer coordinates)
        if (firstValidRow === -1) {
          firstValidRow = bufferIndex
        }
      }
    }

    this._isReady ||= this._totalRowsProcessed >= this._maxWindowSize

    // Return the range of rows that have valid ATR values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
