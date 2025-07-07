import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Schema for individual EMA configuration
 * @property {string} in - Input column name to calculate EMA from
 * @property {string} out - Output column name for the EMA values
 * @property {number} window - Window size (number of periods) for the exponential average
 */
const txSchema = z.object({
  in: z
    .string()
    .regex(/^[a-zA-Z0-9_]{1,20}$/)
    .default('close'),
  out: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
  window: z.number().min(2).max(9_999)
})

/**
 * Main schema for ExponentialMovingAverage transform
 * @property {string} [description] - Optional description of the transform
 * @property {object|array} tx - Single EMA config or array of configs
 *
 * @example
 * // Single EMA
 * { tx: { in: 'close', out: 'ema_12', window: 12 } }
 *
 * @example
 * // Multiple EMAs
 * {
 *   tx: [
 *     { in: 'close', out: 'ema_12', window: 12 },
 *     { in: 'close', out: 'ema_26', window: 26 },
 *     { in: 'high', out: 'ema_high', window: 20 }
 *   ]
 * }
 */
const schema = z
  .object({
    description: z.string().optional(),
    tx: z.union([txSchema, z.array(txSchema)])
  })
  .refine(
    (data) => {
      // Ensure no duplicate output names
      const tx = Array.isArray(data.tx) ? data.tx : [data.tx]
      const emaNames = tx.map((o) => o.out)
      return emaNames.length === new Set(emaNames).size
    },
    {
      message: 'Output names (ema) must be unique'
    }
  )

export interface EmaParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface State {
  readonly in: number;
  readonly window: number;
  readonly mult: number;
  ema: number;
  sum: number;
}

/**
 * Exponential Moving Average (EMA) indicator
 *
 * Calculates an exponentially weighted moving average that gives more weight to recent prices.
 * Uses SMA for the initial value, then applies exponential smoothing formula.
 *
 * **Multiplier**: 2 / (period + 1)
 * **Formula**: EMA = (Current × Multiplier) + (Previous EMA × (1 - Multiplier))
 *
 * More responsive to recent price changes compared to SMA.
 *
 * @example
 * ```typescript
 * const ema = new ExponentialMovingAverage({
 *   tx: [
 *     { in: 'close', out: 'ema_12', window: 12 },
 *     { in: 'close', out: 'ema_26', window: 26 },
 *     { in: 'close', out: 'ema_200', window: 200 }
 *   ]
 * }, inputBuffer)
 * ```
 */
export class ExponentialMovingAverage extends BaseTransform<EmaParams> {
  // State for each output index: [Out index, state]
  private readonly _state = new Map<number, State>()
  private _totalRowsProcessed = 0

  constructor(config: EmaParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'ema',
      'EMA',
      config.description || 'Exponential Moving Average',
      parsed,
      inputSlice
    )

    // Initialize each output column
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    for (const params of tx) {
      // Ensure input column exists
      const inField = params.in
      const inIndex = inputSlice.getColumn(inField)?.index
      if (typeof inIndex !== 'number')
        throw new Error(`Input column '${inField}' not found in input buffer.`)

      // Ensure output column exists (whether new or existing)
      const outIndex: number = this.outputBuffer.ensureColumn(params.out)

      // EMA multiplier = 2 / (period + 1)
      this._state.set(outIndex, {
        in: inIndex,
        window: params.window,
        mult: 2 / (params.window + 1),
        ema: 0,
        sum: 0
      })
    }
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    let firstValidRow = -1

    // Process rows in the buffer range [from, to)
    for (let bufferIndex = from; bufferIndex < to; bufferIndex++) {
      this._totalRowsProcessed += 1

      for (const [outIndex, state] of this._state.entries()) {
        const currentValue = this.outputBuffer.getValue(bufferIndex, state.in) || 0.0

        let ema: number

        // Calculate initial EMA using SMA
        if (this._totalRowsProcessed <= state.window) {
          // Store SUM for first windowSize rows
          state.sum += currentValue

          // But return average for first windowSize rows
          ema = state.sum / this._totalRowsProcessed

          // Store the average once we have enough data
          if (this._totalRowsProcessed === state.window) {
            state.ema = ema
          }
        }
        // EMA = (currentValue * multiplier) + (previousEMA * (1 - multiplier))
        else {
          state.ema = currentValue * state.mult + state.ema * (1 - state.mult)
          ema = state.ema
        }

        this.outputBuffer.updateValue(bufferIndex, outIndex, ema)

        // Track first valid row (in absolute buffer coordinates)
        if (firstValidRow === -1) {
          firstValidRow = bufferIndex
        }
      }
    }

    this._isReady ||= this._totalRowsProcessed >= 2

    // Return the range of rows that have valid EMA values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
