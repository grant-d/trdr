import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Schema for individual SMA configuration
 * @property {string} in - Input column name to calculate SMA from
 * @property {string} out - Output column name for the SMA values
 * @property {number} window - Window size (number of periods) for the moving average
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
 * Main schema for SimpleMovingAverage transform
 * @property {string} [description] - Optional description of the transform
 * @property {object|array} tx - Single SMA config or array of configs
 *
 * @example
 * // Single SMA
 * { tx: { in: 'close', out: 'sma_20', window: 20 } }
 *
 * @example
 * // Multiple SMAs
 * {
 *   tx: [
 *     { in: 'close', out: 'sma_20', window: 20 },
 *     { in: 'close', out: 'sma_50', window: 50 },
 *     { in: 'volume', out: 'sma_vol', window: 10 }
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
      const smaNames = tx.map((o) => o.out)
      return smaNames.length === new Set(smaNames).size
    },
    {
      message: 'Output names (sma) must be unique'
    }
  )

interface SmaParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface State {
  readonly in: number;
  readonly window: number;
  readonly items: number[];
}

/**
 * Simple Moving Average (SMA) indicator
 *
 * Calculates the arithmetic mean of values over a specified window period.
 * The SMA smooths price data by creating a constantly updated average price.
 *
 * **Calculation**: Sum of values over N periods / N
 *
 * @example
 * ```typescript
 * const sma = new SimpleMovingAverage({
 *   tx: [
 *     { in: 'close', out: 'sma_20', window: 20 },
 *     { in: 'close', out: 'sma_50', window: 50 },
 *     { in: 'volume', out: 'vol_sma', window: 10 }
 *   ]
 * }, inputBuffer)
 * ```
 */
export class SimpleMovingAverage extends BaseTransform<SmaParams> {
  // State for each output index: [Out index, {multiplier, previous close, previous ATR}]
  private readonly _state = new Map<number, State>()
  private _totalRowsProcessed = 0

  constructor(config: SmaParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'sma',
      'SMA',
      config.description || 'Simple Moving Average',
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

      // Initialize state for this output
      this._state.set(outIndex, {
        in: inIndex,
        window: params.window,
        items: []
      })
    }
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    let firstValidRow = -1

    // Process rows in the buffer range [from, to)
    for (let bufferIndex = from; bufferIndex < to; bufferIndex++) {
      this._totalRowsProcessed += 1

      for (const [outIndex, state] of Array.from(this._state.entries())) {
        // Get value directly from the buffer using absolute index
        const currentValue = this.outputBuffer.getValue(bufferIndex, state.in) || 0.0

        // Add current value to window
        state.items.push(currentValue)

        // Keep window at max size
        if (state.items.length > state.window) {
          state.items.shift()
        }

        // Only calculate average when we have enough data points
        if (state.items.length >= state.window) {
          const sum = state.items.reduce(
            (acc: number, val: number) => acc + val,
            0
          )
          const avg = sum / state.items.length
          this.outputBuffer.updateValue(bufferIndex, outIndex, avg)

          // Track first row with valid output (in absolute buffer coordinates)
          if (firstValidRow === -1) {
            firstValidRow = bufferIndex
          }
        }
      }
    }

    this._isReady ||= this._totalRowsProcessed >= 2

    // Return the range of rows that have valid SMA values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
