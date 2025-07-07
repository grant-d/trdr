import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { ColumnValue, DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Schema for individual missing value handling configuration
 * @property {string} in - Input column name to check for missing values
 * @property {string} out - Output column name (typically same as input for in-place handling)
 * @property {string} [strategy] - Strategy for handling missing values: 'forward', 'interpolate', 'value' (default: 'forward')
 * @property {number} [fillValue] - Value to use when strategy is 'value' (default: 0)
 */
const txSchema = z
  .object({
    in: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
    out: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
    strategy: z.enum(['forward', 'interpolate', 'value']).default('forward'),
    fillValue: z.number().default(0)
  })
  .refine(
    (data) => {
      if (data.strategy === 'value' && data.fillValue === undefined) {
        return false
      }
      return true
    },
    {
      message: 'fillValue must be provided when using "value" strategy'
    }
  )

/**
 * Main schema for MissingValueHandler transform
 * @property {string} [description] - Optional description of the transform
 * @property {object|array} tx - Single missing value config or array of configs
 *
 * @example
 * // Single column with forward fill
 * { tx: { in: "close", out: "close", strategy: "forward" } }
 *
 * @example
 * // Multiple columns with different strategies
 * {
 *   tx: [
 *     { in: "close", out: "close", strategy: "forward" },
 *     { in: "volume", out: "volume", strategy: "value", fillValue: 0 },
 *     { in: "high", out: "high", strategy: "interpolate" }
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
      const outNames = tx.map((o) => o.out)
      return outNames.length === new Set(outNames).size
    },
    {
      message: 'Output names must be unique'
    }
  )

export interface ImputeParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface State {
  readonly inIndex: number;
  readonly strategy: 'forward' | 'interpolate' | 'value';
  readonly fillValue: number;
  lastValidValue: number;
}

/**
 * Missing Value Handler
 *
 * Handles missing, undefined, or NaN values in the data stream using various strategies.
 * Essential for maintaining data integrity in real-time processing pipelines.
 *
 * **Strategies**:
 * - **forward**: Fill missing values with the last valid value (forward fill)
 * - **interpolate**: Linear interpolation between valid values (falls back to forward in streaming)
 * - **value**: Fill with a constant value
 *
 * **Use Cases**:
 * - Data cleaning in real-time streams
 * - Handling gaps in market data
 * - Preprocessing for ML models that can't handle NaN values
 * - Ensuring technical indicators receive valid inputs
 *
 * @example
 * ```typescript
 * const missingHandler = new MissingValueHandler({
 *   tx: [
 *     // Forward fill price data to maintain continuity
 *     { in: "close", out: "close", strategy: "forward" },
 *     // Fill missing volume with 0
 *     { in: "volume", out: "volume", strategy: "value", fillValue: 0 },
 *     // Forward fill for high/low
 *     { in: "high", out: "high", strategy: "forward" },
 *     { in: "low", out: "low", strategy: "forward" }
 *   ]
 * }, inputBuffer)
 * ```
 *
 * @note Always ready (no warmup period required)
 * @note Interpolation strategy falls back to forward fill in streaming mode
 * @note Preserves column types and handles NaN, undefined, and null values
 */
export class ImputeTransform extends BaseTransform<ImputeParams> {
  // State for each output
  private readonly _state = new Map<number, State>()
  private _totalRowsProcessed = 0

  constructor(config: ImputeParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'impute',
      'MissingValueHandler',
      config.description || 'Missing Value Handler',
      parsed,
      inputSlice
    )

    // Initialize each missing value handler
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    for (const params of tx) {
      // Get input column index
      const inIndex = inputSlice.getColumn(params.in)?.index
      if (typeof inIndex !== 'number') {
        throw new Error(
          `Input column '${params.in}' not found in input buffer.`
        )
      }

      // Ensure output column exists
      const outIndex = this.outputBuffer.ensureColumn(params.out)

      // Store state
      this._state.set(outIndex, {
        inIndex,
        strategy: params.strategy,
        fillValue: params.fillValue,
        lastValidValue: params.fillValue
      })
    }
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    let firstValidRow = -1

    // Process rows in the buffer range [from, to)
    for (let bufferIndex = from; bufferIndex < to; bufferIndex++) {
      this._totalRowsProcessed += 1

      for (const [outIndex, state] of this._state.entries()) {
        const currentValue = this.outputBuffer.getValue(bufferIndex, state.inIndex)

        if (this.isMissingValue(currentValue)) {
          let replacementValue: number

          switch (state.strategy) {
            case 'forward': {
              replacementValue = state.lastValidValue
              break
            }

            case 'value': {
              replacementValue = state.fillValue
              break
            }

            case 'interpolate': {
              // In streaming mode, fall back to forward fill
              replacementValue = state.lastValidValue
              break
            }

            default:
              replacementValue = state.fillValue
          }

          this.outputBuffer.updateValue(bufferIndex, outIndex, replacementValue)

          // Track first row with valid output (in absolute buffer coordinates)
          if (firstValidRow === -1) {
            firstValidRow = bufferIndex
          }
        } else if (typeof currentValue === 'number' && !isNaN(currentValue)) {
          // Update last valid value
          state.lastValidValue = currentValue

          // If in-place modification (in === out), no need to update
          // If different columns, copy the valid value
          if (state.inIndex !== outIndex) {
            this.outputBuffer.updateValue(bufferIndex, outIndex, currentValue)
          }

          // Track first row with valid output (in absolute buffer coordinates)
          if (firstValidRow === -1) {
            firstValidRow = bufferIndex
          }
        }
      }
    }

    // Always ready - no warmup period required
    this._isReady = true

    // Return the range of rows that have valid imputed values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }

  private isMissingValue(value: ColumnValue | undefined): boolean {
    return (
      value === undefined ||
      value === null ||
      (typeof value === 'number' && isNaN(value))
    )
  }
}
