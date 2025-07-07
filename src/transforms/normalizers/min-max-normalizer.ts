import { z } from 'zod'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Schema for individual min-max normalization configuration
 * @property {string} in - Input column name
 * @property {string} out - Output column name for normalized values
 * @property {number} window - Window size for calculating min/max
 * @property {number} [min] - Target minimum value (default: 0)
 * @property {number} [max] - Target maximum value (default: 1)
 */
const txSchema = z
  .object({
    in: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
    out: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
    window: z.number().min(2),
    min: z.number().default(0),
    max: z.number().default(1)
  })
  .refine((data) => data.min < data.max, {
    message: 'Target max must be greater than target min'
  })

/**
 * Main schema for MinMaxNormalizer transform
 * @property {string} [description] - Optional description of the transform
 * @property {object|array} tx - Single normalization config or array of configs
 *
 * @example
 * // Single series normalization to [0, 1]
 * { tx: { in: "rsi", out: "rsi_norm", window: 100 } }
 *
 * @example
 * // Multiple series with custom ranges
 * {
 *   tx: [
 *     { in: "price", out: "price_norm", window: 50 },
 *     { in: "volume", out: "vol_norm", window: 20, min: -1, max: 1 },
 *     { in: "volatility", out: "vol_scaled", window: 100, min: 0, max: 10 }
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
      const outNames = Array.isArray(data.tx) ? data.tx.map((o) => o.out) : [data.tx.out]
      return outNames.length === new Set(outNames).size
    },
    {
      message: 'Output names must be unique'
    }
  )

export interface MinMaxParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface State {
  readonly inIndex: number
  readonly targetMin: number
  readonly targetMax: number
  readonly targetRange: number
  readonly windowSize: number
  window: number[]
}

/**
 * Min-Max Normalizer
 *
 * Scales values to a specified range [min, max] based on the minimum and maximum
 * values observed within a rolling window. This is a feature scaling technique
 * commonly used in machine learning and signal processing.
 *
 * **Formula**:
 * X_norm = min + (X - X_min) / (X_max - X_min) * (max - min)
 *
 * Where:
 * - X is the current value
 * - X_min is the minimum value in the window
 * - X_max is the maximum value in the window
 * - min, max are the target range bounds
 *
 * **Properties**:
 * - Preserves the shape and distribution of data
 * - Bounded output range [min, max]
 * - Sensitive to outliers within the window
 * - Zero information loss (reversible transformation)
 *
 * **Applications**:
 * - Neural network input preprocessing
 * - Feature scaling for ML models
 * - Indicator normalization for consistent ranges
 * - Visualization scaling
 * - Cross-asset comparison
 *
 * @example
 * ```typescript
 * const minMaxNorm = new MinMaxNormalizer({
 *   tx: [
 *     // Normalize RSI to standard [0, 1] range
 *     { in: "rsi", out: "rsi_norm", window: 50 },
 *     // Scale volatility to percentage [0, 100]
 *     { in: "atr", out: "atr_pct", window: 20, min: 0, max: 100 },
 *     // Normalize momentum to [-1, 1] for ML model
 *     { in: "momentum", out: "mom_scaled", window: 100, min: -1, max: 1 }
 *   ]
 * }, inputBuffer)
 * ```
 *
 * @note Returns midpoint of target range when all values in window are identical
 * @note Window size affects responsiveness to changing data ranges
 * @note Not suitable for unbounded data without careful window selection
 */
export class MinMaxNormalizer extends BaseTransform<MinMaxParams> {
  // State for each output
  private readonly _state = new Map<number, State>()
  private _totalRowsProcessed = 0

  constructor(config: MinMaxParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'minMax',
      'MinMax',
      config.description || 'Min-Max Normalizer',
      parsed,
      inputSlice
    )

    // Initialize each normalization
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

      // Calculate target range
      const targetRange = params.max - params.min

      // Store state
      this._state.set(outIndex, {
        inIndex,
        targetMin: params.min,
        targetMax: params.max,
        targetRange,
        windowSize: params.window,
        window: []
      })
    }
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    let firstValidRow = -1

    // Process rows in the buffer range [from, to)
    for (let bufferIndex = from; bufferIndex < to; bufferIndex++) {
      this._totalRowsProcessed += 1

      for (const [outIndex, state] of Array.from(this._state.entries())) {
        const currentValue = Number(
          this.outputBuffer.getValue(bufferIndex, state.inIndex) || 0
        )

        // Add current value to window
        state.window.push(currentValue)

        // Keep window at max size
        if (state.window.length > state.windowSize) {
          state.window.shift()
        }

        // Find min and max in the window
        let min = Number.POSITIVE_INFINITY
        let max = Number.NEGATIVE_INFINITY

        for (const value of state.window) {
          min = Math.min(min, value)
          max = Math.max(max, value)
        }

        const range = max - min

        let normalizedValue: number
        if (range > 0) {
          // Scale to [0, 1] then to target range
          const scaled01 = (currentValue - min) / range
          normalizedValue = state.targetMin + scaled01 * state.targetRange
        } else {
          // All values in window are the same
          normalizedValue = (state.targetMin + state.targetMax) / 2
        }

        this.outputBuffer.updateValue(bufferIndex, outIndex, normalizedValue)

        // Track first row with valid output (in absolute buffer coordinates)
        if (firstValidRow === -1) {
          firstValidRow = bufferIndex
        }
      }
    }

    // Ready when we have at least one full window
    this._isReady ||=
      this._totalRowsProcessed >=
      Math.min(...Array.from(this._state.values()).map((s) => s.windowSize))

    // Return the range of rows that have valid normalized values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
