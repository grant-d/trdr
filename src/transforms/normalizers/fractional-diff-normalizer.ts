import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Schema for individual fractional differentiation configuration
 * @property {string} in - Input column name (time series to differentiate)
 * @property {string} out - Output column name for fractionally differentiated values
 * @property {number} d - Differencing parameter (0 < d < 2, typically 0.3-0.5)
 * @property {number} [maxWeights] - Maximum number of weights to calculate (default: 100)
 * @property {number} [minWeight] - Minimum weight threshold to stop calculation (default: 1e-5)
 */
const txSchema = z.object({
  in: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
  out: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
  d: z.number().min(0).max(2),
  maxWeights: z.number().min(10).max(1000).default(100),
  minWeight: z.number().min(1e-10).max(0.1).default(1e-5)
})

/**
 * Main schema for FractionalDiffNormalizer transform
 * @property {string} [description] - Optional description of the transform
 * @property {object|array} tx - Single fractional diff config or array of configs
 *
 * @example
 * // Single series with standard parameters
 * { tx: { in: "close", out: "close_ffd", d: 0.4 } }
 *
 * @example
 * // Multiple series with different parameters
 * {
 *   tx: [
 *     { in: "close", out: "close_ffd", d: 0.3 },
 *     { in: "volume", out: "volume_ffd", d: 0.5, maxWeights: 200 },
 *     { in: "returns", out: "returns_ffd", d: 0.25, minWeight: 1e-6 }
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
      const outNames = Array.isArray(data.tx)
        ? data.tx.map((o) => o.out)
        : [data.tx.out]
      return outNames.length === new Set(outNames).size
    },
    {
      message: 'Output names must be unique'
    }
  )

export interface FractionalDiffParams
  extends z.infer<typeof schema>,
          BaseTransformParams {
}

interface State {
  readonly inIndex: number
  readonly weights: number[]
  readonly maxLookback: number
  history: number[]
}

/**
 * Fractional Differentiation Normalizer
 *
 * Applies fractional differentiation to time series data, achieving stationarity
 * while preserving maximum memory (autocorrelation structure). This technique
 * is crucial for financial ML where both stationarity and predictive power are needed.
 *
 * **Mathematical Foundation**:
 * The fractional differencing operator (1-L)^d where:
 * - L is the lag operator
 * - d is the differencing parameter (0 < d < 1 for fractional)
 * - d = 0: No differencing (original series)
 * - d = 1: First difference (removes all memory)
 * - 0 < d < 1: Fractional difference (balance between stationarity and memory)
 *
 * **Weight Calculation** (Binomial series expansion):
 * - w₀ = 1
 * - wₖ = -wₖ₋₁ × (d - k + 1) / k, for k ≥ 1
 *
 * **Key Properties**:
 * - Weights decay hyperbolically (slower than exponential)
 * - Sum of absolute weights converges for d < 1
 * - Preserves long-range dependencies unlike integer differencing
 * - Output is weighted sum of historical values
 *
 * **Applications**:
 * - **Financial ML**: Creating stationary features without losing predictive power
 * - **Cointegration**: Testing and modeling long-run equilibrium relationships
 * - **Risk Management**: Preserving volatility clustering while achieving stationarity
 * - **Pairs Trading**: Constructing mean-reverting spreads
 * - **Forecasting**: Balancing bias-variance tradeoff in time series models
 *
 * **Parameter Selection**:
 * - d ∈ [0.2, 0.4]: Light differencing, maximum memory preservation
 * - d ∈ [0.4, 0.6]: Balanced stationarity and memory
 * - d ∈ [0.6, 0.8]: Strong differencing, approaching first difference
 *
 * @example
 * ```typescript
 * const fracDiff = new FractionalDiffNormalizer({
 *   tx: [
 *     // Light differencing for price series
 *     { in: "close", out: "close_ffd", d: 0.3 },
 *     // Stronger differencing for volume
 *     { in: "volume", out: "vol_ffd", d: 0.5 },
 *     // Custom parameters for high-frequency data
 *     { in: "bid", out: "bid_ffd", d: 0.4, maxWeights: 200, minWeight: 1e-6 }
 *   ]
 * }, inputBuffer)
 * ```
 *
 * @note Requires sufficient history (up to maxWeights periods) for full accuracy
 * @note Memory usage scales with maxWeights parameter
 * @note Choose d using ADF test for stationarity vs autocorrelation preservation
 *
 * @reference "Advances in Financial Machine Learning" by Marcos López de Prado
 */
export class FractionalDiffNormalizer extends BaseTransform<FractionalDiffParams> {
  // State for each output
  private readonly _state = new Map<number, State>()
  private _totalRowsProcessed = 0

  constructor(config: FractionalDiffParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'fractionalDiff',
      'FracDiff',
      config.description || 'Fractional Differentiation',
      parsed,
      inputSlice
    )

    // Initialize each fractional diff calculation
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

      // Calculate weights using the expanding window method
      const weights = this.calculateWeights(
        params.d,
        params.maxWeights,
        params.minWeight
      )

      // Store state
      this._state.set(outIndex, {
        inIndex,
        weights,
        maxLookback: weights.length,
        history: []
      })
    }
  }

  /**
   * Calculate fractional differentiation weights using binomial series expansion
   *
   * @param d - Differencing parameter
   * @param maxWeights - Maximum number of weights to calculate
   * @param minWeight - Minimum weight threshold
   * @returns Array of weights in order [w₀, w₁, w₂, ...]
   */
  private calculateWeights(
    d: number,
    maxWeights: number,
    minWeight: number
  ): number[] {
    const weights: number[] = [1]

    for (let k = 1; k < maxWeights; k++) {
      // Recursive formula: wₖ = -wₖ₋₁ × (d - k + 1) / k
      const weight = (-weights[k - 1]! * (d - k + 1)) / k

      // Stop if weight is too small (convergence)
      if (Math.abs(weight) < minWeight) {
        break
      }

      weights.push(weight)
    }

    return weights
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    let firstValidRow = -1

    // Process rows in the buffer range [from, to)
    for (let bufferIndex = from; bufferIndex < to; bufferIndex++) {
      this._totalRowsProcessed += 1

      for (const [outIndex, state] of Array.from(this._state.entries())) {
        const currentValue = this.outputBuffer.getValue(bufferIndex, state.inIndex) || 0

        // Add current value to history
        state.history.push(currentValue)

        // Keep only the necessary history
        if (state.history.length > state.maxLookback) {
          state.history.shift()
        }

        // Calculate fractionally differentiated value
        let diffValue = 0
        const n = Math.min(state.history.length, state.weights.length)

        // Apply weights in reverse order (most recent data first)
        for (let j = 0; j < n; j++) {
          const dataIndex = state.history.length - 1 - j
          const value = state.history[dataIndex]!
          const weight = state.weights[j]!
          diffValue += weight * value
        }

        this.outputBuffer.updateValue(bufferIndex, outIndex, diffValue)

        // Track first row with valid output (in absolute buffer coordinates)
        if (firstValidRow === -1) {
          firstValidRow = bufferIndex
        }
      }
    }

    // Ready immediately as we can start producing output from first value
    this._isReady = true

    // Return the range of rows that have valid fractional diff values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
