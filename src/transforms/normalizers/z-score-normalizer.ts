import { z } from 'zod'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Schema for individual z-score normalization configuration
 * @property {string} in - Input column name
 * @property {string} out - Output column name for z-scores
 * @property {number} window - Window size for calculating mean and standard deviation
 */
const txSchema = z.object({
  in: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
  out: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
  window: z.number().min(2)
})

/**
 * Main schema for ZScoreNormalizer transform
 * @property {string} [description] - Optional description of the transform
 * @property {object|array} tx - Single z-score config or array of configs
 *
 * @example
 * // Single series standardization
 * { tx: { in: "price", out: "price_zscore", window: 20 } }
 *
 * @example
 * // Multiple series standardization
 * {
 *   tx: [
 *     { in: "returns", out: "returns_z", window: 50 },
 *     { in: "volume", out: "volume_z", window: 100 },
 *     { in: "volatility", out: "vol_z", window: 20 }
 *   ]
 * }
 */
const schema = z
  .object({
    description: z.string().optional(),
    tx: z.union([txSchema, z.array(txSchema)])
  })
  .transform((data) => ({
    ...data,
    tx: Array.isArray(data.tx) ? data.tx : [data.tx]
  }))
  .refine(
    (data) => {
      // Ensure no duplicate output names
      const outNames = data.tx.map((o) => o.out)
      return outNames.length === new Set(outNames).size
    },
    {
      message: 'Output names must be unique'
    }
  )

export interface ZScoreParams extends BaseTransformParams {
  description?: string
  tx: z.infer<typeof txSchema> | z.infer<typeof txSchema>[]
}

interface State {
  readonly inIndex: number;
  readonly windowSize: number;
  window: number[];
}

/**
 * Z-Score Normalizer
 *
 * Standardizes values by removing the mean and scaling to unit variance within
 * a rolling window. Also known as standardization or standard score normalization.
 *
 * **Formula**:
 * Z = (X - μ) / σ
 *
 * Where:
 * - X is the current value
 * - μ is the mean of values in the window
 * - σ is the standard deviation of values in the window
 *
 * **Properties**:
 * - Mean of 0, standard deviation of 1 (within each window)
 * - Unbounded output range (typically -4 to +4 for normal data)
 * - Less sensitive to outliers than min-max normalization
 * - Assumes approximately normal distribution
 *
 * **Statistical Interpretation**:
 * - Z = 0: Value equals the window mean
 * - Z = 1: Value is one standard deviation above mean
 * - Z = -2: Value is two standard deviations below mean
 * - |Z| > 3: Potential outlier (rare for normal distributions)
 *
 * **Applications**:
 * - Statistical analysis and hypothesis testing
 * - Anomaly detection (high |Z| indicates outliers)
 * - Feature preprocessing for ML algorithms
 * - Cross-sectional analysis across different scales
 * - Mean-reversion strategies
 * - Risk-adjusted performance metrics
 *
 * @example
 * ```typescript
 * const zScoreNorm = new ZScoreNormalizer({
 *   tx: [
 *     // Standardize returns for statistical analysis
 *     { in: "returns", out: "returns_z", window: 50 },
 *     // Detect volume anomalies
 *     { in: "volume", out: "volume_z", window: 100 },
 *     // Normalize price for mean-reversion signals
 *     { in: "close", out: "price_z", window: 20 }
 *   ]
 * }, inputBuffer)
 * ```
 *
 * @note Returns 0 when standard deviation is 0 (all values identical)
 * @note Window size affects the stability of mean/std estimates
 * @note Suitable for normally distributed data, may distort other distributions
 */
export class ZScoreNormalizer extends BaseTransform<ZScoreParams> {
  // State for each output
  private readonly _state = new Map<number, State>()
  private _totalRowsProcessed = 0

  constructor(config: ZScoreParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'zScore',
      'ZScore',
      config.description || 'Z-Score Normalizer',
      parsed,
      inputSlice
    )

    // Initialize each z-score calculation
    for (const params of parsed.tx) {
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

        // Calculate statistics using the window
        const sum = state.window.reduce((acc: number, val: number) => acc + val, 0)
        const mean = sum / state.window.length

        const sumSquares = state.window.reduce(
          (acc: number, val: number) => acc + val * val,
          0
        )
        const variance = sumSquares / state.window.length - mean * mean
        const std = Math.sqrt(Math.max(0, variance)) // Ensure non-negative

        // Calculate z-score
        const zScore = std > 0 ? (currentValue - mean) / std : 0

        this.outputBuffer.updateValue(bufferIndex, outIndex, zScore)

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

    // Return the range of rows that have valid z-scores (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
