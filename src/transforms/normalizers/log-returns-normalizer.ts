import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Schema for individual log returns configuration
 * @property {string} in - Input column name (price series)
 * @property {string} out - Output column name for log returns
 * @property {'ln' | 'log10'} [base] - Logarithm base: natural log (ln) or base-10 (log10)
 */
const txSchema = z.object({
  in: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
  out: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
  base: z.enum(['ln', 'log10']).default('ln')
})

/**
 * Main schema for LogReturnsNormalizer transform
 * @property {string} [description] - Optional description of the transform
 * @property {object|array} tx - Single log returns config or array of configs
 *
 * @example
 * // Single series with natural log
 * { tx: { in: "close", out: "log_returns" } }
 *
 * @example
 * // Multiple series with different bases
 * {
 *   tx: [
 *     { in: "close", out: "log_returns", base: "ln" },
 *     { in: "volume", out: "log_vol_change", base: "log10" },
 *     { in: "high", out: "high_log_ret" }
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

export interface LogReturnsParams
  extends z.infer<typeof schema>,
          BaseTransformParams {
}

interface State {
  readonly inIndex: number
  readonly logFn: (x: number) => number
  prevValue: number
}

/**
 * Log Returns Normalizer
 *
 * Transforms price series into logarithmic returns, which are preferred in quantitative
 * finance for their mathematical properties and statistical characteristics.
 *
 * **Formula**: log(Pt / Pt-1) = log(Pt) - log(Pt-1)
 *
 * **Advantages of log returns**:
 * - Time-additive: Multi-period returns are sum of single-period returns
 * - Symmetric: +10% and -10% have equal magnitude
 * - Normally distributed (approximately) for many assets
 * - No lower bound constraint (unlike simple returns which can't go below -100%)
 *
 * **Applications**:
 * - Statistical modeling and forecasting
 * - Risk management (VaR, volatility)
 * - Portfolio optimization
 * - Machine learning features
 *
 * @example
 * ```typescript
 * const logReturns = new LogReturnsNormalizer({
 *   tx: [
 *     { in: "close", out: "returns" },
 *     { in: "adj_close", out: "adj_returns" },
 *     { in: "vwap", out: "vwap_returns", base: "log10" }
 *   ]
 * }, inputBuffer)
 * ```
 *
 * @note First row always returns 0 as there's no previous value
 * @note Returns 0 for invalid inputs (negative/zero prices)
 */
export class LogReturnsNormalizer extends BaseTransform<LogReturnsParams> {
  // State for each output
  private readonly _state = new Map<number, State>()
  private _totalRowsProcessed = 0

  constructor(config: LogReturnsParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'logReturns',
      'LogReturns',
      config.description || 'Log Returns Normalizer',
      parsed,
      inputSlice
    )

    // Initialize each log returns calculation
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    for (const params of tx) {
      // Get input column index
      const inIndex = this.inputSlice.getColumn(params.in)?.index
      if (typeof inIndex !== 'number') {
        throw new Error(
          `Input column '${params.in}' not found in input buffer.`
        )
      }

      // Ensure output column exists
      const outIndex = this.outputBuffer.ensureColumn(params.out)

      // Choose log function based on base
      const logFn = params.base === 'log10' ? Math.log10 : Math.log

      // Store state
      this._state.set(outIndex, {
        inIndex,
        logFn,
        prevValue: 0
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
        const currentValue = this.outputBuffer.getValue(bufferIndex, state.inIndex) || 0

        let logReturn = 0

        // Calculate log return if we have valid previous and current values
        if (
          this._totalRowsProcessed > 1 &&
          state.prevValue > 0 &&
          currentValue > 0
        ) {
          logReturn = state.logFn(currentValue / state.prevValue)
        }

        // Store current value for next iteration
        state.prevValue = currentValue

        // Update the output value
        this.outputBuffer.updateValue(bufferIndex, outIndex, logReturn)

        // Track first row with valid output (in absolute buffer coordinates)
        if (firstValidRow === -1) {
          firstValidRow = bufferIndex
        }
      }
    }

    this._isReady ||= this._totalRowsProcessed > 1

    // Return the range of rows that have valid log returns (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
