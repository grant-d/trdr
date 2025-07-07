import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * RSI transaction schema
 * @property in - The input column name (e.g., 'close', 'high', 'low') for RSI calculation
 * @property out - The output column name where RSI values will be stored
 * @property window - The number of periods for RSI calculation (typically 14)
 */
const txSchema = z.object({
  in: z
    .string()
    .regex(/^[a-zA-Z0-9_]{1,20}$/)
    .default('close'),
  out: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
  window: z.number().min(2).max(9_999).default(14)
})

/**
 * Relative Strength Index (RSI) configuration
 *
 * RSI is a momentum oscillator that measures the speed and magnitude of price changes.
 * Values range from 0 to 100, typically:
 * - Above 70: Overbought (potential sell signal)
 * - Below 30: Oversold (potential buy signal)
 *
 * Uses Wilder's smoothing method for calculation:
 * RS = Average Gain / Average Loss
 * RSI = 100 - (100 / (1 + RS))
 *
 * @example
 * // Single RSI (14-period on close)
 * {
 *   tx: { in: 'close', out: 'rsi', window: 14 }
 * }
 *
 * @example
 * // Multiple RSI indicators with different periods
 * {
 *   tx: [
 *     { in: 'close', out: 'rsi9', window: 9 },   // Fast RSI
 *     { in: 'close', out: 'rsi14', window: 14 }, // Standard RSI
 *     { in: 'close', out: 'rsi21', window: 21 }  // Slow RSI
 *   ]
 * }
 *
 * @example
 * // RSI on different price points
 * {
 *   tx: [
 *     { in: 'close', out: 'rsiClose', window: 14 },
 *     { in: 'typicalPrice', out: 'rsiTp', window: 14 }
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
      const rsiNames = Array.isArray(data.tx)
        ? data.tx.map((o) => o.out)
        : [data.tx.out]
      return rsiNames.length === new Set(rsiNames).size
    },
    {
      message: 'Output names (rsi) must be unique'
    }
  )

interface RsiParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface State {
  readonly in: number;
  readonly window: number;
  readonly mult: number;
  prevValue: number;
  avgGain: number;
  avgLoss: number;
  gainSum: number;
  lossSum: number;
}

/**
 * Relative Strength Index (RSI) implementation
 *
 * The RSI is a momentum oscillator that measures the speed and magnitude of directional
 * price movements. It compares the magnitude of recent gains to recent losses to determine
 * overbought and oversold conditions.
 *
 * Calculation method:
 * 1. Calculate price changes between consecutive periods
 * 2. Separate gains (positive changes) and losses (negative changes)
 * 3. Calculate initial average gain and loss using SMA for the first 'window' periods
 * 4. Apply Wilder's smoothing for subsequent periods:
 *    - Avg Gain = (Previous Avg Gain × (window-1) + Current Gain) / window
 *    - Avg Loss = (Previous Avg Loss × (window-1) + Current Loss) / window
 * 5. Calculate Relative Strength (RS) = Avg Gain / Avg Loss
 * 6. Calculate RSI = 100 - (100 / (1 + RS))
 *
 * Trading signals:
 * - RSI > 70: Potentially overbought (bearish signal)
 * - RSI < 30: Potentially oversold (bullish signal)
 * - Divergences between price and RSI can signal potential reversals
 * - Centerline (50) crossovers indicate momentum shifts
 *
 * @example
 * // Standard 14-period RSI
 * const rsi = new RelativeStrengthIndex({
 *   tx: { in: 'close', out: 'rsi14', window: 14 }
 * }, inputBuffer)
 */
export class RelativeStrengthIndex extends BaseTransform<RsiParams> {
  // State for each output index
  private readonly _state = new Map<number, State>()
  private _totalRowsProcessed = 0

  constructor(config: RsiParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'rsi',
      'RSI',
      config.description || 'Relative Strength Index',
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

      // RSI uses Wilder's smoothing (same as EMA with multiplier = 1/period)
      this._state.set(outIndex, {
        in: inIndex,
        window: params.window,
        mult: 1 / params.window,
        prevValue: 0,
        avgGain: 0,
        avgLoss: 0,
        gainSum: 0,
        lossSum: 0
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

        // Need at least 2 values to calculate change
        if (this._totalRowsProcessed === 1) {
          state.prevValue = currentValue
          // Don't set a value yet - we need more data
          continue
        }

        // Calculate change
        const change = currentValue - state.prevValue
        const gain = change > 0 ? change : 0
        const loss = change < 0 ? -change : 0

        // Store current value for next iteration
        state.prevValue = currentValue

        // Calculate initial averages using SMA
        if (this._totalRowsProcessed <= state.window + 1) {
          state.gainSum += gain
          state.lossSum += loss

          // Only calculate and set RSI once we have enough data
          if (this._totalRowsProcessed >= state.window + 1) {
            // Calculate average
            const divisor = state.window // Number of changes for the window
            const avgGain = state.gainSum / divisor
            const avgLoss = state.lossSum / divisor

            // Store the average once we have enough data
            if (this._totalRowsProcessed === state.window + 1) {
              state.avgGain = avgGain
              state.avgLoss = avgLoss
            }

            // Calculate RSI
            let rsi: number
            if (avgLoss === 0) {
              rsi = 100
            } else if (avgGain === 0) {
              rsi = 0
            } else {
              const rs = avgGain / avgLoss
              rsi = 100 - 100 / (1 + rs)
            }

            this.outputBuffer.updateValue(bufferIndex, outIndex, rsi)

            // Track first valid row (in absolute buffer coordinates)
            if (firstValidRow === -1) {
              firstValidRow = bufferIndex
            }
          }
        }
        // Apply Wilder's smoothing
        else {
          state.avgGain = gain * state.mult + state.avgGain * (1 - state.mult)
          state.avgLoss = loss * state.mult + state.avgLoss * (1 - state.mult)

          // Calculate RSI
          let rsi: number
          if (state.avgLoss === 0) {
            rsi = 100
          } else if (state.avgGain === 0) {
            rsi = 0
          } else {
            const rs = state.avgGain / state.avgLoss
            rsi = 100 - 100 / (1 + rs)
          }

          this.outputBuffer.updateValue(bufferIndex, outIndex, rsi)
        }
      }
    }

    // RSI is ready when we have enough data to calculate it (window + 1 values)
    if (!this._isReady) {
      // Check if any state has reached the window + 1 threshold
      for (const state of this._state.values()) {
        if (this._totalRowsProcessed >= state.window + 1) {
          this._isReady = true
          break
        }
      }
    }

    // Return the range of rows that have valid RSI values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
