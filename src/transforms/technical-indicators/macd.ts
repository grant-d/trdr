import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * MACD transaction schema
 * @property in - The input column name (typically 'close') for MACD calculation
 * @property fast - Fast EMA period (default: 12)
 * @property slow - Slow EMA period (default: 26)
 * @property signal - Signal EMA period (default: 9)
 * @property out - Output column names for MACD components
 * @property out.macd - Column name for MACD line (fast EMA - slow EMA)
 * @property out.signal - Column name for signal line (EMA of MACD)
 * @property out.hist - Column name for histogram (MACD - signal)
 */
const txSchema = z.object({
  in: z
    .string()
    .regex(/^[a-zA-Z0-9_]{1,20}$/)
    .default('close'),
  fast: z.number().min(2).max(9_999).default(12),
  slow: z.number().min(2).max(9_999).default(26),
  signal: z.number().min(2).max(9_999).default(9),
  out: z.object({
    macd: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
    signal: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
    hist: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/)
  })
})

/**
 * Moving Average Convergence Divergence (MACD) configuration
 *
 * MACD is a trend-following momentum indicator that shows the relationship between
 * two exponential moving averages (EMAs) of prices. It consists of:
 * - MACD Line: Fast EMA - Slow EMA
 * - Signal Line: EMA of MACD line
 * - Histogram: MACD line - Signal line
 *
 * Trading signals:
 * - MACD crosses above signal: Bullish signal
 * - MACD crosses below signal: Bearish signal
 * - Histogram above zero: Upward momentum
 * - Histogram below zero: Downward momentum
 * - Divergences between price and MACD indicate potential reversals
 *
 * @example
 * // Standard MACD (12,26,9)
 * {
 *   tx: {
 *     in: 'close',
 *     fast: 12,
 *     slow: 26,
 *     signal: 9,
 *     out: {
 *       macd: 'macd',
 *       signal: 'macdSignal',
 *       hist: 'macdHist'
 *     }
 *   }
 * }
 *
 * @example
 * // Multiple MACD configurations for different timeframes
 * {
 *   tx: [
 *     {
 *       in: 'close',
 *       fast: 12,
 *       slow: 26,
 *       signal: 9,
 *       out: { macd: 'macdStd', signal: 'signalStd', hist: 'histStd' }
 *     },
 *     {
 *       in: 'close',
 *       fast: 5,
 *       slow: 13,
 *       signal: 5,
 *       out: { macd: 'macdFast', signal: 'signalFast', hist: 'histFast' }
 *     }
 *   ]
 * }
 *
 * @example
 * // MACD on different price inputs
 * {
 *   tx: [
 *     {
 *       in: 'close',
 *       out: { macd: 'macdClose', signal: 'signalClose', hist: 'histClose' }
 *     },
 *     {
 *       in: 'typicalPrice',
 *       out: { macd: 'macdTp', signal: 'signalTp', hist: 'histTp' }
 *     }
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
      const tx = Array.isArray(data.tx) ? data.tx : [data.tx]
      // Ensure no duplicate output names across all MACD groups
      const allOutputNames: string[] = []
      for (const t of tx) {
        allOutputNames.push(t.out.macd, t.out.signal, t.out.hist)
      }
      return allOutputNames.length === new Set(allOutputNames).size
    },
    {
      message:
        'All output names (macd, signal, hist) must be unique across all MACD configurations'
    }
  )

export interface MacdParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface State {
  readonly in: number;
  readonly fast: number;
  readonly slow: number;
  readonly signal: number;
  readonly fastMult: number;
  readonly slowMult: number;
  readonly signalMult: number;
  readonly out: {
    readonly macd: number;
    readonly signal: number;
    readonly hist: number;
  };
  fastEma: number;
  slowEma: number;
  signalEma: number;
  fastSum: number;
  slowSum: number;
  signalSum: number;
  macdCount: number;
}

/**
 * MACD (Moving Average Convergence Divergence) implementation
 *
 * MACD is a versatile momentum indicator that reveals changes in the strength, direction,
 * momentum, and duration of a trend. It uses the relationship between two EMAs to generate
 * trading signals.
 *
 * Calculation steps:
 * 1. Calculate Fast EMA (typically 12-period) of the input price
 * 2. Calculate Slow EMA (typically 26-period) of the input price
 * 3. MACD Line = Fast EMA - Slow EMA
 * 4. Signal Line = EMA of MACD Line (typically 9-period)
 * 5. Histogram = MACD Line - Signal Line
 *
 * Key interpretations:
 * - MACD above zero: Fast EMA > Slow EMA, indicating upward momentum
 * - MACD below zero: Fast EMA < Slow EMA, indicating downward momentum
 * - Signal line crossovers: Primary trading signals
 * - Histogram expansion/contraction: Momentum acceleration/deceleration
 * - Divergences: When price makes new highs/lows but MACD doesn't
 *
 * @example
 * // Create standard MACD indicator
 * const macd = new Macd({
 *   tx: {
 *     in: 'close',
 *     fast: 12,
 *     slow: 26,
 *     signal: 9,
 *     out: { macd: 'macd', signal: 'signal', hist: 'histogram' }
 *   }
 * }, inputBuffer)
 */
export class Macd extends BaseTransform<MacdParams> {
  // State for each MACD configuration
  private readonly _state: State[] = []
  private _totalRowsProcessed = 0
  private readonly _maxSlowPeriod: number = 0

  constructor(config: MacdParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'macd',
      'MACD',
      config.description || 'Moving Average Convergence Divergence',
      parsed,
      inputSlice
    )

    // Initialize each MACD configuration
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    for (const params of tx) {
      // Ensure input column exists
      const inField = params.in
      const inIndex = inputSlice.getColumn(inField)?.index
      if (typeof inIndex !== 'number')
        throw new Error(`Input column '${inField}' not found in input buffer.`)

      // Ensure output columns exist
      const macdIndex = this.outputBuffer.ensureColumn(params.out.macd)
      const signalIndex = this.outputBuffer.ensureColumn(params.out.signal)
      const histIndex = this.outputBuffer.ensureColumn(params.out.hist)

      // Calculate multipliers
      const fastMult = 2 / (params.fast + 1)
      const slowMult = 2 / (params.slow + 1)
      const signalMult = 2 / (params.signal + 1)

      this._state.push({
        in: inIndex,
        fast: params.fast,
        slow: params.slow,
        signal: params.signal,
        fastMult,
        slowMult,
        signalMult,
        out: {
          macd: macdIndex,
          signal: signalIndex,
          hist: histIndex
        },
        fastEma: 0,
        slowEma: 0,
        signalEma: 0,
        fastSum: 0,
        slowSum: 0,
        signalSum: 0,
        macdCount: 0
      })

      this._maxSlowPeriod = Math.max(this._maxSlowPeriod, params.slow)
    }
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    let firstValidRow = -1

    // Process rows in the buffer range [from, to)
    for (let bufferIndex = from; bufferIndex < to; bufferIndex++) {
      this._totalRowsProcessed += 1

      for (const state of this._state) {
        const value = this.outputBuffer.getValue(bufferIndex, state.in) || 0.0

        // Update fast EMA
        if (this._totalRowsProcessed <= state.fast) {
          state.fastSum += value
          state.fastEma = state.fastSum / this._totalRowsProcessed
        } else {
          state.fastEma =
            value * state.fastMult + state.fastEma * (1 - state.fastMult)
        }

        // Update slow EMA
        if (this._totalRowsProcessed <= state.slow) {
          state.slowSum += value
          state.slowEma = state.slowSum / this._totalRowsProcessed
        } else {
          state.slowEma =
            value * state.slowMult + state.slowEma * (1 - state.slowMult)
        }

        // Calculate MACD line when slow EMA is ready
        if (this._totalRowsProcessed < state.slow) {
          this.outputBuffer.updateValue(bufferIndex, state.out.macd, 0.0)
          this.outputBuffer.updateValue(bufferIndex, state.out.signal, 0.0)
          this.outputBuffer.updateValue(bufferIndex, state.out.hist, 0.0)
          continue
        }

        const macdValue = state.fastEma - state.slowEma
        this.outputBuffer.updateValue(bufferIndex, state.out.macd, macdValue)

        // Track first valid row (in absolute buffer coordinates)
        if (firstValidRow === -1) {
          firstValidRow = bufferIndex
        }

        // Update signal EMA
        state.macdCount++

        if (state.macdCount <= state.signal) {
          state.signalSum += macdValue
          state.signalEma = state.signalSum / state.macdCount

          if (state.macdCount < state.signal) {
            this.outputBuffer.updateValue(bufferIndex, state.out.signal, 0.0)
            this.outputBuffer.updateValue(bufferIndex, state.out.hist, 0.0)
            continue
          }
        } else {
          state.signalEma =
            macdValue * state.signalMult +
            state.signalEma * (1 - state.signalMult)
        }

        // Update signal and histogram
        this.outputBuffer.updateValue(bufferIndex, state.out.signal, state.signalEma)
        const histogram = macdValue - state.signalEma
        this.outputBuffer.updateValue(bufferIndex, state.out.hist, histogram)
      }
    }

    this._isReady ||= this._totalRowsProcessed >= this._maxSlowPeriod

    // Return the range of rows that have valid MACD values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
