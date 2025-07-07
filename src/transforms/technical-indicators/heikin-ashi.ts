import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Schema for individual Heikin-Ashi configuration
 * @property {string} [prefix] - Prefix for output columns (default: 'ha_')
 * @property {object} [in] - Input column names
 * @property {object} out - Output column names
 */
const txSchema = z
  .object({
    prefix: z.string().default('ha_'),
    in: z
      .object({
        open: z.string().default('open'),
        high: z.string().default('high'),
        low: z.string().default('low'),
        close: z.string().default('close')
      })
      .partial()
      .default({}),
    out: z.object({
      open: z.string(),
      high: z.string(),
      low: z.string(),
      close: z.string()
    })
  })
  .transform((data) => ({
    ...data,
    in: {
      open: data.in.open || 'open',
      high: data.in.high || 'high',
      low: data.in.low || 'low',
      close: data.in.close || 'close'
    },
    out: {
      open: data.out.open || `${data.prefix}open`,
      high: data.out.high || `${data.prefix}high`,
      low: data.out.low || `${data.prefix}low`,
      close: data.out.close || `${data.prefix}close`
    }
  }))

/**
 * Main schema for HeikinAshi transform
 */
const schema = z.object({
  description: z.string().optional(),
  tx: z.union([txSchema, z.array(txSchema)])
})

export interface HeikinAshiParams
  extends z.infer<typeof schema>,
          BaseTransformParams {
}

interface State {
  readonly inOpen: number;
  readonly inHigh: number;
  readonly inLow: number;
  readonly inClose: number;
  readonly outOpen: number;
  readonly outHigh: number;
  readonly outLow: number;
  readonly outClose: number;
  prevHAOpen: number;
  prevHAClose: number;
  isFirst: boolean;
}

/**
 * Heikin-Ashi Transform
 *
 * Converts regular OHLC bars to Heikin-Ashi (HA) bars for smoother trend visualization
 * and noise reduction. Heikin-Ashi creates more readable charts by filtering market noise.
 *
 * **Formulas**:
 * - HA_Close = (Open + High + Low + Close) / 4
 * - HA_Open = (Previous HA_Open + Previous HA_Close) / 2
 * - HA_High = Max(High, HA_Open, HA_Close)
 * - HA_Low = Min(Low, HA_Open, HA_Close)
 *
 * **First Bar**: HA_Open = (Open + Close) / 2
 *
 * **Visual Characteristics**:
 * - Uptrend: Long green candles with small lower wicks
 * - Downtrend: Long red candles with small upper wicks
 * - Consolidation: Candles with both upper and lower wicks
 * - Trend Change: Color changes and wick patterns shift
 *
 * **Use Cases**:
 * - Trend identification and analysis
 * - Noise reduction in price action
 * - Signal generation with fewer false positives
 * - Support/resistance identification
 * - Multi-timeframe trend analysis
 *
 * @example
 * ```typescript
 * const heikinAshi = new HeikinAshi({
 *   tx: {
 *     out: {
 *       open: 'ha_open',
 *       high: 'ha_high',
 *       low: 'ha_low',
 *       close: 'ha_close'
 *     }
 *   }
 * }, inputBuffer)
 *
 * // Multiple timeframes
 * const multiHA = new HeikinAshi({
 *   tx: [
 *     { prefix: 'ha5_', out: { open: 'ha5_open', high: 'ha5_high', low: 'ha5_low', close: 'ha5_close' } },
 *     { prefix: 'ha15_', out: { open: 'ha15_open', high: 'ha15_high', low: 'ha15_low', close: 'ha15_close' } }
 *   ]
 * }, inputBuffer)
 * ```
 *
 * @note HA values don't represent actual tradeable prices
 * @note First bar uses special initialization
 * @note State maintained per configuration for proper HA calculation
 */
export class HeikinAshi extends BaseTransform<HeikinAshiParams> {
  // State for each configuration
  private readonly _state = new Map<string, State>()
  private _totalRowsProcessed = 0

  constructor(config: HeikinAshiParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'heikinAshi',
      'HeikinAshi',
      config.description || 'Heikin-Ashi Transform',
      parsed,
      inputSlice
    )

    // Initialize each configuration
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    for (const params of tx) {
      // Get input column indices
      const openCol = inputSlice.getColumn(params.in.open)
      const highCol = inputSlice.getColumn(params.in.high)
      const lowCol = inputSlice.getColumn(params.in.low)
      const closeCol = inputSlice.getColumn(params.in.close)

      if (!openCol || !highCol || !lowCol || !closeCol) {
        throw new Error(
          'Heikin-Ashi requires open, high, low, and close columns'
        )
      }

      // Ensure output columns exist
      const outOpenIdx = this.outputBuffer.ensureColumn(params.out.open)
      const outHighIdx = this.outputBuffer.ensureColumn(params.out.high)
      const outLowIdx = this.outputBuffer.ensureColumn(params.out.low)
      const outCloseIdx = this.outputBuffer.ensureColumn(params.out.close)

      // Create unique key for this configuration
      const stateKey = `${params.out.open}_${params.out.high}_${params.out.low}_${params.out.close}`

      // Store state
      this._state.set(stateKey, {
        inOpen: openCol.index,
        inHigh: highCol.index,
        inLow: lowCol.index,
        inClose: closeCol.index,
        outOpen: outOpenIdx,
        outHigh: outHighIdx,
        outLow: outLowIdx,
        outClose: outCloseIdx,
        prevHAOpen: 0,
        prevHAClose: 0,
        isFirst: true
      })
    }
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    let firstValidRow = -1

    // Process rows in the buffer range [from, to)
    for (let bufferIndex = from; bufferIndex < to; bufferIndex++) {
      this._totalRowsProcessed++

      for (const state of this._state.values()) {
        // Get OHLC values
        const open = this.outputBuffer.getValue(bufferIndex, state.inOpen) || 0
        const high = this.outputBuffer.getValue(bufferIndex, state.inHigh) || 0
        const low = this.outputBuffer.getValue(bufferIndex, state.inLow) || 0
        const close = this.outputBuffer.getValue(bufferIndex, state.inClose) || 0

        // Calculate Heikin-Ashi Close (average price)
        const haClose = (open + high + low + close) / 4

        let haOpen: number

        if (state.isFirst) {
          // First bar - use regular open/close
          haOpen = (open + close) / 2
          state.isFirst = false
        } else {
          // Subsequent bars - use previous HA values
          haOpen = (state.prevHAOpen + state.prevHAClose) / 2
        }

        // Calculate Heikin-Ashi High and Low
        const haHigh = Math.max(high, haOpen, haClose)
        const haLow = Math.min(low, haOpen, haClose)

        // Update output values
        this.outputBuffer.updateValue(bufferIndex, state.outOpen, haOpen)
        this.outputBuffer.updateValue(bufferIndex, state.outHigh, haHigh)
        this.outputBuffer.updateValue(bufferIndex, state.outLow, haLow)
        this.outputBuffer.updateValue(bufferIndex, state.outClose, haClose)

        // Track first valid row (in absolute buffer coordinates)
        if (firstValidRow === -1) {
          firstValidRow = bufferIndex
        }

        // Update state for next bar
        state.prevHAOpen = haOpen
        state.prevHAClose = haClose
      }
    }

    // Ready after first row
    this._isReady ||= this._totalRowsProcessed >= 1

    // Return the range of rows that have valid Heikin-Ashi values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
