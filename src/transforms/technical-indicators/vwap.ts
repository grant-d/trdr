import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * VWAP transaction schema
 * @property out - The output column name where VWAP values will be stored
 * @property window - Optional rolling window period. If not specified, uses cumulative VWAP
 * @property anchor - Time anchor for resetting VWAP calculation (not yet implemented)
 */
const txSchema = z.object({
  out: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
  window: z.number().min(1).max(9_999).optional(),
  anchor: z.enum(['session', 'day', 'week', 'month', 'none']).default('none')
})

/**
 * Volume Weighted Average Price (VWAP) configuration
 *
 * VWAP represents the average price a security has traded at throughout the day,
 * based on both volume and price. It's particularly important for institutional traders
 * who need to assess whether they got a good execution price.
 *
 * Calculation:
 * VWAP = Σ(Price × Volume) / Σ(Volume)
 * where Price = (High + Low + Close) / 3 (typical price)
 *
 * Trading applications:
 * - Price above VWAP: Bullish sentiment (buyers in control)
 * - Price below VWAP: Bearish sentiment (sellers in control)
 * - VWAP acts as support/resistance level
 * - Used as a benchmark for trade execution quality
 * - Institutional traders aim to buy below VWAP and sell above it
 *
 * @example
 * // Standard cumulative VWAP
 * {
 *   in: { high: 'high', low: 'low', close: 'close', volume: 'volume' },
 *   tx: { out: 'vwap' }
 * }
 *
 * @example
 * // Multiple VWAP calculations with different windows
 * {
 *   in: { high: 'high', low: 'low', close: 'close', volume: 'volume' },
 *   tx: [
 *     { out: 'vwapCumulative' },              // Cumulative (standard)
 *     { out: 'vwap20', window: 20 },          // 20-period rolling
 *     { out: 'vwap50', window: 50 },          // 50-period rolling
 *     { out: 'vwap200', window: 200 }         // 200-period rolling
 *   ]
 * }
 *
 * @example
 * // VWAP with custom input columns
 * {
 *   in: {
 *     high: 'askHigh',
 *     low: 'bidLow',
 *     close: 'midClose',
 *     volume: 'tradeVolume'
 *   },
 *   tx: { out: 'customVwap' }
 * }
 */
const schema = z
  .object({
    description: z.string().optional(),
    in: z.object({
      high: z
        .string()
        .regex(/^[a-zA-Z0-9_]{1,20}$/)
        .default('high'),
      low: z
        .string()
        .regex(/^[a-zA-Z0-9_]{1,20}$/)
        .default('low'),
      close: z
        .string()
        .regex(/^[a-zA-Z0-9_]{1,20}$/)
        .default('close'),
      volume: z
        .string()
        .regex(/^[a-zA-Z0-9_]{1,20}$/)
        .default('volume')
    }),
    tx: z.union([txSchema, z.array(txSchema)])
  })
  .refine(
    (data) => {
      // Ensure no duplicate output names
      const vwapNames = Array.isArray(data.tx)
        ? data.tx.map((o) => o.out)
        : [data.tx.out]
      return vwapNames.length === new Set(vwapNames).size
    },
    {
      message: 'Output names (vwap) must be unique'
    }
  )

interface VwapParams extends z.infer<typeof schema>, BaseTransformParams {
}

interface State {
  readonly window?: number;
  readonly anchor: 'session' | 'day' | 'week' | 'month' | 'none';
  pvSum: number;
  vSum: number;
  pvWindow: number[];
  vWindow: number[];
  lastResetTime?: number;
}

/**
 * Volume Weighted Average Price (VWAP) implementation
 *
 * VWAP is a trading benchmark that gives the average price a security has traded at
 * throughout the day, based on both volume and price. It's especially useful for
 * large institutional orders to minimize market impact.
 *
 * Key features:
 * - Cumulative VWAP: Calculates from the start of data (or trading session)
 * - Rolling VWAP: Uses a fixed window of periods
 * - Resets at session/day/week/month boundaries (anchor - not yet implemented)
 *
 * Calculation details:
 * 1. Typical Price = (High + Low + Close) / 3
 * 2. TPV (Typical Price × Volume) for each period
 * 3. Cumulative VWAP = Σ(TPV) / Σ(Volume)
 * 4. Rolling VWAP = Σ(TPV over window) / Σ(Volume over window)
 *
 * Trading strategies:
 * - VWAP bands: Add standard deviations to create bands
 * - Mean reversion: Fade moves away from VWAP
 * - Trend following: Trade in direction of VWAP slope
 * - Execution benchmark: Compare fills to VWAP
 *
 * @example
 * // Create standard VWAP indicator
 * const vwap = new VolumeWeightedAveragePrice({
 *   in: { high: 'high', low: 'low', close: 'close', volume: 'volume' },
 *   tx: { out: 'vwap' }
 * }, inputBuffer)
 */
export class VolumeWeightedAveragePrice extends BaseTransform<VwapParams> {
  // Input column indices (shared across all outputs)
  private readonly _inputs: {
    readonly hi: number;
    readonly lo: number;
    readonly cl: number;
    readonly vol: number;
  }

  // State for each output index
  private readonly _state = new Map<number, State>()
  private _totalRowsProcessed = 0

  constructor(config: VwapParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'vwap',
      'VWAP',
      config.description || 'Volume Weighted Average Price',
      parsed,
      inputSlice
    )

    // All outputs share same input columns
    const hi = inputSlice.getColumn(parsed.in.high)?.index
    if (typeof hi !== 'number')
      throw new Error(
        `Input column '${parsed.in.high}' not found in input buffer.`
      )
    const lo = inputSlice.getColumn(parsed.in.low)?.index
    if (typeof lo !== 'number')
      throw new Error(
        `Input column '${parsed.in.low}' not found in input buffer.`
      )
    const cl = inputSlice.getColumn(parsed.in.close)?.index
    if (typeof cl !== 'number')
      throw new Error(
        `Input column '${parsed.in.close}' not found in input buffer.`
      )
    const vol = inputSlice.getColumn(parsed.in.volume)?.index
    if (typeof vol !== 'number')
      throw new Error(
        `Input column '${parsed.in.volume}' not found in input buffer.`
      )

    this._inputs = { hi, lo, cl, vol }

    // Initialize each output column
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    for (const params of tx) {
      // Ensure output column exists
      const outIndex: number = this.outputBuffer.ensureColumn(params.out)

      this._state.set(outIndex, {
        window: params.window,
        anchor: params.anchor,
        pvSum: 0,
        vSum: 0,
        pvWindow: [],
        vWindow: []
      })
    }
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    let firstValidRow = -1

    // Process rows in the buffer range [from, to)
    for (let bufferIndex = from; bufferIndex < to; bufferIndex++) {
      this._totalRowsProcessed += 1

      // Get input values (shared across all outputs)
      const high = this.outputBuffer.getValue(bufferIndex, this._inputs.hi) || 0.0
      const low = this.outputBuffer.getValue(bufferIndex, this._inputs.lo) || 0.0
      const close = this.outputBuffer.getValue(bufferIndex, this._inputs.cl) || 0.0
      const volume = this.outputBuffer.getValue(bufferIndex, this._inputs.vol) || 0.0

      // Calculate typical price (HLC/3)
      const typicalPrice = (high + low + close) / 3
      const pv = typicalPrice * volume

      for (const [outIndex, state] of this._state) {
        // Handle windowed VWAP
        if (state.window) {
          // Add to window
          state.pvWindow.push(pv)
          state.vWindow.push(volume)

          // Keep window at max size
          if (state.pvWindow.length > state.window) {
            state.pvWindow.shift()
            state.vWindow.shift()
          }

          // Calculate windowed sums
          const pvSum = state.pvWindow.reduce((acc, val) => acc + val, 0)
          const vSum = state.vWindow.reduce((acc, val) => acc + val, 0)
          const vwap = vSum > 0 ? pvSum / vSum : typicalPrice

          this.outputBuffer.updateValue(bufferIndex, outIndex, vwap)

          // Track first valid row (in absolute buffer coordinates)
          if (firstValidRow === -1) {
            firstValidRow = bufferIndex
          }
        }
        // Handle cumulative VWAP (standard)
        else {
          // Add to cumulative sums
          state.pvSum += pv
          state.vSum += volume

          // Calculate VWAP
          const vwap = state.vSum > 0 ? state.pvSum / state.vSum : typicalPrice

          this.outputBuffer.updateValue(bufferIndex, outIndex, vwap)

          // Track first valid row (in absolute buffer coordinates)
          if (firstValidRow === -1) {
            firstValidRow = bufferIndex
          }
        }

        // Note: anchor resets not implemented yet
        // This would require timestamp information to reset at day/week/month boundaries
      }
    }

    this._isReady ||= this._totalRowsProcessed >= 1

    // Return the range of rows that have valid VWAP values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
