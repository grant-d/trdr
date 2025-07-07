import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Bollinger Bands transaction schema
 * @property in - The input column name (typically 'close') for band calculation
 * @property window - The number of periods for the moving average (default: 20)
 * @property std - The number of standard deviations for band width (default: 2)
 * @property out - Output column names for the three bands
 * @property out.upper - Column name for the upper band
 * @property out.middle - Column name for the middle band (SMA)
 * @property out.lower - Column name for the lower band
 */
const txSchema = z.object({
  in: z
    .string()
    .regex(/^[a-zA-Z0-9_]{1,20}$/)
    .default('close'),
  window: z.number().min(2).max(9_999).default(20),
  std: z.number().min(0.1).max(10).default(2),
  out: z.object({
    upper: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
    middle: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
    lower: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/)
  })
})

/**
 * Bollinger Bands configuration
 *
 * Bollinger Bands are volatility bands placed above and below a moving average.
 * They widen during volatile periods and contract during calm periods, making them
 * useful for identifying overbought/oversold conditions and volatility changes.
 *
 * Band calculations:
 * - Middle Band = SMA(price, window)
 * - Upper Band = Middle Band + (std × StdDev)
 * - Lower Band = Middle Band - (std × StdDev)
 *
 * Trading signals:
 * - Price touches upper band: Potentially overbought
 * - Price touches lower band: Potentially oversold
 * - Band squeeze (narrow bands): Low volatility, potential breakout
 * - Band expansion: Increased volatility, trend acceleration
 * - Walking the bands: Strong trend when price hugs one band
 *
 * @example
 * // Standard Bollinger Bands (20,2)
 * {
 *   tx: {
 *     in: 'close',
 *     window: 20,
 *     std: 2,
 *     out: {
 *       upper: 'bbUpper',
 *       middle: 'bbMiddle',
 *       lower: 'bbLower'
 *     }
 *   }
 * }
 *
 * @example
 * // Multiple Bollinger Bands with different parameters
 * {
 *   tx: [
 *     {
 *       in: 'close',
 *       window: 20,
 *       std: 1,
 *       out: { upper: 'bb1Upper', middle: 'bb1Middle', lower: 'bb1Lower' }
 *     },
 *     {
 *       in: 'close',
 *       window: 20,
 *       std: 2,
 *       out: { upper: 'bb2Upper', middle: 'bb2Middle', lower: 'bb2Lower' }
 *     },
 *     {
 *       in: 'close',
 *       window: 20,
 *       std: 3,
 *       out: { upper: 'bb3Upper', middle: 'bb3Middle', lower: 'bb3Lower' }
 *     }
 *   ]
 * }
 *
 * @example
 * // Bollinger Bands on different price types
 * {
 *   tx: [
 *     {
 *       in: 'close',
 *       out: { upper: 'bbCloseUp', middle: 'bbCloseMid', lower: 'bbCloseLow' }
 *     },
 *     {
 *       in: 'typicalPrice',
 *       out: { upper: 'bbTpUp', middle: 'bbTpMid', lower: 'bbTpLow' }
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
      // Ensure no duplicate output names across all BB groups
      const allOutputNames: string[] = []
      for (const t of tx) {
        allOutputNames.push(t.out.upper, t.out.middle, t.out.lower)
      }
      return allOutputNames.length === new Set(allOutputNames).size
    },
    {
      message:
        'All output names (upper, middle, lower) must be unique across all Bollinger Bands configurations'
    }
  )

export interface BollingerBandsParams
  extends z.infer<typeof schema>,
          BaseTransformParams {
}

interface State {
  readonly in: number;
  readonly window: number;
  readonly std: number;
  readonly out: {
    readonly upper: number;
    readonly middle: number;
    readonly lower: number;
  };
  readonly items: number[];
}

/**
 * Bollinger Bands implementation
 *
 * Bollinger Bands consist of a middle band (SMA) with two outer bands that expand
 * and contract based on market volatility. They're one of the most popular technical
 * indicators for identifying overbought/oversold conditions and volatility.
 *
 * Mathematical foundation:
 * 1. Calculate SMA of closing prices over N periods
 * 2. Calculate standard deviation of the same data
 * 3. Upper Band = SMA + (K × standard deviation)
 * 4. Lower Band = SMA - (K × standard deviation)
 * where K is typically 2
 *
 * Key trading concepts:
 * - Bollinger Squeeze: When bands narrow, indicating low volatility and potential breakout
 * - Bollinger Bounce: Price tends to bounce between the bands
 * - W-Bottoms and M-Tops: Chart patterns that form at bands
 * - Band width: Measure of volatility (upper - lower) / middle
 * - %B: Shows where price is relative to the bands
 *
 * Advanced strategies:
 * - Combine with RSI for divergence signals
 * - Use multiple standard deviations (1, 2, 3) for probability zones
 * - Keltner channel squeeze: When BB moves inside Keltner channels
 *
 * @example
 * // Create standard Bollinger Bands
 * const bb = new BollingerBands({
 *   tx: {
 *     in: 'close',
 *     window: 20,
 *     std: 2,
 *     out: { upper: 'bbUp', middle: 'bbMid', lower: 'bbLow' }
 *   }
 * }, inputBuffer)
 */
export class BollingerBands extends BaseTransform<BollingerBandsParams> {
  // State for each BB configuration
  private readonly _state: State[] = []
  private _totalRowsProcessed = 0

  constructor(config: BollingerBandsParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'bollinger',
      'BB',
      config.description || 'Bollinger Bands',
      parsed,
      inputSlice
    )

    // Initialize each BB configuration
    const tx = Array.isArray(parsed.tx) ? parsed.tx : [parsed.tx]
    for (const params of tx) {
      // Ensure input column exists
      const inField = params.in
      const inIndex = inputSlice.getColumn(inField)?.index
      if (typeof inIndex !== 'number')
        throw new Error(`Input column '${inField}' not found in input buffer.`)

      // Ensure output columns exist
      const upperIndex = this.outputBuffer.ensureColumn(params.out.upper)
      const middleIndex = this.outputBuffer.ensureColumn(params.out.middle)
      const lowerIndex = this.outputBuffer.ensureColumn(params.out.lower)

      this._state.push({
        in: inIndex,
        window: params.window,
        std: params.std,
        out: {
          upper: upperIndex,
          middle: middleIndex,
          lower: lowerIndex
        },
        items: []
      })
    }
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    let firstValidRow = -1

    // Process rows in the buffer range [from, to)
    for (let bufferIndex = from; bufferIndex < to; bufferIndex++) {
      this._totalRowsProcessed += 1

      for (const state of this._state) {
        const currentValue = this.outputBuffer.getValue(bufferIndex, state.in) || 0.0

        // Add current value to window
        state.items.push(currentValue)

        // Keep window at max size
        if (state.items.length > state.window) {
          state.items.shift()
        }

        // Calculate SMA (middle band)
        const sum = state.items.reduce((acc, val) => acc + val, 0)
        const sma = sum / state.items.length

        // Calculate standard deviation
        const sumSquares = state.items.reduce((acc, val) => acc + val * val, 0)
        const variance = sumSquares / state.items.length - sma * sma
        const stdDev = Math.sqrt(Math.max(0, variance)) // Ensure non-negative

        // Calculate bands
        const upperBand = sma + state.std * stdDev
        const lowerBand = sma - state.std * stdDev

        // Only update output values once we have enough data
        if (state.items.length >= state.window) {
          this.outputBuffer.updateValue(bufferIndex, state.out.middle, sma)
          this.outputBuffer.updateValue(bufferIndex, state.out.upper, upperBand)
          this.outputBuffer.updateValue(bufferIndex, state.out.lower, lowerBand)

          // Track first valid row (in absolute buffer coordinates)
          if (firstValidRow === -1) {
            firstValidRow = bufferIndex
          }
        }
      }
    }

    this._isReady ||= this._totalRowsProcessed >= 2

    // Return the range of rows that have valid Bollinger Bands values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
