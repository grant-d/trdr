import { z } from 'zod/v4'
import type { BaseTransformParams } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Schema for individual mapping configuration
 * @property {string} in - Input column name to map from
 * @property {string} out - Output column name to map to
 */
const txSchema = z.object({
  in: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/),
  out: z.string().regex(/^[a-zA-Z0-9_]{1,20}$/)
})

/**
 * Main schema for MappingTransform
 * @property {string} [description] - Optional description of the transform
 * @property {object|array} tx - Single mapping config or array of configs
 *
 * @example
 * // Single mapping
 * { tx: { in: 'close', out: 'adj_close' } }
 *
 * @example
 * // Multiple mappings
 * {
 *   tx: [
 *     { in: 'close', out: 'adj_close' },
 *     { in: 'volume', out: 'adj_volume' },
 *     { in: 'high', out: 'h' },
 *     { in: 'low', out: 'l' }
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

export interface MapParams extends z.infer<typeof schema>, BaseTransformParams {
}

/**
 * Mapping Transform
 *
 * A simple transform that copies values from input columns to output columns.
 * Useful for creating column aliases, renaming fields, or duplicating data
 * without any processing or calculation.
 *
 * **Features**:
 * - Zero-copy operation (references same data)
 * - No calculations or aggregations
 * - Instant readiness (no warmup period)
 * - Preserves original values exactly
 *
 * @example
 * ```typescript
 * const mapper = new MappingTransform({
 *   tx: [
 *     { in: 'close', out: 'price' },
 *     { in: 'volume', out: 'vol' },
 *     { in: 'close', out: 'close_copy' }
 *   ]
 * }, inputBuffer)
 * ```
 *
 * Common use cases:
 * - Renaming columns for compatibility
 * - Creating aliases for downstream transforms
 * - Duplicating columns for different processing paths
 * - Standardizing column names across data sources
 */
export class MapTransform extends BaseTransform<MapParams> {
  // Map of output index to input index
  private readonly _out2in = new Map<number, number>()

  constructor(config: MapParams, inputSlice: DataSlice) {
    // Validate config
    const parsed = schema.parse(config)

    // Base class constructor
    super(
      'map',
      'Map',
      config.description || 'Field Mapping Transform',
      parsed,
      inputSlice
    )

    // Initialize each mapping
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

      // Store mapping
      this._out2in.set(outIndex, inIndex)
    }
  }

  /**
   * Mapping transform is always ready - no aggregation or windowing required
   */
  public get isReady(): boolean {
    return true
  }

  protected processBatch(from: number, to: number): { from: number; to: number } {
    let firstValidRow = -1

    // Process rows in the buffer range [from, to)
    for (let bufferIndex = from; bufferIndex < to; bufferIndex++) {
      // Map each input field to its corresponding output field
      this._out2in.forEach((inIndex, outIndex) => {
        const value = this.outputBuffer.getValue(bufferIndex, inIndex) || 0
        this.outputBuffer.updateValue(bufferIndex, outIndex, value)
      })

      // Track first valid row (in absolute buffer coordinates)
      if (firstValidRow === -1) {
        firstValidRow = bufferIndex
      }
    }

    this._isReady = true

    // Return the range of rows that have valid mapped values (in absolute buffer coordinates)
    return {
      from: firstValidRow === -1 ? to : firstValidRow,
      to
    }
  }
}
