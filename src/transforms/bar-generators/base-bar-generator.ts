import { z } from 'zod/v4'
import type { BaseTransformParams, TransformType } from '../../interfaces'
import type { DataSlice } from '../../utils'
import { DataBuffer } from '../../utils'
import { BaseTransform } from '../base-transform'

/**
 * Base schema for bar configurations
 */
export const baseBarSchema = z.object({
  description: z.string().optional()
})

/**
 * Common bar state interface
 */
export interface BaseBarState {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  firstTimestamp: number;
  lastTimestamp: number;
  tickCount: number;
}

/**
 * Base class for bar generators that aggregate tick data into bars
 * Provides common functionality for all bar generator types
 */
export abstract class BaseBarGenerator<
  TConfig extends BaseTransformParams,
  TBarState extends BaseBarState = BaseBarState,
> extends BaseTransform<TConfig> {
  // Output buffer for aggregated bars
  protected _outputBuffer: DataBuffer

  // Column indices for efficient access
  protected readonly _indices: {
    timestamp: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    [key: string]: number | undefined;
  }

  // Current bar state
  protected _currentBar?: TBarState

  // Statistics
  protected _totalRowsProcessed = 0
  protected _barsGenerated = 0

  protected constructor(
    type: TransformType,
    name: string,
    description: string,
    config: TConfig,
    inputSlice: DataSlice
  ) {
    // Base class constructor
    super(type, name, description, config, inputSlice)

    // Create output buffer with same columns as input
    this._outputBuffer = new DataBuffer({
      columns: inputSlice.getColumnDefinitions()
    })

    // Get required column indices
    const timestampCol = inputSlice.getColumn('timestamp')
    const openCol = inputSlice.getColumn('open')
    const highCol = inputSlice.getColumn('high')
    const lowCol = inputSlice.getColumn('low')
    const closeCol = inputSlice.getColumn('close')
    const volumeCol = inputSlice.getColumn('volume')

    if (
      !timestampCol ||
      !openCol ||
      !highCol ||
      !lowCol ||
      !closeCol ||
      !volumeCol
    ) {
      throw new Error(
        `${name} requires timestamp, open, high, low, close, and volume columns`
      )
    }

    this._indices = {
      timestamp: timestampCol.index,
      open: openCol.index,
      high: highCol.index,
      low: lowCol.index,
      close: closeCol.index,
      volume: volumeCol.index
    }

    // Let subclasses add additional indices
    this.initializeAdditionalIndices(inputSlice)
  }

  /**
   * Subclasses can override to add additional column indices
   */
  protected initializeAdditionalIndices(_inputSlice: DataSlice): void {
    // Default: no additional indices
  }

  /**
   * Main batch processing method
   */
  protected processBatch(): { from: number; to: number } {
    let firstValidRow = -1
    const rowCount = this.inputSlice.length()

    for (let rid = 0; rid < rowCount; rid++) {
      this._totalRowsProcessed++

      // Get tick data
      const tickData = this.extractTickData(rid)

      // Process tick
      if (!this._currentBar) {
        // Start new bar
        this._currentBar = this.createNewBar(tickData, rid)
      } else {
        // Update current bar
        this.updateBar(this._currentBar, tickData, rid)

        // Check if bar is complete
        if (this.isBarComplete(this._currentBar, tickData, rid)) {
          // Emit completed bar to output buffer
          this.emitBar(rid, this._currentBar)

          // Track first valid row (in absolute buffer coordinates)
          if (firstValidRow === -1) {
            firstValidRow = this.inputSlice.from + rid
          }

          // Reset for next bar
          this._currentBar = undefined
          this._barsGenerated++
        }
      }
    }

    // Mark as ready after processing first batch
    this._isReady = true

    // Return the range of rows that were processed
    return {
      from: firstValidRow === -1 ? this.inputSlice.to : firstValidRow,
      to: this.inputSlice.to
    }
  }

  /**
   * Extract tick data from buffer row
   */
  protected extractTickData(rid: number): {
    timestamp: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    [key: string]: any;
  } {
    return {
      timestamp: this.inputSlice.getValue(
        rid,
        this._indices.timestamp
      )!,
      open: this.inputSlice.getValue(rid, this._indices.open)!,
      high: this.inputSlice.getValue(rid, this._indices.high)!,
      low: this.inputSlice.getValue(rid, this._indices.low)!,
      close: this.inputSlice.getValue(rid, this._indices.close)!,
      volume: this.inputSlice.getValue(rid, this._indices.volume)!
    }
  }

  /**
   * Emit completed bar to output buffer
   */
  protected emitBar(sourceRid: number, bar: TBarState): void {
    // Create new row object with aggregated bar values
    const newRow: Record<string, number> = {}

    // Copy all columns from source row first to preserve metadata
    const columnNames = this.inputSlice.getColumns()
    for (const colName of columnNames) {
      const colDef = this.inputSlice.getColumn(colName)
      if (colDef) {
        newRow[colName] =
          this.inputSlice.getValue(sourceRid, colDef.index) || 0
      }
    }

    // Then update with aggregated bar values
    newRow.timestamp = bar.lastTimestamp
    newRow.open = bar.open
    newRow.high = bar.high
    newRow.low = bar.low
    newRow.close = bar.close
    newRow.volume = bar.volume

    // Let subclasses add additional fields
    this.addAdditionalBarFields(newRow, bar)

    // Push to output buffer
    this._outputBuffer.push(newRow)
  }

  /**
   * Subclasses can override to add additional fields to emitted bars
   */
  protected addAdditionalBarFields(
    _row: Record<string, number>,
    _bar: TBarState
  ): void {
    // Default: no additional fields
  }

  /**
   * Get statistics about bar generation
   */
  public getStats(): {
    totalTicks: number;
    barsGenerated: number;
    compression: number;
  } {
    return {
      totalTicks: this._totalRowsProcessed,
      barsGenerated: this._barsGenerated,
      compression:
        this._totalRowsProcessed > 0
          ? this._barsGenerated / this._totalRowsProcessed
          : 0
    }
  }

  /**
   * Abstract methods that subclasses must implement
   */

  /**
   * Create a new bar from the first tick
   */
  protected abstract createNewBar(tick: any, rid: number): TBarState;

  /**
   * Update the current bar with new tick data
   */
  protected abstract updateBar(bar: TBarState, tick: any, rid: number): void;

  /**
   * Check if the current bar is complete
   */
  protected abstract isBarComplete(
    bar: TBarState,
    tick: any,
    rid: number
  ): boolean;
}
