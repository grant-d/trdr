import type { ColumnDefinition, ColumnValue, DataBuffer, Row } from './data-buffer'

/**
 * DataSlice provides a logical slice of a DataBuffer
 * It decorates DataBuffer but offsets row numbers according to the from/to range
 */
export class DataSlice {
  private readonly buffer: DataBuffer
  private readonly fromRow: number
  private readonly toRow: number

  constructor(buffer: DataBuffer, from: number, to: number) {
    this.buffer = buffer
    this.fromRow = from
    this.toRow = to
  }

  /**
   * Get the underlying buffer
   */
  get underlyingBuffer(): DataBuffer {
    return this.buffer
  }

  /**
   * Get the start index of this view in the underlying buffer
   */
  get from(): number {
    return this.fromRow
  }

  /**
   * Get the end index of this view in the underlying buffer
   */
  get to(): number {
    return this.toRow
  }

  /**
   * Get column names
   */
  getColumns(): readonly string[] {
    return this.buffer.getColumns()
  }

  /**
   * Get column definitions
   */
  getColumnDefinitions(): Readonly<Record<string, ColumnDefinition>> {
    return this.buffer.getColumnDefinitions()
  }

  /**
   * Get column definition by name
   */
  getColumn(name: string): ColumnDefinition | undefined {
    return this.buffer.getColumn(name)
  }

  /**
   * Check if a column exists
   */
  hasColumn(name: string): boolean {
    return this.buffer.hasColumn(name)
  }

  /**
   * Ensure a column exists, creating it if necessary
   * Returns the column index
   */
  ensureColumn(outField: string): number {
    return this.buffer.ensureColumn(outField)
  }

  /**
   * Get the number of rows in this view
   */
  length(): number {
    return this.toRow - this.fromRow
  }

  /**
   * Check if the view is empty
   */
  isEmpty(): boolean {
    return this.length() === 0
  }

  /**
   * Get a row by index (relative to this view)
   * @param index - Index relative to this view where 0 is the first row in the view
   * @returns The row at the specified index or undefined if out of bounds
   */
  getRow(index: number): Row | undefined {
    const absoluteIndex = this.fromRow + index
    if (absoluteIndex < this.fromRow || absoluteIndex >= this.toRow) {
      return undefined
    }
    return this.buffer.getRow(absoluteIndex)
  }

  /**
   * Get a value from a specific row and column
   * @param rowIndex - Index relative to this view
   * @param columnNameOrIndex - Column name or index
   */
  getValue(
    rowIndex: number,
    columnNameOrIndex: string | number
  ): ColumnValue | undefined {
    const absoluteIndex = this.fromRow + rowIndex
    if (absoluteIndex < this.fromRow || absoluteIndex >= this.toRow) {
      throw new Error(`Row index ${rowIndex} is out of bounds for this view`)
    }
    return this.buffer.getValue(absoluteIndex, columnNameOrIndex)
  }

  /**
   * Update a value in a specific row and column
   * @param rowIndex - Index relative to this view
   * @param columnNameOrIndex - Column name or index
   * @param value - New value to set
   */
  updateValue(
    rowIndex: number,
    columnNameOrIndex: string | number,
    value: ColumnValue
  ): void {
    const absoluteIndex = this.fromRow + rowIndex
    if (absoluteIndex < this.fromRow || absoluteIndex >= this.toRow) {
      throw new Error(`Row index ${rowIndex} is out of bounds for this view`)
    }
    this.buffer.updateValue(absoluteIndex, columnNameOrIndex, value)
  }

  /**
   * Get a window of values from a specific column
   * @param columnName - Name of the column to extract
   * @param size - Optional window size (number of most recent rows in this view)
   * @returns Array of column values (oldest to newest within this view)
   */
  window(columnName: string, size?: number): ColumnValue[] {
    const columnDef = this.buffer.getColumn(columnName)
    if (!columnDef) {
      throw new Error(`Column '${columnName}' does not exist`)
    }

    const viewSize = this.length()
    const windowSize = size !== undefined ? Math.min(size, viewSize) : viewSize
    const startIndex = this.toRow - windowSize

    const values: ColumnValue[] = []
    for (let i = Math.max(startIndex, this.fromRow); i < this.toRow; i++) {
      const value = this.buffer.getValue(i, columnDef.index)
      if (value !== undefined) {
        values.push(value)
      }
    }

    return values
  }

  /**
   * Get all rows in this view as an array (oldest to newest)
   */
  toArray(): Row[] {
    const rows: Row[] = []
    for (let i = 0; i < this.length(); i++) {
      const row = this.getRow(i)
      if (row) {
        rows.push(row)
      }
    }
    return rows
  }

  /**
   * Create a new slice that is a subset of this slice
   * @param from - Start index relative to this slice
   * @param to - End index relative to this slice
   */
  slice(from: number, to: number): DataSlice {
    const absoluteFrom = this.fromRow + from
    const absoluteTo = this.fromRow + to

    // Clamp to current slice bounds
    const clampedFrom = Math.max(absoluteFrom, this.fromRow)
    const clampedTo = Math.min(absoluteTo, this.toRow)

    return new DataSlice(this.buffer, clampedFrom, clampedTo)
  }
}
