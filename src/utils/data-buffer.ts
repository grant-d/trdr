/**
 * Type representing a single column value
 */
export type ColumnValue = number

/**
 * Type representing a row of data with named columns
 */
export type Row = Record<string, ColumnValue>

/**
 * Column definition with metadata
 */
export interface ColumnDefinition {
  /** Column index in the row array */
  index: number
  /** Default value for this column */
  defaultValue?: ColumnValue
}

/**
 * Configuration for the DataBuffer
 */
export interface DataBufferConfig {
  /** Column definitions keyed by column name */
  columns: Record<string, ColumnDefinition>
}

/**
 * A rectangular data buffer that acts as a FIFO queue
 * Ensures all rows have the same columns in the same order
 */
export class DataBuffer {
  private readonly columns: Record<string, ColumnDefinition>
  private readonly columnNames: string[]
  private readonly buffer: ColumnValue[][] = []

  constructor(config: DataBufferConfig) {
    const columnCount = Object.keys(config.columns).length
    if (columnCount === 0) {
      throw new Error('DataBuffer requires at least one column')
    }

    // Validate indices are unique and collect all indices
    const indices = new Set<number>()
    const names: string[] = []
    let maxIndex = -1

    for (const [name, def] of Object.entries(config.columns)) {
      if (def.index < 0) {
        throw new Error(
          `Column '${name}' has invalid index ${def.index}. Index must be non-negative`
        )
      }
      if (indices.has(def.index)) {
        throw new Error(`Duplicate index ${def.index} for column '${name}'`)
      }
      indices.add(def.index)
      names.push(name)
      maxIndex = Math.max(maxIndex, def.index)
    }

    // Check if indices are sequential from 0 to maxIndex
    for (let i = 0; i <= maxIndex; i++) {
      if (!indices.has(i)) {
        throw new Error(`Missing column definition for index ${i}`)
      }
    }

    // Ensure the number of columns matches the highest index + 1
    if (maxIndex + 1 !== columnCount) {
      throw new Error(
        `Invalid column indices: expected indices 0-${columnCount - 1}, but highest index is ${maxIndex}`
      )
    }

    // Sort names by index for consistent ordering
    names.sort((a, b) => config.columns[a]!.index - config.columns[b]!.index)

    this.columns = config.columns
    this.columnNames = names
  }

  ensureColumn(outField: string): number {
    const def = this.getColumn(outField)
    if (def) return def.index

    const index = this.columnNames.length
    this.columns[outField] = { index }
    this.columnNames.push(outField)
    return index
  }

  /**
   * Get the column names
   */
  getColumns(): readonly string[] {
    return this.columnNames
  }

  /**
   * Get column definitions
   */
  getColumnDefinitions(): Readonly<Record<string, ColumnDefinition>> {
    return this.columns
  }

  /**
   * Check if a column exists
   */
  hasColumn(name: string): boolean {
    return name in this.columns
  }

  getColumn(name: string): ColumnDefinition | undefined {
    return this.columns[name]
  }

  /**
   * Get the number of rows in the buffer (alias for length)
   */
  length(): number {
    return this.buffer.length
  }

  /**
   * Check if the buffer is empty
   */
  isEmpty(): boolean {
    return this.buffer.length === 0
  }

  /**
   * Push a row to the front of the buffer (enqueue)
   * Missing columns will be filled with the default value
   */
  push(row: Row): void {
    // Create array with values in column order
    const rowArray: ColumnValue[] = new Array(this.columnNames.length)

    for (const [name, def] of Object.entries(this.columns)) {
      if (name in row) {
        rowArray[def.index] = row[name] || 0.0
      } else {
        rowArray[def.index] = def.defaultValue || 0.0
      }
    }

    // Add to end of buffer (forward order)
    this.buffer.push(rowArray)
  }

  /**
   * Push multiple rows at once
   */
  pushMany(rows: Row[]): void {
    for (const row of rows) {
      this.push(row)
    }
  }

  /**
   * Remove and return a row from the back of the buffer (dequeue)
   * Returns undefined if buffer is empty
   */
  pop(): Row | undefined {
    const rowArray = this.buffer.pop()
    if (!rowArray) {
      return undefined
    }

    return this.arrayToRow(rowArray)
  }

  /**
   * Remove and return multiple rows from the back
   */
  popMany(count: number): Row[] {
    const rows: Row[] = []
    for (let i = 0; i < count && !this.isEmpty(); i++) {
      const row = this.pop()
      if (row) {
        rows.push(row)
      }
    }
    return rows
  }

  /**
   * Peek at the row at the back without removing it
   */
  peekBack(): Row | undefined {
    if (this.isEmpty()) {
      return undefined
    }

    const rowArray = this.buffer[this.buffer.length - 1]!
    return this.arrayToRow(rowArray)
  }

  /**
   * Peek at the row at the front without removing it
   */
  peekFront(): Row | undefined {
    if (this.isEmpty()) {
      return undefined
    }

    const rowArray = this.buffer[0]!
    return this.arrayToRow(rowArray)
  }

  /**
   * Get all rows as an array (oldest to newest)
   */
  toArray(): Row[] {
    // Return in reverse order (back to front)
    return this.buffer
      .slice()
      .reverse()
      .map((rowArray) => this.arrayToRow(rowArray))
  }

  /**
   * Clear all data from the buffer
   */
  clear(): void {
    this.buffer.length = 0
  }

  /**
   * Get a window of values from a specific column
   * @param columnName - Name of the column to extract
   * @param size - Optional window size (number of most recent rows)
   * @returns Array of column values (oldest to newest)
   */
  window(columnName: string, size?: number): ColumnValue[] {
    const columnDef = this.columns[columnName]
    if (!columnDef) {
      throw new Error(`Column '${columnName}' does not exist`)
    }

    const columnIndex = columnDef.index
    const startIndex =
      size !== undefined ? Math.max(0, this.buffer.length - size) : 0

    // Extract column values (oldest to newest)
    const values: ColumnValue[] = []
    for (let i = startIndex; i < this.buffer.length; i++) {
      values.push(this.buffer[i]![columnIndex]!)
    }

    return values
  }

  /**
   * Get a window of values from a column by its index
   * @param columnIndex - Index of the column to extract
   * @param size - Optional window size (number of most recent rows)
   * @returns Array of column values (newest to oldest, where index 0 is most recent)
   */
  windowByIndex(columnIndex: number, size?: number): ColumnValue[] {
    if (columnIndex < 0 || columnIndex >= this.columnNames.length) {
      throw new Error(
        `Column index ${columnIndex} is out of range (0-${this.columnNames.length - 1})`
      )
    }

    const limit =
      size !== undefined
        ? Math.min(size, this.buffer.length)
        : this.buffer.length

    // Extract column values (newest to oldest)
    const values: ColumnValue[] = []
    for (let i = 0; i < limit; i++) {
      values.push(this.buffer[i]![columnIndex]!)
    }

    return values
  }

  /**
   * Get a row by its reverse index (0 = most recent)
   * @param index - Reverse index where 0 is the most recent row
   * @returns The row at the specified index, or undefined if out of bounds
   */
  getRow(index: number): Row | undefined {
    if (index < 0 || index >= this.buffer.length) {
      return undefined
    }

    return this.arrayToRow(this.buffer[index]!)
  }

  getValue(
    rowIndex: number,
    columnNameOrIndex: string | number
  ): ColumnValue | undefined {
    if (rowIndex < 0 || rowIndex >= this.buffer.length) {
      throw new Error(`Row index ${rowIndex} is out of bounds`)
    }

    let columnIndex: number
    if (typeof columnNameOrIndex === 'number') {
      // Direct index provided - validate it
      if (
        columnNameOrIndex < 0 ||
        columnNameOrIndex >= this.columnNames.length
      ) {
        throw new Error(
          `Column index ${columnNameOrIndex} is out of range (0-${this.columnNames.length - 1})`
        )
      }
      columnIndex = columnNameOrIndex
    } else {
      // Column name provided - look up index
      const columnDef = this.columns[columnNameOrIndex]
      if (!columnDef) {
        throw new Error(`Column '${columnNameOrIndex}' does not exist`)
      }
      columnIndex = columnDef.index
    }

    return this.buffer[rowIndex]![columnIndex] ?? 0
  }

  /**
   * Update a value in a specific row and column
   * @param rowIndex - Reverse index where 0 is the most recent row
   * @param columnNameOrIndex - Name of the column or its index for better performance
   * @param value - New value to set
   */
  updateValue(
    rowIndex: number,
    columnNameOrIndex: string | number,
    value: ColumnValue
  ): void {
    if (rowIndex < 0 || rowIndex >= this.buffer.length) {
      throw new Error(`Row index ${rowIndex} is out of bounds`)
    }

    let columnIndex: number
    if (typeof columnNameOrIndex === 'number') {
      // Direct index provided - validate it
      if (
        columnNameOrIndex < 0 ||
        columnNameOrIndex >= this.columnNames.length
      ) {
        throw new Error(
          `Column index ${columnNameOrIndex} is out of range (0-${this.columnNames.length - 1})`
        )
      }
      columnIndex = columnNameOrIndex
    } else {
      // Column name provided - look up index
      const columnDef = this.columns[columnNameOrIndex]
      if (!columnDef) {
        throw new Error(`Column '${columnNameOrIndex}' does not exist`)
      }
      columnIndex = columnDef.index
    }

    this.buffer[rowIndex]![columnIndex] = value
  }

  /**
   * Create a new buffer with additional columns
   */
  withColumns(
    newColumns: Record<
      string,
      Omit<ColumnDefinition, 'index'> & { defaultValue?: ColumnValue }
    >
  ): DataBuffer {
    // Create new column definitions
    const newColumnDefs: Record<string, ColumnDefinition> = { ...this.columns }
    let nextIndex = this.columnNames.length

    for (const [name, def] of Object.entries(newColumns)) {
      if (!(name in newColumnDefs)) {
        newColumnDefs[name] = {
          index: nextIndex++,
          defaultValue: def.defaultValue ?? 0.0
        }
      }
    }

    const newBuffer = new DataBuffer({
      columns: newColumnDefs
    })

    // Copy existing data
    for (const row of this.toArray()) {
      newBuffer.push(row)
    }

    return newBuffer
  }

  /**
   * Convert internal array representation to Row object
   */
  private arrayToRow(rowArray: ColumnValue[]): Row {
    const row: Row = {}
    for (const [name, def] of Object.entries(this.columns)) {
      row[name] = rowArray[def.index] ?? 0
    }
    return row
  }
}
