import { deepEqual, strictEqual, throws } from 'node:assert'
import { describe, it } from 'node:test'
import type { ColumnDefinition, Row } from '../../../src/utils'
import { DataBuffer } from '../../../src/utils'

// Helper to create column definitions from array of names
function createColumns(names: string[]): Record<string, ColumnDefinition> {
  const columns: Record<string, ColumnDefinition> = {}
  names.forEach((name, index) => {
    columns[name] = { index }
  })
  return columns
}

describe('DataBuffer', () => {
  describe('constructor', () => {
    it('should create buffer with valid columns', () => {
      const buffer = new DataBuffer({
        columns: {
          a: { index: 0 },
          b: { index: 1 },
          c: { index: 2 }
        }
      })
      deepEqual(buffer.getColumns(), ['a', 'b', 'c'])
      strictEqual(buffer.length(), 0)
      strictEqual(buffer.isEmpty(), true)
    })

    it('should throw on empty columns', () => {
      throws(
        () => new DataBuffer({ columns: {} }),
        /DataBuffer requires at least one column/
      )
    })

    it('should throw on duplicate indices', () => {
      throws(
        () => new DataBuffer({ columns: { a: { index: 0 }, b: { index: 0 } } }),
        /Duplicate index/
      )
    })

    it('should accept default value configuration', () => {
      const buffer = new DataBuffer({
        columns: {
          a: { index: 0 },
          b: { index: 1, defaultValue: -1 }
        }
      })

      buffer.push({ a: 1 }) // missing 'b'
      const row = buffer.pop()
      deepEqual(row, { a: 1, b: -1 })
    })
  })

  describe('push and pop (FIFO queue)', () => {
    it('should push to front and pop from back', () => {
      const buffer = new DataBuffer({
        columns: createColumns(['id', 'value'])
      })

      buffer.push({ id: 1, value: 1 })
      buffer.push({ id: 2, value: 2 })
      buffer.push({ id: 3, value: 3 })

      strictEqual(buffer.length(), 3)

      // Pop should return oldest (FIFO)
      deepEqual(buffer.pop(), { id: 1, value: 1 })
      deepEqual(buffer.pop(), { id: 2, value: 2 })
      deepEqual(buffer.pop(), { id: 3, value: 3 })
      strictEqual(buffer.pop(), undefined)
    })

    it('should handle missing columns with default null', () => {
      const buffer = new DataBuffer({
        columns: createColumns(['a', 'b', 'c'])
      })

      buffer.push({ a: 1 })
      buffer.push({ b: 2 })
      buffer.push({ a: 3, c: 3 })

      deepEqual(buffer.pop(), { a: 1, b: null, c: null })
      deepEqual(buffer.pop(), { a: null, b: 2, c: null })
      deepEqual(buffer.pop(), { a: 3, b: null, c: 3 })
    })

    it('should ignore extra columns not in schema', () => {
      const buffer = new DataBuffer({ columns: createColumns(['a', 'b']) })

      buffer.push({ a: 1, b: 2, c: 3, d: 4 })
      deepEqual(buffer.pop(), { a: 1, b: 2 })
    })
  })

  describe('pushMany and popMany', () => {
    it('should push multiple rows at once', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })

      buffer.pushMany([{ x: 1 }, { x: 2 }, { x: 3 }])

      strictEqual(buffer.length(), 3)
      deepEqual(buffer.toArray(), [{ x: 1 }, { x: 2 }, { x: 3 }])
    })

    it('should pop multiple rows at once', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })

      buffer.pushMany([{ x: 1 }, { x: 2 }, { x: 3 }, { x: 4 }])

      const rows = buffer.popMany(2)
      deepEqual(rows, [{ x: 1 }, { x: 2 }])
      strictEqual(buffer.length(), 2)
    })

    it('should handle popMany with count > size', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })
      buffer.pushMany([{ x: 1 }, { x: 2 }])

      const rows = buffer.popMany(5)
      deepEqual(rows, [{ x: 1 }, { x: 2 }])
      strictEqual(buffer.isEmpty(), true)
    })
  })

  describe('peek operations', () => {
    it('should peek at front and back without removing', () => {
      const buffer = new DataBuffer({ columns: createColumns(['n']) })

      buffer.push({ n: 1 })
      buffer.push({ n: 2 })
      buffer.push({ n: 3 })

      deepEqual(buffer.peekFront(), { n: 3 }) // newest
      deepEqual(buffer.peekBack(), { n: 1 }) // oldest
      strictEqual(buffer.length(), 3) // unchanged
    })

    it('should return undefined when peeking empty buffer', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })

      strictEqual(buffer.peekFront(), undefined)
      strictEqual(buffer.peekBack(), undefined)
    })
  })

  describe('toArray and toMatrix', () => {
    it('should convert to array in correct order', () => {
      const buffer = new DataBuffer({ columns: createColumns(['id', 'name']) })

      buffer.push({ id: 1, name: 100 })
      buffer.push({ id: 2, name: 200 })
      buffer.push({ id: 3, name: 300 })

      deepEqual(buffer.toArray(), [
        { id: 1, name: 100 },
        { id: 2, name: 200 },
        { id: 3, name: 300 }
      ])
    })

    it('should handle missing values in matrix', () => {
      const buffer = new DataBuffer({
        columns: {
          x: { index: 0, defaultValue: 0 },
          y: { index: 1, defaultValue: 0 }
        }
      })

      buffer.push({ x: 1 })
      buffer.push({ y: 2 })

      // toMatrix method removed - verifying data using toArray
      deepEqual(buffer.toArray(), [
        { x: 1, y: 0 },
        { x: 0, y: 2 }
      ])
    })
  })

  describe('clear', () => {
    it('should remove all data', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })

      buffer.pushMany([{ x: 1 }, { x: 2 }, { x: 3 }])
      strictEqual(buffer.length(), 3)

      buffer.clear()
      strictEqual(buffer.length(), 0)
      strictEqual(buffer.isEmpty(), true)
    })
  })

  describe('withColumns', () => {
    it('should create new buffer with additional columns', () => {
      const buffer = new DataBuffer({ columns: createColumns(['a', 'b']) })
      buffer.push({ a: 1, b: 2 })
      buffer.push({ a: 3, b: 4 })

      const newBuffer = buffer.withColumns({ c: {}, d: {} })

      deepEqual(newBuffer.getColumns(), ['a', 'b', 'c', 'd'])
      deepEqual(newBuffer.toArray(), [
        { a: 1, b: 2, c: null, d: null },
        { a: 3, b: 4, c: null, d: null }
      ])
    })

    it('should not duplicate existing columns', () => {
      const buffer = new DataBuffer({ columns: createColumns(['a', 'b']) })
      const newBuffer = buffer.withColumns({ b: {}, c: {} })

      deepEqual(newBuffer.getColumns(), ['a', 'b', 'c'])
    })

    it('should preserve original buffer unchanged', () => {
      const buffer = new DataBuffer({ columns: createColumns(['a']) })
      buffer.push({ a: 1 })

      const newBuffer = buffer.withColumns({ b: {} })
      newBuffer.push({ a: 2, b: 2 })

      strictEqual(buffer.length(), 1)
      deepEqual(buffer.getColumns(), ['a'])
      deepEqual(buffer.pop(), { a: 1 })

      strictEqual(newBuffer.length(), 2)
      deepEqual(newBuffer.getColumns(), ['a', 'b'])
    })

    it('should use new default value if provided', () => {
      const buffer = new DataBuffer({
        columns: createColumns(['a'])
      })

      const newBuffer = buffer.withColumns({ b: { defaultValue: -999 } })
      newBuffer.push({ a: 1 })

      deepEqual(newBuffer.pop(), { a: 1, b: -999 })
    })
  })

  describe('data type handling', () => {
    it('should handle numeric data types', () => {
      const buffer = new DataBuffer({
        columns: createColumns(['a', 'b', 'c', 'd', 'e'])
      })

      buffer.push({
        a: 1,
        b: 42,
        c: 1, // representing true as 1
        d: 0, // representing null as 0
        e: 0  // representing undefined as 0
      })

      const row = buffer.pop()
      strictEqual(row?.a, 1)
      strictEqual(row?.b, 42)
      strictEqual(row?.c, 1)
      strictEqual(row?.d, 0)
      strictEqual(row?.e, 0)
    })
  })

  describe('window', () => {
    it('should get all values from a column (newest to oldest)', () => {
      const buffer = new DataBuffer({
        columns: createColumns(['a', 'b', 'c'])
      })

      buffer.push({ a: 1, b: 2, c: 3 })
      buffer.push({ a: 4, b: 5, c: 6 })
      buffer.push({ a: 7, b: 8, c: 9 })

      // Values returned newest to oldest
      deepEqual(buffer.window('b'), [8, 5, 2])
      deepEqual(buffer.window('a'), [7, 4, 1])
      deepEqual(buffer.window('c'), [9, 6, 3])
    })

    it('should get column with specified window size', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x', 'y']) })

      for (let i = 0; i < 10; i++) {
        buffer.push({ x: i, y: i * 10 })
      }

      // Get most recent 3 values (newest to oldest)
      deepEqual(buffer.window('x', 3), [9, 8, 7])

      // Get most recent 5 values
      deepEqual(buffer.window('y', 5), [90, 80, 70, 60, 50])

      // Window larger than buffer size returns all values
      deepEqual(buffer.window('x', 20), [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

      // Get just the most recent value
      deepEqual(buffer.window('x', 1), [9])
    })

    it('should return empty array for empty buffer', () => {
      const buffer = new DataBuffer({ columns: createColumns(['a']) })
      deepEqual(buffer.window('a'), [])
      deepEqual(buffer.window('a', 5), [])
    })

    it('should handle window size of zero', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })
      buffer.push({ x: 1 })
      buffer.push({ x: 2 })

      deepEqual(buffer.window('x', 0), [])
    })

    it('should throw for non-existent column', () => {
      const buffer = new DataBuffer({ columns: createColumns(['a', 'b']) })

      throws(() => buffer.window('c'), /Column 'c' does not exist/)
    })

    it('should handle column with mixed types', () => {
      const buffer = new DataBuffer({
        columns: {
          mixed: { index: 0, defaultValue: 0 }
        }
      })

      buffer.push({ mixed: 1 })
      buffer.push({ mixed: 100 })
      buffer.push({ mixed: 1 })
      buffer.push({ mixed: 0 })
      buffer.push({}) // Uses default

      deepEqual(buffer.window('mixed'), [0, 0, 1, 100, 1])
    })

    it('should handle sequential window requests correctly', () => {
      const buffer = new DataBuffer({ columns: createColumns(['price']) })

      // Simulate price data
      const prices = [100, 102, 98, 103, 101, 99, 104]
      prices.forEach((p) => buffer.push({ price: p }))

      // Multiple window requests should be consistent
      const window1 = buffer.window('price', 3)
      const window2 = buffer.window('price', 3)
      deepEqual(window1, window2)
      deepEqual(window1, [104, 99, 101])

      // Add new data
      buffer.push({ price: 106 })

      // Window should reflect new data
      deepEqual(buffer.window('price', 3), [106, 104, 99])
    })
  })

  describe('windowByIndex', () => {
    it('should get column values by index (newest to oldest)', () => {
      const buffer = new DataBuffer({
        columns: createColumns(['a', 'b', 'c'])
      })

      buffer.push({ a: 1, b: 2, c: 3 })
      buffer.push({ a: 4, b: 5, c: 6 })

      deepEqual(buffer.windowByIndex(0), [4, 1]) // column 'a' newest to oldest
      deepEqual(buffer.windowByIndex(1), [5, 2]) // column 'b' newest to oldest
      deepEqual(buffer.windowByIndex(2), [6, 3]) // column 'c' newest to oldest
    })

    it('should support windowing', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })

      for (let i = 0; i < 5; i++) {
        buffer.push({ x: i })
      }

      // Get most recent 2 values
      deepEqual(buffer.windowByIndex(0, 2), [4, 3])

      // Get most recent 3 values
      deepEqual(buffer.windowByIndex(0, 3), [4, 3, 2])

      // Window larger than buffer
      deepEqual(buffer.windowByIndex(0, 10), [4, 3, 2, 1, 0])
    })

    it('should throw for invalid index', () => {
      const buffer = new DataBuffer({ columns: createColumns(['a', 'b']) })

      throws(() => buffer.windowByIndex(2), /Column index 2 is out of range/)

      throws(() => buffer.windowByIndex(-1), /Column index -1 is out of range/)
    })

    it('should return empty for empty buffer', () => {
      const buffer = new DataBuffer({ columns: createColumns(['a', 'b']) })

      deepEqual(buffer.windowByIndex(0), [])
      deepEqual(buffer.windowByIndex(1, 5), [])
    })

    it('should handle zero window size', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })
      buffer.push({ x: 1 })

      deepEqual(buffer.windowByIndex(0, 0), [])
    })
  })

  describe('getRow', () => {
    it('should get rows by reverse index', () => {
      const buffer = new DataBuffer({ columns: createColumns(['a', 'b']) })

      buffer.push({ a: 1, b: 10 })
      buffer.push({ a: 2, b: 20 })
      buffer.push({ a: 3, b: 30 })

      // Index 0 is most recent
      deepEqual(buffer.getRow(0), { a: 3, b: 30 })
      deepEqual(buffer.getRow(1), { a: 2, b: 20 })
      deepEqual(buffer.getRow(2), { a: 1, b: 10 })

      // Out of bounds returns undefined
      strictEqual(buffer.getRow(3), undefined)
      strictEqual(buffer.getRow(-1), undefined)
    })
  })

  describe('updateValue', () => {
    it('should update values in-place', () => {
      const buffer = new DataBuffer({
        columns: createColumns(['a', 'b', 'c'])
      })

      buffer.push({ a: 1, b: 2, c: 3 })
      buffer.push({ a: 4, b: 5, c: 6 })

      // Update value in most recent row
      buffer.updateValue(0, 'b', 99)
      deepEqual(buffer.getRow(0), { a: 4, b: 99, c: 6 })

      // Update value in older row
      buffer.updateValue(1, 'a', 77)
      deepEqual(buffer.getRow(1), { a: 77, b: 2, c: 3 })
    })

    it('should throw on invalid row index', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })
      buffer.push({ x: 1 })

      throws(
        () => buffer.updateValue(2, 'x', 99),
        /Row index 2 is out of bounds/
      )

      throws(
        () => buffer.updateValue(-1, 'x', 99),
        /Row index -1 is out of bounds/
      )
    })

    it('should throw on invalid column name', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })
      buffer.push({ x: 1 })

      throws(() => buffer.updateValue(0, 'y', 99), /Column 'y' does not exist/)
    })

    it('should handle various value types', () => {
      const buffer = new DataBuffer({ columns: createColumns(['val']) })
      buffer.push({ val: 1 })

      // Update to number
      buffer.updateValue(0, 'val', 42)
      strictEqual(buffer.getRow(0)?.val, 42)

      // Update to numeric values only
      buffer.updateValue(0, 'val', 1)
      strictEqual(buffer.getRow(0)?.val, 1)

      // Update to zero
      buffer.updateValue(0, 'val', 0)
      strictEqual(buffer.getRow(0)?.val, 0)

      // Update to negative
      buffer.updateValue(0, 'val', -1)
      strictEqual(buffer.getRow(0)?.val, -1)

      // Update to decimal
      buffer.updateValue(0, 'val', 99.5)
      strictEqual(buffer.getRow(0)?.val, 99.5)
    })

    it('should update values using column index', () => {
      const buffer = new DataBuffer({
        columns: createColumns(['a', 'b', 'c'])
      })

      buffer.push({ a: 1, b: 2, c: 3 })
      buffer.push({ a: 4, b: 5, c: 6 })

      // Update using column index (b is index 1)
      buffer.updateValue(0, 1, 99)
      deepEqual(buffer.getRow(0), { a: 4, b: 99, c: 6 })

      // Update using column index (a is index 0)
      buffer.updateValue(1, 0, 77)
      deepEqual(buffer.getRow(1), { a: 77, b: 2, c: 3 })

      // Update using column index (c is index 2)
      buffer.updateValue(0, 2, 88)
      deepEqual(buffer.getRow(0), { a: 4, b: 99, c: 88 })
    })

    it('should throw on invalid column index', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x', 'y']) })
      buffer.push({ x: 1, y: 2 })

      throws(
        () => buffer.updateValue(0, 2, 99),
        /Column index 2 is out of range \(0-1\)/
      )

      throws(
        () => buffer.updateValue(0, -1, 99),
        /Column index -1 is out of range \(0-1\)/
      )
    })
  })

  describe('edge cases', () => {
    it('should handle single column buffer', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })
      buffer.push({ x: 1 })
      deepEqual(buffer.pop(), { x: 1 })

      // Window operations on single column
      buffer.push({ x: 2 })
      buffer.push({ x: 3 })
      deepEqual(buffer.window('x', 2), [3, 2])
    })

    it('should handle many columns', () => {
      const columns = Array.from({ length: 100 }, (_, i) => `col${i}`)
      const buffer = new DataBuffer({ columns: createColumns(columns) })

      const row: Row = {}
      columns.forEach((col, i) => {
        row[col] = i
      })

      buffer.push(row)
      const result = buffer.pop()

      columns.forEach((col, i) => {
        strictEqual(result?.[col], i)
      })
    })

    it('should maintain rectangular shape with varying input', () => {
      const buffer = new DataBuffer({
        columns: createColumns(['a', 'b', 'c'])
      })

      // Push rows with different fields present
      buffer.push({ a: 1 })
      buffer.push({ b: 2 })
      buffer.push({ c: 3 })
      buffer.push({ a: 4, b: 5 })
      buffer.push({ b: 6, c: 7 })
      buffer.push({ a: 8, c: 9 })
      buffer.push({ a: 10, b: 11, c: 12 })

      // All rows should have all columns
      const array = buffer.toArray()
      strictEqual(array.length, 7) // 7 rows

      // Each row should have same number of columns
      for (const row of array) {
        strictEqual(Object.keys(row).length, 3)
      }
    })

    it('should handle column operations after clear', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x', 'y']) })

      // Add some data
      buffer.push({ x: 1, y: 10 })
      buffer.push({ x: 2, y: 20 })

      // Clear
      buffer.clear()

      // Window operations should return empty
      deepEqual(buffer.window('x'), [])
      deepEqual(buffer.window('y', 5), [])

      // Add new data should work normally
      buffer.push({ x: 3, y: 30 })
      deepEqual(buffer.window('x'), [3])
    })

    it('should handle pushMany with empty array', () => {
      const buffer = new DataBuffer({ columns: createColumns(['a']) })
      buffer.pushMany([])
      strictEqual(buffer.length(), 0)
    })

    it('should handle popMany with count larger than buffer', () => {
      const buffer = new DataBuffer({ columns: createColumns(['x']) })
      buffer.push({ x: 1 })
      buffer.push({ x: 2 })

      const results = buffer.popMany(5)
      strictEqual(results.length, 2)
      strictEqual(buffer.length(), 0)
    })

    it('should maintain column order in toArray', () => {
      const buffer = new DataBuffer({
        columns: {
          z: { index: 2 },
          a: { index: 0 },
          m: { index: 1 }
        }
      })

      buffer.push({ z: 1, a: 10, m: 100 })
      buffer.push({ z: 2, a: 20, m: 200 })

      const arr = buffer.toArray()
      // Should have proper keys despite insertion order
      deepEqual(Object.keys(arr[0]!), ['z', 'a', 'm'])
      deepEqual(arr[0], { z: 1, a: 10, m: 100 })
      deepEqual(arr[1], { z: 2, a: 20, m: 200 })
    })

    it('should throw on invalid column definition', () => {
      // Missing index 1
      throws(
        () =>
          new DataBuffer({
            columns: {
              a: { index: 0 },
              b: { index: 2 }
            }
          }),
        /Missing column definition for index 1/
      )

      // Negative index
      throws(
        () =>
          new DataBuffer({
            columns: {
              a: { index: -1 }
            }
          }),
        /invalid index/
      )

      // Index out of range
      throws(
        () =>
          new DataBuffer({
            columns: {
              a: { index: 0 },
              b: { index: 3 } // Missing indices 1 and 2
            }
          }),
        /Missing column definition for index 1/
      )

      // Duplicate index
      throws(
        () =>
          new DataBuffer({
            columns: {
              a: { index: 0 },
              b: { index: 0 }
            }
          }),
        /Duplicate index/
      )
    })

    it('should preserve undefined values correctly', () => {
      const buffer = new DataBuffer({
        columns: {
          a: { index: 0, defaultValue: -1 },
          b: { index: 1, defaultValue: 0 }
        }
      })

      // Test with numeric values only
      buffer.push({ a: 999, b: 888 })
      const row = buffer.pop()
      strictEqual(row?.a, 999)
      strictEqual(row?.b, 888)

      // Missing values should use defaults
      buffer.push({})
      const row2 = buffer.pop()
      strictEqual(row2?.a, -1)
      strictEqual(row2?.b, 0)
    })
  })
})
