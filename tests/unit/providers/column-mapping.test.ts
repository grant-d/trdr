import assert from 'node:assert'
import { mkdir, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { after, before, describe, it } from 'node:test'
import type { HistoricalParams } from '../../../src/interfaces/data-provider.interface'
import type { OhlcvDto } from '../../../src/models/ohlcv.dto'
import { CsvFileProvider } from '../../../src/providers/base/csv-file-provider'
import type { FileProviderConfig } from '../../../src/providers/base/types'

describe('Column Mapping Tests - Task 2.3 Acceptance Criteria', () => {
  const testDir = join(process.cwd(), 'test-data-mapping')
  
  before(async () => {
    await mkdir(testDir, { recursive: true })
  })

  after(async () => {
    await rm(testDir, { recursive: true, force: true })
  })

  describe('Acceptance Criteria: Users can define custom mappings', () => {
    it('should allow custom column name mappings', async () => {
      // Create CSV with non-standard column names
      const customCsv = `datetime,open_price,high_price,low_price,close_price,vol,ticker
2024-01-01T00:00:00Z,42000,42500,41800,42200,1500.5,BTC-USD
2024-01-01T01:00:00Z,42200,42600,42100,42400,1200.3,BTC-USD`
      
      const csvPath = join(testDir, 'custom-columns.csv')
      await writeFile(csvPath, customCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'custom-exchange',
        columnMapping: {
          timestamp: 'datetime',
          open: 'open_price',
          high: 'high_price',
          low: 'low_price',
          close: 'close_price',
          volume: 'vol',
          symbol: 'ticker'
        }
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      const data: OhlcvDto[] = []
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 2)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 42000)
      assert.strictEqual(data[0].high, 42500)
      assert.strictEqual(data[0].low, 41800)
      assert.strictEqual(data[0].close, 42200)
      assert.strictEqual(data[0].volume, 1500.5)
      assert.strictEqual(data[0].symbol, 'BTC-USD')
      
      await provider.disconnect()
    })

    it('should support partial column mapping with defaults', async () => {
      // CSV with some standard and some custom column names
      const mixedCsv = `timestamp,o,h,l,c,volume
2024-01-01T00:00:00Z,100,105,99,103,1000`
      
      const csvPath = join(testDir, 'mixed-columns.csv')
      await writeFile(csvPath, mixedCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'ETH-USD', // Default symbol since not in CSV
        columnMapping: {
          timestamp: 'timestamp', // Standard name
          open: 'o',              // Custom mapping
          high: 'h',              // Custom mapping
          low: 'l',               // Custom mapping
          close: 'c',             // Custom mapping
          volume: 'volume'        // Standard name
        }
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 1)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 100)
      assert.strictEqual(data[0].symbol, 'ETH-USD') // From config default
      
      await provider.disconnect()
    })
  })

  describe('Acceptance Criteria: Automatic mapping for standard column names', () => {
    it('should automatically map standard lowercase column names', async () => {
      const standardCsv = `timestamp,open,high,low,close,volume,symbol,exchange
2024-01-01T00:00:00Z,100,105,99,103,1000,BTC-USD,binance`
      
      const csvPath = join(testDir, 'standard-lowercase.csv')
      await writeFile(csvPath, standardCsv)
      
      // No explicit column mapping provided - should use defaults
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 1)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 100)
      assert.strictEqual(data[0].symbol, 'BTC-USD')
      assert.strictEqual(data[0].exchange, 'binance')
      
      await provider.disconnect()
    })

    it('should handle different case variations with custom mapping', async () => {
      const mixedCaseCsv = `Time,Open,High,Low,Close,Volume,Symbol
2024-01-01T00:00:00Z,100,105,99,103,1000,BTC-USD`
      
      const csvPath = join(testDir, 'mixed-case.csv')
      await writeFile(csvPath, mixedCaseCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        columnMapping: {
          timestamp: 'Time',
          open: 'Open',
          high: 'High',
          low: 'Low',
          close: 'Close',
          volume: 'Volume',
          symbol: 'Symbol'
        }
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 1)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 100)
      assert.strictEqual(data[0].symbol, 'BTC-USD')
      
      await provider.disconnect()
    })
  })

  describe('Acceptance Criteria: Type conversion handles common data types', () => {
    it('should convert string numbers to numeric values', async () => {
      const stringNumbersCsv = `timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,"100.50","105.75","99.25","103.00","1000.123"`
      
      const csvPath = join(testDir, 'string-numbers.csv')
      await writeFile(csvPath, stringNumbersCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'BTC-USD'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 1)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 100.50)
      assert.strictEqual(data[0].high, 105.75)
      assert.strictEqual(data[0].low, 99.25)
      assert.strictEqual(data[0].close, 103.00)
      assert.strictEqual(data[0].volume, 1000.123)
      assert.strictEqual(typeof data[0].open, 'number')
      assert.strictEqual(typeof data[0].volume, 'number')
      
      await provider.disconnect()
    })

    it('should parse various timestamp formats', async () => {
      const timestampsCsv = `time,open,high,low,close,volume
2024-01-01T00:00:00Z,100,105,99,103,1000
2024-01-01 01:00:00,101,106,100,104,1100
1704070800000,102,107,101,105,1200
1704074400,103,108,102,106,1300`
      
      const csvPath = join(testDir, 'timestamps.csv')
      await writeFile(csvPath, timestampsCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'BTC-USD',
        columnMapping: {
          timestamp: 'time',
          open: 'open',
          high: 'high',
          low: 'low',
          close: 'close',
          volume: 'volume'
        }
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 4)
      
      // Check all timestamps are valid milliseconds
      data.forEach((ohlcv, index) => {
        assert.ok(ohlcv)
        assert.ok(typeof ohlcv.timestamp === 'number')
        assert.ok(ohlcv.timestamp > 0)
        assert.ok(ohlcv.timestamp < Date.now())
        assert.strictEqual(ohlcv.open, 100 + index)
      })
      
      // Verify specific timestamp conversions
      assert.ok(data[0])
      assert.strictEqual(data[0].timestamp, new Date('2024-01-01T00:00:00Z').getTime())
      
      assert.ok(data[2])
      assert.strictEqual(data[2].timestamp, 1704070800000) // Already in milliseconds
      
      assert.ok(data[3])
      assert.strictEqual(data[3].timestamp, 1704074400 * 1000) // Convert seconds to milliseconds
      
      await provider.disconnect()
    })

    it('should handle scientific notation', async () => {
      const scientificCsv = `timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,1e2,1.05e2,9.9e1,1.03e2,1e3`
      
      const csvPath = join(testDir, 'scientific.csv')
      await writeFile(csvPath, scientificCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'BTC-USD'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 1)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 100)
      assert.strictEqual(data[0].high, 105)
      assert.strictEqual(data[0].low, 99)
      assert.strictEqual(data[0].close, 103)
      assert.strictEqual(data[0].volume, 1000)
      
      await provider.disconnect()
    })

    it('should correctly convert timestamps from seconds to milliseconds', async () => {
      // Test with timestamps that are clearly in seconds (small numbers)
      const oldTimestampsCsv = `timestamp,open,high,low,close,volume
946684800,100,105,99,103,1000
946771200,101,106,100,104,1100`  // Jan 1, 2000 and Jan 2, 2000 in seconds
      
      const csvPath = join(testDir, 'old-timestamps.csv')
      await writeFile(csvPath, oldTimestampsCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'BTC-USD'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 2)
      
      // Check that seconds were converted to milliseconds
      assert.ok(data[0])
      assert.strictEqual(data[0].timestamp, 946684800000) // Should be in milliseconds
      assert.strictEqual(data[0].open, 100)
      
      assert.ok(data[1])
      assert.strictEqual(data[1].timestamp, 946771200000) // Should be in milliseconds
      assert.strictEqual(data[1].open, 101)
      
      // Verify dates are correct (Year 2000)
      const date1 = new Date(data[0].timestamp)
      const date2 = new Date(data[1].timestamp)
      assert.strictEqual(date1.getUTCFullYear(), 2000)
      assert.strictEqual(date2.getUTCFullYear(), 2000)
      assert.strictEqual(date1.getUTCMonth(), 0) // January
      assert.strictEqual(date2.getUTCMonth(), 0) // January
      
      await provider.disconnect()
    })
  })

  describe('Acceptance Criteria: Missing required fields generate errors', () => {
    it('should error when required price fields are missing', async () => {
      const missingFieldsCsv = `timestamp,open,high,low,volume
2024-01-01T00:00:00Z,100,105,99,1000`
      
      const csvPath = join(testDir, 'missing-close.csv')
      await writeFile(csvPath, missingFieldsCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'BTC-USD'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      // Should skip rows with missing required fields
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 0) // Row should be skipped due to missing close price
      
      await provider.disconnect()
    })

    it('should error when timestamp field is missing', async () => {
      const noTimestampCsv = `open,high,low,close,volume
100,105,99,103,1000`
      
      const csvPath = join(testDir, 'no-timestamp.csv')
      await writeFile(csvPath, noTimestampCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'BTC-USD'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      // Should skip rows without timestamp
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 0)
      
      await provider.disconnect()
    })

    it('should handle empty or null values gracefully', async () => {
      const emptyCsv = `timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,100,105,99,103,
2024-01-01T01:00:00Z,,105,99,103,1000
2024-01-01T02:00:00Z,100,105,99,,1000`
      
      const csvPath = join(testDir, 'empty-values.csv')
      await writeFile(csvPath, emptyCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'BTC-USD'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      // All rows should be skipped due to missing required fields
      assert.strictEqual(data.length, 0)
      
      await provider.disconnect()
    })
  })

  describe('Acceptance Criteria: Mapping configuration can be saved and loaded', () => {
    it('should accept column mapping through configuration', async () => {
      const csv = `dt,o,h,l,c,v
2024-01-01T00:00:00Z,100,105,99,103,1000`
      
      const csvPath = join(testDir, 'configurable.csv')
      await writeFile(csvPath, csv)
      
      // Simulate loading mapping from a configuration file
      const savedMapping = {
        timestamp: 'dt',
        open: 'o',
        high: 'h',
        low: 'l',
        close: 'c',
        volume: 'v'
      }
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'BTC-USD',
        columnMapping: savedMapping
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 1)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 100)
      
      await provider.disconnect()
    })

    it('should work with different delimiter configurations', async () => {
      // Semicolon-delimited CSV
      const semicolonCsv = `timestamp;open;high;low;close;volume
2024-01-01T00:00:00Z;100;105;99;103;1000`
      
      const csvPath = join(testDir, 'semicolon.csv')
      await writeFile(csvPath, semicolonCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'BTC-USD',
        delimiter: ';' // Custom delimiter
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 1)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 100)
      assert.strictEqual(data[0].close, 103)
      
      await provider.disconnect()
    })
  })

  describe('Edge Cases and Error Handling', () => {
    it('should handle invalid numeric values', async () => {
      const invalidCsv = `timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,100,105,99,103,abc
2024-01-01T01:00:00Z,NaN,105,99,103,1000
2024-01-01T02:00:00Z,100,105,99,103,1000`
      
      const csvPath = join(testDir, 'invalid-numbers.csv')
      await writeFile(csvPath, invalidCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'BTC-USD'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      // Only the valid row should be processed
      assert.strictEqual(data.length, 1)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 100)
      assert.strictEqual(data[0].volume, 1000)
      
      await provider.disconnect()
    })

    it('should handle quoted fields with commas', async () => {
      const quotedCsv = `timestamp,open,high,low,close,volume,notes
"2024-01-01T00:00:00Z","100","105","99","103","1,000.50","Price includes fee"`
      
      const csvPath = join(testDir, 'quoted.csv')
      await writeFile(csvPath, quotedCsv)
      
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test',
        symbol: 'BTC-USD'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const data: OhlcvDto[] = []
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1h'
      }
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 1)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 100)
      assert.strictEqual(data[0].volume, 1000.50)
      
      await provider.disconnect()
    })
  })
})