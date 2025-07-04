import assert from 'node:assert'
import { join } from 'node:path'
import { describe, it } from 'node:test'
import type { HistoricalParams } from '../../../src/interfaces/data-provider.interface'
import type { OhlcvDto } from '../../../src/models/ohlcv.dto'
import { CsvFileProvider } from '../../../src/providers/base/csv-file-provider'
import { JsonlFileProvider } from '../../../src/providers/base/jsonl-file-provider'
import { FileProviderConfig } from '../../../src/providers/base/types'

describe('Real Data Provider Tests', () => {
  const testDir = join(process.cwd(), 'tests/unit/providers')
  const csvPath = join(testDir, 'BTCUSD-short.csv')
  const jsonlPath = join(testDir, 'BTCUSD-short.jsonl')
  
  // For unit tests, we'll limit data reads to improve performance
  const TEST_ROW_LIMIT = 50

  describe('CSV Provider with Real BTC-USD Data', () => {
    it('should read Yahoo Finance format CSV', async () => {
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'yahoo',
        symbol: 'BTC-USD',
        columnMapping: {
          timestamp: 'Date',
          open: 'Open',
          high: 'High',
          low: 'Low',
          close: 'Close',
          volume: 'Volume',
        }
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1d'
      }
      
      const data: OhlcvDto[] = []
      const iterator = provider.getHistoricalData(params)
      
      // For unit tests, limit rows to avoid timeout
      let count = 0
      for await (const ohlcv of iterator) {
        data.push(ohlcv)
        count++
        if (count >= TEST_ROW_LIMIT) break
      }
      
      // Verify we read the expected number of rows
      assert.strictEqual(data.length, TEST_ROW_LIMIT)
      
      // Check first row (2017-10-04)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 4214.84)
      assert.strictEqual(data[0].high, 4241.15)
      assert.strictEqual(data[0].low, 4151.0)
      assert.strictEqual(data[0].close, 4165.0)
      assert.strictEqual(data[0].volume, 1295.29733578)
      assert.strictEqual(data[0].symbol, 'BTC-USD')
      assert.strictEqual(data[0].exchange, 'yahoo')
      
      // Check date parsing
      const firstDate = new Date(data[0].timestamp)
      assert.strictEqual(firstDate.getUTCFullYear(), 2017)
      assert.strictEqual(firstDate.getUTCMonth(), 9) // October is month 9
      assert.strictEqual(firstDate.getUTCDate(), 4)
      
      await provider.disconnect()
    })

    it('should filter by date range', async () => {
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'yahoo',
        symbol: 'BTC-USD',
        columnMapping: {
          timestamp: 'Date',
          open: 'Open',
          high: 'High',
          low: 'Low',
          close: 'Close',
          volume: 'Volume',
        }
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      // Filter to January 2018
      const params: HistoricalParams = {
        symbols: [],
        start: new Date('2018-01-01').getTime(),
        end: new Date('2018-01-31').getTime(),
        timeframe: '1d'
      }
      
      const data: OhlcvDto[] = []
      const iterator = provider.getHistoricalData(params)
      for await (const ohlcv of iterator) {
        data.push(ohlcv)
      }
      
      // Should have January 2018 data only
      assert.ok(data.length > 0)
      assert.ok(data.length < 50) // Should be around 31 days
      
      // All dates should be in January 2018
      for (const ohlcv of data) {
        const date = new Date(ohlcv.timestamp)
        assert.strictEqual(date.getUTCFullYear(), 2018)
        assert.strictEqual(date.getUTCMonth(), 0) // January
      }
      
      await provider.disconnect()
    })

    it('should handle chunk processing efficiently', async () => {
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'yahoo',
        symbol: 'BTC-USD',
        chunkSize: 50, // Small chunks to test streaming
        columnMapping: {
          timestamp: 'Date',
          open: 'Open',
          high: 'High',
          low: 'Low',
          close: 'Close',
          volume: 'Volume',
        }
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1d'
      }
      
      let count = 0
      const startMemory = process.memoryUsage().heapUsed
      
      const iterator = provider.getHistoricalData(params)
      for await (const ohlcv of iterator) {
        count++
        // Verify data integrity
        assert.ok(ohlcv.high >= ohlcv.low)
        assert.ok(ohlcv.high >= ohlcv.open)
        assert.ok(ohlcv.high >= ohlcv.close)
        assert.ok(ohlcv.low <= ohlcv.open)
        assert.ok(ohlcv.low <= ohlcv.close)
        
        // Limit for unit test performance
        if (count >= TEST_ROW_LIMIT) break
      }
      
      const endMemory = process.memoryUsage().heapUsed
      const memoryIncrease = (endMemory - startMemory) / 1024 / 1024 // MB
      
      assert.strictEqual(count, TEST_ROW_LIMIT)
      assert.ok(memoryIncrease < 20, `Memory increase should be minimal, but was ${memoryIncrease}MB`)
      
      await provider.disconnect()
    })
  })

  describe('Jsonl Provider with Real BTC-USD Data', () => {
    it('should read Jsonl file created from CSV', async () => {
      const config: FileProviderConfig = {
        path: jsonlPath,
        format: 'jsonl',
        exchange: 'yahoo',
        symbol: 'BTC-USD'
      }
      
      const provider = new JsonlFileProvider(config)
      await provider.connect()
      
      const params: HistoricalParams = {
        symbols: [],
        start: 0,
        end: Date.now(),
        timeframe: '1d'
      }
      
      const data: OhlcvDto[] = []
      const iterator = provider.getHistoricalData(params)
      
      // For unit tests, limit rows to avoid timeout
      let count = 0
      for await (const ohlcv of iterator) {
        data.push(ohlcv)
        count++
        if (count >= TEST_ROW_LIMIT) break
      }
      
      // Verify we read the expected number of rows
      assert.strictEqual(data.length, TEST_ROW_LIMIT)
      
      // Check first row matches CSV data
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 4214.84)
      assert.strictEqual(data[0].high, 4241.15)
      assert.strictEqual(data[0].low, 4151.0)
      assert.strictEqual(data[0].close, 4165.0)
      assert.strictEqual(data[0].volume, 1295.29733578)
      assert.strictEqual(data[0].symbol, 'BTC-USD')
      assert.strictEqual(data[0].exchange, 'yahoo')
      
      await provider.disconnect()
    })

    it('should filter Jsonl data by date range', async () => {
      const config: FileProviderConfig = {
        path: jsonlPath,
        format: 'jsonl',
        exchange: 'yahoo',
        symbol: 'BTC-USD'
      }
      
      const provider = new JsonlFileProvider(config)
      await provider.connect()
      
      // Filter to December 2017
      const params: HistoricalParams = {
        symbols: [],
        start: new Date('2017-12-01').getTime(),
        end: new Date('2017-12-31').getTime(),
        timeframe: '1d'
      }
      
      const data: OhlcvDto[] = []
      const iterator = provider.getHistoricalData(params)
      for await (const ohlcv of iterator) {
        data.push(ohlcv)
      }
      
      // Should have December 2017 data only
      assert.ok(data.length > 0)
      assert.ok(data.length < 50) // Should be around 31 days
      
      // All dates should be in December 2017
      for (const ohlcv of data) {
        const date = new Date(ohlcv.timestamp)
        assert.strictEqual(date.getUTCFullYear(), 2017)
        assert.strictEqual(date.getUTCMonth(), 11) // December
      }
      
      await provider.disconnect()
    })

    it('should read only required columns from Jsonl', async () => {
      const config: FileProviderConfig = {
        path: jsonlPath,
        format: 'jsonl',
        exchange: 'yahoo',
        symbol: 'BTC-USD'
      }
      
      const provider = new JsonlFileProvider(config)
      await provider.connect()
      
      // Small date range to test column projection
      const params: HistoricalParams = {
        symbols: ['BTC-USD'],
        start: new Date('2017-10-01').getTime(),
        end: new Date('2017-10-10').getTime(),
        timeframe: '1d'
      }
      
      const data: OhlcvDto[] = []
      const iterator = provider.getHistoricalData(params)
      for await (const ohlcv of iterator) {
        data.push(ohlcv)
      }
      
      // Should have filtered data
      assert.ok(data.length > 0)
      assert.ok(data.length < 10)
      
      // Verify all required fields are present
      for (const ohlcv of data) {
        assert.ok(typeof ohlcv.timestamp === 'number')
        assert.ok(typeof ohlcv.open === 'number')
        assert.ok(typeof ohlcv.high === 'number')
        assert.ok(typeof ohlcv.low === 'number')
        assert.ok(typeof ohlcv.close === 'number')
        assert.ok(typeof ohlcv.volume === 'number')
        assert.ok(typeof ohlcv.symbol === 'string')
        assert.ok(typeof ohlcv.exchange === 'string')
      }
      
      await provider.disconnect()
    })
  })

  describe('CSV vs Jsonl Comparison', () => {
    it('should produce identical data from CSV and Jsonl', async () => {
      // Read from CSV
      const csvConfig: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'yahoo',
        symbol: 'BTC-USD',
        columnMapping: {
          timestamp: 'Date',
          open: 'Open',
          high: 'High',
          low: 'Low',
          close: 'Close',
          volume: 'Volume',
        }
      }
      
      const csvProvider = new CsvFileProvider(csvConfig)
      await csvProvider.connect()
      
      const params: HistoricalParams = {
        symbols: [],
        start: new Date('2018-01-01').getTime(),
        end: new Date('2018-01-31').getTime(),
        timeframe: '1d'
      }
      
      const csvData: OhlcvDto[] = []
      const csvIterator = csvProvider.getHistoricalData(params)
      for await (const ohlcv of csvIterator) {
        csvData.push(ohlcv)
      }
      await csvProvider.disconnect()
      
      // Read from Jsonl
      const jsonlConfig: FileProviderConfig = {
        path: jsonlPath,
        format: 'jsonl',
        exchange: 'yahoo',
        symbol: 'BTC-USD'
      }
      
      const jsonlProvider = new JsonlFileProvider(jsonlConfig)
      await jsonlProvider.connect()
      
      const jsonlData: OhlcvDto[] = []
      const jsonlIterator = jsonlProvider.getHistoricalData(params)
      for await (const ohlcv of jsonlIterator) {
        jsonlData.push(ohlcv)
      }
      await jsonlProvider.disconnect()
      
      // Compare results
      assert.strictEqual(csvData.length, jsonlData.length)
      
      for (let i = 0; i < csvData.length; i++) {
        const cd = csvData[i]
        assert.ok(cd)
        const pd = jsonlData[i]
        assert.ok(pd)
        assert.strictEqual(cd.timestamp, pd.timestamp)
        assert.strictEqual(cd.open, pd.open)
        assert.strictEqual(cd.high, pd.high)
        assert.strictEqual(cd.low, pd.low)
        assert.strictEqual(cd.close, pd.close)
        assert.strictEqual(cd.volume, pd.volume)
        assert.strictEqual(cd.symbol, pd.symbol)
        assert.strictEqual(cd.exchange, pd.exchange)
      }
    })
  })
})