import assert from 'node:assert'
import { mkdir, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { after, before, describe, it } from 'node:test'
import type { HistoricalParams } from '../../../src/interfaces'
import type { OhlcvDto } from '../../../src/models'
import type { FileProviderConfig } from '../../../src/providers'
import { CsvFileProvider } from '../../../src/providers/base'

describe('FileProvider Tests', () => {
  const testDir = join(process.cwd(), 'test-data')
  const csvPath = join(testDir, 'test.csv')
  
  // Sample CSV data
  const csvContent = `timestamp,open,high,low,close,volume,symbol
2024-01-01T00:00:00Z,100,105,99,103,1000,BTC-USD
2024-01-01T01:00:00Z,103,107,102,106,1200,BTC-USD
2024-01-01T02:00:00Z,106,108,104,105,800,BTC-USD
2024-01-01T03:00:00Z,105,106,103,104,900,BTC-USD
2024-01-01T04:00:00Z,104,110,104,109,1500,BTC-USD`

  before(async () => {
    // Create test directory and files
    await mkdir(testDir, { recursive: true })
    await writeFile(csvPath, csvContent)
  })

  after(async () => {
    // Clean up test files
    await rm(testDir, { recursive: true, force: true })
  })

  describe('CsvFileProvider', () => {
    it('should connect to a valid CSV file', async () => {
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test-exchange'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      assert.strictEqual(provider.isConnected(), true)
      await provider.disconnect()
    })

    it('should throw error when connecting to non-existent file', async () => {
      const config: FileProviderConfig = {
        path: join(testDir, 'non-existent.csv'),
        format: 'csv',
        exchange: 'test-exchange'
      }
      
      const provider = new CsvFileProvider(config)
      await assert.rejects(
        async () => await provider.connect(),
        /Cannot access file/
      )
    })

    it('should read and parse CSV data correctly', async () => {
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test-exchange'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const params: HistoricalParams = {
        symbols: ['BTC-USD'],
        start: new Date('2024-01-01T00:00:00Z').getTime(),
        end: new Date('2024-01-01T23:59:59Z').getTime(),
        timeframe: '1h'
      }
      
      const data: OhlcvDto[] = []
      const iterator = provider.getHistoricalData(params)
      for await (const ohlcv of iterator) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 5)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 100)
      assert.strictEqual(data[0].high, 105)
      assert.strictEqual(data[0].low, 99)
      assert.strictEqual(data[0].close, 103)
      assert.strictEqual(data[0].volume, 1000)
      assert.strictEqual(data[0].symbol, 'BTC-USD')
      
      await provider.disconnect()
    })

    it('should filter data by timestamp range', async () => {
      const config: FileProviderConfig = {
        path: csvPath,
        format: 'csv',
        exchange: 'test-exchange'
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const params: HistoricalParams = {
        symbols: ['BTC-USD'],
        start: new Date('2024-01-01T02:00:00Z').getTime(),
        end: new Date('2024-01-01T03:00:00Z').getTime(),
        timeframe: '1h'
      }
      
      const data: OhlcvDto[] = []
      const iterator = provider.getHistoricalData(params)
      for await (const ohlcv of iterator) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 2)
      assert.ok(data[0])
      assert.strictEqual(data[0].close, 105)
      assert.ok(data[1])
      assert.strictEqual(data[1].close, 104)
      
      await provider.disconnect()
    })

    it('should handle custom column mapping', async () => {
      // Create CSV with different column names
      const customCsv = `time,o,h,l,c,v,pair
2024-01-01T00:00:00Z,100,105,99,103,1000,BTC-USD`
      const customPath = join(testDir, 'custom.csv')
      await writeFile(customPath, customCsv)
      
      const config: FileProviderConfig = {
        path: customPath,
        format: 'csv',
        exchange: 'test-exchange',
        columnMapping: {
          timestamp: 'time',
          open: 'o',
          high: 'h',
          low: 'l',
          close: 'c',
          volume: 'v',
          symbol: 'pair'
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
      const iterator = provider.getHistoricalData(params)
      for await (const ohlcv of iterator) {
        data.push(ohlcv)
      }
      
      assert.strictEqual(data.length, 1)
      assert.ok(data[0])
      assert.strictEqual(data[0].open, 100)
      assert.strictEqual(data[0].symbol, 'BTC-USD')
      
      await provider.disconnect()
    })

    it('should handle malformed CSV rows gracefully', async () => {
      const malformedCsv = `timestamp,open,high,low,close,volume,symbol
2024-01-01T00:00:00Z,100,105,99,103,1000,BTC-USD
2024-01-01T01:00:00Z,103,107,102
2024-01-01T02:00:00Z,106,108,104,105,800,BTC-USD`
      
      const malformedPath = join(testDir, 'malformed.csv')
      await writeFile(malformedPath, malformedCsv)
      
      const config: FileProviderConfig = {
        path: malformedPath,
        format: 'csv',
        exchange: 'test-exchange'
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
      const iterator = provider.getHistoricalData(params)
      for await (const ohlcv of iterator) {
        data.push(ohlcv)
      }
      
      // Should skip the malformed row
      assert.strictEqual(data.length, 2)
      
      await provider.disconnect()
    })

    it('should validate OHLC relationships', async () => {
      const invalidCsv = `timestamp,open,high,low,close,volume,symbol
2024-01-01T00:00:00Z,100,90,99,103,1000,BTC-USD`  // high < low
      
      const invalidPath = join(testDir, 'invalid.csv')
      await writeFile(invalidPath, invalidCsv)
      
      const config: FileProviderConfig = {
        path: invalidPath,
        format: 'csv',
        exchange: 'test-exchange'
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
      const iterator = provider.getHistoricalData(params)
      for await (const ohlcv of iterator) {
        data.push(ohlcv)
      }
      
      // Should skip invalid data
      assert.strictEqual(data.length, 0)
      
      await provider.disconnect()
    })

    it('should handle large files with streaming', async () => {
      // Create a larger CSV file
      let largeCsv = 'timestamp,open,high,low,close,volume,symbol\n'
      const baseTime = new Date('2024-01-01T00:00:00Z').getTime()
      
      for (let i = 0; i < 10000; i++) {
        const timestamp = new Date(baseTime + i * 60000).toISOString() // 1 minute intervals
        const price = 100 + Math.sin(i / 100) * 10
        largeCsv += `${timestamp},${price},${price + 1},${price - 1},${price},${1000 + i},BTC-USD\n`
      }
      
      const largePath = join(testDir, 'large.csv')
      await writeFile(largePath, largeCsv)
      
      const config: FileProviderConfig = {
        path: largePath,
        format: 'csv',
        exchange: 'test-exchange',
        chunkSize: 100 // Small chunks to test streaming
      }
      
      const provider = new CsvFileProvider(config)
      await provider.connect()
      
      const params: HistoricalParams = {
        symbols: [],
        start: baseTime,
        end: baseTime + 1000 * 60000, // First 1000 minutes
        timeframe: '1m'
      }
      
      let count = 0
      const startMemory = process.memoryUsage().heapUsed
      
      for await (const ohlcv of provider.getHistoricalData(params)) {
        count++
        assert.ok(ohlcv.timestamp >= params.start)
        assert.ok(ohlcv.timestamp <= params.end)
      }
      
      const endMemory = process.memoryUsage().heapUsed
      const memoryIncrease = (endMemory - startMemory) / 1024 / 1024 // MB
      
      assert.strictEqual(count, 1001) // Includes both start and end
      assert.ok(memoryIncrease < 50, `Memory increase should be minimal, but was ${memoryIncrease}MB`)
      
      await provider.disconnect()
    })
  })
})