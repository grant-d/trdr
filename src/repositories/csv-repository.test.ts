import * as assert from 'node:assert'
import { describe, it, beforeEach, afterEach } from 'node:test'
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'
import { CsvRepository } from './csv-repository'
import type { OhlcvDto } from '../models'

describe('CsvRepository', () => {
  let testDir: string
  let testFile: string
  let repository: CsvRepository

  beforeEach(async () => {
    // Create unique test directory for each test
    testDir = path.join(tmpdir(), `csv-repo-test-${Date.now()}-${Math.random().toString(36).slice(2)}`)
    testFile = path.join(testDir, 'test.csv')
    await fs.mkdir(testDir, { recursive: true })
    repository = new CsvRepository()
  })

  afterEach(async () => {
    // Clean up test files
    try {
      await repository.close()
    } catch {
      // Ignore errors
    }
    try {
      await fs.rm(testDir, { recursive: true, force: true })
    } catch {
      // Ignore errors
    }
  })

  describe('single file output', () => {
    it('should write to a single file, not create directory structure', async () => {
      await repository.initialize({
        connectionString: testFile,
        options: {}
      })

      const data: OhlcvDto = {
        timestamp: Date.now(),
        symbol: 'AAPL',
        exchange: 'test',
        open: 100,
        high: 105,
        low: 99,
        close: 104,
        volume: 1000
      }

      await repository.save(data)
      await repository.flush()

      // Check that the file exists
      const stats = await fs.stat(testFile)
      assert.ok(stats.isFile(), 'Should create a file, not a directory')

      // Read and verify content
      const content = await fs.readFile(testFile, 'utf-8')
      assert.ok(content.includes('timestamp,symbol,exchange'), 'Should contain CSV headers')
      assert.ok(content.includes('AAPL'), 'Should contain the symbol')
    })
  })

  describe('overwrite functionality', () => {
    it('should overwrite existing file when overwrite is true', async () => {
      // Create initial file with data
      await repository.initialize({
        connectionString: testFile,
        options: { overwrite: false }
      })

      const data1: OhlcvDto = {
        timestamp: 1704067200000,
        symbol: 'AAPL',
        exchange: 'test',
        open: 100,
        high: 105,
        low: 99,
        close: 104,
        volume: 1000
      }

      await repository.save(data1)
      await repository.flush()

      // Re-initialize with overwrite=true
      await repository.close()
      repository = new CsvRepository()
      await repository.initialize({
        connectionString: testFile,
        options: { overwrite: true }
      })

      const data2: OhlcvDto = {
        timestamp: 1704067201000,
        symbol: 'AAPL',
        exchange: 'test',
        open: 105,
        high: 110,
        low: 104,
        close: 109,
        volume: 2000
      }

      await repository.save(data2)
      await repository.flush()

      // Read file and check it only contains the second record
      const content = await fs.readFile(testFile, 'utf-8')
      const lines = content.trim().split('\n')
      assert.strictEqual(lines.length, 2, 'Should have header + 1 data row')
      assert.ok(!content.includes('1704067200000'), 'Should not contain first timestamp')
      assert.ok(content.includes('1704067201000'), 'Should contain second timestamp')
    })

    it('should append to existing file when overwrite is false', async () => {
      // Create initial file with data
      await repository.initialize({
        connectionString: testFile,
        options: { overwrite: false }
      })

      const data1: OhlcvDto = {
        timestamp: 1704067200000,
        symbol: 'AAPL',
        exchange: 'test',
        open: 100,
        high: 105,
        low: 99,
        close: 104,
        volume: 1000
      }

      await repository.save(data1)
      await repository.flush()

      // Re-initialize with overwrite=false
      await repository.close()
      repository = new CsvRepository()
      await repository.initialize({
        connectionString: testFile,
        options: { overwrite: false }
      })

      const data2: OhlcvDto = {
        timestamp: 1704067201000,
        symbol: 'AAPL',
        exchange: 'test',
        open: 105,
        high: 110,
        low: 104,
        close: 109,
        volume: 2000
      }

      await repository.save(data2)
      await repository.flush()

      // Read file and check it contains both records
      const content = await fs.readFile(testFile, 'utf-8')
      const lines = content.trim().split('\n')
      assert.strictEqual(lines.length, 3, 'Should have header + 2 data rows')
      assert.ok(content.includes('1704067200000'), 'Should contain first timestamp')
      assert.ok(content.includes('1704067201000'), 'Should contain second timestamp')
    })
  })

  describe('single symbol/exchange enforcement', () => {
    it('should reject data with different symbol', async () => {
      await repository.initialize({
        connectionString: testFile,
        options: {}
      })

      const data1: OhlcvDto = {
        timestamp: 1704067200000,
        symbol: 'AAPL',
        exchange: 'test',
        open: 100,
        high: 105,
        low: 99,
        close: 104,
        volume: 1000
      }

      await repository.save(data1)

      const data2: OhlcvDto = {
        timestamp: 1704067201000,
        symbol: 'GOOGL',
        exchange: 'test',
        open: 200,
        high: 210,
        low: 199,
        close: 209,
        volume: 2000
      }

      await assert.rejects(
        async () => repository.save(data2),
        /CSV file can only contain data for one symbol\/exchange/,
        'Should reject different symbol'
      )
    })

    it('should reject data with different exchange', async () => {
      await repository.initialize({
        connectionString: testFile,
        options: {}
      })

      const data1: OhlcvDto = {
        timestamp: 1704067200000,
        symbol: 'AAPL',
        exchange: 'nasdaq',
        open: 100,
        high: 105,
        low: 99,
        close: 104,
        volume: 1000
      }

      await repository.save(data1)

      const data2: OhlcvDto = {
        timestamp: 1704067201000,
        symbol: 'AAPL',
        exchange: 'nyse',
        open: 100,
        high: 110,
        low: 99,
        close: 109,
        volume: 2000
      }

      await assert.rejects(
        async () => repository.save(data2),
        /CSV file can only contain data for one symbol\/exchange/,
        'Should reject different exchange'
      )
    })

    it('should accept data with same symbol and exchange', async () => {
      await repository.initialize({
        connectionString: testFile,
        options: {}
      })

      const data1: OhlcvDto = {
        timestamp: 1704067200000,
        symbol: 'AAPL',
        exchange: 'test',
        open: 100,
        high: 105,
        low: 99,
        close: 104,
        volume: 1000
      }

      const data2: OhlcvDto = {
        timestamp: 1704067201000,
        symbol: 'AAPL',
        exchange: 'test',
        open: 105,
        high: 110,
        low: 104,
        close: 109,
        volume: 2000
      }

      await repository.save(data1)
      await repository.save(data2)
      await repository.flush()

      const content = await fs.readFile(testFile, 'utf-8')
      const lines = content.trim().split('\n')
      assert.strictEqual(lines.length, 3, 'Should have header + 2 data rows')
    })
  })
})
