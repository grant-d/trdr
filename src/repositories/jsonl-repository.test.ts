import * as assert from 'node:assert'
import { describe, it, beforeEach, afterEach } from 'node:test'
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'
import { JsonlRepository } from './jsonl-repository'
import type { OhlcvDto } from '../models'

describe('JsonlRepository', () => {
  let testDir: string
  let testFile: string
  let repository: JsonlRepository

  beforeEach(async () => {
    // Create unique test directory for each test
    testDir = path.join(tmpdir(), `jsonl-repo-test-${Date.now()}-${Math.random().toString(36).slice(2)}`)
    testFile = path.join(testDir, 'test.jsonl')
    await fs.mkdir(testDir, { recursive: true })
    repository = new JsonlRepository()
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
    it('should write to a single JSONL file', async () => {
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
      assert.ok(stats.isFile(), 'Should create a file')

      // Read and verify content
      const content = await fs.readFile(testFile, 'utf-8')
      const lines = content.trim().split('\n')
      assert.strictEqual(lines.length, 1, 'Should have one line')
      
      const parsed = JSON.parse(lines[0]!)
      assert.strictEqual(parsed.s, 'AAPL', 'Should contain symbol in abbreviated format')
      assert.strictEqual(parsed.x, 'test', 'Should contain exchange in abbreviated format')
    })

    it('should write multiple records to same file', async () => {
      await repository.initialize({
        connectionString: testFile,
        options: {}
      })

      const records: OhlcvDto[] = []
      for (let i = 0; i < 19; i++) {
        records.push({
          timestamp: 1704067200000 + i,
          symbol: 'AAPL',
          exchange: 'test',
          open: 100 + i,
          high: 105 + i,
          low: 99 + i,
          close: 104 + i,
          volume: 1000 + i
        })
      }

      await repository.saveMany(records)
      await repository.flush()

      // Read and verify content
      const content = await fs.readFile(testFile, 'utf-8')
      const lines = content.trim().split('\n')
      assert.strictEqual(lines.length, 19, 'Should have 19 lines')
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
      repository = new JsonlRepository()
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
      assert.strictEqual(lines.length, 1, 'Should have only 1 record')
      
      const parsed = JSON.parse(lines[0]!)
      assert.strictEqual(parsed.t, 1704067201000, 'Should contain only second timestamp')
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
      repository = new JsonlRepository()
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
      assert.strictEqual(lines.length, 2, 'Should have 2 records')
      
      const parsed1 = JSON.parse(lines[0]!)
      const parsed2 = JSON.parse(lines[1]!)
      assert.strictEqual(parsed1.t, 1704067200000, 'Should contain first timestamp')
      assert.strictEqual(parsed2.t, 1704067201000, 'Should contain second timestamp')
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
        /JSONL file can only contain data for one symbol\/exchange/,
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
        /JSONL file can only contain data for one symbol\/exchange/,
        'Should reject different exchange'
      )
    })
  })

  describe('abbreviated format', () => {
    it('should write data in abbreviated format', async () => {
      await repository.initialize({
        connectionString: testFile,
        options: {}
      })

      const data: OhlcvDto = {
        timestamp: 1704067200000,
        symbol: 'AAPL',
        exchange: 'nasdaq',
        open: 100.5,
        high: 105.25,
        low: 99.75,
        close: 104.0,
        volume: 1000000
      }

      await repository.save(data)
      await repository.flush()

      const content = await fs.readFile(testFile, 'utf-8')
      const parsed = JSON.parse(content.trim())

      // Check abbreviated property names
      assert.strictEqual(parsed.t, 1704067200000, 'timestamp -> t')
      assert.strictEqual(parsed.s, 'AAPL', 'symbol -> s')
      assert.strictEqual(parsed.x, 'nasdaq', 'exchange -> x')
      assert.strictEqual(parsed.o, 100.5, 'open -> o')
      assert.strictEqual(parsed.h, 105.25, 'high -> h')
      assert.strictEqual(parsed.l, 99.75, 'low -> l')
      assert.strictEqual(parsed.c, 104.0, 'close -> c')
      assert.strictEqual(parsed.v, 1000000, 'volume -> v')

      // Should not have full property names
      assert.strictEqual(parsed.timestamp, undefined)
      assert.strictEqual(parsed.symbol, undefined)
    })

    it('should read both abbreviated and full formats', async () => {
      // Write file with mixed formats
      const mixedContent = [
        JSON.stringify({ t: 1704067200000, s: 'AAPL', x: 'test', o: 100, h: 105, l: 99, c: 104, v: 1000 }),
        JSON.stringify({ timestamp: 1704067201000, symbol: 'AAPL', exchange: 'test', open: 105, high: 110, low: 104, close: 109, volume: 2000 })
      ].join('\n')

      await fs.writeFile(testFile, mixedContent)

      await repository.initialize({
        connectionString: testFile,
        options: {}
      })

      const results = await repository.query({})
      assert.strictEqual(results.length, 2, 'Should read both formats')
      assert.strictEqual(results[0]!.timestamp, 1704067200000)
      assert.strictEqual(results[1]!.timestamp, 1704067201000)
    })
  })
})
