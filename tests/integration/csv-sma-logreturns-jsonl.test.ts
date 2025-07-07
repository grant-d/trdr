import { deepStrictEqual, strictEqual } from 'node:assert'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { after, before, describe, it } from 'node:test'
import { BufferPipeline } from '../../src/pipeline'
import { CsvFileProvider } from '../../src/providers'
import { CsvRepository } from '../../src/repositories'
import { LogReturnsNormalizer, SimpleMovingAverage } from '../../src/transforms'
import { DataBuffer, DataSlice } from '../../src/utils'

describe('CSV → SMA → LogReturns → JSONL Integration', () => {
  const testDir = join(process.cwd(), 'test-integration')
  const csvPath = join(testDir, 'test-data.csv')
  const outputPath = join(testDir, 'output.csv')

  // Hardcoded test data: 20 rows with predictable values
  const testCsvData = `timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,100,105,95,100,1000
2024-01-01T01:00:00Z,100,105,95,101,1100
2024-01-01T02:00:00Z,101,106,96,102,1200
2024-01-01T03:00:00Z,102,107,97,103,1300
2024-01-01T04:00:00Z,103,108,98,104,1400
2024-01-01T05:00:00Z,104,109,99,105,1500
2024-01-01T06:00:00Z,105,110,100,106,1600
2024-01-01T07:00:00Z,106,111,101,107,1700
2024-01-01T08:00:00Z,107,112,102,108,1800
2024-01-01T09:00:00Z,108,113,103,109,1900
2024-01-01T10:00:00Z,109,114,104,110,2000
2024-01-01T11:00:00Z,110,115,105,111,2100
2024-01-01T12:00:00Z,111,116,106,112,2200
2024-01-01T13:00:00Z,112,117,107,113,2300
2024-01-01T14:00:00Z,113,118,108,114,2400
2024-01-01T15:00:00Z,114,119,109,115,2500
2024-01-01T16:00:00Z,115,120,110,116,2600
2024-01-01T17:00:00Z,116,121,111,117,2700
2024-01-01T18:00:00Z,117,122,112,118,2800
2024-01-01T19:00:00Z,118,123,113,119,2900`

  before(async () => {
    await mkdir(testDir, { recursive: true })
    await writeFile(csvPath, testCsvData)
  })

  after(async () => {
    await rm(testDir, { recursive: true, force: true })
  })

  it('should process CSV data through SMA and LogReturns transforms with batch size 3', async () => {
    // Create shared buffer for the pipeline
    const sharedBuffer = new DataBuffer({
      columns: {
        timestamp: { index: 0 },
        open: { index: 1 },
        high: { index: 2 },
        low: { index: 3 },
        close: { index: 4 },
        volume: { index: 5 }
      }
    })

    // Create CSV provider with batch size 3
    const csvProvider = new CsvFileProvider({
      path: csvPath,
      format: 'csv',
      chunkSize: 3 // Process 3 rows at a time
    })

    // Set the buffer on the provider
    csvProvider.setBuffer(sharedBuffer)

    // Create transforms that operate on the shared buffer
    // Note: transforms will be given proper slices by the pipeline
    const fullSlice = new DataSlice(sharedBuffer, 0, 1000) // Large enough slice
    
    const smaTransform = new SimpleMovingAverage({
      tx: { in: 'close', out: 'sma_5', window: 5 }
    }, fullSlice)

    const logReturnsTransform = new LogReturnsNormalizer({
      tx: { in: 'close', out: 'log_returns', base: 'ln' }
    }, fullSlice)

    // Create output repository
    const repository = new CsvRepository()

    // Initialize repository
    await repository.initialize({
      connectionString: outputPath,
      options: { overwrite: true }
    })

    // Create pipeline
    const pipeline = new BufferPipeline({
      provider: csvProvider,
      transforms: [smaTransform, logReturnsTransform],
      repository,
      batchSize: 3,
      initialBuffer: sharedBuffer,
      finalBuffer: sharedBuffer
    })

    // Execute the pipeline
    const result = await pipeline.execute()

    // Verify results
    strictEqual(result.recordsProcessed, 20, 'Should process 20 records')
    strictEqual(result.recordsWritten, 20, 'Should write 20 records')
    strictEqual(result.errors, 0, 'Should have no errors')

    // Verify buffer has expected columns
    const columns = sharedBuffer.getColumns()
    deepStrictEqual(columns, [
      'timestamp', 'open', 'high', 'low', 'close', 'volume', 'sma_5', 'log_returns'
    ], 'Buffer should have all expected columns')

    // Buffer should be empty after flushing to repository
    strictEqual(sharedBuffer.length(), 0, 'Buffer should be empty after pipeline execution')

    // Read and verify the output CSV file
    const outputContent = await readFile(outputPath, 'utf8')
    const lines = outputContent.trim().split('\n')
    
    // Should have header + 20 data rows
    strictEqual(lines.length, 21, 'Output file should have header + 20 data rows')
    
    // Check header contains our new columns
    const header = lines[0]!
    strictEqual(header.includes('sma_5'), true, 'Header should contain sma_5 column')
    strictEqual(header.includes('log_returns'), true, 'Header should contain log_returns column')
    
    // Parse a few sample rows to verify calculations
    const row5 = lines[5]!.split(',') // 5th data row
    const close5 = parseFloat(row5[4]!) // close column
    const sma5 = parseFloat(row5[6]!) // sma_5 column  
    const logReturn5 = parseFloat(row5[7]!) // log_returns column
    
    strictEqual(close5, 104, 'Row 5 close should be 104')
    strictEqual(sma5, 102, 'Row 5 SMA should be 102 (average of 100,101,102,103,104)')
    
    // Log return should be ln(104/103) for row 5
    const expectedLogReturn = Math.log(104 / 103)
    strictEqual(
      Math.abs(logReturn5 - expectedLogReturn) < 0.0001,
      true,
      `Row 5 log return should be approximately ${expectedLogReturn}`
    )
  })

  it('should verify batch processing happens in correct order', async () => {
    // Track processing calls
    const processingCalls: Array<{ from: number; to: number }> = []

    // Create shared buffer
    const sharedBuffer = new DataBuffer({
      columns: {
        timestamp: { index: 0 },
        open: { index: 1 },
        high: { index: 2 },
        low: { index: 3 },
        close: { index: 4 },
        volume: { index: 5 }
      }
    })

    // Create CSV provider
    const csvProvider = new CsvFileProvider({
      path: csvPath,
      format: 'csv',
      chunkSize: 3
    })
    csvProvider.setBuffer(sharedBuffer)

    // Create a mock transform that tracks processing calls
    class MockTransform {
      constructor(public outputBuffer = sharedBuffer) {}
      
      next(from: number, to: number) {
        processingCalls.push({ from, to })
        return new DataSlice(this.outputBuffer, from, to)
      }
    }

    const mockTransform = new MockTransform()

    // Create and initialize repository
    const mockRepository = new CsvRepository()
    await mockRepository.initialize({
      connectionString: outputPath,
      options: { overwrite: true }
    })

    // Create pipeline with mock transform
    const pipeline = new BufferPipeline({
      provider: csvProvider,
      transforms: [mockTransform as any],
      repository: mockRepository,
      batchSize: 3,
      initialBuffer: sharedBuffer,
      finalBuffer: sharedBuffer
    })

    await pipeline.execute()

    // Verify batch processing calls
    // 20 rows with batch size 3: (0,3), (3,6), (6,9), (9,12), (12,15), (15,18), (18,20)
    strictEqual(processingCalls.length, 7, 'Should have 7 batch processing calls')
    
    deepStrictEqual(processingCalls[0], { from: 0, to: 3 }, 'First batch: 0-3')
    deepStrictEqual(processingCalls[1], { from: 3, to: 6 }, 'Second batch: 3-6')
    deepStrictEqual(processingCalls[2], { from: 6, to: 9 }, 'Third batch: 6-9')
    deepStrictEqual(processingCalls[3], { from: 9, to: 12 }, 'Fourth batch: 9-12')
    deepStrictEqual(processingCalls[4], { from: 12, to: 15 }, 'Fifth batch: 12-15')
    deepStrictEqual(processingCalls[5], { from: 15, to: 18 }, 'Sixth batch: 15-18')
    deepStrictEqual(processingCalls[6], { from: 18, to: 20 }, 'Seventh batch: 18-20')
  })
})
