#!/usr/bin/env tsx
import { CsvFileProvider } from '../src/providers/base/csv-file-provider'
import type { OhlcvDto } from '../src/models/ohlcv.dto'

/**
 * Script to convert CSV files to Parquet format
 * Usage: tsx scripts/csv-to-parquet.ts <input.csv> <output.parquet>
 */
async function convertCsvToParquet(): Promise<void> {
  const args = process.argv.slice(2)
  if (args.length !== 2) {
    console.error('Usage: tsx scripts/csv-to-parquet.ts <input.csv> <output.parquet>')
    process.exit(1)
  }

  const inputPath = args[0]
  const outputPath = args[1]
  
  if (!inputPath || !outputPath) {
    console.error('Error: Missing arguments')
    process.exit(1)
  }
  console.log(`Converting ${inputPath} to ${outputPath}...`)

  try {
    // Create CSV provider with column mapping for Yahoo Finance format
    const provider = new CsvFileProvider({
      path: inputPath,
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
    })

    await provider.connect()

    // Read all data
    const data: OhlcvDto[] = []
    const params = {
      symbols: [],
      start: 0,
      end: Date.now(),
      timeframe: '1d'
    }

    for await (const ohlcv of provider.getHistoricalData(params)) {
      data.push(ohlcv)
    }

    await provider.disconnect()
    console.log(`Read ${data.length} rows from CSV`)

    // Write to parquet file using dynamic import
    const { parquetWriteFile } = await import('hyparquet-writer')
    parquetWriteFile({
      filename: outputPath,
      columnData: [
        { name: 'timestamp', data: data.map(d => BigInt(d.timestamp)), type: 'INT64' },
        { name: 'open', data: data.map(d => d.open), type: 'DOUBLE' },
        { name: 'high', data: data.map(d => d.high), type: 'DOUBLE' },
        { name: 'low', data: data.map(d => d.low), type: 'DOUBLE' },
        { name: 'close', data: data.map(d => d.close), type: 'DOUBLE' },
        { name: 'volume', data: data.map(d => d.volume), type: 'DOUBLE' },
        { name: 'symbol', data: data.map(d => d.symbol), type: 'STRING' },
        { name: 'exchange', data: data.map(d => d.exchange), type: 'STRING' }
      ]
    })

    console.log(`Successfully wrote ${outputPath}`)
  } catch (error) {
    console.error('Error converting file:', error)
    process.exit(1)
  }
}

// Run the conversion
void convertCsvToParquet()