#!/usr/bin/env tsx

import { createReadStream, createWriteStream } from 'node:fs'
import { join } from 'node:path'
import { createInterface } from 'node:readline'

const csvPath = join(process.cwd(), 'tests/unit/providers/BTCUSD-short.csv')
const jsonlPath = join(process.cwd(), 'tests/unit/providers/BTCUSD-short.jsonl')

async function convertCsvToJsonl() {
  const readStream = createReadStream(csvPath)
  const writeStream = createWriteStream(jsonlPath)
  const rl = createInterface({
    input: readStream,
    crlfDelay: Infinity
  })

  let isFirstLine = true
  let headers: string[] = []

  for await (const line of rl) {
    if (isFirstLine) {
      headers = line.split(',')
      isFirstLine = false
      continue
    }

    const values = line.split(',')
    const row: any = {}
    
    headers.forEach((header, index) => {
      row[header] = values[index]
    })

    // Convert to OHLCV format
    const timestamp = new Date(row.Date).getTime()
    
    const jsonlRecord = {
      timestamp,
      symbol: 'BTC-USD',
      exchange: 'yahoo',
      open: parseFloat(row.Open),
      high: parseFloat(row.High),
      low: parseFloat(row.Low),
      close: parseFloat(row.Close),
      volume: parseFloat(row.Volume),
    }
    
    writeStream.write(JSON.stringify(jsonlRecord) + '\n')
  }

  await new Promise((resolve, reject) => {
    writeStream.end((err: Error | undefined) => {
      if (err) reject(err)
      else resolve(undefined)
    })
  })

  console.log(`Converted ${csvPath} to ${jsonlPath}`)
}

convertCsvToJsonl().catch(console.error)