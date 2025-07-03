#!/usr/bin/env node

/**
 * Utility script to download historical price data from Yahoo Finance
 * Saves data in CSV format compatible with the trading demo
 */

import { writeFileSync, mkdirSync } from 'fs'
import { join, dirname } from 'path'
import chalk from 'chalk'
import ora from 'ora'

// Yahoo Finance API endpoint
const YAHOO_FINANCE_API = 'https://query1.finance.yahoo.com/v7/finance/download'

interface YahooFinanceOptions {
  symbol: string
  period1: number  // Start timestamp
  period2: number  // End timestamp
  interval: string // 1d, 1wk, 1mo, etc.
  events: string   // history, div, split
}

async function downloadFromYahoo(options: YahooFinanceOptions, retries = 3): Promise<string> {
  const params = new URLSearchParams({
    period1: options.period1.toString(),
    period2: options.period2.toString(),
    interval: options.interval,
    events: options.events,
    includeAdjustedClose: 'true'
  })
  
  const url = `${YAHOO_FINANCE_API}/${options.symbol}?${params}`
  
  console.log(chalk.gray(`Fetching: ${url}`))
  
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      // Add delay between attempts to avoid rate limiting
      if (attempt > 1) {
        console.log(chalk.yellow(`Retry attempt ${attempt}/${retries}...`))
        await new Promise(resolve => setTimeout(resolve, 2000 * attempt))
      }
      
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          'Accept': 'text/csv,application/csv,*/*',
          'Accept-Language': 'en-US,en;q=0.9',
          'Connection': 'keep-alive',
          'Cache-Control': 'no-cache'
        }
      })
      
      if (response.ok) {
        return await response.text()
      }
      
      if (response.status === 429 && attempt < retries) {
        console.log(chalk.yellow(`Rate limited (429), waiting before retry...`))
        continue
      }
      
      throw new Error(`Failed to fetch data: ${response.status} ${response.statusText}`)
      
    } catch (error) {
      if (attempt === retries) {
        throw error
      }
      console.log(chalk.yellow(`Attempt ${attempt} failed, retrying...`))
    }
  }
  
  throw new Error('All retry attempts failed')
}

function parseTimeframe(timeframe: string): { period1: number, period2: number } {
  const now = Math.floor(Date.now() / 1000)
  
  switch (timeframe.toLowerCase()) {
    case '1d':
    case '1day':
      return { period1: now - (24 * 60 * 60), period2: now }
    
    case '1w':
    case '1week':
      return { period1: now - (7 * 24 * 60 * 60), period2: now }
    
    case '1m':
    case '1month':
      return { period1: now - (30 * 24 * 60 * 60), period2: now }
    
    case '3m':
    case '3months':
      return { period1: now - (90 * 24 * 60 * 60), period2: now }
    
    case '6m':
    case '6months':
      return { period1: now - (180 * 24 * 60 * 60), period2: now }
    
    case '1y':
    case '1year':
      return { period1: now - (365 * 24 * 60 * 60), period2: now }
    
    case '2y':
    case '2years':
      return { period1: now - (2 * 365 * 24 * 60 * 60), period2: now }
    
    case '5y':
    case '5years':
      return { period1: now - (5 * 365 * 24 * 60 * 60), period2: now }
    
    case 'max':
      return { period1: 0, period2: now }
    
    default:
      // Try to parse as date range like "2020-01-01:2023-12-31"
      if (timeframe.includes(':')) {
        const [start, end] = timeframe.split(':')
        if (!start || !end) {
          throw new Error(`Invalid date range format: ${timeframe}`)
        }
        return {
          period1: Math.floor(new Date(start).getTime() / 1000),
          period2: Math.floor(new Date(end).getTime() / 1000)
        }
      }
      
      throw new Error(`Invalid timeframe: ${timeframe}. Use 1d, 1w, 1m, 3m, 6m, 1y, 2y, 5y, max, or YYYY-MM-DD:YYYY-MM-DD`)
  }
}

function normalizeSymbol(symbol: string): string {
  // Convert common crypto symbols to Yahoo Finance format
  const cryptoMap: Record<string, string> = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'BITCOIN': 'BTC-USD',
    'ETHEREUM': 'ETH-USD',
    'ADA': 'ADA-USD',
    'SOL': 'SOL-USD',
    'DOT': 'DOT-USD',
    'MATIC': 'MATIC-USD',
    'AVAX': 'AVAX-USD',
    'ATOM': 'ATOM-USD'
  }
  
  const upperSymbol = symbol.toUpperCase()
  return cryptoMap[upperSymbol] || symbol
}

async function main() {
  console.log(chalk.cyan('Yahoo Finance Data Downloader\n'))
  
  // Parse command line arguments
  const args = process.argv.slice(2)
  
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(chalk.yellow('Usage:'))
    console.log('  yarn download-data <symbol> [options]\n')
    console.log(chalk.yellow('Arguments:'))
    console.log('  symbol          Stock/crypto symbol (e.g., AAPL, BTC, BTC-USD)\n')
    console.log(chalk.yellow('Options:'))
    console.log('  --timeframe, -t  Time period (1d, 1w, 1m, 3m, 6m, 1y, 2y, 5y, max) [default: 1y]')
    console.log('  --interval, -i   Data interval (1d, 1wk, 1mo) [default: 1d]')
    console.log('  --output, -o     Output file path [default: csv/<symbol>.csv]')
    console.log('  --help, -h       Show this help\n')
    console.log(chalk.yellow('Examples:'))
    console.log('  yarn download-data AAPL --timeframe 2y')
    console.log('  yarn download-data BTC --timeframe 5y --interval 1d')
    console.log('  yarn download-data SOL --timeframe 2y --output csv/solana-2y.csv')
    console.log('  yarn download-data TSLA --timeframe 2020-01-01:2023-12-31')
    console.log('  yarn download-data ETH --output ./my-data/ethereum.csv\n')
    console.log(chalk.yellow('Note:'))
    console.log('  Yahoo Finance may rate limit requests. If downloads fail,')
    console.log('  try again later or use a different symbol/timeframe.')
    return
  }
  
  const symbol = normalizeSymbol(args[0]!)
  
  // Helper function to get argument value
  const getArgValue = (argName: string, shortName?: string, defaultValue?: string): string => {
    // Check for --arg=value format
    const longFormat = args.find(arg => arg.startsWith(`--${argName}=`))
    if (longFormat) return longFormat.split('=')[1]!
    
    // Check for --arg value format
    const longIndex = args.indexOf(`--${argName}`)
    if (longIndex >= 0 && longIndex + 1 < args.length && !args[longIndex + 1]!.startsWith('-')) {
      return args[longIndex + 1]!
    }
    
    // Check for -x value format
    if (shortName) {
      const shortIndex = args.indexOf(`-${shortName}`)
      if (shortIndex >= 0 && shortIndex + 1 < args.length && !args[shortIndex + 1]!.startsWith('-')) {
        return args[shortIndex + 1]!
      }
    }
    
    return defaultValue || ''
  }
  
  const timeframe = getArgValue('timeframe', 't', '1y')
  const interval = getArgValue('interval', 'i', '1d')
  const outputPath = getArgValue('output', 'o') || join(process.cwd(), 'csv', `${symbol.toLowerCase()}.csv`)
  
  console.log(chalk.gray('Configuration:'))
  console.log(chalk.gray(`- Symbol: ${symbol}`))
  console.log(chalk.gray(`- Timeframe: ${timeframe}`))
  console.log(chalk.gray(`- Interval: ${interval}`))
  console.log(chalk.gray(`- Output: ${outputPath}\n`))
  
  const spinner = ora('Downloading data from Yahoo Finance...').start()
  
  try {
    // Parse timeframe
    const { period1, period2 } = parseTimeframe(timeframe)
    
    // Download data
    const csvData = await downloadFromYahoo({
      symbol,
      period1,
      period2,
      interval,
      events: 'history'
    })
    
    // Validate data
    const lines = csvData.trim().split('\n')
    if (lines.length < 2) {
      throw new Error('No data returned or invalid CSV format')
    }
    
    spinner.text = 'Saving data to file...'
    
    // Ensure output directory exists
    mkdirSync(dirname(outputPath), { recursive: true })
    
    // Write CSV file
    writeFileSync(outputPath, csvData, 'utf8')
    
    spinner.succeed(`Downloaded ${lines.length - 1} records`)
    
    // Show summary
    console.log(chalk.green('\n✓ Download completed!\n'))
    console.log(chalk.cyan('Summary:'))
    console.log(`- Records: ${lines.length - 1}`)
    console.log(`- File: ${outputPath}`)
    console.log(`- Size: ${(csvData.length / 1024).toFixed(1)} KB`)
    
    // Show date range from data
    if (lines.length > 1) {
      const firstDataLine = lines[1]!.split(',')
      const lastDataLine = lines[lines.length - 1]!.split(',')
      
      if (firstDataLine.length > 0 && lastDataLine.length > 0) {
        console.log(`- Date Range: ${firstDataLine[0]} to ${lastDataLine[0]}`)
        
        if (firstDataLine.length >= 5 && lastDataLine.length >= 5) {
          const startPrice = parseFloat(firstDataLine[4]!) // Close price
          const endPrice = parseFloat(lastDataLine[4]!)   // Close price
          
          if (!isNaN(startPrice) && !isNaN(endPrice)) {
            const priceChange = ((endPrice - startPrice) / startPrice) * 100
            console.log(`- Price Change: ${startPrice.toFixed(2)} → ${endPrice.toFixed(2)} (${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%)`)
          }
        }
      }
    }
    
    console.log(chalk.gray('\nTo use with the demo:'))
    console.log(chalk.gray(`yarn demo:quick ${outputPath}`))
    
  } catch (error) {
    spinner.fail('Download failed')
    console.error(chalk.red('\nError:'), error)
    
    if (error instanceof Error && error.message.includes('Failed to fetch data')) {
      console.log(chalk.yellow('\nTips:'))
      console.log('- Check if the symbol is valid (e.g., AAPL, MSFT, BTC-USD)')
      console.log('- Try a different timeframe or interval')
      console.log('- Some symbols may not be available on Yahoo Finance')
    }
    
    process.exit(1)
  }
}

// Run the script
main().catch(console.error)