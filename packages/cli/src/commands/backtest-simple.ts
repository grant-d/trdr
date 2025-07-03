import { parse } from 'csv-parse'
import { createReadStream } from 'fs'
import chalk from 'chalk'
import ora from 'ora'
import { table } from 'table'
import path from 'path'
import { 
  EventBus, 
  BacktestDataFeed,
  GridManager,
  TrailingOrderManager,
  AgentOrchestrator
} from '@trdr/core'
import { 
  type Candle, 
  toStockSymbol
} from '@trdr/shared'
import { EventTypes } from '@trdr/core/dist/events/types'
// Agents temporarily disabled due to API mismatches
// import { RsiAgent } from '../agents/rsi-agent'
// import { MacdAgent } from '../agents/macd-agent'
// import { BollingerBandsAgent } from '../agents/bollinger-agent'

interface BacktestOptions {
  file: string
  symbol: string
  capital: string
  gridSpacing: string
  gridLevels: string
  agents: string
  startDate?: string
  endDate?: string
  verbose: boolean
}

interface CSVRow {
  Date?: string
  Open?: string
  High?: string
  Low?: string
  Close?: string
  'Adj Close'?: string
  Volume?: string
  // Alternative formats
  timestamp?: string
  date?: string
  time?: string
  open?: string
  high?: string
  low?: string
  close?: string
  volume?: string
}

export async function runBacktest(options: BacktestOptions): Promise<void> {
  const spinner = ora('Setting up backtest environment...').start()
  
  try {
    // Parse options
    const symbol = toStockSymbol(options.symbol)
    const initialCapital = parseFloat(options.capital)
    const gridSpacing = parseFloat(options.gridSpacing)
    const gridLevels = parseInt(options.gridLevels)
    const agentTypes = options.agents.split(',').map(a => a.trim())
    
    // Load CSV data
    spinner.text = 'Loading price data from CSV...'
    const candles = await loadCandlesFromCSV(options.file)
    
    if (candles.length === 0) {
      throw new Error('No valid price data found in CSV')
    }
    
    spinner.text = `Loaded ${candles.length} candles`
    
    // Filter by date range if specified
    let filteredCandles = candles
    if (options.startDate || options.endDate) {
      const startTime = options.startDate ? new Date(options.startDate).getTime() : 0
      const endTime = options.endDate ? new Date(options.endDate).getTime() : Infinity
      filteredCandles = candles.filter(c => c.timestamp >= startTime && c.timestamp <= endTime)
    }
    
    // Initialize event bus
    spinner.text = 'Initializing event system...'
    const eventBus = EventBus.getInstance()
    Object.values(EventTypes).forEach(eventType => {
      eventBus.registerEvent(eventType)
    })
    
    // Create backtest data feed
    const startCandle = filteredCandles[0]!
    const endCandle = filteredCandles[filteredCandles.length - 1]!
    
    const dataFeed = new BacktestDataFeed({
      symbol: symbol,
      feedType: 'backtest',
      dataSource: 'csv',
      speed: 1000,
      startDate: startCandle.timestamp,
      endDate: endCandle.timestamp,
      networkDelay: 0,
      failureRate: 0,
      debug: false
    })
    
    // Initialize core components
    spinner.text = 'Initializing trading components...'
    
    const trailingOrderManager = new TrailingOrderManager(eventBus)
    
    // Initialize grid manager
    const gridManager = new GridManager(
      eventBus,
      trailingOrderManager
    )
    
    // Initialize agents
    spinner.text = 'Initializing trading agents...'
    const agentOrchestrator = new AgentOrchestrator(eventBus)
    
    // Agents temporarily disabled due to API mismatches
    console.log(chalk.yellow('⚠ Note: Agents temporarily disabled for build stability'))
    console.log(chalk.gray(`Requested agents: ${agentTypes.join(', ')}`))
    
    // TODO: Re-enable agents after fixing API compatibility
    /*
    if (agentTypes.includes('rsi')) {
      const rsiAgent = new RSIAgent({
        id: 'rsi-agent',
        name: 'RSI Agent',
        version: '1.0.0',
        description: 'RSI-based trading signals',
        type: 'momentum'
      })
      await agentOrchestrator.registerAgent(rsiAgent, 1.0)
    }
    
    if (agentTypes.includes('macd')) {
      const macdAgent = new MACDAgent({
        id: 'macd-agent',
        name: 'MACD Agent',
        version: '1.0.0',
        description: 'MACD-based trading signals',
        type: 'momentum'
      })
      await agentOrchestrator.registerAgent(macdAgent, 1.0)
    }
    
    if (agentTypes.includes('bollinger')) {
      const bollingerAgent = new BollingerBandsAgent({
        id: 'bollinger-agent',
        name: 'Bollinger Bands Agent',
        version: '1.0.0',
        description: 'Bollinger Bands trading signals',
        type: 'volatility'
      })
      await agentOrchestrator.registerAgent(bollingerAgent, 1.0)
    }
    */
    
    // Track performance metrics
    let totalTrades = 0
    let winningTrades = 0
    let totalPnL = 0
    let currentCapital = initialCapital
    const trades: any[] = []
    
    // Subscribe to order filled events
    eventBus.subscribe(EventTypes.ORDER_FILLED, (data: unknown) => {
      const event = data as any
      if (event.order) {
        totalTrades++
        const pnl = event.order.side === 'sell' 
          ? (event.order.filledPrice - event.order.price) * event.order.filledSize
          : (event.order.price - event.order.filledPrice) * event.order.filledSize
        
        if (pnl > 0) winningTrades++
        totalPnL += pnl
        currentCapital += pnl
        
        trades.push({
          time: new Date(event.timestamp),
          side: event.order.side,
          price: event.order.filledPrice,
          size: event.order.filledSize,
          pnl
        })
        
        if (options.verbose) {
          console.log(chalk.blue(`[TRADE] ${event.order.side.toUpperCase()} ${event.order.filledSize} @ ${event.order.filledPrice} | PnL: ${pnl > 0 ? chalk.green(`+${pnl.toFixed(2)}`) : chalk.red(pnl.toFixed(2))}`))
        }
      }
    })
    
    // Create initial grid
    spinner.text = 'Creating trading grid...'
    const currentPrice = filteredCandles[0]?.close || 50000
    
    const gridConfig = {
      gridSpacing,
      gridLevels,
      trailPercent: 0.5,
      minOrderSize: 0.001,
      maxOrderSize: currentCapital * 0.05 / currentPrice, // Max 5% per order
      rebalanceThreshold: 0.1
    }
    
    const gridParams = {
      symbol,
      allocatedCapital: currentCapital * 0.8, // Use 80% for grid
      baseAmount: 0,
      quoteAmount: currentCapital * 0.8,
      riskLevel: 0.5
    }
    
    const grid = await gridManager.createGrid(gridConfig, gridParams)
    await gridManager.activateNearbyGrids(grid.gridId, currentPrice)
    
    spinner.succeed('Backtest environment ready')
    
    // Run backtest
    console.log(chalk.yellow('\nRunning backtest...'))
    const progressBar = ora('Processing candles...').start()
    
    let processedCandles = 0
    const startTime = Date.now()
    
    // Subscribe to market data
    await dataFeed.subscribe([symbol])
    
    eventBus.subscribe(EventTypes.MARKET_CANDLE, async (data: any) => {
      if (data.symbol !== symbol) return
      const candle = data as Candle
      processedCandles++
      progressBar.text = `Processing candle ${processedCandles}/${filteredCandles.length} (${((processedCandles/filteredCandles.length) * 100).toFixed(1)}%)`
      
      // Get agent consensus
      const marketContext = {
        symbol,
        currentPrice: candle.close,
        candles: filteredCandles.slice(Math.max(0, processedCandles - 100), processedCandles)
      }
      
      try {
        const consensus = await agentOrchestrator.getConsensus(marketContext)
        
        if (options.verbose && consensus.action !== 'hold') {
          console.log(chalk.gray(`[CONSENSUS] ${consensus.action} - confidence: ${consensus.confidence.toFixed(2)}`))
        }
      } catch (error) {
        // Continue even if consensus fails
      }
      
      // Update grid with new price
      await gridManager.updateGrid(grid.gridId, candle.close)
      await gridManager.processMarketUpdate(grid.gridId, candle.close, candle.volume)
      
      // Process trailing orders
      await trailingOrderManager.processMarketUpdate(symbol, candle.close)
    })
    
    // Start data feed
    await dataFeed.start()
    
    // Wait for completion
    await new Promise(resolve => {
      const checkInterval = setInterval(() => {
        if (processedCandles >= filteredCandles.length) {
          clearInterval(checkInterval)
          resolve(undefined)
        }
      }, 100)
    })
    
    progressBar.succeed('Backtest completed')
    
    // Calculate statistics
    const endTime = Date.now()
    const duration = (endTime - startTime) / 1000
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0
    const avgTrade = totalTrades > 0 ? totalPnL / totalTrades : 0
    const roi = ((currentCapital - initialCapital) / initialCapital) * 100
    
    // Display results
    console.log(chalk.cyan('\n=== Backtest Results ===\n'))
    
    const summaryData = [
      ['Metric', 'Value'],
      ['Initial Capital', `$${initialCapital.toFixed(2)}`],
      ['Final Capital', `$${currentCapital.toFixed(2)}`],
      ['Total P&L', totalPnL >= 0 ? chalk.green(`+$${totalPnL.toFixed(2)}`) : chalk.red(`-$${Math.abs(totalPnL).toFixed(2)}`)],
      ['ROI', roi >= 0 ? chalk.green(`+${roi.toFixed(2)}%`) : chalk.red(`${roi.toFixed(2)}%`)],
      ['Total Trades', totalTrades.toString()],
      ['Winning Trades', winningTrades.toString()],
      ['Win Rate', `${winRate.toFixed(2)}%`],
      ['Avg Trade P&L', avgTrade >= 0 ? chalk.green(`+$${avgTrade.toFixed(2)}`) : chalk.red(`-$${Math.abs(avgTrade).toFixed(2)}`)],
      ['Duration', `${duration.toFixed(1)}s`],
      ['Candles/sec', `${(filteredCandles.length / duration).toFixed(0)}`]
    ]
    
    console.log(table(summaryData))
    
    // Show price range
    const minPrice = Math.min(...filteredCandles.map(c => c.low))
    const maxPrice = Math.max(...filteredCandles.map(c => c.high))
    const priceRange = ((maxPrice - minPrice) / minPrice) * 100
    
    console.log(chalk.cyan('\n=== Market Summary ===\n'))
    console.log(`Price Range: $${minPrice.toFixed(2)} - $${maxPrice.toFixed(2)} (${priceRange.toFixed(2)}%)`)
    console.log(`Total Candles: ${filteredCandles.length}`)
    
    // Cleanup
    await dataFeed.stop()
    await gridManager.shutdown()
    await agentOrchestrator.shutdown()
    
  } catch (error) {
    spinner.fail('Backtest failed')
    console.error(chalk.red('\nError:'), error)
    process.exit(1)
  }
}

async function loadCandlesFromCSV(filePath: string): Promise<Candle[]> {
  const candles: Candle[] = []
  const absolutePath = path.resolve(filePath)
  
  return new Promise((resolve, reject) => {
    const parser = parse({
      columns: true,
      skip_empty_lines: true,
      cast: false, // We'll parse manually
      trim: true
    })
    
    parser.on('data', (row: CSVRow) => {
      try {
        // Handle different CSV formats
        let timestamp: number
        let open: number
        let high: number
        let low: number
        let close: number
        let volume: number
        
        // Yahoo Finance format
        if (row.Date && row.Open && row.High && row.Low && row.Close) {
          timestamp = new Date(row.Date).getTime()
          open = parseFloat(row.Open)
          high = parseFloat(row.High)
          low = parseFloat(row.Low)
          // Use Adj Close if available, otherwise Close
          close = row['Adj Close'] ? parseFloat(row['Adj Close']) : parseFloat(row.Close)
          volume = parseFloat(row.Volume || '0')
        }
        // Standard format
        else if (row.timestamp || (row.date && row.time) || row.date) {
          if (row.timestamp) {
            timestamp = new Date(row.timestamp).getTime()
          } else if (row.date && row.time) {
            timestamp = new Date(`${row.date} ${row.time}`).getTime()
          } else {
            timestamp = new Date(row.date!).getTime()
          }
          open = parseFloat(row.open!)
          high = parseFloat(row.high!)
          low = parseFloat(row.low!)
          close = parseFloat(row.close!)
          volume = parseFloat(row.volume || '0')
        } else {
          throw new Error('Unrecognized CSV format')
        }
        
        // Validate candle data
        if (isNaN(timestamp) || isNaN(open) || isNaN(close) || isNaN(high) || isNaN(low)) {
          throw new Error('Invalid price data')
        }
        
        const candle: Candle = {
          timestamp: timestamp as any, // Type assertion for epoch time
          open,
          high,
          low,
          close,
          volume: isNaN(volume) ? 0 : volume
        }
        
        candles.push(candle)
      } catch (error) {
        // Skip header or invalid rows silently
      }
    })
    
    parser.on('error', reject)
    parser.on('end', () => {
      // Sort candles by timestamp
      candles.sort((a, b) => a.timestamp - b.timestamp)
      resolve(candles)
    })
    
    createReadStream(absolutePath).pipe(parser)
  })
}