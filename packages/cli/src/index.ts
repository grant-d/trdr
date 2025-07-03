#!/usr/bin/env node

import { Command } from 'commander'
import { runBacktest } from './commands/backtest-simple'
import { runLive } from './commands/live'
import { runPaperTrade } from './commands/paper-trade'

const program = new Command()

program
  .name('trdr')
  .description('TRDR - Advanced Trading System CLI')
  .version('1.0.0')

program
  .command('backtest')
  .description('Run backtest with historical data from CSV')
  .option('-f, --file <path>', 'Path to CSV file with price data', './data.csv')
  .option('-s, --symbol <symbol>', 'Trading symbol (e.g., BTC-USD)', 'BTC-USD')
  .option('-c, --capital <amount>', 'Initial capital', '10000')
  .option('--grid-spacing <percent>', 'Grid spacing percentage', '2')
  .option('--grid-levels <count>', 'Number of grid levels', '10')
  .option('--agents <list>', 'Comma-separated list of agents to use', 'rsi,macd,bollinger')
  .option('--start-date <date>', 'Start date (YYYY-MM-DD)')
  .option('--end-date <date>', 'End date (YYYY-MM-DD)')
  .option('-v, --verbose', 'Verbose output')
  .action(runBacktest)

program
  .command('paper')
  .description('Run paper trading (simulated live trading)')
  .option('-s, --symbol <symbol>', 'Trading symbol (e.g., BTC-USD)', 'BTC-USD')
  .option('-c, --capital <amount>', 'Initial capital', '10000')
  .option('--grid-spacing <percent>', 'Grid spacing percentage', '2')
  .option('--grid-levels <count>', 'Number of grid levels', '10')
  .option('--agents <list>', 'Comma-separated list of agents to use', 'rsi,macd,bollinger')
  .option('-v, --verbose', 'Verbose output')
  .action(runPaperTrade)

program
  .command('live')
  .description('Run live trading (REAL MONEY - USE WITH CAUTION)')
  .option('-s, --symbol <symbol>', 'Trading symbol (e.g., BTC-USD)', 'BTC-USD')
  .option('-c, --capital <amount>', 'Trading capital', '1000')
  .option('--grid-spacing <percent>', 'Grid spacing percentage', '2')
  .option('--grid-levels <count>', 'Number of grid levels', '10')
  .option('--agents <list>', 'Comma-separated list of agents to use', 'rsi,macd,bollinger')
  .option('--api-key <key>', 'Exchange API key (or use EXCHANGE_API_KEY env var)')
  .option('--api-secret <secret>', 'Exchange API secret (or use EXCHANGE_API_SECRET env var)')
  .option('-v, --verbose', 'Verbose output')
  .option('--confirm', 'Confirm you want to trade with real money')
  .action(runLive)

program.parse()