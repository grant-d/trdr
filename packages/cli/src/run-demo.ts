#!/usr/bin/env node

// yarn demo --verbose --adaptive --liquidate ./csv/BTCUSD-feed.csv -s BTC -st -r aggressive --analyze  --normalize=frac-diff

import {
  AgentOrchestrator,
  EventBus,
  TrailingOrderManager
} from '@trdr/core'
import type { ConsensusResult } from '@trdr/core/dist/agents/types'
import { EventTypes } from '@trdr/core/dist/events/types'
import {
  type Candle,
  toEpochDate,
  toStockSymbol
} from '@trdr/shared'
import chalk from 'chalk'
import { parse } from 'csv-parse'
import { createReadStream } from 'fs'
import path from 'path'
import { table } from 'table'
import { AdaptiveBollingerBandsAgent } from './agents/adaptive-bollinger-agent'
import { AdaptiveMacdAgent } from './agents/adaptive-macd-agent'
import { AdaptiveMomentumAgent } from './agents/adaptive-momentum-agent'
import { AdaptiveRsiAgent } from './agents/adaptive-rsi-agent'
import { AdaptiveVolumeProfileAgent } from './agents/adaptive-volume-profile-agent'
import { BollingerBandsAgent } from './agents/bollinger-agent'
import { ClaudePatternAgent } from './agents/claude-pattern-agent'
// import { LorentzianDistanceAgent } from './agents/lorentzian-distance-agent'
import { LorentzianFixedAgent } from './agents/lorentzian-fixed-agent'
import { MacdAgent } from './agents/macd-agent'
import { MarketBreathingAgent } from './agents/market-breathing-agent'
import { MarketMemoryAgent } from './agents/market-memory-agent'
import { MathematicalHarmonyAgent } from './agents/mathematical-harmony-agent'
import { MatrixProfileAgent } from './agents/matrix-profile-agent'
import { MomentumAgent } from './agents/momentum-agent'
import { QuantumSuperpositionAgent } from './agents/quantum-superposition-agent'
import { RelativisticFieldAgent } from './agents/relativistic-field-agent'
import { RsiAgent } from './agents/rsi-agent'
import { SwarmIntelligenceAgent } from './agents/swarm-intelligence-agent'
import { TimeDecayAgent } from './agents/time-decay-agent'
import { TopologicalShapeAgent } from './agents/topological-shape-agent'
import { VolumeProfileAgent } from './agents/volume-profile-agent'
import { SelfTuningRelativisticAgent } from './agents/self-tuning-relativistic-agent'
import { AgentPerformanceAnalyzer } from './utils/agent-performance-analyzer'
import { PriceNormalizer, createAutoNormalizer, type NormalizedCandle } from './utils/price-normalizer'

const CANDLE_COUNT = 1000

// Helper to get original price from candle (normalized or not)
function getOriginalPrice(candle: Candle | NormalizedCandle): number {
  return 'originalClose' in candle ? candle.originalClose : candle.close
}

// Helper to check if candle is normalized
function isNormalizedCandle(candle: Candle | NormalizedCandle): candle is NormalizedCandle {
  return 'originalClose' in candle
}

async function runDemo() {
  console.log(chalk.cyan('TRDR Trading System Demo\n'))
  
  // Parse command line arguments
  const args = process.argv.slice(2)
  
  // Show help if requested
  if (args.includes('--help') || args.includes('-h')) {
    console.log(chalk.yellow('Usage:'))
    console.log('  yarn demo [csv-file] [options]\n')
    console.log(chalk.yellow('Arguments:'))
    console.log('  csv-file        Path to CSV data file [default: btc-usd.csv]\n')
    console.log(chalk.yellow('Options:'))
    console.log('  --symbol, -s        Trading symbol (e.g., BTC-USD, AAPL, SOL-USD) [default: BTC-USD]')
    console.log('  --duration          Demo duration in milliseconds [default: 5000]')
    console.log('  --trail-percent, -t Trail percentage for stop-loss orders [default: auto-tuned]')
    console.log('  --risk-profile, -r  Risk profile: conservative, moderate, aggressive [default: moderate]')
    console.log('  --agents            Enable trading agents (rsi,macd,bollinger,momentum,volume,memory,quantum,topological,timedecay,breathing,swarm,claude,lorentzian,mathematical,relativistic,matrix) [default: all]')
    console.log('  --adaptive, -a      Use adaptive agents that adjust to market conditions')
    console.log('  --self-tuning, -st  Use self-tuning agents that optimize their parameters')
    console.log(`  --candles, -c       Number of candles to process [default: ${CANDLE_COUNT}]`)
    console.log('  --verbose           Show detailed order information')
    console.log('  --liquidate         Liquidate all positions at end')
    console.log('  --analyze           Show detailed agent performance analysis')
    console.log('  --normalize, -n     Normalize prices (auto,z-score,min-max,log,percent,frac-diff) [default: none]')
    console.log('  --help, -h          Show this help\n')
    console.log(chalk.yellow('Examples:'))
    console.log('  yarn demo csv/aapl-sample.csv --symbol AAPL --liquidate')
    console.log('  yarn demo csv/solana-sample.csv --symbol SOL-USD --duration 3000')
    console.log('  yarn demo --symbol ETH-USD --verbose --liquidate')
    console.log('  yarn demo csv/btc-usd.csv --adaptive --agents=rsi,macd,bollinger')
    console.log('  yarn demo --candles 500 --agents relativistic,matrix --verbose')
    return
  }
  const csvFile = args.find(arg => !arg.startsWith('--') && !arg.startsWith('-')) || '/Users/grantdickinson/repos/trdr/csv/btc-usd.csv'
  const durationArg = args.find(arg => arg.startsWith('--duration'))
  const duration = durationArg ? parseInt(durationArg.includes('=') ? durationArg.split('=')[1]! : args[args.indexOf(durationArg) + 1]!) : 5000
  const verbose = args.includes('--verbose')
  const liquidateAtEnd = args.includes('--liquidate')
  const showAnalysis = args.includes('--analyze')
  
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
  
  // Configuration
  const symbolArg = getArgValue('symbol', 's', 'BTC-USD')
  const symbol = toStockSymbol(symbolArg)
  const initialCapital = 10000
  const riskFreeRate = 0.05 // 5% annual risk-free rate (Treasury bills)
  
  // Get tuning parameters
  const trailPercentArg = getArgValue('trail-percent', 't')
  const riskProfileArg = getArgValue('risk-profile', 'r', 'moderate')
  const agentsArg = getArgValue('agents', '', 'rsi,macd,bollinger,momentum,volume,memory,quantum,topological,timedecay,breathing,swarm,lorentzian,mathematical,relativistic,matrix') // Not claude
  const useAdaptive = args.includes('--adaptive') || args.includes('-a')
  const useSelfTuning = args.includes('--self-tuning') || args.includes('-st')
  const candleCountArg = getArgValue('candles', 'c', `${CANDLE_COUNT}`)
  const candleCount = parseInt(candleCountArg) || CANDLE_COUNT
  const normalizeArg = getArgValue('normalize', 'n', 'none')

  // Adaptive tuning that updates parameters as new data arrives
  class AdaptiveTuner {
    private readonly recentCandles: Candle[] = []
    private readonly windowSize = 20 // Rolling window for volatility calculation
    private readonly symbol: string
    private readonly riskProfile: string
    
    constructor(symbol: string, riskProfile: string) {
      this.symbol = symbol
      this.riskProfile = riskProfile
    }
    
    addCandle(candle: Candle) {
      this.recentCandles.push(candle)
      if (this.recentCandles.length > this.windowSize) {
        this.recentCandles.shift() // Keep only recent candles
      }
    }
    
    getCurrentParameters() {
      if (this.recentCandles.length < 2) {
        // Not enough data, use conservative defaults
        return this.getDefaultParameters()
      }
      
      // Calculate market behavior metrics
      const marketMetrics = this.analyzeMarketBehavior()
      
      // Detect asset type
      const isCrypto = this.symbol.includes('BTC') || this.symbol.includes('ETH') || this.symbol.includes('SOL')
      
      // Base parameters by asset type
      const baseTrail = isCrypto ? 0.005 : 0.002   // 0.5% for crypto, 0.2% for stocks
      
      // Risk profile adjustments
      const riskProfiles = {
        conservative: { trailMult: 1.5 },
        moderate: { trailMult: 1.0 },
        aggressive: { trailMult: 0.7 }
      }
      
      const profile = riskProfiles[this.riskProfile as keyof typeof riskProfiles] || riskProfiles.moderate
      
      return {
        trailPercent: baseTrail * profile.trailMult * 100, // Convert to percentage
        volatility: marketMetrics.volatility,
        assetType: isCrypto ? 'crypto' : 'stock',
        sampleSize: this.recentCandles.length,
        marketType: marketMetrics.marketType,
        meanReversionStrength: marketMetrics.meanReversionStrength,
        trendStrength: marketMetrics.trendStrength
      }
    }
    
    private analyzeMarketBehavior() {
      if (this.recentCandles.length < 5) {
        return {
          volatility: 0,
          marketType: 'unknown' as const,
          meanReversionStrength: 0.5,
          trendStrength: 0
        }
      }
      
      const prices = this.recentCandles.map((c: Candle) => c?.close ?? 0)
      const returns = prices.slice(1).map((price, i) => (price - prices[i]!) / (prices[i]! ?? 1))
      
      // Calculate volatility
      const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length
      const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
      const volatility = Math.sqrt(variance)
      
      // Calculate trend strength (using linear regression slope)
      const n = prices.length
      const x = Array.from({length: n}, (_, i) => i)
      const sumX = x.reduce((a, b) => a + b, 0)
      const sumY = prices.reduce((a, b) => a + b, 0)
      const sumXY = x.reduce((sum, xi, i) => sum + xi * prices[i]!, 0)
      const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0)
      
      const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
      const normalizedSlope = slope / (prices[0] || 1) // Normalize by initial price
      const trendStrength = Math.abs(normalizedSlope)
      
      // Calculate mean reversion strength (how often price returns to recent average)
      const recentAvg = prices.slice(-Math.min(10, prices.length)).reduce((a, b) => a + b, 0) / Math.min(10, prices.length)
      const deviations = prices.slice(-5).map(p => Math.abs(p - recentAvg) / recentAvg)
      const avgDeviation = deviations.reduce((a, b) => a + b, 0) / deviations.length
      const meanReversionStrength = Math.max(0, Math.min(1, 1 - avgDeviation * 10)) // Invert deviation
      
      // Determine market type
      let marketType: 'trending' | 'ranging' | 'choppy' | 'unknown'
      
      if (trendStrength > 0.002 && meanReversionStrength < 0.4) {
        marketType = 'trending'
      } else if (trendStrength < 0.001 && meanReversionStrength > 0.6) {
        marketType = 'ranging'
      } else if (volatility > 0.03) {
        marketType = 'choppy'
      } else {
        marketType = 'ranging' // Default for calm markets
      }
      
      return {
        volatility,
        marketType,
        meanReversionStrength,
        trendStrength
      }
    }
    
    private getDefaultParameters() {
      const isCrypto = this.symbol.includes('BTC') || this.symbol.includes('ETH') || this.symbol.includes('SOL')
      
      const riskProfiles = {
        conservative: { trailMult: 1.5 },
        moderate: { trailMult: 1.0 },
        aggressive: { trailMult: 0.7 }
      }
      
      const profile = riskProfiles[this.riskProfile as keyof typeof riskProfiles] || riskProfiles.moderate
      
      return {
        trailPercent: (isCrypto ? 0.5 : 0.2) * profile.trailMult,
        volatility: 0,
        assetType: isCrypto ? 'crypto' : 'stock',
        sampleSize: 0,
        marketType: 'unknown' as const,
        meanReversionStrength: 0.5,
        trendStrength: 0
      }
    }
  }
  
  console.log(chalk.gray('Configuration:'))
  console.log(chalk.gray(`- CSV File: ${csvFile}`))
  console.log(chalk.gray(`- Symbol: ${symbol}`))
  console.log(chalk.gray(`- Initial Capital: $${initialCapital}`))
  console.log(chalk.gray(`- Demo Duration: ${duration}ms`))
  console.log(chalk.gray(`- Candles to Process: ${candleCount}`))
  console.log(chalk.gray(`- Adaptive Agents: ${useAdaptive ? 'ON' : 'OFF'}`))
  console.log(chalk.gray(`- Self-Tuning Agents: ${useSelfTuning ? 'ON' : 'OFF'}`))
  console.log(chalk.gray(`- Verbose Mode: ${verbose ? 'ON' : 'OFF'}`))
  console.log(chalk.gray(`- Liquidate at End: ${liquidateAtEnd ? 'ON' : 'OFF'}\n`))
  
  try {
    // Load CSV data
    console.log(chalk.yellow('Loading price data...'))
    const candles = await loadCandlesFromCSV(csvFile)
    console.log(chalk.green(`✓ Loaded ${candles.length} candles\n`))
    
    if (candles.length === 0) {
      throw new Error('No valid price data found in CSV')
    }
    
    // Initialize price normalizer
    let normalizer: PriceNormalizer
    let normalizedCandles: (Candle | NormalizedCandle)[] = candles
    
    if (normalizeArg !== 'none') {
      console.log(chalk.yellow('Initializing price normalization...'))
      
      // Create normalizer based on option
      if (normalizeArg === 'auto') {
        normalizer = createAutoNormalizer(candles)
      } else {
        const normMethod = normalizeArg as 'z-score' | 'min-max' | 'log' | 'percent-change' | 'frac-diff'
        normalizer = new PriceNormalizer({ method: normMethod })
        normalizer.initialize(candles)
      }
      
      // Normalize all candles
      normalizedCandles = normalizer.normalizeCandles(candles)
      console.log(chalk.green(`✓ Normalized prices using ${normalizer.describe()}\n`))
    } else {
      normalizer = new PriceNormalizer({ method: 'none' })
    }
    
    // Initialize adaptive tuner
    const adaptiveTuner = new AdaptiveTuner(symbol, riskProfileArg)
    
    // Start with default parameters (no future data)
    const currentParams = adaptiveTuner.getCurrentParameters()
    
    // Use manual overrides if provided, otherwise use adaptive values
    const trailPercent = trailPercentArg ? parseFloat(trailPercentArg) : currentParams.trailPercent
    
    // Show data range (using original prices)
    const startDate = new Date(candles[0]!.timestamp)
    const endDate = new Date(candles[candles.length - 1]!.timestamp)
    const startPrice = candles[0]!.close
    const endPrice = candles[candles.length - 1]!.close
    
    console.log(chalk.gray('Data Range:'))
    console.log(chalk.gray(`- Start: ${startDate.toISOString().split('T')[0]} @ $${startPrice.toFixed(2)}`))
    console.log(chalk.gray(`- End: ${endDate.toISOString().split('T')[0]} @ $${endPrice.toFixed(2)}`))
    console.log(chalk.gray(`- Price Change: ${((endPrice - startPrice) / startPrice * 100).toFixed(2)}%`))
    
    console.log(chalk.gray('\nAdaptive Tuning Parameters:'))
    console.log(chalk.gray(`- Asset Type: ${currentParams.assetType}`))
    console.log(chalk.gray(`- Initial Volatility: ${currentParams.sampleSize > 0 ? (currentParams.volatility * 100).toFixed(3) : 'N/A'}%`))
    console.log(chalk.gray(`- Trail Percent: ${trailPercent.toFixed(3)}% ${trailPercentArg ? '(manual)' : '(auto)'}`))
    console.log(chalk.gray(`- Risk Profile: ${riskProfileArg}\n`))
    
    // Initialize event bus
    console.log(chalk.yellow('Initializing system...'))
    const eventBus = EventBus.getInstance()
    Object.values(EventTypes).forEach(eventType => {
      eventBus.registerEvent(eventType)
    })
    
    // Create components  
    const trailingOrderManager = new TrailingOrderManager(eventBus)
    
    // Initialize agents with custom consensus config
    console.log(chalk.yellow('Initializing trading agents...'))
    const agentOrchestrator = new AgentOrchestrator(eventBus, {
      minConfidence: 0.2, // Lower threshold to allow more signals through
      useAdaptiveWeights: true,
      weightUpdateFrequency: 10 // Update more frequently
    })
    
    // Use confidence-weighted consensus for better signal aggregation
    agentOrchestrator.setConsensusStrategy('confidence-weighted')
    
    const enabledAgents = agentsArg.split(',').map(a => a.trim().toLowerCase())
    
    if (enabledAgents.includes('rsi')) {
      if (useAdaptive) {
        const adaptiveRsiAgent = new AdaptiveRsiAgent({
          id: 'rsi-agent',
          name: 'Adaptive RSI Agent',
          version: '1.0.0',
          description: 'Market-adaptive RSI signals with regime detection',
          type: 'momentum',
          defaultConfig: {}
        })
        await adaptiveRsiAgent.initialize()
        await agentOrchestrator.registerAgent(adaptiveRsiAgent, 1.0)
        console.log(chalk.green('✓ Adaptive RSI Agent initialized'))
      } else {
        const rsiAgent = new RsiAgent({
          id: 'rsi-agent',
          name: 'RSI Agent',
          version: '1.0.0',
          description: 'Enhanced RSI-based trading signals with divergence detection',
          type: 'momentum',
          defaultConfig: {}
        })
        await rsiAgent.initialize()
        await agentOrchestrator.registerAgent(rsiAgent, 1.0)
        console.log(chalk.green('✓ RSI Agent initialized'))
      }
    }
    
    if (enabledAgents.includes('macd')) {
      if (useAdaptive) {
        const adaptiveMacdAgent = new AdaptiveMacdAgent({
          id: 'macd-agent',
          name: 'Adaptive MACD Agent',
          version: '1.0.0',
          description: 'Market-adaptive MACD signals with regime detection',
          type: 'momentum',
          defaultConfig: {}
        })
        await adaptiveMacdAgent.initialize()
        await agentOrchestrator.registerAgent(adaptiveMacdAgent, 1.0)
        console.log(chalk.green('✓ Adaptive MACD Agent initialized'))
      } else {
        const macdAgent = new MacdAgent({
          id: 'macd-agent',
          name: 'MACD Agent',
          version: '1.0.0',
          description: 'Enhanced MACD-based trading signals with crossover and divergence detection',
          type: 'momentum',
          defaultConfig: {}
        })
        await macdAgent.initialize()
        await agentOrchestrator.registerAgent(macdAgent, 1.0)
        console.log(chalk.green('✓ MACD Agent initialized'))
      }
    }
    
    if (enabledAgents.includes('bollinger')) {
      if (useAdaptive) {
        const adaptiveBollingerAgent = new AdaptiveBollingerBandsAgent({
          id: 'bollinger-agent',
          name: 'Adaptive Bollinger Bands Agent',
          version: '1.0.0',
          description: 'Market-adaptive Bollinger Bands with dynamic parameters',
          type: 'volatility',
          defaultConfig: {}
        })
        await adaptiveBollingerAgent.initialize()
        await agentOrchestrator.registerAgent(adaptiveBollingerAgent, 1.0)
        console.log(chalk.green('✓ Adaptive Bollinger Bands Agent initialized'))
      } else {
        const bollingerAgent = new BollingerBandsAgent({
          id: 'bollinger-agent',
          name: 'Bollinger Bands Agent',
          version: '1.0.0',
          description: 'Enhanced Bollinger Bands trading signals with squeeze and %B analysis',
          type: 'volatility',
          defaultConfig: {}
        })
        await bollingerAgent.initialize()
        await agentOrchestrator.registerAgent(bollingerAgent, 1.0)
        console.log(chalk.green('✓ Bollinger Bands Agent initialized'))
      }
    }
    
    if (enabledAgents.includes('momentum')) {
      if (useAdaptive) {
        const adaptiveMomentumAgent = new AdaptiveMomentumAgent({
          id: 'momentum-agent',
          name: 'Adaptive Momentum Agent',
          version: '1.0.0',
          description: 'Market-adaptive momentum analysis with dynamic RSI/MACD parameters',
          type: 'momentum',
          defaultConfig: {}
        })
        await adaptiveMomentumAgent.initialize()
        await agentOrchestrator.registerAgent(adaptiveMomentumAgent, 1.0)
        console.log(chalk.green('✓ Adaptive Momentum Agent initialized'))
      } else {
        const momentumAgent = new MomentumAgent({
          id: 'momentum-agent',
          name: 'Momentum Agent',
          version: '1.0.0',
          description: 'Comprehensive momentum analysis combining RSI and MACD with advanced divergence detection',
          type: 'momentum',
          defaultConfig: {}
        })
        await momentumAgent.initialize()
        await agentOrchestrator.registerAgent(momentumAgent, 1.0)
        console.log(chalk.green('✓ Momentum Agent initialized'))
      }
    }
    
    if (enabledAgents.includes('volume')) {
      if (useAdaptive) {
        const adaptiveVolumeProfileAgent = new AdaptiveVolumeProfileAgent({
          id: 'volume-profile-agent',
          name: 'Adaptive Volume Profile Agent',
          version: '1.0.0',
          description: 'Market-adaptive volume analysis with dynamic threshold adjustment',
          type: 'volume',
          defaultConfig: {}
        })
        await adaptiveVolumeProfileAgent.initialize()
        await agentOrchestrator.registerAgent(adaptiveVolumeProfileAgent, 1.0)
        console.log(chalk.green('✓ Adaptive Volume Profile Agent initialized'))
      } else {
        const volumeProfileAgent = new VolumeProfileAgent({
          id: 'volume-profile-agent',
          name: 'Volume Profile Agent',
          version: '1.0.0',
          description: 'Volume-based analysis for support/resistance and significant volume events',
          type: 'volume',
          defaultConfig: {}
        })
        await volumeProfileAgent.initialize()
        await agentOrchestrator.registerAgent(volumeProfileAgent, 1.0)
        console.log(chalk.green('✓ Volume Profile Agent initialized'))
      }
    }
    
    if (enabledAgents.includes('memory')) {
      const marketMemoryAgent = new MarketMemoryAgent({
        id: 'memory-agent',
        name: 'Market Memory Agent',
        version: '1.0.0',
        description: 'Innovative memory-based trading using psychological price imprinting',
        type: 'custom',
        defaultConfig: {}
      })
      await marketMemoryAgent.initialize()
      await agentOrchestrator.registerAgent(marketMemoryAgent, 1.0)
      console.log(chalk.green('✓ Market Memory Agent initialized'))
    }
    
    // Exotic agents
    if (enabledAgents.includes('quantum')) {
      const quantumAgent = new QuantumSuperpositionAgent({
        id: 'quantum-agent',
        name: 'Quantum Superposition Agent',
        version: '1.0.0',
        description: 'Models price as quantum probability waves until observed by volume',
        type: 'custom',
        defaultConfig: {}
      })
      await quantumAgent.initialize()
      await agentOrchestrator.registerAgent(quantumAgent, 1.0)
      console.log(chalk.green('✓ Quantum Superposition Agent initialized'))
    }
    
    if (enabledAgents.includes('topological')) {
      const topologicalAgent = new TopologicalShapeAgent({
        id: 'topological-agent',
        name: 'Topological Shape Agent',
        version: '1.0.0',
        description: 'Uses persistent homology to find price holes and voids',
        type: 'custom',
        defaultConfig: {}
      })
      await topologicalAgent.initialize()
      await agentOrchestrator.registerAgent(topologicalAgent, 1.0)
      console.log(chalk.green('✓ Topological Shape Agent initialized'))
    }
    
    if (enabledAgents.includes('timedecay')) {
      const timeDecayAgent = new TimeDecayAgent({
        id: 'timedecay-agent',
        name: 'Time Decay Agent',
        version: '1.0.0',
        description: 'Tracks time at grid levels and tightens trails on stale prices',
        type: 'custom',
        defaultConfig: {}
      })
      await timeDecayAgent.initialize()
      await agentOrchestrator.registerAgent(timeDecayAgent, 1.0)
      console.log(chalk.green('✓ Time Decay Agent initialized'))
    }
    
    if (enabledAgents.includes('breathing')) {
      const breathingAgent = new MarketBreathingAgent({
        id: 'breathing-agent',
        name: 'Market Breathing Agent',
        version: '1.0.0',
        description: 'Models market as breathing cycles with expansion and contraction',
        type: 'custom',
        defaultConfig: {}
      })
      await breathingAgent.initialize()
      await agentOrchestrator.registerAgent(breathingAgent, 1.0)
      console.log(chalk.green('✓ Market Breathing Agent initialized'))
    }
    
    if (enabledAgents.includes('swarm')) {
      const swarmAgent = new SwarmIntelligenceAgent({
        id: 'swarm-agent',
        name: 'Swarm Intelligence Agent',
        version: '1.0.0',
        description: 'Models market participants as swarm with emergent behavior',
        type: 'custom',
        defaultConfig: {}
      })
      await swarmAgent.initialize()
      await agentOrchestrator.registerAgent(swarmAgent, 1.0)
      console.log(chalk.green('✓ Swarm Intelligence Agent initialized'))
    }
    
    if (enabledAgents.includes('claude')) {
      const claudeAgent = new ClaudePatternAgent({
        id: 'claude-agent',
        name: 'Claude Pattern Agent',
        version: '1.0.0',
        description: 'Connects to Claude API for advanced pattern recognition',
        type: 'ai',
        defaultConfig: {}
      }, { debug: true })
      await claudeAgent.initialize()
      await agentOrchestrator.registerAgent(claudeAgent, 1.0)
      console.log(chalk.green('✓ Claude Pattern Agent initialized'))
    }
    
    if (enabledAgents.includes('lorentzian')) {
      const lorentzianAgent = new LorentzianFixedAgent({
        id: 'lorentzian-agent',
        name: 'Lorentzian Fixed Agent',
        version: '1.0.0',
        description: 'Fixed Lorentzian distance pattern matching',
        type: 'custom',
        defaultConfig: {}
      })
      await lorentzianAgent.initialize()
      await agentOrchestrator.registerAgent(lorentzianAgent, 1.0)
      console.log(chalk.green('✓ Lorentzian Fixed Agent initialized'))
    }
    
    if (enabledAgents.includes('mathematical')) {
      const mathematicalAgent = new MathematicalHarmonyAgent({
        id: 'mathematical-agent',
        name: 'Mathematical Harmony Agent',
        version: '1.0.0',
        description: 'Uses prime numbers, golden ratio, Fibonacci, and Riemann zeta function for pattern analysis',
        type: 'custom',
        defaultConfig: {}
      })
      await mathematicalAgent.initialize()
      await agentOrchestrator.registerAgent(mathematicalAgent, 1.0)
      console.log(chalk.green('✓ Mathematical Harmony Agent initialized'))
    }
    
    if (enabledAgents.includes('relativistic')) {
      if (useSelfTuning) {
        const selfTuningRelativisticAgent = new SelfTuningRelativisticAgent({
          id: 'relativistic-agent',
          name: 'Self-Tuning Relativistic Field Agent',
          version: '1.0.0',
          description: 'Self-tuning relativistic momentum field analysis with adaptive parameters',
          type: 'custom',
          defaultConfig: {}
        })
        await selfTuningRelativisticAgent.initialize()
        await agentOrchestrator.registerAgent(selfTuningRelativisticAgent, 1.0)
        console.log(chalk.green('✓ Self-Tuning Relativistic Field Agent initialized'))
      } else {
        const relativisticAgent = new RelativisticFieldAgent({
          id: 'relativistic-agent',
          name: 'Relativistic Field Agent',
          version: '1.0.0',
          description: 'Treats market as relativistic momentum field using Lorentzian distance metrics',
          type: 'custom',
          defaultConfig: {}
        })
        await relativisticAgent.initialize()
        await agentOrchestrator.registerAgent(relativisticAgent, 1.0)
        console.log(chalk.green('✓ Relativistic Field Agent initialized'))
      }
    }
    
    if (enabledAgents.includes('matrix')) {
      const matrixAgent = new MatrixProfileAgent({
        id: 'matrix-agent',
        name: 'Matrix Profile Agent',
        version: '1.0.0',
        description: 'Time series pattern discovery using matrix profile algorithm',
        type: 'custom',
        defaultConfig: {}
      })
      await matrixAgent.initialize()
      await agentOrchestrator.registerAgent(matrixAgent, 1.0)
      console.log(chalk.green('✓ Matrix Profile Agent initialized'))
    }
    
    console.log(chalk.green(`✓ ${enabledAgents.length} agents ready\n`))
    
    // Track performance
    let orderCount = 0
    let filledCount = 0
    let buyVolume = 0 // Will be set when grid is created
    let sellVolume = 0
    let totalBuyValue = 0 // Will be set when grid is created
    let totalSellValue = 0
    const activeOrders = new Map()
    const trades: Array<{side: string, price: number, size: number, timestamp: number, consensus?: any}> = []
    
    // Initialize comprehensive agent performance analyzer
    const performanceAnalyzer = new AgentPerformanceAnalyzer()
    
    // Keep the basic tracker for compatibility, but also use the analyzer
    interface TrackedSignal {
      consensus: ConsensusResult
      entryPrice: number
      timestamp: number
      lookAheadPeriod: number
    }
    
    const agentPerformanceTracker = {
      signals: new Map<string, TrackedSignal[]>(),
      
      // Track individual agent vote usefulness (0-1 score)
      trackSignalOutcome(agentId: string, consensus: ConsensusResult, entryPrice: number, currentPrice: number, _lookAheadPeriod: number): number {
        const signal = consensus.agentSignals[agentId]
        if (!signal) return 0.5
        
        // For hold signals, check if holding was the right decision
        if (signal.action === 'hold') {
          const priceChange = Math.abs((currentPrice - entryPrice) / entryPrice)
          // More lenient scoring for hold signals
          if (priceChange < 0.005) return 0.9      // Excellent hold (< 0.5%)
          if (priceChange < 0.015) return 0.7      // Good hold (< 1.5%)
          if (priceChange < 0.025) return 0.5      // Okay hold (< 2.5%)
          return 0.4                                // Price moved significantly
        }
        
        const priceChange = (currentPrice - entryPrice) / entryPrice
        const expectedDirection = signal.action === 'buy' ? 1 : -1
        const actualDirection = Math.sign(priceChange)
        
        // Calculate usefulness score based on:
        // 1. Direction correctness
        // 2. Magnitude alignment with confidence
        // 3. Partial credit for reasonable decisions
        
        // More nuanced direction scoring
        let directionScore = 0.5 // Base score
        if (actualDirection === expectedDirection) {
          directionScore = 0.7 + Math.min(0.2, Math.abs(priceChange) * 10) // 0.7-0.9
        } else if (Math.abs(priceChange) < 0.01) {
          directionScore = 0.5 // Neutral if price barely moved
        } else {
          directionScore = 0.3 // Wrong direction but still some credit
        }
        
        // Confidence alignment bonus
        const confidenceBonus = signal.confidence > 0.7 && actualDirection === expectedDirection ? 0.1 : 0
        
        return Math.max(0.1, Math.min(1, directionScore + confidenceBonus))
      },
      
      // Update agent weights based on recent performance
      updateAgentWeights(agentOrchestrator: AgentOrchestrator, _currentPrice: number) {
        const performanceWindow = 15 // Use last 15 signals for weight calculation (rolling window)
        
        for (const [agentId, signals] of this.signals) {
          // Find signals that are old enough to evaluate
          const evaluableSignals = signals.filter(s => s.lookAheadPeriod <= 0)
          
          if (evaluableSignals.length >= 2) { // Need minimum sample size
            // Calculate recent performance scores
            const recentScores = evaluableSignals
              .slice(-performanceWindow)
              .map(s => this.trackSignalOutcome(agentId, s.consensus, s.entryPrice, _currentPrice, s.lookAheadPeriod))
            
            const avgScore = recentScores.reduce((sum, score) => sum + score, 0) / recentScores.length
            const consistency = 1 - (recentScores.reduce((sum, score) => sum + Math.pow(score - avgScore, 2), 0) / recentScores.length)
            
            // Weight combines average performance and consistency
            const newWeight = (avgScore * 0.7 + consistency * 0.3) * 2 // Scale to 0-2 range
            
            try {
              agentOrchestrator.setAgentWeight(agentId, Math.max(0.1, newWeight)) // Minimum weight of 0.1
            } catch (error) {
              // Agent might not exist, ignore error
            }
          }
          
          // Age signals (decrement lookAheadPeriod)
          signals.forEach(s => s.lookAheadPeriod--)
          
          // Also age signals in the analyzer
          performanceAnalyzer.ageSignals()
          
          // Remove old evaluated signals to prevent memory growth
          if (signals.length > performanceWindow * 2) {
            signals.splice(0, signals.length - performanceWindow * 2)
          }
        }
      },
      
      // Record a new consensus signal for tracking
      recordSignal(consensus: ConsensusResult, entryPrice: number, timestamp: number) {
        const lookAheadPeriod = 3 // Will evaluate after 3 candles
        
        // Also record in the comprehensive analyzer
        performanceAnalyzer.recordSignal(consensus, entryPrice, timestamp, lookAheadPeriod)
        
        for (const agentId in consensus.agentSignals) {
          if (!this.signals.has(agentId)) {
            this.signals.set(agentId, [])
          }
          
          this.signals.get(agentId)!.push({
            consensus,
            entryPrice,
            timestamp,
            lookAheadPeriod
          })
        }
      },
      
      // Get current performance stats for display
      getPerformanceStats(currentPrice?: number): Record<string, {avgScore: number, recentSignals: number, currentWeight: number}> {
        const stats: Record<string, {avgScore: number, recentSignals: number, currentWeight: number}> = {}
        
        for (const [agentId, signals] of this.signals) {
          const evaluableSignals = signals.filter(s => s.lookAheadPeriod <= 0)
          
          let recentScores: number[] = []
          if (evaluableSignals.length > 0 && currentPrice) {
            recentScores = evaluableSignals.slice(-10).map(s => 
              this.trackSignalOutcome(agentId, s.consensus, s.entryPrice, currentPrice, 0)
            )
          }
          
          stats[agentId] = {
            avgScore: recentScores.length > 0 ? recentScores.reduce((sum, s) => sum + s, 0) / recentScores.length : 0.5,
            recentSignals: evaluableSignals.length,
            currentWeight: 1.0 // Will be updated by orchestrator
          }
        }
        
        return stats
      }
    }
    
    // Subscribe to events
    eventBus.subscribe(EventTypes.ORDER_CREATED, (data: any) => {
      orderCount++
      if (data.order) {
        activeOrders.set(data.order.id, data.order)
        if (verbose) {
          // Display appropriate price for different order types
          const displayPrice = data.order.price ?? data.order.limitPrice ?? data.order.triggerPrice
          console.log(chalk.cyan(`[ORDER] ${data.order.side.toUpperCase()} @ $${displayPrice?.toFixed(2)} (${data.order.type})`))
        }
      }
    })
    
    eventBus.subscribe(EventTypes.ORDER_FILLED, (data: any) => {
      filledCount++
      if (data.order) {
        activeOrders.delete(data.order.id)
        const side = data.order.side
        const price = data.order.filledPrice
        const size = data.order.filledSize || data.order.size
        
        // Track trade data
        trades.push({
          side,
          price,
          size,
          timestamp: Date.now()
        })
        
        // Update volume and value tracking
        if (side === 'buy') {
          buyVolume += size
          totalBuyValue += price * size
        } else {
          sellVolume += size
          totalSellValue += price * size
        }
        
        console.log(chalk.green(`[FILLED] ${side.toUpperCase()} ${size.toFixed(4)} ${symbol} @ $${price?.toFixed(2)}`))
      }
    })
    
    // Prepare for trading
    console.log(chalk.yellow('\nPreparing trading system...'))
    const currentPrice = candles[0]!.close
    
    // Start with 50/50 allocation to avoid short selling
    const halfCapital = initialCapital * 0.4 // Use 80% total (40% each)
    const startingBtc = halfCapital / currentPrice
    
    const tradingParams = {
      symbol,
      allocatedCapital: initialCapital * 0.8,
      baseAmount: startingBtc, // Start with BTC to enable selling
      quoteAmount: halfCapital, // And matching USD
      riskLevel: 0.5,
      centerPrice: currentPrice // Start trading around actual market price
    }
    
    console.log(chalk.gray(`✓ Trading configuration ready`))
    console.log(chalk.gray(`  - Start Price: $${currentPrice.toFixed(2)} (actual market price)`))
    console.log(chalk.gray(`  - Trail Percent: ${trailPercent.toFixed(2)}%`))
    console.log(chalk.yellow(`  - Trading will activate after agents build history (20 candles)\n`))
    
    // Process market data directly from our CSV instead of using BacktestDataFeed
    console.log(chalk.yellow('Processing market data...'))
    
    // Use a subset of our data for the demo (normalized if applicable)
    const demoCandles = normalizedCandles.slice(0, Math.min(candleCount, normalizedCandles.length))
    let processedCandles = 0
    let tradingActive = false // Will activate after 20 candles
    
    for (const candle of demoCandles) {
      // Get original price for this candle (for orders and display)
      const originalPrice = getOriginalPrice(candle)
      
      // Add candle to adaptive tuner (no future data leak)
      adaptiveTuner.addCandle(candle)
      
      // Update parameters every 10 candles (adaptive rebalancing)
      if (processedCandles > 0 && processedCandles % 10 === 0 && !trailPercentArg) {
        const newParams = adaptiveTuner.getCurrentParameters()
        if (verbose) {
          console.log(chalk.yellow(`[ADAPT] ${newParams.marketType?.toUpperCase()} | Vol: ${(newParams.volatility * 100).toFixed(2)}% | Trail: ${newParams.trailPercent.toFixed(2)}% | MeanRev: ${(newParams.meanReversionStrength * 100).toFixed(0)}%`))
        }
        
        // Record market condition for performance analysis
        if (newParams.marketType && newParams.marketType !== 'unknown') {
          performanceAnalyzer.recordMarketCondition({
            type: newParams.marketType,
            volatility: newParams.volatility,
            trendStrength: newParams.trendStrength,
            timestamp: candle.timestamp
          })
        }
        
        // Note: In a real system, you'd update trading parameters
        // For this demo, we'll just track the parameter evolution
      }
      
      if (verbose) {
        console.log(chalk.blue(`[CANDLE] Price: $${originalPrice.toFixed(2)}, Volume: ${candle.volume.toFixed(0)}`))
      }
      
      // Start active trading after building enough history for agents
      if (processedCandles === 20) {
        console.log(chalk.green('\n✓ Sufficient history built - Activating trading system...'))
        
        // Update trading center price to current market price (original, not normalized)
        tradingParams.centerPrice = originalPrice
        // console.log(chalk.gray(`  - Center Price: $${candle.close.toFixed(2)} (current market price)`))
        // console.log(chalk.gray(`  - Levels: ${grid.totalLevels}`))
        console.log(chalk.gray(`  - Starting BTC: ${startingBtc.toFixed(4)}`))
        console.log(chalk.gray(`  - Starting USD: $${halfCapital.toFixed(2)}\n`))
        
        // Set initial position tracking
        buyVolume = startingBtc
        totalBuyValue = halfCapital
        tradingActive = true
      }
      
      // Get agent consensus using recent candles (build up history gradually)
      if (processedCandles >= 20) { // Need enough history for indicators
        const recentCandles = demoCandles.slice(0, processedCandles + 1) // Use only candles up to current point
        
        // Calculate current position (positive = long, negative = short which shouldn't happen)
        const currentPosition = buyVolume - sellVolume
        
        const marketContext = {
          symbol,
          currentPrice: candle.close, // Use normalized price for agent analysis
          candles: recentCandles, // These are normalized candles
          volume: candle.volume,
          timestamp: candle.timestamp,
          currentPosition // Pass position so agents can enforce no-shorting
        }
        
        try {
          const consensus = await agentOrchestrator.getConsensus(marketContext)
          
          // Record signal for post-facto performance tracking (use original price)
          agentPerformanceTracker.recordSignal(consensus, originalPrice, candle.timestamp)
          
          // Update agent weights every 10 candles based on recent performance
          if (processedCandles % 10 === 0) {
            agentPerformanceTracker.updateAgentWeights(agentOrchestrator, originalPrice)
            
            // Get updated weights for display
            const currentWeights = agentOrchestrator.getAgentWeights()
            const performanceStats = agentPerformanceTracker.getPerformanceStats(originalPrice)
            
            // Update current weights in stats for display
            for (const [agentId, stats] of Object.entries(performanceStats)) {
              stats.currentWeight = currentWeights.get(agentId) || 1.0
            }
            
            if (verbose && Object.keys(performanceStats).length > 0) {
              const weightSummary = Object.entries(performanceStats)
                .map(([agentId, stats]) => `${agentId}: ${stats.currentWeight.toFixed(2)}w (${(stats.avgScore * 100).toFixed(0)}% useful)`)
                .join(', ')
              console.log(chalk.cyan(`[WEIGHTS] ${weightSummary}`))
            }
          }
          
          if (verbose && consensus.action !== 'hold') {
            const agentSignals = Object.entries(consensus.agentSignals)
            const signalSummary = agentSignals
              .filter(([, signal]) => signal.action !== 'hold')
              .map(([agentId, signal]) => `${agentId}: ${signal.action.toUpperCase()} (${(signal.confidence * 100).toFixed(0)}%)`)
              .join(', ')
            
            if (signalSummary) {
              console.log(chalk.magenta(`[AGENTS] Consensus: ${consensus.action.toUpperCase()} (${(consensus.confidence * 100).toFixed(0)}%) | ${signalSummary}`))
            }
          }
          
          // Create orders based on consensus
          if (tradingActive && consensus.action !== 'hold' && consensus.confidence >= 0.3) {
            // Calculate order size based on position and risk
            const currentNetPosition = buyVolume - sellVolume
            const maxPositionSize = startingBtc * 2 // Max 2x initial position
            
            let orderSize = 0
            if (consensus.action === 'buy' && currentNetPosition < maxPositionSize) {
              // Buy if not at max position
              orderSize = Math.min(startingBtc * 0.1, maxPositionSize - currentNetPosition)
            } else if (consensus.action === 'sell' && currentNetPosition > 0) {
              // Sell only if we have position (no shorting)
              orderSize = Math.min(startingBtc * 0.1, currentNetPosition)
            }
            
            if (orderSize > 0.0001) { // Minimum order size
              const orderId = `order-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`
              const order = {
                id: orderId,
                side: consensus.action,
                size: orderSize,
                price: originalPrice, // Use original price for orders
                type: 'market',
                status: 'pending',
                timestamp: Date.now()
              }
              
              // Emit order created event
              eventBus.emit(EventTypes.ORDER_CREATED, {
                timestamp: toEpochDate(Date.now()),
                order
              })
              
              // Immediately fill market orders at current price
              setTimeout(() => {
                eventBus.emit(EventTypes.ORDER_FILLED, {
                  timestamp: toEpochDate(Date.now()),
                  order: {
                    ...order,
                    filledPrice: originalPrice,
                    filledSize: orderSize,
                    status: 'filled'
                  }
                })
              }, 10)
            }
          }
        } catch (error) {
          // Agent errors shouldn't stop the demo
          if (verbose) {
            console.log(chalk.red(`[AGENTS] Error: ${error}`))
          }
        }
      }
      
      // Only process orders after trading is active
      if (tradingActive) {
        // Simple order execution: check if any active orders should be filled
        for (const [, order] of activeOrders.entries()) {
          // Get the execution price based on order type
          const executionPrice = order.price ?? order.limitPrice ?? order.triggerPrice
          
          if (!executionPrice) {
            // Skip orders without a valid execution price
            continue
          }
          
          const shouldFill = (
            (order.side === 'buy' && originalPrice <= executionPrice) ||
            (order.side === 'sell' && originalPrice >= executionPrice)
          )
          
          if (shouldFill) {
            // Simulate order fill
            eventBus.emit(EventTypes.ORDER_FILLED, {
              timestamp: toEpochDate(Date.now()),
              order: {
                ...order,
                filledPrice: executionPrice,
                filledSize: order.size,
                status: 'filled'
              }
            })
          }
        }
        
        // Update grid with real market data
        // await gridManager.updateGrid(grid.gridId, candle.close)
        // await gridManager.processMarketUpdate(grid.gridId, candle.close, candle.volume)
        
        // Process trailing orders (using original price)
        await trailingOrderManager.processMarketUpdate(symbol, originalPrice)
      }
      
      // Emit market events for other systems
      eventBus.emit(EventTypes.MARKET_CANDLE, {
        symbol,
        ...candle
      })
      
      processedCandles++
      
      // Add small delay for demo effect
      if (processedCandles % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 50))
      }
      
      // Break when we've processed all requested candles
      if (processedCandles >= demoCandles.length) break
    }
    
    console.log(chalk.green('\n✓ Demo completed!\n'))
    
    // Calculate net position (positive = long, negative = short)
    const currentNetVolume = buyVolume - sellVolume
    
    // Liquidate remaining position if requested
    if (liquidateAtEnd && currentNetVolume !== 0) {
      const liquidationPrice = candles[Math.min(processedCandles - 1, candles.length - 1)]?.close || 0
      // If we have a long position (positive), we need to sell. If short (negative, shouldn't happen), we need to buy
      const liquidationSide = currentNetVolume > 0 ? 'sell' : 'buy'
      const liquidationVolume = Math.abs(currentNetVolume)
      
      console.log(chalk.yellow(`Liquidating position: ${liquidationSide.toUpperCase()} ${liquidationVolume.toFixed(4)} ${symbol} @ $${liquidationPrice.toFixed(2)}`))
      
      // Add liquidation trade to our tracking
      trades.push({
        side: liquidationSide,
        price: liquidationPrice,
        size: liquidationVolume,
        timestamp: Date.now()
      })
      
      // Update volume and value tracking for liquidation
      if (liquidationSide === 'sell') {
        sellVolume += liquidationVolume
        totalSellValue += liquidationPrice * liquidationVolume
      } else {
        buyVolume += liquidationVolume
        totalBuyValue += liquidationPrice * liquidationVolume
      }
      
      console.log(chalk.green(`✓ Position liquidated\n`))
    }
    
    // Calculate P&L and trading metrics
    const netVolume = buyVolume - sellVolume
    const avgBuyPrice = buyVolume > 0 ? totalBuyValue / buyVolume : 0
    const avgSellPrice = sellVolume > 0 ? totalSellValue / sellVolume : 0
    
    // Calculate P&L: simple cash flow approach
    const totalCashIn = totalBuyValue  // Money spent buying
    const totalCashOut = totalSellValue // Money received selling
    const realizedPnL = totalCashOut - totalCashIn
    
    // Calculate unrealized P&L from remaining position
    const finalPrice = candles[Math.min(processedCandles - 1, candles.length - 1)]?.close || 0
    const unrealizedPnL = liquidateAtEnd ? 0 : netVolume * finalPrice
    
    const calculatedPnL = realizedPnL + unrealizedPnL
    const roi = initialCapital > 0 ? (calculatedPnL / initialCapital) * 100 : 0
    
    // Show summary
    console.log(chalk.cyan('Trading Summary:'))
    console.log(`- Candles Processed: ${processedCandles}`)
    console.log(`- Orders Created: ${orderCount}`)
    console.log(`- Orders Filled: ${filledCount}`)
    console.log(`- Fill Rate: ${orderCount > 0 ? ((filledCount / orderCount) * 100).toFixed(1) : 0}%`)
    
    console.log(chalk.cyan('\nP&L Analysis:'))
    const pnlData = [
      ['Metric', 'Value'],
      ['Buy Volume', `${buyVolume.toFixed(4)} ${symbol}`],
      ['Sell Volume', `${sellVolume.toFixed(4)} ${symbol}`],
      ['Avg Buy Price', `$${avgBuyPrice.toFixed(2)}`],
      ['Avg Sell Price', `$${avgSellPrice.toFixed(2)}`],
      ['Net Position', `${netVolume.toFixed(4)} ${symbol}`],
      ['Realized P&L', realizedPnL >= 0 ? chalk.green(`+$${realizedPnL.toFixed(2)}`) : chalk.red(`-$${Math.abs(realizedPnL).toFixed(2)}`)],
      ['Unrealized P&L', unrealizedPnL >= 0 ? chalk.green(`+$${unrealizedPnL.toFixed(2)}`) : chalk.red(`-$${Math.abs(unrealizedPnL).toFixed(2)}`)],
      ['Total P&L', calculatedPnL >= 0 ? chalk.green(`+$${calculatedPnL.toFixed(2)}`) : chalk.red(`-$${Math.abs(calculatedPnL).toFixed(2)}`)],
      ['ROI', roi >= 0 ? chalk.green(`+${roi.toFixed(2)}%`) : chalk.red(`${roi.toFixed(2)}%`)]
    ]
    console.log(table(pnlData))
    
    // Calculate benchmark comparisons
    const benchmarkStartPrice = candles[0]?.close || 0
    const benchmarkEndPrice = finalPrice
    
    // Buy & Hold calculation
    const btcAmount = initialCapital / benchmarkStartPrice
    const buyHoldValue = btcAmount * benchmarkEndPrice
    const buyHoldPnL = buyHoldValue - initialCapital
    const buyHoldROI = (buyHoldPnL / initialCapital) * 100
    
    // Risk-free rate calculation (time-based)
    const startTime = candles[0]?.timestamp || 0
    const endTime = candles[Math.min(processedCandles - 1, candles.length - 1)]?.timestamp || 0
    const timeHours = (endTime - startTime) / (1000 * 60 * 60)
    const timeDays = timeHours / 24
    const timeYears = timeDays / 365.25
    
    const riskFreeValue = initialCapital * Math.pow(1 + riskFreeRate, timeYears)
    const riskFreePnL = riskFreeValue - initialCapital
    const riskFreeROI = (riskFreePnL / initialCapital) * 100
    
    // Calculate daily returns for Sharpe and Sortino ratios
    const dailyReturns: number[] = []
    let previousValue = initialCapital
    let currentValue = initialCapital
    let btcBalance = startingBtc // Start with initial BTC position
    let cashBalance = halfCapital // Start with initial cash
    
    if (verbose) {
      console.log(chalk.gray(`\nPortfolio Debug:`))
      console.log(chalk.gray(`  Starting BTC: ${btcBalance.toFixed(4)}`))
      console.log(chalk.gray(`  Starting Cash: $${cashBalance.toFixed(2)}`))
      console.log(chalk.gray(`  Total Trades: ${trades.length}`))
    }
    
    // If we have trades, use them to reconstruct portfolio
    if (trades.length > 0) {
      for (const trade of trades) {
        const tradeValue = trade.price * trade.size
        
        if (trade.side === 'buy') {
          btcBalance += trade.size
          cashBalance -= tradeValue
        } else if (trade.side === 'sell') {
          btcBalance -= trade.size
          cashBalance += tradeValue
        }
        
        // Calculate current portfolio value
        currentValue = cashBalance + (btcBalance * trade.price)
        const dailyReturn = (currentValue - previousValue) / previousValue
        if (!isNaN(dailyReturn) && isFinite(dailyReturn)) {
          dailyReturns.push(dailyReturn)
        }
        previousValue = currentValue
      }
    }
    
    // Always calculate returns based on daily price movements for the held position
    // This ensures we have enough data points for Sharpe/Sortino even with few trades
    if (dailyReturns.length < 2 && demoCandles.length > 20) {
      // Calculate returns from candle data
      const startIdx = Math.max(0, Math.min(20, demoCandles.length - 1)) // Start after agents activate
      const endIdx = Math.min(processedCandles, demoCandles.length)
      
      // Reset to calculate from candles
      dailyReturns.length = 0
      btcBalance = startingBtc
      cashBalance = halfCapital
      
      for (let i = startIdx + 1; i < endIdx; i++) {
        const prevPrice = demoCandles[i - 1]!.close
        const currPrice = demoCandles[i]!.close
        
        // Portfolio value = cash + btc * price
        const prevValue = cashBalance + (btcBalance * prevPrice)
        const currValue = cashBalance + (btcBalance * currPrice)
        
        const dailyReturn = (currValue - prevValue) / prevValue
        if (!isNaN(dailyReturn) && isFinite(dailyReturn) && prevValue > 0) {
          dailyReturns.push(dailyReturn)
        }
      }
    }
    
    // Calculate Sharpe ratio
    let sharpeRatio = 0
    let sortinoRatio = 0
    
    if (verbose) {
      console.log(chalk.gray(`  Daily Returns Array Length: ${dailyReturns.length}`))
      if (dailyReturns.length > 0) {
        console.log(chalk.gray(`  Sample Returns: ${dailyReturns.slice(0, 5).map(r => (r * 100).toFixed(2) + '%').join(', ')}...`))
      }
    }
    
    if (dailyReturns.length > 1 && timeYears > 0) {
      // Calculate average daily return
      const avgDailyReturn = dailyReturns.reduce((sum, r) => sum + r, 0) / dailyReturns.length
      const annualizedReturn = avgDailyReturn * 252 // Annualize using trading days
      
      // Calculate standard deviation of returns
      const variance = dailyReturns.reduce((sum, r) => sum + Math.pow(r - avgDailyReturn, 2), 0) / dailyReturns.length
      const stdDev = Math.sqrt(variance)
      const annualizedStdDev = stdDev * Math.sqrt(252) // Annualize using trading days
      
      if (verbose) {
        console.log(chalk.gray(`\nRisk Calculation Debug:`))
        console.log(chalk.gray(`  Daily Returns: ${dailyReturns.length} samples`))
        console.log(chalk.gray(`  Avg Daily Return: ${(avgDailyReturn * 100).toFixed(4)}%`))
        console.log(chalk.gray(`  Daily Std Dev: ${(stdDev * 100).toFixed(4)}%`))
        console.log(chalk.gray(`  Annualized Return: ${(annualizedReturn * 100).toFixed(2)}%`))
        console.log(chalk.gray(`  Annualized Std Dev: ${(annualizedStdDev * 100).toFixed(2)}%`))
      }
      
      // Sharpe ratio = (Return - Risk Free Rate) / Standard Deviation
      sharpeRatio = annualizedStdDev > 0 ? (annualizedReturn - riskFreeRate) / annualizedStdDev : 0
      
      // Calculate downside deviation for Sortino ratio
      const downsideThreshold = riskFreeRate / 252 // Daily risk-free rate (trading days)
      const downsideReturns = dailyReturns.filter(r => r < downsideThreshold)
      
      if (downsideReturns.length > 0) {
        const downsideDeviations = downsideReturns.map(r => Math.pow(r - downsideThreshold, 2))
        const downsideVariance = downsideDeviations.reduce((sum, d) => sum + d, 0) / downsideReturns.length
        const downsideDeviation = Math.sqrt(downsideVariance)
        const annualizedDownsideDeviation = downsideDeviation * Math.sqrt(252) // Use trading days
        
        sortinoRatio = annualizedDownsideDeviation > 0 ? (annualizedReturn - riskFreeRate) / annualizedDownsideDeviation : 0
      }
    }
    
    // Performance comparison
    const outperformanceVsBuyHold = roi - buyHoldROI
    const outperformanceVsRiskFree = roi - riskFreeROI
    
    console.log(chalk.cyan('Benchmark Comparison:'))
    const benchmarkData = [
      ['Benchmark', 'P&L', 'ROI', 'vs Strategy'],
      ['Strategy', calculatedPnL >= 0 ? chalk.green(`+$${calculatedPnL.toFixed(2)}`) : chalk.red(`-$${Math.abs(calculatedPnL).toFixed(2)}`), roi >= 0 ? chalk.green(`+${roi.toFixed(2)}%`) : chalk.red(`${roi.toFixed(2)}%`), '-'],
      ['Buy & Hold', buyHoldPnL >= 0 ? chalk.green(`+$${buyHoldPnL.toFixed(2)}`) : chalk.red(`-$${Math.abs(buyHoldPnL).toFixed(2)}`), `${buyHoldROI >= 0 ? '+' : ''}${buyHoldROI.toFixed(2)}%`, outperformanceVsBuyHold >= 0 ? chalk.green(`+${outperformanceVsBuyHold.toFixed(2)}pp`) : chalk.red(`${outperformanceVsBuyHold.toFixed(2)}pp`)],
      ['Risk-Free', riskFreePnL >= 0 ? chalk.green(`+$${riskFreePnL.toFixed(2)}`) : chalk.red(`-$${Math.abs(riskFreePnL).toFixed(2)}`), `${riskFreeROI >= 0 ? '+' : ''}${riskFreeROI.toFixed(2)}%`, outperformanceVsRiskFree >= 0 ? chalk.green(`+${outperformanceVsRiskFree.toFixed(2)}pp`) : chalk.red(`${outperformanceVsRiskFree.toFixed(2)}pp`)]
    ]
    console.log(table(benchmarkData))
    
    console.log(chalk.cyan('Risk Metrics:'))
    const riskData = [
      ['Metric', 'Value'],
      ['Sharpe Ratio', sharpeRatio.toFixed(2)],
      ['Sortino Ratio', sortinoRatio.toFixed(2)],
      ['Time Period', `${timeDays.toFixed(1)} days (${timeYears.toFixed(3)} years)`],
      ['Risk-Free Rate', `${(riskFreeRate * 100).toFixed(1)}% APY`]
    ]
    console.log(table(riskData))
    
    // Agent Performance Summary
    const lastPrice = demoCandles[demoCandles.length - 1]!.close
    const finalPerformanceStats = agentPerformanceTracker.getPerformanceStats(lastPrice)
    const finalWeights = agentOrchestrator.getAgentWeights()
    
    // Update final weights
    for (const [agentId, stats] of Object.entries(finalPerformanceStats)) {
      stats.currentWeight = finalWeights.get(agentId) || 1.0
    }
    
    if (Object.keys(finalPerformanceStats).length > 0) {
      console.log(chalk.cyan('Agent Performance:'))
      const agentData = [
        ['Agent', 'Usefulness', 'Signals', 'Final Weight'],
        ...Object.entries(finalPerformanceStats).map(([agentId, stats]) => [
          agentId.replace('-agent', '').toUpperCase(),
          `${(stats.avgScore * 100).toFixed(1)}%`,
          stats.recentSignals.toString(),
          stats.currentWeight.toFixed(2)
        ])
      ]
      console.log(table(agentData))
    }
    
    // Generate comprehensive performance analysis report if requested
    if (showAnalysis) {
      const performanceReport = performanceAnalyzer.generateReport(lastPrice)
      console.log(performanceReport)
    }
    
    // Get trading status
    if (tradingActive) {
      console.log(chalk.cyan('\nTrading Status:'))
      console.log(`- Trading Active: Yes`)
      console.log(`- Current Price: $${finalPrice.toFixed(2)}`)
    } else {
      console.log(chalk.yellow('\nTrading Status:'))
      console.log(`- Trading not activated (insufficient history)`)
      console.log(`- Current Price: $${finalPrice.toFixed(2)}`)
    }
    
    // Cleanup
  } catch (error) {
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
      trim: true
    })
    
    parser.on('data', (row: any) => {
      try {
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
          close = row['Adj Close'] ? parseFloat(row['Adj Close']) : parseFloat(row.Close)
          volume = parseFloat(row.Volume || '0')
        } else {
          return // Skip invalid rows
        }
        
        if (!isNaN(timestamp) && !isNaN(open) && !isNaN(close) && !isNaN(high) && !isNaN(low)) {
          candles.push({
            timestamp: toEpochDate(timestamp),
            open,
            high,
            low,
            close,
            volume: isNaN(volume) ? 0 : volume
          })
        }
      } catch (error) {
        // Skip invalid rows
      }
    })
    
    parser.on('error', reject)
    parser.on('end', () => {
      candles.sort((a, b) => a.timestamp - b.timestamp)
      resolve(candles)
    })
    
    createReadStream(absolutePath).pipe(parser)
  })
}

// Run the demo
runDemo().catch(console.error)