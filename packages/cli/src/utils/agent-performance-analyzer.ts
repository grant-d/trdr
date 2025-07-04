import type { ConsensusResult } from '@trdr/core/dist/agents/types'
import chalk from 'chalk'
import { table } from 'table'

export interface AgentSignalRecord {
  agentId: string
  timestamp: number
  action: 'buy' | 'sell' | 'hold'
  confidence: number
  entryPrice: number
  consensus: ConsensusResult
  lookAheadPeriod: number
}

export interface AgentPerformanceMetrics {
  agentId: string
  totalSignals: number
  buySignals: number
  sellSignals: number
  holdSignals: number
  avgConfidence: number
  winRate: number
  avgReturn: number
  sharpeRatio: number
  consistency: number
  signalBalance: number // 0 = perfect balance, 1 = all one direction
  confidenceAccuracy: number // How well confidence correlates with outcomes
  marketConditionPerformance: {
    trending: { winRate: number; signalCount: number }
    ranging: { winRate: number; signalCount: number }
    choppy: { winRate: number; signalCount: number }
  }
  weaknesses: string[]
  strengths: string[]
  recommendations: string[]
}

export interface MarketCondition {
  type: 'trending' | 'ranging' | 'choppy'
  volatility: number
  trendStrength: number
  timestamp: number
}

export class AgentPerformanceAnalyzer {
  private readonly signalHistory = new Map<string, AgentSignalRecord[]>()
  private readonly marketConditions: MarketCondition[] = []
  private readonly lookbackWindow = 100 // Analyze last 100 signals per agent
  
  recordSignal(
    consensus: ConsensusResult,
    entryPrice: number,
    timestamp: number,
    lookAheadPeriod = 5
  ): void {
    for (const [agentId, signal] of Object.entries(consensus.agentSignals)) {
      if (!this.signalHistory.has(agentId)) {
        this.signalHistory.set(agentId, [])
      }
      
      const record: AgentSignalRecord = {
        agentId,
        timestamp,
        action: signal.action,
        confidence: signal.confidence,
        entryPrice,
        consensus,
        lookAheadPeriod
      }
      
      const history = this.signalHistory.get(agentId)!
      history.push(record)
      
      // Keep only recent history to prevent memory growth
      if (history.length > this.lookbackWindow * 2) {
        history.splice(0, history.length - this.lookbackWindow * 2)
      }
    }
  }
  
  recordMarketCondition(condition: MarketCondition): void {
    this.marketConditions.push(condition)
    
    // Keep only recent conditions
    if (this.marketConditions.length > 1000) {
      this.marketConditions.splice(0, this.marketConditions.length - 1000)
    }
  }
  
  evaluateSignal(
    record: AgentSignalRecord,
    exitPrice: number
  ): { isWin: boolean; return: number } {
    const priceChange = (exitPrice - record.entryPrice) / record.entryPrice
    
    if (record.action === 'hold') {
      // Hold is successful if price doesn't move much
      const isWin = Math.abs(priceChange) < 0.01 // Less than 1% movement
      return { isWin, return: 0 }
    }
    
    const expectedDirection = record.action === 'buy' ? 1 : -1
    const actualDirection = Math.sign(priceChange)
    
    // Consider it a win if:
    // 1. Direction is correct and movement is meaningful (> 0.2%)
    // 2. For small movements, at least didn't lose money
    const isWin = actualDirection === expectedDirection && Math.abs(priceChange) > 0.002
    const return_ = record.action === 'buy' ? priceChange : -priceChange
    
    return { isWin, return: return_ }
  }
  
  analyzeAgent(agentId: string, currentPrice: number): AgentPerformanceMetrics | null {
    const history = this.signalHistory.get(agentId)
    if (!history || history.length < 10) {
      return null // Not enough data
    }
    
    // Get evaluable signals (old enough to assess outcome)
    const evaluableSignals = history
      .filter(s => s.lookAheadPeriod <= 0)
      .slice(-this.lookbackWindow)
    
    if (evaluableSignals.length === 0) {
      return null
    }
    
    // Calculate basic metrics
    const totalSignals = evaluableSignals.length
    const buySignals = evaluableSignals.filter(s => s.action === 'buy').length
    const sellSignals = evaluableSignals.filter(s => s.action === 'sell').length
    const holdSignals = evaluableSignals.filter(s => s.action === 'hold').length
    
    const avgConfidence = evaluableSignals.reduce((sum, s) => sum + s.confidence, 0) / totalSignals
    
    // Evaluate performance
    const outcomes = evaluableSignals.map(signal => this.evaluateSignal(signal, currentPrice))
    const wins = outcomes.filter(o => o.isWin).length
    const winRate = wins / totalSignals
    
    const returns = outcomes.map(o => o.return)
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length
    
    // Calculate Sharpe ratio (simplified)
    const returnStdDev = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    )
    const sharpeRatio = returnStdDev > 0 ? (avgReturn * 252) / (returnStdDev * Math.sqrt(252)) : 0
    
    // Calculate consistency (lower variance in confidence when winning)
    const winningSignals = evaluableSignals.filter((_, i) => outcomes[i]?.isWin ?? false)
    const losingSignals = evaluableSignals.filter((_, i) => !(outcomes[i]?.isWin ?? false))
    
    const winConfidenceVar = this.calculateVariance(winningSignals.map(s => s.confidence))
    const loseConfidenceVar = this.calculateVariance(losingSignals.map(s => s.confidence))
    const consistency = 1 - (winConfidenceVar + loseConfidenceVar) / 2
    
    // Calculate signal balance (0 = perfect balance, 1 = all one direction)
    const signalBalance = Math.abs(buySignals - sellSignals) / Math.max(buySignals + sellSignals, 1)
    
    // Calculate confidence accuracy (correlation between confidence and success)
    const confidenceAccuracy = this.calculateConfidenceAccuracy(evaluableSignals, outcomes)
    
    // Analyze by market condition
    const marketConditionPerformance = this.analyzeByMarketCondition(evaluableSignals, outcomes)
    
    // Identify weaknesses and strengths
    const analysis = this.identifyStrengthsAndWeaknesses({
      winRate,
      avgReturn,
      sharpeRatio,
      consistency,
      signalBalance,
      confidenceAccuracy,
      buySignals,
      sellSignals,
      holdSignals,
      totalSignals,
      marketConditionPerformance
    })
    
    return {
      agentId,
      totalSignals,
      buySignals,
      sellSignals,
      holdSignals,
      avgConfidence,
      winRate,
      avgReturn,
      sharpeRatio,
      consistency,
      signalBalance,
      confidenceAccuracy,
      marketConditionPerformance,
      weaknesses: analysis.weaknesses,
      strengths: analysis.strengths,
      recommendations: analysis.recommendations
    }
  }
  
  private calculateVariance(values: number[]): number {
    if (values.length === 0) return 0
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length
    return values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length
  }
  
  private calculateConfidenceAccuracy(
    signals: AgentSignalRecord[],
    outcomes: { isWin: boolean; return: number }[]
  ): number {
    // Group by confidence buckets and calculate win rate per bucket
    const buckets = new Map<number, { wins: number; total: number }>()
    
    signals.forEach((signal, i) => {
      const bucket = Math.floor(signal.confidence * 10) / 10 // Round to nearest 0.1
      if (!buckets.has(bucket)) {
        buckets.set(bucket, { wins: 0, total: 0 })
      }
      
      const bucketData = buckets.get(bucket)!
      bucketData.total++
      if (outcomes[i]?.isWin) {
        bucketData.wins++
      }
    })
    
    // Calculate correlation between confidence and win rate
    let correlation = 0
    let totalWeight = 0
    
    buckets.forEach((data, confidence) => {
      const winRate = data.total > 0 ? data.wins / data.total : 0
      const weight = data.total
      
      // Perfect accuracy would mean confidence matches win rate
      const accuracy = 1 - Math.abs(confidence - winRate)
      correlation += accuracy * weight
      totalWeight += weight
    })
    
    return totalWeight > 0 ? correlation / totalWeight : 0
  }
  
  private analyzeByMarketCondition(
    signals: AgentSignalRecord[],
    outcomes: { isWin: boolean; return: number }[]
  ): AgentPerformanceMetrics['marketConditionPerformance'] {
    const performance = {
      trending: { winRate: 0, signalCount: 0 },
      ranging: { winRate: 0, signalCount: 0 },
      choppy: { winRate: 0, signalCount: 0 }
    }
    
    // Match signals to market conditions
    signals.forEach((signal, i) => {
      const condition = this.getMarketConditionAt(signal.timestamp)
      if (!condition) return
      
      const bucket = performance[condition.type]
      bucket.signalCount++
      if (outcomes[i]?.isWin) {
        bucket.winRate++
      }
    })
    
    // Convert counts to rates
    Object.values(performance).forEach(bucket => {
      if (bucket.signalCount > 0) {
        bucket.winRate = bucket.winRate / bucket.signalCount
      }
    })
    
    return performance
  }
  
  private getMarketConditionAt(timestamp: number): MarketCondition | null {
    // Find the market condition closest to the timestamp
    let closest: MarketCondition | null = null
    let minDiff = Infinity
    
    for (const condition of this.marketConditions) {
      const diff = Math.abs(condition.timestamp - timestamp)
      if (diff < minDiff) {
        minDiff = diff
        closest = condition
      }
    }
    
    return closest
  }
  
  private identifyStrengthsAndWeaknesses(metrics: {
    winRate: number
    avgReturn: number
    sharpeRatio: number
    consistency: number
    signalBalance: number
    confidenceAccuracy: number
    buySignals: number
    sellSignals: number
    holdSignals: number
    totalSignals: number
    marketConditionPerformance: AgentPerformanceMetrics['marketConditionPerformance']
  }): { weaknesses: string[]; strengths: string[]; recommendations: string[] } {
    const weaknesses: string[] = []
    const strengths: string[] = []
    const recommendations: string[] = []
    
    // Win rate analysis
    if (metrics.winRate < 0.45) {
      weaknesses.push('Low win rate (< 45%)')
      recommendations.push('Review signal generation logic and thresholds')
    } else if (metrics.winRate > 0.55) {
      strengths.push(`Strong win rate (${(metrics.winRate * 100).toFixed(1)}%)`)
    }
    
    // Return analysis
    if (metrics.avgReturn < 0) {
      weaknesses.push('Negative average returns')
      recommendations.push('Consider inverting signals or adjusting entry/exit timing')
    } else if (metrics.avgReturn > 0.005) {
      strengths.push(`Positive returns (${(metrics.avgReturn * 100).toFixed(2)}% avg)`)
    }
    
    // Signal balance analysis
    if (metrics.signalBalance > 0.7) {
      weaknesses.push('Heavily biased signals (mostly one direction)')
      recommendations.push('Adjust parameters to generate more balanced buy/sell signals')
    } else if (metrics.signalBalance < 0.3) {
      strengths.push('Well-balanced buy/sell signals')
    }
    
    // Confidence accuracy
    if (metrics.confidenceAccuracy < 0.5) {
      weaknesses.push('Poor confidence calibration')
      recommendations.push('Recalibrate confidence scoring to match actual performance')
    } else if (metrics.confidenceAccuracy > 0.7) {
      strengths.push('Accurate confidence predictions')
    }
    
    // Market condition analysis
    const conditionPerf = metrics.marketConditionPerformance
    if (conditionPerf.trending.winRate < 0.4 && conditionPerf.trending.signalCount > 5) {
      weaknesses.push('Poor performance in trending markets')
      recommendations.push('Add trend-following components or filters')
    }
    if (conditionPerf.ranging.winRate < 0.4 && conditionPerf.ranging.signalCount > 5) {
      weaknesses.push('Poor performance in ranging markets')
      recommendations.push('Implement mean-reversion strategies for sideways markets')
    }
    if (conditionPerf.choppy.winRate < 0.4 && conditionPerf.choppy.signalCount > 5) {
      weaknesses.push('Poor performance in choppy markets')
      recommendations.push('Add volatility filters or reduce position sizing in high volatility')
    }
    
    // Consistency analysis
    if (metrics.consistency < 0.3) {
      weaknesses.push('Inconsistent performance')
      recommendations.push('Implement more stable signal generation methods')
    } else if (metrics.consistency > 0.7) {
      strengths.push('Highly consistent performance')
    }
    
    // Sharpe ratio analysis
    if (metrics.sharpeRatio < 0) {
      weaknesses.push('Negative risk-adjusted returns')
    } else if (metrics.sharpeRatio > 1) {
      strengths.push(`Excellent risk-adjusted returns (Sharpe: ${metrics.sharpeRatio.toFixed(2)})`)
    }
    
    return { weaknesses, strengths, recommendations }
  }
  
  generateReport(currentPrice: number): string {
    const allMetrics: AgentPerformanceMetrics[] = []
    
    for (const agentId of this.signalHistory.keys()) {
      const metrics = this.analyzeAgent(agentId, currentPrice)
      if (metrics) {
        allMetrics.push(metrics)
      }
    }
    
    if (allMetrics.length === 0) {
      return 'No agent performance data available yet.'
    }
    
    // Sort by win rate descending
    allMetrics.sort((a, b) => b.winRate - a.winRate)
    
    let report = chalk.cyan('\n=== Agent Performance Analysis Report ===\n\n')
    
    // Summary table
    const summaryData = [
      ['Agent', 'Signals', 'Win Rate', 'Avg Return', 'Sharpe', 'Balance', 'Confidence Acc'],
      ...allMetrics.map(m => [
        m.agentId.replace('-agent', '').toUpperCase(),
        m.totalSignals.toString(),
        `${(m.winRate * 100).toFixed(1)}%`,
        `${(m.avgReturn * 100).toFixed(3)}%`,
        m.sharpeRatio.toFixed(2),
        `${(m.signalBalance * 100).toFixed(0)}%`,
        `${(m.confidenceAccuracy * 100).toFixed(0)}%`
      ])
    ]
    
    report += chalk.yellow('Performance Summary:\n')
    report += table(summaryData)
    
    // Detailed analysis for each agent
    report += chalk.yellow('\nDetailed Agent Analysis:\n')
    
    for (const metrics of allMetrics) {
      report += chalk.cyan(`\n${metrics.agentId.toUpperCase()}:\n`)
      
      // Signal distribution
      report += chalk.gray(`  Signal Distribution: Buy ${metrics.buySignals} | Sell ${metrics.sellSignals} | Hold ${metrics.holdSignals}\n`)
      report += chalk.gray(`  Average Confidence: ${(metrics.avgConfidence * 100).toFixed(1)}%\n`)
      report += chalk.gray(`  Consistency Score: ${(metrics.consistency * 100).toFixed(1)}%\n`)
      
      // Market condition performance
      report += chalk.gray('  Market Performance:\n')
      const mcp = metrics.marketConditionPerformance
      if (mcp.trending.signalCount > 0) {
        report += chalk.gray(`    Trending: ${(mcp.trending.winRate * 100).toFixed(1)}% (${mcp.trending.signalCount} signals)\n`)
      }
      if (mcp.ranging.signalCount > 0) {
        report += chalk.gray(`    Ranging: ${(mcp.ranging.winRate * 100).toFixed(1)}% (${mcp.ranging.signalCount} signals)\n`)
      }
      if (mcp.choppy.signalCount > 0) {
        report += chalk.gray(`    Choppy: ${(mcp.choppy.winRate * 100).toFixed(1)}% (${mcp.choppy.signalCount} signals)\n`)
      }
      
      // Strengths
      if (metrics.strengths.length > 0) {
        report += chalk.green('  Strengths:\n')
        metrics.strengths.forEach(s => report += chalk.green(`    ✓ ${s}\n`))
      }
      
      // Weaknesses
      if (metrics.weaknesses.length > 0) {
        report += chalk.red('  Weaknesses:\n')
        metrics.weaknesses.forEach(w => report += chalk.red(`    ✗ ${w}\n`))
      }
      
      // Recommendations
      if (metrics.recommendations.length > 0) {
        report += chalk.yellow('  Recommendations:\n')
        metrics.recommendations.forEach(r => report += chalk.yellow(`    → ${r}\n`))
      }
    }
    
    // Overall recommendations
    report += chalk.cyan('\n=== Overall Recommendations ===\n')
    
    const weakAgents = allMetrics.filter(m => m.winRate < 0.45)
    if (weakAgents.length > 0) {
      report += chalk.red(`\nWeak Performers (consider removing or retuning):\n`)
      weakAgents.forEach(agent => {
        report += chalk.red(`  - ${agent.agentId}: ${(agent.winRate * 100).toFixed(1)}% win rate\n`)
      })
    }
    
    const strongAgents = allMetrics.filter(m => m.winRate > 0.55 && m.avgReturn > 0)
    if (strongAgents.length > 0) {
      report += chalk.green(`\nStrong Performers (consider increasing weight):\n`)
      strongAgents.forEach(agent => {
        report += chalk.green(`  - ${agent.agentId}: ${(agent.winRate * 100).toFixed(1)}% win rate, ${(agent.avgReturn * 100).toFixed(3)}% avg return\n`)
      })
    }
    
    // Portfolio balance recommendation
    const avgBalance = allMetrics.reduce((sum, m) => sum + m.signalBalance, 0) / allMetrics.length
    if (avgBalance > 0.6) {
      report += chalk.yellow('\nPortfolio is biased toward one direction. Consider adding agents with opposing strategies.\n')
    }
    
    return report
  }
  
  // Age signals by decrementing lookAheadPeriod
  ageSignals(): void {
    for (const signals of this.signalHistory.values()) {
      signals.forEach(signal => signal.lookAheadPeriod--)
    }
  }
}