import type { PerformanceHistory, PerformanceSummary, WeightAdjustment } from './performance-tracker'
import type { EpochDate} from '@trdr/shared'
import { epochDateNow } from '@trdr/shared'

/**
 * Performance chart data point
 */
export interface PerformanceDataPoint {
  timestamp: EpochDate
  value: number
  label?: string
}

/**
 * Performance chart series
 */
export interface PerformanceSeries {
  name: string
  data: PerformanceDataPoint[]
  color?: string
  type?: 'line' | 'bar' | 'area'
}

/**
 * Agent performance dashboard data
 */
export interface AgentDashboard {
  agentId: string
  summary: PerformanceSummary
  charts: {
    returns: PerformanceSeries[]
    winRate: PerformanceSeries[]
    sharpeRatio: PerformanceSeries[]
    weights: PerformanceSeries[]
  }
  metrics: {
    label: string
    value: string | number
    change?: number
    color?: string
  }[]
}

/**
 * Comparative performance data
 */
export interface ComparativePerformance {
  agents: string[]
  series: {
    returns: PerformanceSeries[]
    winRates: PerformanceSeries[]
    sharpeRatios: PerformanceSeries[]
  }
  rankings: {
    metric: string
    rankings: Array<{
      agentId: string
      value: number
      rank: number
    }>
  }[]
}

/**
 * Visualizes agent performance data
 */
export class PerformanceVisualizer {
  /**
   * Generate dashboard data for an agent
   */
  generateAgentDashboard(
    agentId: string,
    history: PerformanceHistory,
    weightHistory: WeightAdjustment[]
  ): AgentDashboard {
    const summary = history.summary
    
    // Generate time series data
    const returnsSeries = this.generateReturnsSeries(history)
    const winRateSeries = this.generateWinRateSeries(history)
    const sharpeRatioSeries = this.generateSharpeRatioSeries(history)
    const weightsSeries = this.generateWeightsSeries(weightHistory)
    
    // Generate metrics
    const metrics = this.generateMetrics(summary)
    
    return {
      agentId,
      summary,
      charts: {
        returns: returnsSeries,
        winRate: winRateSeries,
        sharpeRatio: sharpeRatioSeries,
        weights: weightsSeries
      },
      metrics
    }
  }
  
  /**
   * Generate comparative performance data
   */
  generateComparativePerformance(
    performanceData: Map<string, PerformanceHistory>
  ): ComparativePerformance {
    const agents = Array.from(performanceData.keys())
    
    // Generate comparative series
    const returnsSeries: PerformanceSeries[] = []
    const winRateSeries: PerformanceSeries[] = []
    const sharpeRatioSeries: PerformanceSeries[] = []
    
    for (const [agentId, history] of performanceData) {
      returnsSeries.push({
        name: agentId,
        data: this.generateCumulativeReturns(history),
        type: 'line'
      })
      
      winRateSeries.push({
        name: agentId,
        data: this.generateRollingWinRate(history),
        type: 'line'
      })
      
      sharpeRatioSeries.push({
        name: agentId,
        data: this.generateRollingSharpe(history),
        type: 'line'
      })
    }
    
    // Generate rankings
    const rankings = this.generateRankings(performanceData)
    
    return {
      agents,
      series: {
        returns: returnsSeries,
        winRates: winRateSeries,
        sharpeRatios: sharpeRatioSeries
      },
      rankings
    }
  }
  
  /**
   * Generate returns time series
   */
  private generateReturnsSeries(history: PerformanceHistory): PerformanceSeries[] {
    const cumulativeReturns: PerformanceDataPoint[] = []
    const dailyReturns: PerformanceDataPoint[] = []
    
    let cumReturn = 0
    for (const metric of history.metrics) {
      if (metric.outcome) {
        cumReturn += metric.outcome.pnlPercent
        cumulativeReturns.push({
          timestamp: metric.timestamp,
          value: cumReturn * 100, // Convert to percentage
          label: `${(cumReturn * 100).toFixed(2)}%`
        })
        
        dailyReturns.push({
          timestamp: metric.timestamp,
          value: metric.outcome.pnlPercent * 100,
          label: `${(metric.outcome.pnlPercent * 100).toFixed(2)}%`
        })
      }
    }
    
    return [
      {
        name: 'Cumulative Returns',
        data: cumulativeReturns,
        type: 'area',
        color: '#10b981'
      },
      {
        name: 'Daily Returns',
        data: dailyReturns,
        type: 'bar',
        color: '#3b82f6'
      }
    ]
  }
  
  /**
   * Generate win rate time series
   */
  private generateWinRateSeries(history: PerformanceHistory): PerformanceSeries[] {
    const winRateData: PerformanceDataPoint[] = []
    const rollingWinRate: PerformanceDataPoint[] = []
    
    let wins = 0
    let total = 0
    const windowSize = 20 // 20-trade rolling window
    
    for (let i = 0; i < history.metrics.length; i++) {
      const metric = history.metrics[i]
      if (metric?.outcome) {
        total++
        if (metric.outcome.success) wins++
        
        const winRate = total > 0 ? (wins / total) * 100 : 0
        winRateData.push({
          timestamp: metric.timestamp,
          value: winRate,
          label: `${winRate.toFixed(1)}%`
        })
        
        // Calculate rolling win rate
        if (i >= windowSize) {
          const windowMetrics = history.metrics.slice(i - windowSize + 1, i + 1)
          const windowWins = windowMetrics.filter(m => m.outcome?.success).length
          const rollingRate = (windowWins / windowSize) * 100
          
          rollingWinRate.push({
            timestamp: metric.timestamp,
            value: rollingRate,
            label: `${rollingRate.toFixed(1)}%`
          })
        }
      }
    }
    
    return [
      {
        name: 'Overall Win Rate',
        data: winRateData,
        type: 'line',
        color: '#8b5cf6'
      },
      {
        name: '20-Trade Rolling Win Rate',
        data: rollingWinRate,
        type: 'line',
        color: '#f59e0b'
      }
    ]
  }
  
  /**
   * Generate Sharpe ratio time series
   */
  private generateSharpeRatioSeries(history: PerformanceHistory): PerformanceSeries[] {
    const sharpeData: PerformanceDataPoint[] = []
    const windowSize = 30 // 30-trade window for Sharpe calculation
    
    for (let i = windowSize; i < history.metrics.length; i++) {
      const windowMetrics = history.metrics.slice(i - windowSize + 1, i + 1)
      const returns = windowMetrics
        .filter(m => m.outcome)
        .map(m => m.outcome!.pnlPercent)
      
      if (returns.length >= windowSize / 2) { // At least half the window has data
        const sharpe = this.calculateSharpe(returns)
        
        const metric = history.metrics[i]
        if (metric) {
          sharpeData.push({
            timestamp: metric.timestamp,
            value: sharpe,
            label: sharpe.toFixed(2)
          })
        }
      }
    }
    
    return [
      {
        name: '30-Trade Rolling Sharpe Ratio',
        data: sharpeData,
        type: 'line',
        color: '#ef4444'
      }
    ]
  }
  
  /**
   * Generate weights time series
   */
  private generateWeightsSeries(weightHistory: WeightAdjustment[]): PerformanceSeries[] {
    const weightsData: PerformanceDataPoint[] = []
    
    for (const adjustment of weightHistory) {
      weightsData.push({
        timestamp: epochDateNow(), // Would need actual timestamp
        value: adjustment.newWeight,
        label: `${adjustment.newWeight.toFixed(2)} (${adjustment.reason})`
      })
    }
    
    return [
      {
        name: 'Agent Weight',
        data: weightsData,
        type: 'line',
        color: '#06b6d4'
      }
    ]
  }
  
  /**
   * Generate cumulative returns for comparison
   */
  private generateCumulativeReturns(history: PerformanceHistory): PerformanceDataPoint[] {
    const data: PerformanceDataPoint[] = []
    let cumReturn = 0
    
    for (const metric of history.metrics) {
      if (metric.outcome) {
        cumReturn += metric.outcome.pnlPercent
        data.push({
          timestamp: metric.timestamp,
          value: cumReturn * 100
        })
      }
    }
    
    return data
  }
  
  /**
   * Generate rolling win rate for comparison
   */
  private generateRollingWinRate(history: PerformanceHistory): PerformanceDataPoint[] {
    const data: PerformanceDataPoint[] = []
    const windowSize = 20
    
    for (let i = windowSize; i < history.metrics.length; i++) {
      const windowMetrics = history.metrics.slice(i - windowSize + 1, i + 1)
      const wins = windowMetrics.filter(m => m.outcome?.success).length
      const total = windowMetrics.filter(m => m.outcome).length
      
      if (total > 0) {
        const metric = history.metrics[i]
        if (metric) {
          data.push({
            timestamp: metric.timestamp,
            value: (wins / total) * 100
          })
        }
      }
    }
    
    return data
  }
  
  /**
   * Generate rolling Sharpe for comparison
   */
  private generateRollingSharpe(history: PerformanceHistory): PerformanceDataPoint[] {
    const data: PerformanceDataPoint[] = []
    const windowSize = 30
    
    for (let i = windowSize; i < history.metrics.length; i++) {
      const windowMetrics = history.metrics.slice(i - windowSize + 1, i + 1)
      const returns = windowMetrics
        .filter(m => m.outcome)
        .map(m => m.outcome!.pnlPercent)
      
      if (returns.length >= windowSize / 2) {
        const sharpe = this.calculateSharpe(returns)
        const metric = history.metrics[i]
        if (metric) {
          data.push({
            timestamp: metric.timestamp,
            value: sharpe
          })
        }
      }
    }
    
    return data
  }
  
  /**
   * Generate performance metrics display
   */
  private generateMetrics(summary: PerformanceSummary): Array<{
    label: string
    value: string | number
    change?: number
    color?: string
  }> {
    return [
      {
        label: 'Win Rate',
        value: `${(summary.winRate * 100).toFixed(1)}%`,
        change: summary.recentPerformance.last24h.winRate - summary.winRate,
        color: summary.winRate >= 0.5 ? '#10b981' : '#ef4444'
      },
      {
        label: 'Average Return',
        value: `${(summary.averageReturn * 100).toFixed(2)}%`,
        color: summary.averageReturn >= 0 ? '#10b981' : '#ef4444'
      },
      {
        label: 'Sharpe Ratio',
        value: summary.sharpeRatio.toFixed(2),
        color: summary.sharpeRatio >= 1 ? '#10b981' : summary.sharpeRatio >= 0 ? '#f59e0b' : '#ef4444'
      },
      {
        label: 'Max Drawdown',
        value: `${(summary.maxDrawdown * 100).toFixed(1)}%`,
        color: summary.maxDrawdown <= 0.1 ? '#10b981' : summary.maxDrawdown <= 0.2 ? '#f59e0b' : '#ef4444'
      },
      {
        label: 'Profit Factor',
        value: summary.profitFactor === Infinity ? '∞' : summary.profitFactor.toFixed(2),
        color: summary.profitFactor >= 1.5 ? '#10b981' : summary.profitFactor >= 1 ? '#f59e0b' : '#ef4444'
      },
      {
        label: 'Consistency',
        value: `${(summary.consistency * 100).toFixed(0)}%`,
        color: summary.consistency >= 0.7 ? '#10b981' : summary.consistency >= 0.5 ? '#f59e0b' : '#ef4444'
      },
      {
        label: 'Total Signals',
        value: summary.totalSignals
      },
      {
        label: 'Avg Hold Time',
        value: `${(summary.avgHoldTime / 1000 / 60).toFixed(1)} min`
      },
      {
        label: '24h Performance',
        value: `${(summary.recentPerformance.last24h.returns * 100).toFixed(2)}%`,
        color: summary.recentPerformance.last24h.returns >= 0 ? '#10b981' : '#ef4444'
      }
    ]
  }
  
  /**
   * Generate agent rankings
   */
  private generateRankings(
    performanceData: Map<string, PerformanceHistory>
  ): Array<{
    metric: string
    rankings: Array<{
      agentId: string
      value: number
      rank: number
    }>
  }> {
    const rankings = []
    
    // Win rate ranking
    const winRateRanking = Array.from(performanceData.entries())
      .map(([agentId, history]) => ({
        agentId,
        value: history.summary.winRate * 100
      }))
      .sort((a, b) => b.value - a.value)
      .map((item, index) => ({ ...item, rank: index + 1 }))
    
    rankings.push({
      metric: 'Win Rate',
      rankings: winRateRanking
    })
    
    // Sharpe ratio ranking
    const sharpeRanking = Array.from(performanceData.entries())
      .map(([agentId, history]) => ({
        agentId,
        value: history.summary.sharpeRatio
      }))
      .sort((a, b) => b.value - a.value)
      .map((item, index) => ({ ...item, rank: index + 1 }))
    
    rankings.push({
      metric: 'Sharpe Ratio',
      rankings: sharpeRanking
    })
    
    // Total returns ranking
    const returnsRanking = Array.from(performanceData.entries())
      .map(([agentId, history]) => ({
        agentId,
        value: history.summary.averageReturn * history.summary.totalSignals * 100
      }))
      .sort((a, b) => b.value - a.value)
      .map((item, index) => ({ ...item, rank: index + 1 }))
    
    rankings.push({
      metric: 'Total Returns',
      rankings: returnsRanking
    })
    
    return rankings
  }
  
  /**
   * Calculate Sharpe ratio for returns array
   */
  private calculateSharpe(returns: number[]): number {
    if (returns.length < 2) return 0
    
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length
    const stdDev = Math.sqrt(variance)
    
    return stdDev > 0 ? mean / stdDev * Math.sqrt(252) : 0
  }
  
  /**
   * Export dashboard data as JSON
   */
  exportDashboardData(dashboard: AgentDashboard): string {
    return JSON.stringify(dashboard, null, 2)
  }
  
  /**
   * Export comparative data as JSON
   */
  exportComparativeData(comparative: ComparativePerformance): string {
    return JSON.stringify(comparative, null, 2)
  }
}