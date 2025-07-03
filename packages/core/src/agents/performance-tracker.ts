import type { AgentSignal, Trade } from './types'
import { epochDateNow, type EpochDate } from '@trdr/shared'
import type { Logger } from '@trdr/types'
import type { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

/**
 * Performance metric with timestamp
 */
export interface PerformanceMetric {
  timestamp: EpochDate
  signal: AgentSignal
  outcome?: TradeOutcome
  executionTime: number
  marketConditions?: MarketSnapshot
}

/**
 * Trade outcome for a signal
 */
export interface TradeOutcome {
  trade: Trade
  pnl: number
  pnlPercent: number
  holdTime: number
  maxDrawdown: number
  success: boolean
}

/**
 * Market snapshot at signal time
 */
export interface MarketSnapshot {
  price: number
  volume: number
  volatility: number
  trend: 'up' | 'down' | 'sideways'
  momentum: number
}

/**
 * Historical performance data
 */
export interface PerformanceHistory {
  agentId: string
  metrics: PerformanceMetric[]
  summary: PerformanceSummary
  lastUpdated: EpochDate
}

/**
 * Performance summary statistics
 */
export interface PerformanceSummary {
  totalSignals: number
  successfulSignals: number
  winRate: number
  averageReturn: number
  sharpeRatio: number
  maxDrawdown: number
  profitFactor: number
  avgHoldTime: number
  consistency: number
  recentPerformance: RecentPerformance
}

/**
 * Recent performance metrics
 */
export interface RecentPerformance {
  last24h: { winRate: number; returns: number; signals: number }
  last7d: { winRate: number; returns: number; signals: number }
  last30d: { winRate: number; returns: number; signals: number }
}

/**
 * Weight adjustment result
 */
export interface WeightAdjustment {
  agentId: string
  oldWeight: number
  newWeight: number
  reason: string
  performanceFactors: {
    winRate: number
    sharpeRatio: number
    consistency: number
    recentTrend: number
  }
}

/**
 * Configuration for performance tracking
 */
export interface PerformanceTrackerConfig {
  /** Maximum metrics to store per agent */
  maxMetricsPerAgent: number
  
  /** Time decay factor for old metrics (0-1) */
  timeDecayFactor: number
  
  /** Minimum signals required for weight adjustment */
  minSignalsForAdjustment: number
  
  /** Weight adjustment sensitivity (0-1) */
  adjustmentSensitivity: number
  
  /** Maximum weight multiplier */
  maxWeightMultiplier: number
  
  /** Minimum weight multiplier */
  minWeightMultiplier: number
  
  /** Performance evaluation window (ms) */
  evaluationWindow: number
  
  /** Enable automatic weight updates */
  enableAutoWeightUpdate: boolean
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: PerformanceTrackerConfig = {
  maxMetricsPerAgent: 1000,
  timeDecayFactor: 0.95,
  minSignalsForAdjustment: 20,
  adjustmentSensitivity: 0.2,
  maxWeightMultiplier: 3.0,
  minWeightMultiplier: 0.1,
  evaluationWindow: 7 * 24 * 60 * 60 * 1000, // 7 days
  enableAutoWeightUpdate: true
}

/**
 * Tracks agent performance and calculates weight adjustments
 */
export class PerformanceTracker {
  private readonly performanceData = new Map<string, PerformanceHistory>()
  private readonly config: PerformanceTrackerConfig
  private readonly signalToTradeMap = new Map<string, string>() // signalId -> tradeId
  
  constructor(
    private readonly eventBus: EventBus,
    config?: Partial<PerformanceTrackerConfig>,
    private readonly logger?: Logger
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config }
    this.subscribeToEvents()
  }
  
  /**
   * Record a new signal from an agent
   */
  recordSignal(
    agentId: string,
    signal: AgentSignal,
    executionTime: number,
    marketSnapshot?: MarketSnapshot
  ): void {
    const history = this.getOrCreateHistory(agentId)
    
    const metric: PerformanceMetric = {
      timestamp: signal.timestamp,
      signal,
      executionTime,
      marketConditions: marketSnapshot
    }
    
    // Add metric
    history.metrics.push(metric)
    
    // Maintain max metrics limit
    if (history.metrics.length > this.config.maxMetricsPerAgent) {
      history.metrics.shift()
    }
    
    // Update summary
    this.updateSummary(history)
    
    // Generate unique signal ID for tracking
    const signalId = `${agentId}-${signal.timestamp}`
    this.signalToTradeMap.set(signalId, '')
    
    this.logger?.debug(`Recorded signal for agent ${agentId}`, {
      action: signal.action,
      confidence: signal.confidence
    })
  }
  
  /**
   * Record trade outcome for a signal
   */
  recordTradeOutcome(
    agentId: string,
    signalTimestamp: EpochDate,
    outcome: TradeOutcome
  ): void {
    const history = this.performanceData.get(agentId)
    if (!history) {
      this.logger?.warn(`No history found for agent ${agentId}`)
      return
    }
    
    // Find the corresponding metric
    const metric = history.metrics.find(m => m.signal.timestamp === signalTimestamp)
    if (!metric) {
      this.logger?.warn(`No signal found for timestamp ${signalTimestamp}`)
      return
    }
    
    // Update metric with outcome
    metric.outcome = outcome
    
    // Update summary
    this.updateSummary(history)
    
    this.logger?.info(`Recorded trade outcome for agent ${agentId}`, {
      pnl: outcome.pnl,
      success: outcome.success
    })
    
    // Emit performance update event
    this.eventBus.emit(EventTypes.AGENT_PERFORMANCE_UPDATED, {
      timestamp: epochDateNow(),
      agentId,
      performance: history.summary
    })
  }
  
  /**
   * Get performance history for an agent
   */
  getPerformanceHistory(agentId: string): PerformanceHistory | undefined {
    return this.performanceData.get(agentId)
  }
  
  /**
   * Get performance summary for an agent
   */
  getPerformanceSummary(agentId: string): PerformanceSummary | undefined {
    return this.performanceData.get(agentId)?.summary
  }
  
  /**
   * Calculate weight adjustments for all agents
   */
  calculateWeightAdjustments(
    currentWeights: Map<string, number>
  ): Map<string, WeightAdjustment> {
    const adjustments = new Map<string, WeightAdjustment>()
    
    for (const [agentId, currentWeight] of currentWeights) {
      const history = this.performanceData.get(agentId)
      if (!history || history.metrics.length < this.config.minSignalsForAdjustment) {
        continue
      }
      
      const adjustment = this.calculateAgentWeightAdjustment(
        agentId,
        currentWeight,
        history.summary
      )
      
      if (adjustment.newWeight !== currentWeight) {
        adjustments.set(agentId, adjustment)
      }
    }
    
    return adjustments
  }
  
  /**
   * Calculate weight adjustment for a single agent
   */
  private calculateAgentWeightAdjustment(
    agentId: string,
    currentWeight: number,
    summary: PerformanceSummary
  ): WeightAdjustment {
    // Calculate performance factors
    const winRateFactor = this.calculateWinRateFactor(summary.winRate)
    const sharpeFactor = this.calculateSharpeFactor(summary.sharpeRatio)
    const consistencyFactor = this.calculateConsistencyFactor(summary.consistency)
    const recentTrendFactor = this.calculateRecentTrendFactor(summary.recentPerformance)
    
    // Combine factors with weights
    const combinedFactor = 
      winRateFactor * 0.3 +
      sharpeFactor * 0.3 +
      consistencyFactor * 0.2 +
      recentTrendFactor * 0.2
    
    // Apply time decay to historical performance
    const decayedFactor = this.applyTimeDecay(combinedFactor, summary)
    
    // Calculate new weight
    let weightMultiplier = 1 + (decayedFactor - 1) * this.config.adjustmentSensitivity
    weightMultiplier = Math.max(
      this.config.minWeightMultiplier,
      Math.min(this.config.maxWeightMultiplier, weightMultiplier)
    )
    
    const newWeight = currentWeight * weightMultiplier
    
    return {
      agentId,
      oldWeight: currentWeight,
      newWeight,
      reason: this.generateAdjustmentReason(winRateFactor, sharpeFactor, consistencyFactor, recentTrendFactor),
      performanceFactors: {
        winRate: winRateFactor,
        sharpeRatio: sharpeFactor,
        consistency: consistencyFactor,
        recentTrend: recentTrendFactor
      }
    }
  }
  
  /**
   * Calculate win rate factor (0-2)
   */
  private calculateWinRateFactor(winRate: number): number {
    // Map win rate to factor
    // 0% -> 0.0, 50% -> 1.0, 100% -> 2.0
    return winRate * 2
  }
  
  /**
   * Calculate Sharpe ratio factor (0-2)
   */
  private calculateSharpeFactor(sharpeRatio: number): number {
    // Map Sharpe ratio to factor
    // < 0 -> 0.2, 0 -> 0.5, 1 -> 1.0, 2 -> 1.5, 3+ -> 2.0
    if (sharpeRatio < 0) return 0.2
    if (sharpeRatio === 0) return 0.5
    if (sharpeRatio >= 3) return 2.0
    
    return 0.5 + (sharpeRatio / 3) * 1.5
  }
  
  /**
   * Calculate consistency factor (0-2)
   */
  private calculateConsistencyFactor(consistency: number): number {
    // Consistency is already 0-1, map to 0.5-1.5
    return 0.5 + consistency
  }
  
  /**
   * Calculate recent trend factor (0-2)
   */
  private calculateRecentTrendFactor(recent: RecentPerformance): number {
    // Weight recent performance more heavily
    const day1Weight = 0.5
    const day7Weight = 0.3
    const day30Weight = 0.2
    
    const day1Factor = recent.last24h.winRate * 2
    const day7Factor = recent.last7d.winRate * 2
    const day30Factor = recent.last30d.winRate * 2
    
    return day1Factor * day1Weight + day7Factor * day7Weight + day30Factor * day30Weight
  }
  
  /**
   * Apply time decay to performance factor
   */
  private applyTimeDecay(factor: number, summary: PerformanceSummary): number {
    // Recent performance should have more weight
    const recentWeight = 0.6
    const historicalWeight = 0.4
    
    const recentFactor = this.calculateRecentTrendFactor(summary.recentPerformance)
    
    return factor * historicalWeight + recentFactor * recentWeight
  }
  
  /**
   * Generate human-readable adjustment reason
   */
  private generateAdjustmentReason(
    winRate: number,
    sharpe: number,
    consistency: number,
    trend: number
  ): string {
    const factors: string[] = []
    
    if (winRate > 1.5) factors.push('high win rate')
    else if (winRate < 0.5) factors.push('low win rate')
    
    if (sharpe > 1.5) factors.push('excellent risk-adjusted returns')
    else if (sharpe < 0.5) factors.push('poor risk-adjusted returns')
    
    if (consistency > 1.3) factors.push('consistent performance')
    else if (consistency < 0.7) factors.push('inconsistent performance')
    
    if (trend > 1.5) factors.push('strong recent performance')
    else if (trend < 0.5) factors.push('weak recent performance')
    
    return factors.length > 0 ? `Based on ${factors.join(', ')}` : 'Based on overall performance'
  }
  
  /**
   * Update performance summary
   */
  private updateSummary(history: PerformanceHistory): void {
    const metrics = history.metrics
    const now = epochDateNow()
    
    // Calculate basic stats
    const totalSignals = metrics.length
    const metricsWithOutcome = metrics.filter(m => m.outcome)
    const successfulSignals = metricsWithOutcome.filter(m => m.outcome!.success).length
    const winRate = metricsWithOutcome.length > 0 ? successfulSignals / metricsWithOutcome.length : 0
    
    // Calculate returns
    const returns = metricsWithOutcome.map(m => m.outcome!.pnlPercent)
    const averageReturn = returns.length > 0 
      ? returns.reduce((a, b) => a + b, 0) / returns.length 
      : 0
    
    // Calculate Sharpe ratio
    const sharpeRatio = this.calculateSharpeRatio(returns)
    
    // Calculate max drawdown
    const maxDrawdown = this.calculateMaxDrawdown(metricsWithOutcome)
    
    // Calculate profit factor
    const profitFactor = this.calculateProfitFactor(metricsWithOutcome)
    
    // Calculate average hold time
    const avgHoldTime = metricsWithOutcome.length > 0
      ? metricsWithOutcome.reduce((sum, m) => sum + m.outcome!.holdTime, 0) / metricsWithOutcome.length
      : 0
    
    // Calculate consistency
    const consistency = this.calculateConsistency(returns)
    
    // Calculate recent performance
    const recentPerformance = this.calculateRecentPerformance(metrics, now)
    
    history.summary = {
      totalSignals,
      successfulSignals,
      winRate,
      averageReturn,
      sharpeRatio,
      maxDrawdown,
      profitFactor,
      avgHoldTime,
      consistency,
      recentPerformance
    }
    
    history.lastUpdated = now
  }
  
  /**
   * Calculate Sharpe ratio
   */
  private calculateSharpeRatio(returns: number[]): number {
    if (returns.length < 2) return 0
    
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length
    const stdDev = Math.sqrt(variance)
    
    return stdDev > 0 ? mean / stdDev * Math.sqrt(252) : 0 // Annualized
  }
  
  /**
   * Calculate maximum drawdown
   */
  private calculateMaxDrawdown(metrics: PerformanceMetric[]): number {
    if (metrics.length === 0) return 0
    
    let peak = 0
    let maxDrawdown = 0
    let cumReturn = 0
    
    for (const metric of metrics) {
      if (metric.outcome) {
        cumReturn += metric.outcome.pnlPercent
        if (cumReturn > peak) {
          peak = cumReturn
        }
        const drawdown = (peak - cumReturn) / (1 + peak)
        if (drawdown > maxDrawdown) {
          maxDrawdown = drawdown
        }
      }
    }
    
    return maxDrawdown
  }
  
  /**
   * Calculate profit factor
   */
  private calculateProfitFactor(metrics: PerformanceMetric[]): number {
    let grossProfit = 0
    let grossLoss = 0
    
    for (const metric of metrics) {
      if (metric.outcome) {
        if (metric.outcome.pnl > 0) {
          grossProfit += metric.outcome.pnl
        } else {
          grossLoss += Math.abs(metric.outcome.pnl)
        }
      }
    }
    
    return grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0
  }
  
  /**
   * Calculate consistency score (0-1)
   */
  private calculateConsistency(returns: number[]): number {
    if (returns.length < 2) return 0
    
    // Calculate coefficient of variation (lower is more consistent)
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length
    if (mean === 0) return 0
    
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length
    const stdDev = Math.sqrt(variance)
    const cv = Math.abs(stdDev / mean)
    
    // Map CV to consistency score (0 CV = 1 consistency)
    return Math.max(0, 1 - Math.min(cv, 1))
  }
  
  /**
   * Calculate recent performance metrics
   */
  private calculateRecentPerformance(
    metrics: PerformanceMetric[],
    now: EpochDate
  ): RecentPerformance {
    const day1Cutoff = now - 24 * 60 * 60 * 1000
    const day7Cutoff = now - 7 * 24 * 60 * 60 * 1000
    const day30Cutoff = now - 30 * 24 * 60 * 60 * 1000
    
    const calculatePeriod = (cutoff: number) => {
      const periodMetrics = metrics.filter(m => m.timestamp >= cutoff && m.outcome)
      const successful = periodMetrics.filter(m => m.outcome!.success).length
      const winRate = periodMetrics.length > 0 ? successful / periodMetrics.length : 0
      const returns = periodMetrics.length > 0
        ? periodMetrics.reduce((sum, m) => sum + m.outcome!.pnlPercent, 0)
        : 0
      
      return {
        winRate,
        returns,
        signals: periodMetrics.length
      }
    }
    
    return {
      last24h: calculatePeriod(day1Cutoff),
      last7d: calculatePeriod(day7Cutoff),
      last30d: calculatePeriod(day30Cutoff)
    }
  }
  
  /**
   * Get or create history for agent
   */
  private getOrCreateHistory(agentId: string): PerformanceHistory {
    let history = this.performanceData.get(agentId)
    
    if (!history) {
      history = {
        agentId,
        metrics: [],
        summary: {
          totalSignals: 0,
          successfulSignals: 0,
          winRate: 0,
          averageReturn: 0,
          sharpeRatio: 0,
          maxDrawdown: 0,
          profitFactor: 0,
          avgHoldTime: 0,
          consistency: 0,
          recentPerformance: {
            last24h: { winRate: 0, returns: 0, signals: 0 },
            last7d: { winRate: 0, returns: 0, signals: 0 },
            last30d: { winRate: 0, returns: 0, signals: 0 }
          }
        },
        lastUpdated: epochDateNow()
      }
      
      this.performanceData.set(agentId, history)
    }
    
    return history
  }
  
  /**
   * Subscribe to relevant events
   */
  private subscribeToEvents(): void {
    // Listen for consensus events to track signals
    this.eventBus.subscribe(EventTypes.CONSENSUS_REACHED, (data) => {
      const consensusData = data as unknown as { 
        consensus: { 
          agentSignals: Record<string, AgentSignal>
        }
        executionTime: number 
      }
      
      if (consensusData.consensus?.agentSignals) {
        for (const [agentId, signal] of Object.entries(consensusData.consensus.agentSignals)) {
          this.recordSignal(agentId, signal, consensusData.executionTime || 0)
        }
      }
    })
    
    // Listen for trade execution to track outcomes
    this.eventBus.subscribe(EventTypes.TRADE_EXECUTED, (_data) => {
      // TODO: Match trade to signal and record outcome
      // This would require additional correlation logic
    })
  }
  
  /**
   * Export performance data for visualization
   */
  exportPerformanceData(): Record<string, PerformanceHistory> {
    const data: Record<string, PerformanceHistory> = {}
    
    for (const [agentId, history] of this.performanceData) {
      data[agentId] = history
    }
    
    return data
  }
  
  /**
   * Clear performance data for an agent
   */
  clearAgentData(agentId: string): void {
    this.performanceData.delete(agentId)
    
    // Clear signal mappings
    for (const [signalId] of this.signalToTradeMap) {
      if (signalId.startsWith(agentId)) {
        this.signalToTradeMap.delete(signalId)
      }
    }
  }
  
  /**
   * Clear all performance data
   */
  clearAllData(): void {
    this.performanceData.clear()
    this.signalToTradeMap.clear()
  }
}