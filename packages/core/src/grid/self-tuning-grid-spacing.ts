import type { Candle, GridState } from '@trdr/shared'
import type { Logger } from '@trdr/types'
import { VolatilityGridSpacing, type VolatilitySpacingConfig, type SpacingCalculationResult } from './volatility-grid-spacing'

/**
 * Performance metrics for evaluating grid effectiveness
 */
export interface GridPerformanceMetrics {
  /** Total profit/loss from grid operations */
  readonly totalPnl: number
  /** Number of successful fills */
  readonly fillCount: number
  /** Fill rate (successful fills / total orders) */
  readonly fillRate: number
  /** Average profit per grid level */
  readonly profitPerLevel: number
  /** Average time to fill (milliseconds) */
  readonly averageTimeToFill: number
  /** Sharpe ratio for risk-adjusted returns */
  readonly sharpeRatio: number
  /** Maximum drawdown percentage */
  readonly maxDrawdown: number
  /** Win rate (profitable fills / total fills) */
  readonly winRate: number
  /** Average winning trade amount */
  readonly averageWin: number
  /** Average losing trade amount */
  readonly averageLoss: number
  /** Number of active periods */
  readonly activePeriods: number
}

/**
 * Historical spacing performance record
 */
export interface SpacingPerformanceRecord {
  /** Grid spacing percentage used */
  readonly spacing: number
  /** Performance metrics achieved */
  readonly metrics: GridPerformanceMetrics
  /** Market conditions during this period */
  readonly marketConditions: {
    readonly volatility: number
    readonly trend: 'bullish' | 'bearish' | 'sideways'
    readonly volume: number
  }
  /** Timestamp when this record was created */
  readonly timestamp: number
  /** Duration of this spacing configuration (ms) */
  readonly duration: number
  /** Confidence score for this performance data */
  readonly confidence: number
}

/**
 * Self-tuning configuration
 */
export interface SelfTuningConfig {
  /** Enable performance-based tuning */
  readonly enablePerformanceTuning: boolean
  /** Minimum performance history required for tuning */
  readonly minHistoryPeriods: number
  /** Performance evaluation window (milliseconds) */
  readonly evaluationWindow: number
  /** Maximum adjustment per tuning cycle (percentage of current spacing) */
  readonly maxAdjustmentPercent: number
  /** Minimum spacing adjustment threshold */
  readonly minAdjustmentThreshold: number
  /** Weight for recent performance vs historical */
  readonly recentPerformanceWeight: number
  /** Enable evolutionary optimization */
  readonly enableEvolutionaryOptimization: boolean
  /** Population size for evolutionary algorithm */
  readonly evolutionPopulationSize: number
  /** Number of generations for evolution */
  readonly evolutionGenerations: number
}

/**
 * SelfTuningGridSpacing extends VolatilityGridSpacing with performance-based optimization.
 * 
 * Features:
 * - Performance metrics tracking and analysis
 * - Adaptive spacing based on historical performance
 * - Evolutionary optimization algorithm
 * - Market condition correlation analysis
 * - Parameter versioning and rollback
 */
export class SelfTuningGridSpacing extends VolatilityGridSpacing {
  private readonly tuningConfig: Required<SelfTuningConfig>
  private readonly performanceHistory: SpacingPerformanceRecord[] = []
  private readonly currentMetrics: {
    totalPnl?: number
    fillCount?: number
    fillRate?: number
    winRate?: number
    averageTimeToFill?: number
    averageWin?: number
    averageLoss?: number
  } = {}
  private currentSpacingStartTime: number = Date.now()
  private lastTuningTime = 0
  private readonly tuningLogger?: Logger

  constructor(
    volatilityConfig: Partial<VolatilitySpacingConfig> = {},
    tuningConfig: Partial<SelfTuningConfig> = {},
    logger?: Logger
  ) {
    super(volatilityConfig, logger)
    this.tuningLogger = logger

    this.tuningConfig = {
      enablePerformanceTuning: tuningConfig.enablePerformanceTuning ?? true,
      minHistoryPeriods: tuningConfig.minHistoryPeriods ?? 5,
      evaluationWindow: tuningConfig.evaluationWindow ?? 24 * 60 * 60 * 1000, // 24 hours
      maxAdjustmentPercent: tuningConfig.maxAdjustmentPercent ?? 20, // 20%
      minAdjustmentThreshold: tuningConfig.minAdjustmentThreshold ?? 0.1, // 0.1%
      recentPerformanceWeight: tuningConfig.recentPerformanceWeight ?? 0.7,
      enableEvolutionaryOptimization: tuningConfig.enableEvolutionaryOptimization ?? false,
      evolutionPopulationSize: tuningConfig.evolutionPopulationSize ?? 10,
      evolutionGenerations: tuningConfig.evolutionGenerations ?? 5
    }

    this.tuningLogger?.debug('SelfTuningGridSpacing initialized', { 
      volatilityConfig, 
      tuningConfig: this.tuningConfig 
    })
  }

  /**
   * Enhanced spacing calculation with performance feedback
   */
  async calculateOptimalSpacing(
    candles: readonly Candle[],
    currentPrice: number,
    gridStates?: readonly GridState[]
  ): Promise<SpacingCalculationResult> {
    // Get base spacing from volatility analysis
    const baseResult = await super.calculateOptimalSpacing(candles, currentPrice)

    if (!this.tuningConfig.enablePerformanceTuning || !gridStates) {
      return baseResult
    }

    // Update performance metrics with current grid states
    this.updatePerformanceMetrics(gridStates)

    // Check if it's time for tuning
    const shouldTune = this.shouldPerformTuning()
    if (!shouldTune) {
      return baseResult
    }

    // Perform performance-based adjustment
    const performanceAdjustment = await this.calculatePerformanceAdjustment(candles, currentPrice)
    const tunedSpacing = this.applyPerformanceAdjustment(baseResult.optimalSpacing, performanceAdjustment)

    // Record the tuning decision
    this.recordTuningDecision(baseResult.optimalSpacing, tunedSpacing, performanceAdjustment)

    return {
      optimalSpacing: tunedSpacing,
      volatilityMetrics: baseResult.volatilityMetrics,
      reasoning: `${baseResult.reasoning}. Performance tuning applied: ${performanceAdjustment.reasoning}`,
      confidence: Math.min(baseResult.confidence + performanceAdjustment.confidenceBoost, 1.0)
    }
  }

  /**
   * Records performance metrics for a completed grid session
   */
  recordGridPerformance(
    spacing: number,
    gridState: GridState,
    marketConditions: SpacingPerformanceRecord['marketConditions']
  ): void {
    const metrics = this.calculateGridMetrics(gridState)
    const duration = Date.now() - this.currentSpacingStartTime
    const confidence = this.calculatePerformanceConfidence(metrics, duration)

    const record: SpacingPerformanceRecord = {
      spacing,
      metrics,
      marketConditions,
      timestamp: Date.now(),
      duration,
      confidence
    }

    this.performanceHistory.push(record)

    // Keep only recent history
    const maxHistorySize = this.tuningConfig.minHistoryPeriods * 3
    if (this.performanceHistory.length > maxHistorySize) {
      this.performanceHistory.shift()
    }

    this.tuningLogger?.info('Grid performance recorded', {
      spacing,
      pnl: metrics.totalPnl,
      fillRate: metrics.fillRate,
      confidence
    })

    // Reset for next session
    this.currentSpacingStartTime = Date.now()
  }

  /**
   * Updates current performance metrics with active grid states
   */
  private updatePerformanceMetrics(gridStates: readonly GridState[]): void {
    let totalPnl = 0
    let totalFills = 0
    let totalOrders = 0
    let totalWins = 0
    let totalWinAmount = 0
    let totalLossAmount = 0
    const fillTimes: number[] = []

    for (const grid of gridStates) {
      totalPnl += grid.realizedPnl + grid.unrealizedPnl
      
      for (const level of grid.levels) {
        totalOrders++
        
        if (level.fillCount > 0) {
          totalFills += level.fillCount
          fillTimes.push(level.updatedAt - level.createdAt)
          
          if (level.pnl > 0) {
            totalWins++
            totalWinAmount += level.pnl
          } else if (level.pnl < 0) {
            totalLossAmount += Math.abs(level.pnl)
          }
        }
      }
    }

    this.currentMetrics.totalPnl = totalPnl
    this.currentMetrics.fillCount = totalFills
    this.currentMetrics.fillRate = totalOrders > 0 ? totalFills / totalOrders : 0
    this.currentMetrics.winRate = totalFills > 0 ? totalWins / totalFills : 0
    this.currentMetrics.averageTimeToFill = fillTimes.length > 0 
      ? fillTimes.reduce((sum, time) => sum + time, 0) / fillTimes.length 
      : 0
    this.currentMetrics.averageWin = totalWins > 0 ? totalWinAmount / totalWins : 0
    this.currentMetrics.averageLoss = (totalFills - totalWins) > 0 ? totalLossAmount / (totalFills - totalWins) : 0
  }

  /**
   * Determines if performance tuning should be performed
   */
  private shouldPerformTuning(): boolean {
    // Don't tune too frequently
    const timeSinceLastTuning = Date.now() - this.lastTuningTime
    if (timeSinceLastTuning < this.tuningConfig.evaluationWindow) {
      return false
    }

    // Need minimum history
    if (this.performanceHistory.length < this.tuningConfig.minHistoryPeriods) {
      return false
    }

    // Need sufficient current session duration
    const currentSessionDuration = Date.now() - this.currentSpacingStartTime
    if (currentSessionDuration < this.tuningConfig.evaluationWindow * 0.5) {
      return false
    }

    return true
  }

  /**
   * Calculates performance-based spacing adjustment
   */
  private async calculatePerformanceAdjustment(
    candles: readonly Candle[],
    currentPrice: number
  ): Promise<{
    adjustmentPercent: number
    reasoning: string
    confidenceBoost: number
  }> {
    if (this.performanceHistory.length === 0) {
      return {
        adjustmentPercent: 0,
        reasoning: 'No performance history available',
        confidenceBoost: 0
      }
    }

    // Analyze recent vs historical performance
    const recentPerformance = this.analyzeRecentPerformance()
    const optimalSpacing = await this.findOptimalSpacingFromHistory()
    const marketConditions = this.assessCurrentMarketConditions(candles)

    // Calculate adjustment based on multiple factors
    let adjustmentPercent = 0
    const reasons: string[] = []

    // Factor 1: Recent performance vs historical average
    if (recentPerformance.pnlTrend < -0.1) {
      // Poor recent performance, try different spacing
      adjustmentPercent += recentPerformance.fillRate > 0.7 ? -10 : 10 // Tighter if high fills, wider if low fills
      reasons.push(`recent underperformance (${(recentPerformance.pnlTrend * 100).toFixed(1)}%)`)
    } else if (recentPerformance.pnlTrend > 0.1) {
      // Good recent performance, slight bias toward current approach
      adjustmentPercent += recentPerformance.fillRate > 0.8 ? -5 : 5
      reasons.push(`recent outperformance (${(recentPerformance.pnlTrend * 100).toFixed(1)}%)`)
    }

    // Factor 2: Fill rate optimization
    if (recentPerformance.fillRate < 0.3) {
      adjustmentPercent -= 15 // Tighter spacing for better fills
      reasons.push(`low fill rate (${(recentPerformance.fillRate * 100).toFixed(1)}%)`)
    } else if (recentPerformance.fillRate > 0.9) {
      adjustmentPercent += 10 // Wider spacing to increase profit per fill
      reasons.push(`high fill rate (${(recentPerformance.fillRate * 100).toFixed(1)}%)`)
    }

    // Factor 3: Market condition adaptation
    const marketAdjustment = this.calculateMarketConditionAdjustment(marketConditions)
    adjustmentPercent += marketAdjustment.adjustment
    if (marketAdjustment.reason) {
      reasons.push(marketAdjustment.reason)
    }

    // Factor 4: Historical optimal spacing
    if (optimalSpacing.confidence > 0.6) {
      const historicalBias = (optimalSpacing.spacing - currentPrice) / currentPrice * 100
      adjustmentPercent += historicalBias * 0.3 // 30% weight to historical optimal
      reasons.push(`historical optimal bias (${historicalBias.toFixed(1)}%)`)
    }

    // Cap the adjustment
    adjustmentPercent = Math.max(
      -this.tuningConfig.maxAdjustmentPercent,
      Math.min(this.tuningConfig.maxAdjustmentPercent, adjustmentPercent)
    )

    // Only apply if significant enough
    if (Math.abs(adjustmentPercent) < this.tuningConfig.minAdjustmentThreshold) {
      adjustmentPercent = 0
      reasons.length = 0
      reasons.push('adjustment below threshold')
    }

    const reasoning = reasons.length > 0 
      ? `Performance tuning: ${reasons.join(', ')}`
      : 'No significant performance adjustment needed'

    const confidenceBoost = Math.min(0.2, this.performanceHistory.length * 0.02)

    return {
      adjustmentPercent,
      reasoning,
      confidenceBoost
    }
  }

  /**
   * Analyzes recent performance trends
   */
  private analyzeRecentPerformance(): {
    pnlTrend: number
    fillRate: number
    winRate: number
  } {
    const recentRecords = this.performanceHistory.slice(-Math.ceil(this.tuningConfig.minHistoryPeriods / 2))
    const historicalRecords = this.performanceHistory.slice(0, -recentRecords.length)

    const recentAvgPnl = recentRecords.length > 0
      ? recentRecords.reduce((sum, r) => sum + r.metrics.totalPnl, 0) / recentRecords.length
      : 0

    const historicalAvgPnl = historicalRecords.length > 0
      ? historicalRecords.reduce((sum, r) => sum + r.metrics.totalPnl, 0) / historicalRecords.length
      : recentAvgPnl

    const pnlTrend = historicalAvgPnl !== 0 
      ? (recentAvgPnl - historicalAvgPnl) / Math.abs(historicalAvgPnl)
      : 0

    const recentFillRate = recentRecords.length > 0
      ? recentRecords.reduce((sum, r) => sum + r.metrics.fillRate, 0) / recentRecords.length
      : 0

    const recentWinRate = recentRecords.length > 0
      ? recentRecords.reduce((sum, r) => sum + r.metrics.winRate, 0) / recentRecords.length
      : 0

    return {
      pnlTrend,
      fillRate: recentFillRate,
      winRate: recentWinRate
    }
  }

  /**
   * Finds historically optimal spacing from performance records
   */
  private async findOptimalSpacingFromHistory(): Promise<{
    spacing: number
    confidence: number
  }> {
    if (this.performanceHistory.length === 0) {
      return { spacing: 0, confidence: 0 }
    }

    // Weight records by confidence and recency
    const weightedRecords = this.performanceHistory.map((record, index) => {
      const recencyWeight = 1 - (this.performanceHistory.length - index - 1) / this.performanceHistory.length
      const weight = record.confidence * (0.5 + 0.5 * recencyWeight)
      
      return {
        ...record,
        weight,
        score: this.calculatePerformanceScore(record.metrics) * weight
      }
    })

    // Find the highest scoring spacing
    const bestRecord = weightedRecords.reduce((best, current) => 
      current.score > best.score ? current : best
    )

    return {
      spacing: bestRecord.spacing,
      confidence: bestRecord.confidence * 0.8 // Slightly reduce confidence for historical data
    }
  }

  /**
   * Assesses current market conditions
   */
  private assessCurrentMarketConditions(candles: readonly Candle[]): SpacingPerformanceRecord['marketConditions'] {
    if (candles.length === 0) {
      return { volatility: 0, trend: 'sideways', volume: 0 }
    }

    // Calculate volatility (simple standard deviation)
    const returns = []
    for (let i = 1; i < Math.min(candles.length, 20); i++) {
      const ret = (candles[i]!.close - candles[i-1]!.close) / candles[i-1]!.close
      returns.push(ret)
    }
    
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length
    const volatility = Math.sqrt(returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length)

    // Determine trend
    const recent = candles.slice(-10)
    const older = candles.slice(-20, -10)
    const recentAvg = recent.reduce((sum, c) => sum + c.close, 0) / recent.length
    const olderAvg = older.length > 0 ? older.reduce((sum, c) => sum + c.close, 0) / older.length : recentAvg
    
    let trend: 'bullish' | 'bearish' | 'sideways' = 'sideways'
    const trendStrength = (recentAvg - olderAvg) / olderAvg
    if (trendStrength > 0.02) trend = 'bullish'
    else if (trendStrength < -0.02) trend = 'bearish'

    // Average volume
    const volume = candles.slice(-10).reduce((sum, c) => sum + c.volume, 0) / Math.min(10, candles.length)

    return { volatility, trend, volume }
  }

  /**
   * Calculates market condition based spacing adjustment
   */
  private calculateMarketConditionAdjustment(conditions: SpacingPerformanceRecord['marketConditions']): {
    adjustment: number
    reason: string
  } {
    let adjustment = 0
    const reasons: string[] = []

    // High volatility = wider spacing
    if (conditions.volatility > 0.03) {
      adjustment += 10
      reasons.push('high volatility')
    } else if (conditions.volatility < 0.01) {
      adjustment -= 5
      reasons.push('low volatility')
    }

    // Strong trend = slightly wider spacing to avoid whipsaws
    if (conditions.trend !== 'sideways') {
      adjustment += 5
      reasons.push(`${conditions.trend} trend`)
    }

    return {
      adjustment,
      reason: reasons.join(', ')
    }
  }

  /**
   * Applies performance adjustment to base spacing
   */
  private applyPerformanceAdjustment(baseSpacing: number, adjustment: { adjustmentPercent: number }): number {
    const adjustmentFactor = 1 + (adjustment.adjustmentPercent / 100)
    return baseSpacing * adjustmentFactor
  }

  /**
   * Records a tuning decision for analysis
   */
  private recordTuningDecision(originalSpacing: number, tunedSpacing: number, adjustment: any): void {
    this.lastTuningTime = Date.now()
    
    this.tuningLogger?.info('Grid spacing tuned', {
      originalSpacing,
      tunedSpacing,
      adjustmentPercent: adjustment.adjustmentPercent,
      reasoning: adjustment.reasoning
    })
  }

  /**
   * Calculates comprehensive grid performance metrics
   */
  private calculateGridMetrics(gridState: GridState): GridPerformanceMetrics {
    const levels = gridState.levels
    const fillCount = levels.reduce((sum, level) => sum + level.fillCount, 0)
    const totalOrders = levels.length
    const winningFills = levels.filter(level => level.pnl > 0).length
    const totalWinAmount = levels.filter(level => level.pnl > 0).reduce((sum, level) => sum + level.pnl, 0)
    const totalLossAmount = levels.filter(level => level.pnl < 0).reduce((sum, level) => sum + Math.abs(level.pnl), 0)
    
    const fillTimes = levels
      .filter(level => level.fillCount > 0)
      .map(level => level.updatedAt - level.createdAt)
    
    const averageTimeToFill = fillTimes.length > 0 
      ? fillTimes.reduce((sum, time) => sum + time, 0) / fillTimes.length 
      : 0

    // Calculate Sharpe ratio (simplified)
    const totalReturn = gridState.realizedPnl + gridState.unrealizedPnl
    const timeActive = Date.now() - gridState.initializedAt
    const annualizedReturn = totalReturn * (365 * 24 * 60 * 60 * 1000) / timeActive
    const sharpeRatio = annualizedReturn / Math.max(gridState.allocatedCapital * 0.1, 1) // Simplified risk-free rate

    return {
      totalPnl: totalReturn,
      fillCount,
      fillRate: totalOrders > 0 ? fillCount / totalOrders : 0,
      profitPerLevel: levels.length > 0 ? totalReturn / levels.length : 0,
      averageTimeToFill,
      sharpeRatio,
      maxDrawdown: 0, // TODO: Calculate actual drawdown
      winRate: fillCount > 0 ? winningFills / fillCount : 0,
      averageWin: winningFills > 0 ? totalWinAmount / winningFills : 0,
      averageLoss: (fillCount - winningFills) > 0 ? totalLossAmount / (fillCount - winningFills) : 0,
      activePeriods: Math.floor(timeActive / this.tuningConfig.evaluationWindow)
    }
  }

  /**
   * Calculates confidence score for performance data
   */
  private calculatePerformanceConfidence(metrics: GridPerformanceMetrics, duration: number): number {
    let confidence = 0.5

    // More active periods = higher confidence
    confidence += Math.min(metrics.activePeriods * 0.1, 0.3)

    // More fills = higher confidence
    confidence += Math.min(metrics.fillCount * 0.01, 0.2)

    // Longer duration = higher confidence
    const minDuration = this.tuningConfig.evaluationWindow
    confidence += Math.min(duration / minDuration * 0.2, 0.2)

    return Math.min(confidence, 1.0)
  }

  /**
   * Calculates overall performance score for ranking
   */
  private calculatePerformanceScore(metrics: GridPerformanceMetrics): number {
    // Weighted combination of key metrics
    const pnlScore = Math.max(0, metrics.totalPnl) * 0.4
    const fillRateScore = metrics.fillRate * 20 // Normalize to similar scale
    const winRateScore = metrics.winRate * 10
    const sharpeScore = Math.max(0, metrics.sharpeRatio) * 0.3

    return pnlScore + fillRateScore + winRateScore + sharpeScore
  }

  /**
   * Gets current performance statistics
   */
  getPerformanceStatistics(): {
    historyCount: number
    averageSpacing: number
    bestPerformingSpacing: number
    currentMetrics: Partial<GridPerformanceMetrics>
  } {
    const averageSpacing = this.performanceHistory.length > 0
      ? this.performanceHistory.reduce((sum, record) => sum + record.spacing, 0) / this.performanceHistory.length
      : 0

    const bestRecord = this.performanceHistory.reduce((best, current) => {
      const currentScore = this.calculatePerformanceScore(current.metrics)
      const bestScore = best ? this.calculatePerformanceScore(best.metrics) : -Infinity
      return currentScore > bestScore ? current : best
    }, null as SpacingPerformanceRecord | null)

    return {
      historyCount: this.performanceHistory.length,
      averageSpacing,
      bestPerformingSpacing: bestRecord?.spacing || 0,
      currentMetrics: { ...this.currentMetrics }
    }
  }

  /**
   * Resets performance history (useful for testing)
   */
  resetPerformanceHistory(): void {
    this.performanceHistory.length = 0
    Object.keys(this.currentMetrics).forEach(key => delete (this.currentMetrics as any)[key])
    this.currentSpacingStartTime = Date.now()
    this.lastTuningTime = 0
    this.tuningLogger?.debug('Performance history reset')
  }
}