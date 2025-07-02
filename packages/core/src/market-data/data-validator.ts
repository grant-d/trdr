import type { EpochDate, StockSymbol } from '@trdr/shared'
import { epochDateNow, toEpochDate } from '@trdr/shared'
import type { Candle, Logger, MarketDataRepository } from '@trdr/types'

export interface ValidationIssue {
  readonly type: 'gap' | 'outlier' | 'duplicate' | 'invalid' | 'corrupted'
  readonly severity: 'low' | 'medium' | 'high' | 'critical'
  readonly symbol: StockSymbol
  readonly interval: string
  readonly startTime: EpochDate
  readonly endTime: EpochDate
  readonly description: string
  readonly affectedCandles?: number
  readonly suggestedAction?: 'backfill' | 'interpolate' | 'remove' | 'manual_review'
  readonly metadata?: Record<string, unknown>
}

export interface ValidationReport {
  readonly symbol: StockSymbol
  readonly interval: string
  readonly startTime: EpochDate
  readonly endTime: EpochDate
  readonly totalCandles: number
  readonly issues: ValidationIssue[]
  readonly validationTime: EpochDate
  readonly overallHealth: 'healthy' | 'warning' | 'critical'
}

export interface ValidationConfig {
  readonly gapThresholdMultiplier?: number // How many intervals constitute a gap (default: 2)
  readonly outlierMethod?: 'zscore' | 'iqr' | 'isolation' // Outlier detection method
  readonly zscoreThreshold?: number // Z-score threshold for outliers (default: 3)
  readonly iqrMultiplier?: number // IQR multiplier for outliers (default: 1.5)
  readonly minDataPoints?: number // Minimum data points for validation (default: 10)
  readonly autoRepair?: boolean // Automatically attempt repairs (default: false)
  readonly maxRepairAttempts?: number // Max repair attempts (default: 3)
}

export interface DataValidatorDependencies {
  readonly repository: MarketDataRepository
  readonly logger?: Logger
}

export class DataValidator {
  private readonly repository: MarketDataRepository
  private readonly logger?: Logger
  private readonly defaultConfig: Required<ValidationConfig> = {
    gapThresholdMultiplier: 2,
    outlierMethod: 'zscore',
    zscoreThreshold: 3,
    iqrMultiplier: 1.5,
    minDataPoints: 10,
    autoRepair: false,
    maxRepairAttempts: 3
  }

  constructor(deps: DataValidatorDependencies) {
    this.repository = deps.repository
    this.logger = deps.logger
  }

  /**
   * Validate data continuity to detect gaps
   */
  async validateDataContinuity(
    symbol: StockSymbol,
    interval: string,
    startTime: EpochDate,
    endTime: EpochDate,
    config?: ValidationConfig
  ): Promise<ValidationIssue[]> {
    const mergedConfig = { ...this.defaultConfig, ...config }
    const issues: ValidationIssue[] = []

    try {
      const candles = await this.repository.getCandles(
        symbol,
        interval,
        startTime,
        endTime
      )

      if (candles.length < mergedConfig.minDataPoints) {
        this.logger?.warn('Insufficient data points for validation', {
          symbol: String(symbol),
          interval,
          found: candles.length,
          required: mergedConfig.minDataPoints
        })
      }

      // Sort candles by timestamp
      const sortedCandles = [...candles].sort((a, b) => Number(a.timestamp) - Number(b.timestamp))

      // Calculate expected interval in milliseconds
      const intervalMs = this.getIntervalMilliseconds(interval)
      const gapThreshold = intervalMs * mergedConfig.gapThresholdMultiplier

      // Check for gaps
      for (let i = 1; i < sortedCandles.length; i++) {
        const prevCandle = sortedCandles[i - 1]!
        const currCandle = sortedCandles[i]!
        const timeDiff = Number(currCandle.timestamp) - Number(prevCandle.timestamp)
        

        if (timeDiff > gapThreshold) {
          const missingCandles = Math.floor(timeDiff / intervalMs) - 1

          issues.push({
            type: 'gap',
            severity: this.calculateGapSeverity(missingCandles),
            symbol,
            interval,
            startTime: prevCandle.closeTime,
            endTime: currCandle.openTime,
            description: `Data gap detected: ${missingCandles} candles missing`,
            affectedCandles: missingCandles,
            suggestedAction: 'backfill',
            metadata: {
              gapDurationMs: timeDiff - intervalMs,
              expectedIntervalMs: intervalMs
            }
          })
        }
      }

      return issues
    } catch (error) {
      this.logger?.error('Error validating data continuity', {
        symbol: String(symbol),
        interval,
        error: (error as Error).message
      })
      throw error
    }
  }

  /**
   * Detect outliers in the data
   */
  async detectOutliers(
    candles: Candle[],
    config?: ValidationConfig
  ): Promise<ValidationIssue[]> {
    const mergedConfig = { ...this.defaultConfig, ...config }
    const issues: ValidationIssue[] = []

    if (candles.length < mergedConfig.minDataPoints) {
      return issues
    }

    try {
      switch (mergedConfig.outlierMethod) {
        case 'zscore':
          return this.detectOutliersZScore(candles, mergedConfig)
        case 'iqr':
          return this.detectOutliersIQR(candles, mergedConfig)
        case 'isolation':
          // Simplified isolation forest approach
          return this.detectOutliersIsolation(candles, mergedConfig)
        default:
          throw new Error(`Unknown outlier method: ${mergedConfig.outlierMethod}`)
      }
    } catch (error) {
      this.logger?.error('Error detecting outliers', {
        method: mergedConfig.outlierMethod,
        error: (error as Error).message
      })
      throw error
    }
  }

  /**
   * Verify overall data integrity
   */
  async verifyDataIntegrity(
    symbol: StockSymbol,
    interval: string,
    startTime?: EpochDate,
    endTime?: EpochDate,
    config?: ValidationConfig
  ): Promise<ValidationReport> {
    const now = epochDateNow()
    const actualEndTime = endTime || now
    const actualStartTime = startTime || toEpochDate(Number(actualEndTime) - 30 * 24 * 60 * 60 * 1000) // 30 days

    try {
      const candles = await this.repository.getCandles(
        symbol,
        interval,
        actualStartTime,
        actualEndTime
      )

      // Run all validation checks
      const continuityIssues = await this.validateDataContinuity(
        symbol,
        interval,
        actualStartTime,
        actualEndTime,
        config
      )

      const outlierIssues = await this.detectOutliers(candles, config)
      const duplicateIssues = this.detectDuplicates(candles)
      const invalidIssues = this.detectInvalidData(candles)

      const allIssues = [
        ...continuityIssues,
        ...outlierIssues,
        ...duplicateIssues,
        ...invalidIssues
      ]

      // Calculate overall health
      const criticalCount = allIssues.filter(i => i.severity === 'critical').length
      const highCount = allIssues.filter(i => i.severity === 'high').length
      
      let overallHealth: 'healthy' | 'warning' | 'critical' = 'healthy'
      if (criticalCount > 0 || highCount > 5) {
        overallHealth = 'critical'
      } else if (highCount > 0 || allIssues.length > 10) {
        overallHealth = 'warning'
      }

      const report: ValidationReport = {
        symbol,
        interval,
        startTime: actualStartTime,
        endTime: actualEndTime,
        totalCandles: candles.length,
        issues: allIssues,
        validationTime: now,
        overallHealth
      }

      this.logger?.info('Data integrity validation completed', {
        symbol: String(symbol),
        interval,
        issueCount: allIssues.length,
        health: overallHealth
      })

      return report
    } catch (error) {
      this.logger?.error('Error verifying data integrity', {
        symbol: String(symbol),
        interval,
        error: (error as Error).message
      })
      throw error
    }
  }

  /**
   * Repair corrupted data based on identified issues
   */
  async repairCorruptedData(
    symbol: StockSymbol,
    interval: string,
    issues: ValidationIssue[],
    config?: ValidationConfig
  ): Promise<{ repaired: number; failed: number; results: Array<{ issue: ValidationIssue; success: boolean; error?: string }> }> {
    const mergedConfig = { ...this.defaultConfig, ...config }
    const results: Array<{ issue: ValidationIssue; success: boolean; error?: string }> = []
    let repaired = 0
    let failed = 0

    if (!mergedConfig.autoRepair) {
      this.logger?.warn('Auto-repair is disabled, skipping repairs', {
        symbol: String(symbol),
        interval,
        issueCount: issues.length
      })
      return { repaired: 0, failed: issues.length, results: issues.map(issue => ({ issue, success: false, error: 'Auto-repair disabled' })) }
    }

    for (const issue of issues) {
      try {
        let success = false
        let error: string | undefined

        switch (issue.suggestedAction) {
          case 'backfill':
            // This would trigger a backfill process
            this.logger?.info('Backfill required for gap', {
              symbol: String(symbol),
              interval,
              startTime: Number(issue.startTime),
              endTime: Number(issue.endTime)
            })
            // In a real implementation, this would call the backfill service
            error = 'Backfill not implemented in validator'
            break

          case 'interpolate':
            success = await this.interpolateMissingData(symbol, interval, issue)
            break

          case 'remove':
            success = await this.removeInvalidData(symbol, interval, issue)
            break

          case 'manual_review':
            this.logger?.warn('Manual review required', {
              symbol: String(symbol),
              interval,
              issue: issue.type,
              description: issue.description
            })
            error = 'Manual review required'
            break
        }

        if (success) {
          repaired++
        } else {
          failed++
        }

        results.push({ issue, success, error })
      } catch (err) {
        failed++
        results.push({ issue, success: false, error: (err as Error).message })
      }
    }

    this.logger?.info('Data repair completed', {
      symbol: String(symbol),
      interval,
      repaired,
      failed,
      total: issues.length
    })

    return { repaired, failed, results }
  }

  private detectOutliersZScore(candles: Candle[], config: Required<ValidationConfig>): ValidationIssue[] {
    const issues: ValidationIssue[] = []
    
    // Calculate statistics for each price field
    const prices = {
      open: candles.map(c => c.open),
      high: candles.map(c => c.high),
      low: candles.map(c => c.low),
      close: candles.map(c => c.close),
      volume: candles.map(c => c.volume)
    }

    // Calculate mean and std deviation for each field
    const stats = Object.entries(prices).reduce((acc, [field, values]) => {
      const mean = values.reduce((sum, v) => sum + v, 0) / values.length
      const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length
      const stdDev = Math.sqrt(variance)
      acc[field] = { mean, stdDev }
      return acc
    }, {} as Record<string, { mean: number; stdDev: number }>)

    // Check each candle for outliers
    candles.forEach((candle, index) => {
      const outlierFields: string[] = []
      
      Object.entries(stats).forEach(([field, { mean, stdDev }]) => {
        const value = (candle as any)[field]
        const zScore = Math.abs((value - mean) / stdDev)
        
        if (zScore > config.zscoreThreshold) {
          outlierFields.push(`${field} (z=${zScore.toFixed(2)})`)
        }
      })

      if (outlierFields.length > 0) {
        issues.push({
          type: 'outlier',
          severity: outlierFields.length > 2 ? 'high' : 'medium',
          symbol: candle.symbol,
          interval: candle.interval,
          startTime: candle.openTime,
          endTime: candle.closeTime,
          description: `Outlier detected in fields: ${outlierFields.join(', ')}`,
          suggestedAction: 'manual_review',
          metadata: {
            outlierFields,
            candleIndex: index
          }
        })
      }
    })

    return issues
  }

  private detectOutliersIQR(candles: Candle[], config: Required<ValidationConfig>): ValidationIssue[] {
    const issues: ValidationIssue[] = []
    
    // Helper to calculate quartiles
    const getQuartiles = (values: number[]) => {
      const sorted = [...values].sort((a, b) => a - b)
      const n = sorted.length
      
      // Calculate Q1 (25th percentile)
      const q1Pos = (n - 1) * 0.25
      const q1Lower = Math.floor(q1Pos)
      const q1Upper = Math.ceil(q1Pos)
      const q1 = sorted[q1Lower]! + (sorted[q1Upper]! - sorted[q1Lower]!) * (q1Pos - q1Lower)
      
      // Calculate Q3 (75th percentile)
      const q3Pos = (n - 1) * 0.75
      const q3Lower = Math.floor(q3Pos)
      const q3Upper = Math.ceil(q3Pos)
      const q3 = sorted[q3Lower]! + (sorted[q3Upper]! - sorted[q3Lower]!) * (q3Pos - q3Lower)
      
      const iqr = q3 - q1
      return { q1, q3, iqr }
    }

    // Calculate IQR for close prices (most important)
    const closePrices = candles.map(c => c.close)
    const { q1, q3, iqr } = getQuartiles(closePrices)
    // Handle edge case where IQR is 0 (all values are the same)
    const adjustedIqr = iqr === 0 ? 1 : iqr
    const lowerBound = q1 - (config.iqrMultiplier * adjustedIqr)
    const upperBound = q3 + (config.iqrMultiplier * adjustedIqr)

    candles.forEach((candle, index) => {
      if (candle.close < lowerBound || candle.close > upperBound) {
        issues.push({
          type: 'outlier',
          severity: 'medium',
          symbol: candle.symbol,
          interval: candle.interval,
          startTime: candle.openTime,
          endTime: candle.closeTime,
          description: `Price outlier: ${candle.close} outside range [${lowerBound.toFixed(2)}, ${upperBound.toFixed(2)}]`,
          suggestedAction: 'manual_review',
          metadata: {
            price: candle.close,
            lowerBound,
            upperBound,
            candleIndex: index
          }
        })
      }
    })

    return issues
  }

  private detectOutliersIsolation(candles: Candle[], _config: Required<ValidationConfig>): ValidationIssue[] {
    // Simplified isolation approach - detect candles that are very different from their neighbors
    const issues: ValidationIssue[] = []
    const windowSize = 5 // Look at 5 candles on each side

    candles.forEach((candle, index) => {
      if (index < windowSize || index >= candles.length - windowSize) {
        return // Skip edge cases
      }

      // Get surrounding candles
      const window = candles.slice(index - windowSize, index + windowSize + 1)
      const windowWithoutCurrent = [...window.slice(0, windowSize), ...window.slice(windowSize + 1)]
      
      // Calculate average of surrounding candles
      const avgClose = windowWithoutCurrent.reduce((sum, c) => sum + c.close, 0) / windowWithoutCurrent.length
      const avgVolume = windowWithoutCurrent.reduce((sum, c) => sum + c.volume, 0) / windowWithoutCurrent.length
      
      // Check for significant deviations
      const closeDeviation = Math.abs(candle.close - avgClose) / avgClose
      const volumeDeviation = Math.abs(candle.volume - avgVolume) / avgVolume
      
      if (closeDeviation > 0.1 || volumeDeviation > 2) { // 10% price or 200% volume deviation
        issues.push({
          type: 'outlier',
          severity: closeDeviation > 0.2 ? 'high' : 'medium',
          symbol: candle.symbol,
          interval: candle.interval,
          startTime: candle.openTime,
          endTime: candle.closeTime,
          description: `Isolated anomaly: ${closeDeviation > 0.1 ? `price deviation ${(closeDeviation * 100).toFixed(1)}%` : ''} ${volumeDeviation > 2 ? `volume deviation ${(volumeDeviation * 100).toFixed(1)}%` : ''}`,
          suggestedAction: 'manual_review',
          metadata: {
            closeDeviation,
            volumeDeviation,
            candleIndex: index
          }
        })
      }
    })

    return issues
  }

  private detectDuplicates(candles: Candle[]): ValidationIssue[] {
    const issues: ValidationIssue[] = []
    const seen = new Map<string, number>()

    candles.forEach((candle, index) => {
      const key = `${candle.symbol}_${candle.interval}_${candle.timestamp}`
      const previousIndex = seen.get(key)
      
      if (previousIndex !== undefined) {
        issues.push({
          type: 'duplicate',
          severity: 'medium',
          symbol: candle.symbol,
          interval: candle.interval,
          startTime: candle.openTime,
          endTime: candle.closeTime,
          description: `Duplicate candle found at indices ${previousIndex} and ${index}`,
          suggestedAction: 'remove',
          metadata: {
            duplicateIndices: [previousIndex, index]
          }
        })
      } else {
        seen.set(key, index)
      }
    })

    return issues
  }

  private detectInvalidData(candles: Candle[]): ValidationIssue[] {
    const issues: ValidationIssue[] = []

    candles.forEach((candle, index) => {
      const problems: string[] = []

      // Check for invalid price relationships
      if (candle.high < candle.low) {
        problems.push('high < low')
      }
      if (candle.open > candle.high || candle.open < candle.low) {
        problems.push('open outside high/low range')
      }
      if (candle.close > candle.high || candle.close < candle.low) {
        problems.push('close outside high/low range')
      }

      // Check for negative values
      if (candle.open < 0 || candle.high < 0 || candle.low < 0 || candle.close < 0) {
        problems.push('negative price')
      }
      if (candle.volume < 0) {
        problems.push('negative volume')
      }

      // Check for zero prices (except volume which can be 0)
      if (candle.open === 0 || candle.high === 0 || candle.low === 0 || candle.close === 0) {
        problems.push('zero price')
      }

      // Check timestamp validity
      if (Number(candle.openTime) >= Number(candle.closeTime)) {
        problems.push('invalid time range')
      }

      if (problems.length > 0) {
        issues.push({
          type: 'invalid',
          severity: problems.length > 2 ? 'critical' : 'high',
          symbol: candle.symbol,
          interval: candle.interval,
          startTime: candle.openTime,
          endTime: candle.closeTime,
          description: `Invalid data: ${problems.join(', ')}`,
          suggestedAction: 'remove',
          metadata: {
            problems,
            candleIndex: index,
            candle: {
              open: candle.open,
              high: candle.high,
              low: candle.low,
              close: candle.close,
              volume: candle.volume
            }
          }
        })
      }
    })

    return issues
  }

  private calculateGapSeverity(missingCandles: number): 'low' | 'medium' | 'high' | 'critical' {
    if (missingCandles >= 100) return 'critical'
    if (missingCandles >= 20) return 'high'
    if (missingCandles >= 5) return 'medium'
    return 'low'
  }

  private getIntervalMilliseconds(interval: string): number {
    // Parse interval string (e.g., "1m", "5m", "1h", "1d")
    const match = /^(\d+)([mhd])$/.exec(interval)
    if (!match) {
      throw new Error(`Invalid interval format: ${interval}`)
    }

    const [, valueStr, unit] = match
    const value = parseInt(valueStr!, 10)

    switch (unit) {
      case 'm': return value * 60 * 1000
      case 'h': return value * 60 * 60 * 1000
      case 'd': return value * 24 * 60 * 60 * 1000
      default: throw new Error(`Unknown interval unit: ${unit}`)
    }
  }

  private async interpolateMissingData(
    symbol: StockSymbol,
    interval: string,
    issue: ValidationIssue
  ): Promise<boolean> {
    // In a real implementation, this would interpolate missing values
    // For now, just log the intention
    this.logger?.info('Would interpolate missing data', {
      symbol: String(symbol),
      interval,
      startTime: Number(issue.startTime),
      endTime: Number(issue.endTime)
    })
    return false
  }

  private async removeInvalidData(
    symbol: StockSymbol,
    interval: string,
    issue: ValidationIssue
  ): Promise<boolean> {
    // In a real implementation, this would remove invalid data from the repository
    // For now, just log the intention
    this.logger?.info('Would remove invalid data', {
      symbol: String(symbol),
      interval,
      startTime: Number(issue.startTime),
      endTime: Number(issue.endTime),
      metadata: issue.metadata
    })
    return false
  }
}