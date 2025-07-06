import type { OhlcvDto } from '../models'

/**
 * Gap type enumeration
 */
export enum GapType {
  TIME_GAP = 'time_gap',
  PRICE_GAP = 'price_gap',
  VOLUME_GAP = 'volume_gap',
  MISSING_DATA = 'missing_data'
}

/**
 * Gap information
 */
export interface Gap {
  type: GapType
  symbol: string
  startTime: number
  endTime: number
  expectedRecords?: number
  actualRecords?: number
  severity: 'low' | 'medium' | 'high'
  description: string
}

/**
 * Gap detection configuration
 */
export interface GapDetectorConfig {
  /** Expected time interval between records in milliseconds */
  expectedInterval?: number
  /** Tolerance for time gaps (multiple of expected interval) */
  timeGapTolerance?: number
  /** Minimum price gap percentage to report */
  priceGapThreshold?: number
  /** Minimum volume drop percentage to report */
  volumeDropThreshold?: number
  /** Time window for volume analysis in milliseconds */
  volumeWindow?: number
  /** Enable detection of specific gap types */
  detectTimeGaps?: boolean
  detectPriceGaps?: boolean
  detectVolumeGaps?: boolean
}

/**
 * Gap detector for identifying missing or anomalous data
 */
export class GapDetector {
  private readonly config: Required<GapDetectorConfig>
  private symbolState: Map<string, {
    lastRecord?: OhlcvDto
    volumeHistory: Array<{ time: number; volume: number }>
    gaps: Gap[]
  }> = new Map()

  constructor(config: GapDetectorConfig = {}) {
    this.config = {
      expectedInterval: config.expectedInterval ?? 60000, // 1 minute default
      timeGapTolerance: config.timeGapTolerance ?? 2,
      priceGapThreshold: config.priceGapThreshold ?? 0.05, // 5%
      volumeDropThreshold: config.volumeDropThreshold ?? 0.9, // 90% drop
      volumeWindow: config.volumeWindow ?? 3600000, // 1 hour
      detectTimeGaps: config.detectTimeGaps ?? true,
      detectPriceGaps: config.detectPriceGaps ?? true,
      detectVolumeGaps: config.detectVolumeGaps ?? true
    }
  }

  /**
   * Process a record and detect gaps
   */
  processRecord(record: OhlcvDto): Gap[] {
    const gaps: Gap[] = []
    const state = this.getOrCreateState(record.symbol)

    if (state.lastRecord) {
      // Detect time gaps
      if (this.config.detectTimeGaps) {
        const timeGap = this.detectTimeGap(state.lastRecord, record)
        if (timeGap) gaps.push(timeGap)
      }

      // Detect price gaps
      if (this.config.detectPriceGaps) {
        const priceGap = this.detectPriceGap(state.lastRecord, record)
        if (priceGap) gaps.push(priceGap)
      }
    }

    // Update volume history and detect volume anomalies
    this.updateVolumeHistory(state, record)
    if (this.config.detectVolumeGaps) {
      const volumeGap = this.detectVolumeGap(state, record)
      if (volumeGap) gaps.push(volumeGap)
    }

    // Update state
    state.lastRecord = record
    state.gaps.push(...gaps)

    return gaps
  }

  /**
   * Detect time gaps between records
   */
  private detectTimeGap(lastRecord: OhlcvDto, currentRecord: OhlcvDto): Gap | null {
    const timeDiff = currentRecord.timestamp - lastRecord.timestamp
    const expectedDiff = this.config.expectedInterval
    const tolerance = expectedDiff * this.config.timeGapTolerance

    if (timeDiff > tolerance) {
      const missedIntervals = Math.floor(timeDiff / expectedDiff) - 1
      const severity = missedIntervals > 10 ? 'high' : missedIntervals > 3 ? 'medium' : 'low'

      return {
        type: GapType.TIME_GAP,
        symbol: currentRecord.symbol,
        startTime: lastRecord.timestamp,
        endTime: currentRecord.timestamp,
        expectedRecords: missedIntervals,
        actualRecords: 0,
        severity,
        description: `Missing ${missedIntervals} expected records between ${new Date(lastRecord.timestamp).toISOString()} and ${new Date(currentRecord.timestamp).toISOString()}`
      }
    }

    return null
  }

  /**
   * Detect price gaps
   */
  private detectPriceGap(lastRecord: OhlcvDto, currentRecord: OhlcvDto): Gap | null {
    const gapPercent = Math.abs(currentRecord.open - lastRecord.close) / lastRecord.close

    if (gapPercent > this.config.priceGapThreshold) {
      const severity = gapPercent > 0.15 ? 'high' : gapPercent > 0.10 ? 'medium' : 'low'

      return {
        type: GapType.PRICE_GAP,
        symbol: currentRecord.symbol,
        startTime: lastRecord.timestamp,
        endTime: currentRecord.timestamp,
        severity,
        description: `Price gap of ${(gapPercent * 100).toFixed(2)}% detected. Previous close: ${lastRecord.close}, Current open: ${currentRecord.open}`
      }
    }

    return null
  }

  /**
   * Detect volume anomalies
   */
  private detectVolumeGap(state: ReturnType<typeof this.getOrCreateState>, record: OhlcvDto): Gap | null {
    const recentVolumes = state.volumeHistory
      .filter(v => v.time > record.timestamp - this.config.volumeWindow)
      .map(v => v.volume)

    if (recentVolumes.length < 5) {
      return null // Not enough history
    }

    const avgVolume = recentVolumes.reduce((sum, v) => sum + v, 0) / recentVolumes.length
    const volumeDrop = 1 - (record.volume / avgVolume)

    if (volumeDrop > this.config.volumeDropThreshold) {
      return {
        type: GapType.VOLUME_GAP,
        symbol: record.symbol,
        startTime: record.timestamp,
        endTime: record.timestamp,
        severity: volumeDrop > 0.95 ? 'high' : 'medium',
        description: `Volume dropped ${(volumeDrop * 100).toFixed(2)}% below average. Current: ${record.volume}, Average: ${avgVolume.toFixed(0)}`
      }
    }

    return null
  }

  /**
   * Update volume history
   */
  private updateVolumeHistory(state: ReturnType<typeof this.getOrCreateState>, record: OhlcvDto): void {
    state.volumeHistory.push({ time: record.timestamp, volume: record.volume })
    
    // Remove old entries outside the window
    const cutoff = record.timestamp - this.config.volumeWindow
    state.volumeHistory = state.volumeHistory.filter(v => v.time > cutoff)
  }

  /**
   * Get or create state for a symbol
   */
  private getOrCreateState(symbol: string) {
    let state = this.symbolState.get(symbol)
    if (!state) {
      state = {
        volumeHistory: [],
        gaps: []
      }
      this.symbolState.set(symbol, state)
    }
    return state
  }

  /**
   * Get all detected gaps
   */
  getGaps(symbol?: string): Gap[] {
    if (symbol) {
      return this.symbolState.get(symbol)?.gaps || []
    }

    const allGaps: Gap[] = []
    for (const state of this.symbolState.values()) {
      allGaps.push(...state.gaps)
    }
    return allGaps
  }

  /**
   * Get gap summary
   */
  getGapSummary(): {
    total: number
    byType: Record<GapType, number>
    bySeverity: Record<string, number>
    bySymbol: Record<string, number>
  } {
    const gaps = this.getGaps()
    const summary = {
      total: gaps.length,
      byType: {} as Record<GapType, number>,
      bySeverity: { low: 0, medium: 0, high: 0 },
      bySymbol: {} as Record<string, number>
    }

    for (const gap of gaps) {
      // By type
      summary.byType[gap.type] = (summary.byType[gap.type] || 0) + 1

      // By severity
      summary.bySeverity[gap.severity]++

      // By symbol
      summary.bySymbol[gap.symbol] = (summary.bySymbol[gap.symbol] || 0) + 1
    }

    return summary
  }

  /**
   * Clear gaps for a symbol or all symbols
   */
  clearGaps(symbol?: string): void {
    if (symbol) {
      const state = this.symbolState.get(symbol)
      if (state) {
        state.gaps = []
      }
    } else {
      for (const state of this.symbolState.values()) {
        state.gaps = []
      }
    }
  }

  /**
   * Reset detector state
   */
  reset(): void {
    this.symbolState.clear()
  }
}

/**
 * Create a gap detection transform
 */
export function createGapDetectionTransform(config?: GapDetectorConfig) {
  const detector = new GapDetector(config)

  return async function* gapDetectionTransform(
    data: AsyncIterator<OhlcvDto>
  ): AsyncGenerator<OhlcvDto & { gaps?: Gap[] }> {
    let result = await data.next()
    
    while (!result.done) {
      const record = result.value
      const gaps = detector.processRecord(record)

      // Only add gaps if there are any
      if (gaps.length > 0) {
        yield {
          ...record,
          gaps
        } as OhlcvDto & { gaps: Gap[] }
      } else {
        yield record
      }

      result = await data.next()
    }
  }
}