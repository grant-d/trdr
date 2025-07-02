import { epochDateNow, toEpochDate, type EpochDate } from '@trdr/shared'

/**
 * Time source abstraction for consistent time access.
 * Supports both real-time and simulated time for backtesting.
 */
export interface TimeSource {
  /**
   * Get current time as EpochDate (milliseconds since Unix epoch)
   */
  nowEpoch(): EpochDate

  /**
   * Get current time as Date
   */
  nowDate(): Date

  /**
   * Reset time source (for testing/backtesting)
   */
  reset?(): void
}

/**
 * Real-time source for live trading
 */
export class RealTimeSource implements TimeSource {
  nowEpoch(): EpochDate {
    return epochDateNow()
  }

  nowDate(): Date {
    return new Date()
  }
}

/**
 * Simulated time source for backtesting
 */
export class SimulatedTimeSource implements TimeSource {
  private currentTime: EpochDate
  private readonly startTime: EpochDate
  private speed = 1 // Speed multiplier

  constructor(startTime: EpochDate = epochDateNow()) {
    this.startTime = startTime
    this.currentTime = startTime
  }

  nowEpoch(): EpochDate {
    return this.currentTime
  }

  nowDate(): Date {
    return new Date(this.currentTime)
  }

  /**
   * Advance time by specified milliseconds
   */
  advance(milliseconds: number): void {
    const advance = milliseconds * this.speed
    this.currentTime = (this.currentTime + advance) as EpochDate
  }

  /**
   * Advance time to specific date
   */
  advanceTo(date: EpochDate | Date): void {
    const ms = date instanceof Date ? toEpochDate(date) : date
    if (ms < this.currentTime) {
      throw new Error('Cannot move time backwards')
    }
    this.currentTime = ms
  }

  /**
   * Set simulation speed multiplier
   */
  setSpeed(speed: number): void {
    if (speed <= 0) {
      throw new Error('Speed must be positive')
    }
    this.speed = speed
  }

  /**
   * Reset to start time
   */
  reset(): void {
    this.currentTime = this.startTime
    this.speed = 1
  }

  /**
   * Get elapsed time since start
   */
  getElapsed(): number {
    return this.currentTime - this.startTime
  }
}

/**
 * Time source manager for global time access
 */
export class TimeSourceManager {
  private static instance: TimeSourceManager
  private timeSource: TimeSource

  private constructor() {
    this.timeSource = new RealTimeSource()
  }

  static getInstance(): TimeSourceManager {
    if (!TimeSourceManager.instance) {
      TimeSourceManager.instance = new TimeSourceManager()
    }
    return TimeSourceManager.instance
  }

  /**
   * Set the time source
   */
  setTimeSource(source: TimeSource): void {
    this.timeSource = source
  }

  /**
   * Get current time source
   */
  getTimeSource(): TimeSource {
    return this.timeSource
  }

  /**
   * Get current time as EpochDate (milliseconds since Unix epoch)
   */
  nowEpoch(): EpochDate {
    return this.timeSource.nowEpoch()
  }

  /**
   * Get current time as Date
   */
  nowDate(): Date {
    return this.timeSource.nowDate()
  }

  /**
   * Reset to real-time source
   */
  useRealTime(): void {
    this.timeSource = new RealTimeSource()
  }

  /**
   * Switch to simulated time
   */
  useSimulatedTime(startTime?: EpochDate): SimulatedTimeSource {
    const simulated = new SimulatedTimeSource(startTime)
    this.timeSource = simulated
    return simulated
  }
}

// Export singleton instance
export const timeSourceManager = TimeSourceManager.getInstance()
