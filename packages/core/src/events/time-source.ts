/**
 * Time source abstraction for consistent time access.
 * Supports both real-time and simulated time for backtesting.
 */
export interface TimeSource {
  /**
   * Get current time
   */
  now(): Date

  /**
   * Get current timestamp in milliseconds
   */
  nowMs(): number

  /**
   * Reset time source (for testing/backtesting)
   */
  reset?(): void
}

/**
 * Real-time source for live trading
 */
export class RealTimeSource implements TimeSource {
  now(): Date {
    return new Date()
  }

  nowMs(): number {
    return Date.now()
  }
}

/**
 * Simulated time source for backtesting
 */
export class SimulatedTimeSource implements TimeSource {
  private currentTime: Date
  private readonly startTime: Date
  private speed: number = 1 // Speed multiplier

  constructor(startTime: Date = new Date()) {
    this.startTime = startTime
    this.currentTime = new Date(startTime)
  }

  now(): Date {
    return new Date(this.currentTime)
  }

  nowMs(): number {
    return this.currentTime.getTime()
  }

  /**
   * Advance time by specified milliseconds
   */
  advance(milliseconds: number): void {
    const advance = milliseconds * this.speed
    this.currentTime = new Date(this.currentTime.getTime() + advance)
  }

  /**
   * Advance time to specific date
   */
  advanceTo(date: Date): void {
    if (date < this.currentTime) {
      throw new Error('Cannot move time backwards')
    }
    this.currentTime = new Date(date)
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
    this.currentTime = new Date(this.startTime)
    this.speed = 1
  }

  /**
   * Get elapsed time since start
   */
  getElapsed(): number {
    return this.currentTime.getTime() - this.startTime.getTime()
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
   * Convenience method to get current time
   */
  now(): Date {
    return this.timeSource.now()
  }

  /**
   * Convenience method to get current timestamp
   */
  nowMs(): number {
    return this.timeSource.nowMs()
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
  useSimulatedTime(startTime?: Date): SimulatedTimeSource {
    const simulated = new SimulatedTimeSource(startTime)
    this.timeSource = simulated
    return simulated
  }
}

// Export singleton instance
export const timeSourceManager = TimeSourceManager.getInstance()
