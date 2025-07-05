import { stdout } from 'node:process'

/**
 * Progress bar configuration
 */
export interface ProgressConfig {
  /** Total width of the progress bar */
  width?: number
  /** Character to use for completed portion */
  completeChar?: string
  /** Character to use for incomplete portion */
  incompleteChar?: string
  /** Whether to show percentage */
  showPercentage?: boolean
  /** Whether to show current/total counts */
  showCounts?: boolean
  /** Whether to show time elapsed */
  showTime?: boolean
  /** Whether to show ETA */
  showEta?: boolean
  /** Update interval in milliseconds */
  updateInterval?: number
}

/**
 * Terminal progress indicator for CLI operations
 */
export class ProgressIndicator {
  private readonly config: Required<ProgressConfig>
  private startTime: number
  private lastUpdate: number
  private currentValue: number
  private totalValue?: number
  private message?: string
  private isActive: boolean
  private lastLineLength: number

  constructor(config: ProgressConfig = {}) {
    this.config = {
      width: config.width ?? 40,
      completeChar: config.completeChar ?? '█',
      incompleteChar: config.incompleteChar ?? '░',
      showPercentage: config.showPercentage ?? true,
      showCounts: config.showCounts ?? true,
      showTime: config.showTime ?? true,
      showEta: config.showEta ?? true,
      updateInterval: config.updateInterval ?? 100,
    }

    this.startTime = Date.now()
    this.lastUpdate = 0
    this.currentValue = 0
    this.isActive = false
    this.lastLineLength = 0
  }

  /**
   * Start the progress indicator
   */
  public start(total?: number, message?: string): void {
    this.startTime = Date.now()
    this.currentValue = 0
    this.totalValue = total
    this.message = message
    this.isActive = true
    this.render()
  }

  /**
   * Update progress
   */
  public update(current: number, total?: number, message?: string): void {
    this.currentValue = current

    if (total !== undefined) {
      this.totalValue = total
    }

    if (message !== undefined) {
      this.message = message
    }

    // Throttle updates
    const now = Date.now()
    if (now - this.lastUpdate < this.config.updateInterval) {
      return
    }

    this.lastUpdate = now
    this.render()
  }

  /**
   * Complete the progress indicator
   */
  public complete(message?: string): void {
    if (this.totalValue) {
      this.currentValue = this.totalValue
    }

    if (message) {
      this.message = message
    }

    this.render()
    this.isActive = false
    stdout.write('\n')
  }

  /**
   * Stop and clear the progress indicator
   */
  public stop(): void {
    this.clear()
    this.isActive = false
  }

  /**
   * Render the progress bar
   */
  private render(): void {
    if (!this.isActive) return

    const parts: string[] = []

    // Add message if present
    if (this.message) {
      parts.push(this.message)
    }

    // Calculate percentage and bar
    if (this.totalValue && this.totalValue > 0) {
      const percentage = Math.min(100, Math.floor((this.currentValue / this.totalValue) * 100))
      const filled = Math.floor((percentage / 100) * this.config.width)
      const empty = this.config.width - filled

      // Progress bar
      const bar = this.config.completeChar.repeat(filled) +
        this.config.incompleteChar.repeat(empty)
      parts.push(`[${bar}]`)

      // Percentage
      if (this.config.showPercentage) {
        parts.push(`${percentage}%`)
      }
    } else {
      // Indeterminate progress - spinner
      const spinnerChars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
      const spinnerIndex = Math.floor((Date.now() / 100) % spinnerChars.length)
      parts.push(spinnerChars[spinnerIndex] ?? '⠋')
    }

    // Counts
    if (this.config.showCounts) {
      if (this.totalValue) {
        parts.push(`${this.currentValue}/${this.totalValue}`)
      } else {
        parts.push(`${this.currentValue}`)
      }
    }

    // Time elapsed
    if (this.config.showTime) {
      const elapsed = Date.now() - this.startTime
      parts.push(`[${this.formatTime(elapsed)}]`)
    }

    // ETA
    if (this.config.showEta && this.totalValue && this.currentValue > 0) {
      const elapsed = Date.now() - this.startTime
      const rate = this.currentValue / elapsed
      const remaining = (this.totalValue - this.currentValue) / rate
      parts.push(`ETA: ${this.formatTime(remaining)}`)
    }

    // Build and write line
    const line = parts.join(' ')
    this.clearLine()
    stdout.write(line)
    this.lastLineLength = line.length
  }

  /**
   * Clear the current line
   */
  private clear(): void {
    this.clearLine()
    this.lastLineLength = 0
  }

  /**
   * Clear the current line in the terminal
   */
  private clearLine(): void {
    stdout.write('\r' + ' '.repeat(this.lastLineLength) + '\r')
  }

  /**
   * Format time duration
   */
  private formatTime(ms: number): string {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)

    if (hours > 0) {
      const m = minutes % 60
      const s = seconds % 60
      return `${hours}h${m}m${s}s`
    } else if (minutes > 0) {
      const s = seconds % 60
      return `${minutes}m${s}s`
    } else {
      return `${seconds}s`
    }
  }
}

/**
 * Simple spinner for indeterminate progress
 */
export class Spinner {
  private readonly frames: string[]
  private frameIndex: number
  private intervalId?: NodeJS.Timeout
  private message?: string

  constructor(frames?: string[]) {
    this.frames = frames ?? ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    this.frameIndex = 0
  }

  /**
   * Start the spinner
   */
  public start(message?: string): void {
    this.message = message
    this.frameIndex = 0

    // Render first frame immediately
    this.render()

    // Then set up interval for subsequent frames
    this.intervalId = setInterval(() => {
      this.frameIndex = (this.frameIndex + 1) % this.frames.length
      this.render()
    }, 80)
  }

  /**
   * Update spinner message
   */
  public update(message: string): void {
    this.message = message
  }

  /**
   * Stop the spinner
   */
  public stop(finalMessage?: string): void {
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = undefined
    }

    stdout.write('\r' + ' '.repeat(80) + '\r')

    if (finalMessage) {
      stdout.write(finalMessage + '\n')
    }
  }

  /**
   * Render the spinner
   */
  private render(): void {
    const frame = this.frames[this.frameIndex]
    const message = this.message ? ` ${this.message}` : ''
    stdout.write(`\r${frame}${message}`)
  }
}

/**
 * Multi-progress bar for tracking multiple operations
 */
export class MultiProgress {
  private readonly indicators: Map<string, ProgressIndicator>
  private readonly order: string[]

  constructor() {
    this.indicators = new Map()
    this.order = []
  }

  /**
   * Add a new progress bar
   */
  public add(id: string, config?: ProgressConfig): ProgressIndicator {
    const indicator = new ProgressIndicator(config)
    this.indicators.set(id, indicator)

    if (!this.order.includes(id)) {
      this.order.push(id)
    }

    return indicator
  }

  /**
   * Get a progress indicator by ID
   */
  public get(id: string): ProgressIndicator | undefined {
    return this.indicators.get(id)
  }

  /**
   * Remove a progress indicator
   */
  public remove(id: string): void {
    const indicator = this.indicators.get(id)
    if (indicator) {
      indicator.stop()
      this.indicators.delete(id)
      const index = this.order.indexOf(id)
      if (index >= 0) {
        this.order.splice(index, 1)
      }
    }
  }

  /**
   * Clear all progress indicators
   */
  public clear(): void {
    for (const indicator of this.indicators.values()) {
      indicator.stop()
    }
    this.indicators.clear()
    this.order.length = 0
  }
}
