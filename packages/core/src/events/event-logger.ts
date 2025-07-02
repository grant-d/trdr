import type { EventType, EventData } from './types'
import { eventBus } from './event-bus'
import { timeSourceManager } from './time-source'

/**
 * Event log entry
 */
export interface EventLogEntry<T extends EventData = EventData> {
  readonly id: string
  readonly type: EventType
  readonly timestamp: Date
  readonly data: T
  readonly metadata?: Record<string, unknown>
}

/**
 * Options for event replay
 */
export interface ReplayOptions {
  /** Filter events by type */
  eventTypes?: EventType[]
  /** Start time for replay */
  startTime?: Date
  /** End time for replay */
  endTime?: Date
  /** Speed multiplier for replay */
  speed?: number
  /** Whether to use original timestamps */
  preserveTimestamps?: boolean
}

/**
 * Event logger for recording and replaying events
 */
export class EventLogger {
  private events: EventLogEntry[] = []
  private isRecording = false
  private eventCounter = 0
  private readonly subscriptions = new Map<EventType, { unsubscribe: () => void }>()

  /**
   * Start recording events
   */
  startRecording(eventTypes?: EventType[]): void {
    if (this.isRecording) {
      throw new Error('Already recording')
    }

    this.isRecording = true
    const typesToRecord = eventTypes || eventBus.getRegisteredEvents()

    // Subscribe to all specified event types
    typesToRecord.forEach(eventType => {
      const subscription = eventBus.subscribe(
        eventType,
        (data: EventData) => this.logEvent(eventType, data),
        { priority: -1000 }, // Low priority to record after processing
      )
      this.subscriptions.set(eventType, subscription)
    })
  }

  /**
   * Stop recording events
   */
  stopRecording(): void {
    if (!this.isRecording) {
      return
    }

    this.isRecording = false

    // Unsubscribe from all events
    this.subscriptions.forEach(subscription => {
      subscription.unsubscribe()
    })
    this.subscriptions.clear()
  }

  /**
   * Log an event
   */
  private logEvent<T extends EventData>(eventType: EventType, data: T): void {
    const entry: EventLogEntry<T> = {
      id: `event_${++this.eventCounter}`,
      type: eventType,
      timestamp: data.timestamp || timeSourceManager.now(),
      data: { ...data }, // Create a copy to avoid mutations
    }
    this.events.push(entry)
  }

  /**
   * Get all logged events
   */
  getEvents(): EventLogEntry[] {
    return [...this.events]
  }

  /**
   * Get events filtered by criteria
   */
  getFilteredEvents(options: ReplayOptions): EventLogEntry[] {
    let filtered = [...this.events]

    if (options.eventTypes && options.eventTypes.length > 0) {
      filtered = filtered.filter(e => options.eventTypes!.includes(e.type))
    }

    if (options.startTime) {
      filtered = filtered.filter(e => e.timestamp >= options.startTime!)
    }

    if (options.endTime) {
      filtered = filtered.filter(e => e.timestamp <= options.endTime!)
    }

    return filtered
  }

  /**
   * Clear all logged events
   */
  clear(): void {
    this.events = []
    this.eventCounter = 0
  }

  /**
   * Replay logged events
   */
  async replay(options: ReplayOptions = {}): Promise<void> {
    const events = this.getFilteredEvents(options)
    if (events.length === 0) {
      return
    }

    const speed = options.speed || 1
    const preserveTimestamps = options.preserveTimestamps ?? false

    // Sort events by timestamp
    events.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())

    // If preserving timestamps, emit all events with original timestamps
    if (preserveTimestamps) {
      for (const event of events) {
        eventBus.emit(event.type, event.data)
      }
      await eventBus.waitForAsyncHandlers()
      return
    }

    // Otherwise, replay with timing
    const firstEvent = events[0]
    if (!firstEvent) return

    const startTime = firstEvent.timestamp.getTime()
    const replayStartTime = timeSourceManager.nowMs()

    for (const event of events) {
      if (!event) continue

      const eventOffset = event.timestamp.getTime() - startTime
      const targetTime = replayStartTime + (eventOffset / speed)
      const currentTime = timeSourceManager.nowMs()
      const delay = targetTime - currentTime

      if (delay > 0) {
        await this.sleep(delay)
      }

      eventBus.emit(event.type, event.data)
    }

    await eventBus.waitForAsyncHandlers()
  }

  /**
   * Export events to JSON
   */
  exportToJSON(): string {
    return JSON.stringify(this.events, null, 2)
  }

  /**
   * Import events from JSON
   */
  importFromJSON(json: string): void {
    const parsed = JSON.parse(json) as unknown
    if (!Array.isArray(parsed)) {
      throw new Error('Invalid event log format')
    }

    // Convert date strings back to Date objects
    this.events = parsed.map(entry => {
      const eventEntry = entry as { timestamp: string; [key: string]: unknown }
      return {
        ...eventEntry,
        timestamp: new Date(eventEntry.timestamp),
      } as EventLogEntry
    })

    // Update event counter
    this.eventCounter = this.events.length
  }

  /**
   * Get event statistics
   */
  getStatistics(): {
    totalEvents: number
    eventTypes: Record<EventType, number>
    timeRange: { start: Date | null; end: Date | null }
  } {
    const stats: Record<EventType, number> = {}
    let minTime: Date | null = null
    let maxTime: Date | null = null

    this.events.forEach(event => {
      stats[event.type] = (stats[event.type] || 0) + 1

      if (!minTime || event.timestamp < minTime) {
        minTime = event.timestamp
      }
      if (!maxTime || event.timestamp > maxTime) {
        maxTime = event.timestamp
      }
    })

    return {
      totalEvents: this.events.length,
      eventTypes: stats,
      timeRange: { start: minTime, end: maxTime },
    }
  }

  /**
   * Sleep for specified milliseconds
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }
}

// Export singleton instance
export const eventLogger = new EventLogger()
