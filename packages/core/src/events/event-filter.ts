import { toEpochDate, type EpochDate } from '@trdr/shared'
import type { EventData, EventType } from './types'

/**
 * Event filter function type
 */
export type EventFilter<T extends EventData> = (data: T) => boolean

/**
 * Event filter configuration
 */
export interface EventFilterConfig<T extends EventData> {
  /** Event type to filter */
  readonly eventType: EventType
  /** Filter function */
  readonly filter: EventFilter<T>
  /** Filter priority (higher numbers execute first) */
  readonly priority?: number
  /** Filter name for debugging */
  readonly name?: string
}

/**
 * Predefined common filters for market data events
 */
export class MarketDataFilters {
  /**
   * Filter events by symbol
   */
  static bySymbol(symbols: string[]): EventFilter<EventData & { symbol?: string }> {
    const symbolSet = new Set(symbols)
    return (data) => !data.symbol || symbolSet.has(data.symbol)
  }

  /**
   * Filter events by price range
   */
  static byPriceRange(min: number, max: number): EventFilter<EventData & { price?: number }> {
    return (data) => {
      if (typeof data.price !== 'number') return true
      return data.price >= min && data.price <= max
    }
  }

  /**
   * Filter events by volume threshold
   */
  static byVolumeThreshold(minVolume: number): EventFilter<EventData & { volume?: number }> {
    return (data) => {
      if (typeof data.volume !== 'number') return true
      return data.volume >= minVolume
    }
  }

  /**
   * Filter events by time range
   */
  static byTimeRange(startTime: Date | EpochDate, endTime: Date | EpochDate): EventFilter<EventData> {
    const start: EpochDate = startTime instanceof Date ? toEpochDate(startTime) : startTime
    const end: EpochDate = endTime instanceof Date ? toEpochDate(endTime) : endTime

    return (data) => {
      const timestamp = data.timestamp
      if (!timestamp) return true
      return timestamp >= start && timestamp <= end
    }
  }

  /**
   * Filter tick events by price change threshold
   */
  static byPriceChangeThreshold(threshold: number, lastPrices: Map<string, number>): EventFilter<EventData & { symbol?: string; price?: number }> {
    return (data) => {
      if (!data.symbol || typeof data.price !== 'number') return true

      const lastPrice = lastPrices.get(data.symbol)
      if (!lastPrice) {
        lastPrices.set(data.symbol, data.price)
        return true
      }

      const priceChange = Math.abs((data.price - lastPrice) / lastPrice)
      if (priceChange >= threshold) {
        lastPrices.set(data.symbol, data.price)
        return true
      }

      return false
    }
  }

  /**
   * Filter candle events by interval
   */
  static byInterval(intervals: string[]): EventFilter<EventData & { interval?: string }> {
    const intervalSet = new Set(intervals)
    return (data) => !data.interval || intervalSet.has(data.interval)
  }

  /**
   * Rate limit filter - only allow events at specified frequency
   */
  static rateLimit(maxEventsPerSecond: number, eventTimes: Map<string, number[]>): EventFilter<EventData & { symbol?: string }> {
    return (data) => {
      const key = data.symbol || 'global'
      const now = Date.now()

      if (!eventTimes.has(key)) {
        eventTimes.set(key, [])
      }

      const times = eventTimes.get(key)
      if (!times) {
        eventTimes.set(key, [now])
        return true
      }

      // Remove events older than 1 second
      const cutoff = now - 1000
      let i = 0
      while (i < times.length && (times[i] ?? 0) < cutoff) {
        i++
      }
      times.splice(0, i)

      // Check if we're within the rate limit
      if (times.length >= maxEventsPerSecond) {
        return false
      }

      times.push(now)
      return true
    }
  }
}

/**
 * System filters for critical events
 */
export class SystemFilters {
  /**
   * Filter system events by severity
   */
  static bySeverity(minSeverity: 'low' | 'medium' | 'high' | 'critical'): EventFilter<EventData & { severity?: 'low' | 'medium' | 'high' | 'critical' }> {
    const severityLevels = { low: 1, medium: 2, high: 3, critical: 4 }
    const minLevel = severityLevels[minSeverity]

    return (data) => {
      if (!data.severity) return true
      return severityLevels[data.severity] >= minLevel
    }
  }

  /**
   * Filter events by context
   */
  static byContext(contexts: string[]): EventFilter<EventData & { context?: string }> {
    const contextSet = new Set(contexts)
    return (data) => !data.context || contextSet.has(data.context)
  }
}

/**
 * Trading filters for order and trade events
 */
export class TradingFilters {
  /**
   * Filter by order side
   */
  static bySide(sides: Array<'buy' | 'sell'>): EventFilter<EventData & { side?: 'buy' | 'sell' }> {
    const sideSet = new Set(sides)
    return (data) => !data.side || sideSet.has(data.side)
  }

  /**
   * Filter by order size range
   */
  static bySizeRange(min: number, max: number): EventFilter<EventData & { size?: number }> {
    return (data) => {
      if (typeof data.size !== 'number') return true
      return data.size >= min && data.size <= max
    }
  }

  /**
   * Filter by order status
   */
  static byStatus(statuses: string[]): EventFilter<EventData & { status?: string }> {
    const statusSet = new Set(statuses)
    return (data) => !data.status || statusSet.has(data.status)
  }
}

/**
 * Composite filter builder for complex filtering scenarios
 */
export class FilterBuilder<T extends EventData> {
  private filters: EventFilter<T>[] = []

  /**
   * Add an AND condition
   */
  and(filter: EventFilter<T>): FilterBuilder<T> {
    this.filters.push(filter)
    return this
  }

  /**
   * Create OR condition with another filter builder
   */
  or(otherBuilder: FilterBuilder<T>): FilterBuilder<T> {
    const otherFilters = [...otherBuilder.filters]
    const currentFilters = [...this.filters]

    // Create a new filter that returns true if either set of filters passes
    this.filters = [(data: T) => {
      const currentPasses = currentFilters.length === 0 || currentFilters.every(filter => filter(data))
      const otherPasses = otherFilters.length === 0 || otherFilters.every(filter => filter(data))
      return currentPasses || otherPasses
    }]
    return this
  }

  /**
   * Create NOT condition
   */
  not(): FilterBuilder<T> {
    const currentFilters = [...this.filters]
    this.filters = [(data: T) => {
      if (currentFilters.length === 0) return false
      return !currentFilters.every(filter => filter(data))
    }]
    return this
  }

  /**
   * Build the final composite filter
   */
  build(): EventFilter<T> {
    if (this.filters.length === 0) {
      return () => true
    }

    return (data: T) => {
      return this.filters.every(filter => filter(data))
    }
  }

  /**
   * Create a new filter builder
   */
  static create<T extends EventData>(): FilterBuilder<T> {
    return new FilterBuilder<T>()
  }
}
