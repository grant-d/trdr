import type { EpochDate, StockSymbol } from '@trdr/shared'
import type { EventData } from './types'

/**
 * Enhanced market data event with metadata
 */
export interface EnhancedMarketDataEvent extends EventData {
  /** Trading symbol */
  readonly symbol: StockSymbol
  /** Data source identifier */
  readonly source: string
  /** Feed type that generated the event */
  readonly feedType: 'live' | 'paper' | 'backtest'
  /** Sequence number for ordering */
  readonly sequence?: number
  /** Event priority for processing */
  readonly priority: 'low' | 'normal' | 'high' | 'critical'
  /** Original timestamp from data source */
  readonly sourceTimestamp?: EpochDate
  /** Processing latency in milliseconds */
  readonly latency?: number
}

/**
 * Enhanced tick event with spread information
 */
export interface EnhancedTickEvent extends EnhancedMarketDataEvent {
  readonly type: 'market.tick.enhanced'
  /** Current price */
  readonly price: number
  /** Trading volume */
  readonly volume?: number
  /** Bid price */
  readonly bid?: number
  /** Ask price */
  readonly ask?: number
  /** Spread (ask - bid) */
  readonly spread?: number
  /** Price change from previous tick */
  readonly priceChange?: number
  /** Price change percentage */
  readonly priceChangePercent?: number
  /** Time since last tick in milliseconds */
  readonly timeSinceLastTick?: number
}

/**
 * Enhanced candle event with technical indicators
 */
export interface EnhancedCandleEvent extends EnhancedMarketDataEvent {
  readonly type: 'market.candle.enhanced'
  /** OHLCV data */
  readonly open: number
  readonly high: number
  readonly low: number
  readonly close: number
  readonly volume: number
  /** Candle interval */
  readonly interval: string
  /** Volume-weighted average price */
  readonly vwap?: number
  /** Number of trades in this candle */
  readonly tradeCount?: number
  /** Typical price ((high + low + close) / 3) */
  readonly typicalPrice?: number
  /** Price range (high - low) */
  readonly range: number
  /** Body size (abs(close - open)) */
  readonly bodySize: number
  /** Upper wick size */
  readonly upperWick: number
  /** Lower wick size */
  readonly lowerWick: number
  /** Candle color */
  readonly candleType: 'bullish' | 'bearish' | 'doji'
}

/**
 * Order book level data
 */
export interface OrderBookLevel {
  readonly price: number
  readonly size: number
  readonly orderCount?: number
}

/**
 * Enhanced order book event
 */
export interface EnhancedOrderBookEvent extends EnhancedMarketDataEvent {
  readonly type: 'market.orderbook.enhanced'
  /** Bid levels (sorted by price descending) */
  readonly bids: readonly OrderBookLevel[]
  /** Ask levels (sorted by price ascending) */
  readonly asks: readonly OrderBookLevel[]
  /** Best bid price */
  readonly bestBid?: number
  /** Best ask price */
  readonly bestAsk?: number
  /** Current spread */
  readonly spread?: number
  /** Mid price */
  readonly midPrice?: number
  /** Total bid volume */
  readonly totalBidVolume: number
  /** Total ask volume */
  readonly totalAskVolume: number
  /** Order book depth (number of levels) */
  readonly depth: number
  /** Book imbalance ratio */
  readonly imbalance?: number
}

/**
 * Market status event
 */
export interface MarketStatusEvent extends EnhancedMarketDataEvent {
  readonly type: 'market.status'
  /** Market status */
  readonly status: 'open' | 'closed' | 'pre-market' | 'after-hours' | 'unknown'
  /** Trading session information */
  readonly session?: {
    readonly name: string
    readonly start: Date
    readonly end: Date
  }
  /** Market statistics */
  readonly stats?: {
    readonly volume24h: number
    readonly high24h: number
    readonly low24h: number
    readonly priceChange24h: number
    readonly priceChangePercent24h: number
  }
}

/**
 * Connection status event for data feeds
 */
export interface ConnectionStatusEvent extends EventData {
  readonly type: 'market.connection'
  /** Data source identifier */
  readonly source: string
  /** Feed type */
  readonly feedType: 'live' | 'paper' | 'backtest'
  /** Connection status */
  readonly status: 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting'
  /** Connection details */
  readonly details?: {
    readonly reconnectAttempts?: number
    readonly uptime?: number
    readonly lastError?: string
    readonly subscriptions?: string[]
  }
}

/**
 * Event serialization utilities
 */
export class EventSerializer {
  /**
   * Serialize event to JSON string
   */
  static serialize(event: unknown): string {
    return JSON.stringify(event)
  }


  /**
   * Deserialize event from JSON string
   */
  static deserialize<T = unknown>(json: string): T {
    return JSON.parse(json) as T
  }

  /**
   * Create a serializable snapshot of event data
   */
  static createSnapshot(event: Record<string, unknown>): Record<string, unknown> {
    const snapshot = { ...event }

    // Ensure timestamp is serializable
    if (snapshot.timestamp instanceof Date) {
      snapshot.timestamp = snapshot.timestamp.toISOString()
    }

    // Handle nested dates
    if (snapshot.sourceTimestamp instanceof Date) {
      snapshot.sourceTimestamp = snapshot.sourceTimestamp.toISOString()
    }

    return snapshot
  }

  /**
   * Validate event structure
   */
  static validate(event: unknown): boolean {
    if (!event || typeof event !== 'object') {
      return false
    }

    const eventObj = event as Record<string, unknown>

    // Check required fields
    if (!eventObj.type || !eventObj.timestamp) {
      return false
    }

    // Check timestamp format
    if (!(eventObj.timestamp instanceof Date) &&
      typeof eventObj.timestamp !== 'string' &&
      typeof eventObj.timestamp !== 'number') {
      return false
    }

    return true
  }

  /**
   * Calculate event size in bytes (approximate)
   */
  static getEventSize(event: unknown): number {
    try {
      return new TextEncoder().encode(this.serialize(event)).length
    } catch {
      return 0
    }
  }
}

/**
 * Event compression utilities for high-frequency data
 */
export class EventCompressor {
  /**
   * Compress multiple tick events into a summary
   */
  static compressTickEvents(ticks: EnhancedTickEvent[]): EnhancedTickEvent | null {
    if (ticks.length === 0) return null
    if (ticks.length === 1) return ticks[0] ?? null

    const first = ticks[0]
    const last = ticks[ticks.length - 1]

    if (!first || !last) return null

    const totalVolume = ticks.reduce((sum, tick) => sum + (tick.volume || 0), 0)

    return {
      ...last,
      type: 'market.tick.enhanced',
      price: last.price,
      volume: totalVolume,
      priceChange: last.price - first.price,
      priceChangePercent: first.price > 0 ? ((last.price - first.price) / first.price) * 100 : 0,
      timeSinceLastTick: last.timestamp - first.timestamp,
      sequence: last.sequence,
      priority: 'normal',
      source: 'compressed',
    } as EnhancedTickEvent
  }

  /**
   * Detect if events can be compressed based on time window
   */
  static canCompress(events: EnhancedMarketDataEvent[], windowMs = 1000): boolean {
    if (events.length < 2) return false

    const lastEvent = events[events.length - 1]
    const firstEvent = events[0]
    if (!lastEvent || !firstEvent) return false

    const timeSpan = lastEvent.timestamp - firstEvent.timestamp
    return timeSpan <= windowMs
  }
}

/**
 * Event priority classifier
 */
export class EventPriorityClassifier {
  /**
   * Classify event priority based on content
   */
  static classifyPriority(event: EnhancedMarketDataEvent): 'low' | 'normal' | 'high' | 'critical' {
    // Market status changes are critical
    if (event.type === 'market.status') {
      return 'critical'
    }

    // Connection events are high priority
    if (event.type === 'market.connection') {
      return 'high'
    }

    // Large price movements are high priority
    if ('priceChangePercent' in event && typeof event.priceChangePercent === 'number' && Math.abs(event.priceChangePercent) > 5) {
      return 'high'
    }

    // Large volume spikes are high priority
    if ('volume' in event && typeof event.volume === 'number' && event.volume > 10000) {
      return 'high'
    }

    // Wide spreads indicate low liquidity - high priority
    if ('spread' in event && 'price' in event &&
      typeof event.spread === 'number' && typeof event.price === 'number' && event.price > 0) {
      const spreadBps = (event.spread / event.price) * 10000
      if (spreadBps > 100) { // > 1% spread
        return 'high'
      }
    }

    return 'normal'
  }
}

// Enhanced event types registry
export const EnhancedEventTypes = {
  // Enhanced market data
  MARKET_TICK_ENHANCED: 'market.tick.enhanced',
  MARKET_CANDLE_ENHANCED: 'market.candle.enhanced',
  MARKET_ORDERBOOK_ENHANCED: 'market.orderbook.enhanced',
  MARKET_STATUS: 'market.status',
  MARKET_CONNECTION: 'market.connection',

  // Aggregated events
  MARKET_SUMMARY: 'market.summary',
  MARKET_STATISTICS: 'market.statistics',
} as const
