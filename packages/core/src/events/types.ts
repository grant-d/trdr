import type { EpochDate, StockSymbol } from '@trdr/shared'

/**
 * Base event type
 */
export type EventType = string

/**
 * Base event data interface
 */
export interface EventData {
  readonly timestamp: EpochDate
  [key: string]: unknown
}

/**
 * Event handler function type
 */
export type EventHandler<T extends EventData> = (data: T) => void | Promise<void>

/**
 * Event subscription returned when subscribing
 */
export interface EventSubscription {
  readonly id: number
  readonly eventType: EventType
  unsubscribe(): void
}

/**
 * Base event interface
 */
export interface BaseEvent<T extends EventData = EventData> {
  readonly id: string
  readonly type: EventType
  readonly timestamp: EpochDate
  readonly data: T
}

// Market data events
export interface MarketDataEvent extends EventData {
  readonly symbol: StockSymbol
  readonly price: number
  readonly volume: number
}

export interface CandleEvent extends EventData {
  readonly symbol: StockSymbol
  readonly open: number
  readonly high: number
  readonly low: number
  readonly close: number
  readonly volume: number
  readonly interval: string
}

// Order events
export interface OrderEvent extends EventData {
  readonly orderId: string
  readonly symbol: StockSymbol
  readonly side: 'buy' | 'sell'
  readonly price: number
  readonly size: number
  readonly status: string
}

export interface OrderCreatedEvent extends OrderEvent {
  readonly type: 'order.created'
}

export interface OrderFilledEvent extends OrderEvent {
  readonly type: 'order.filled'
  readonly fillPrice: number
  readonly fillSize: number
}

export interface OrderCancelledEvent extends OrderEvent {
  readonly type: 'order.cancelled'
  readonly reason?: string
}

// Trade events
export interface TradeEvent extends EventData {
  readonly tradeId: string
  readonly orderId: string
  readonly symbol: StockSymbol
  readonly side: 'buy' | 'sell'
  readonly price: number
  readonly size: number
  readonly pnl?: number
  readonly fees: number
}

// Agent events
export interface AgentSignalEvent extends EventData {
  readonly agentId: string
  readonly agentType: string
  readonly action: 'TRAIL_BUY' | 'TRAIL_SELL' | 'HOLD'
  readonly confidence: number
  readonly trailDistance: number
  readonly reasoning: Record<string, unknown>
}

export interface AgentConsensusEvent extends EventData {
  readonly decision: 'buy' | 'sell' | 'hold'
  readonly confidence: number
  readonly dissent: number
  readonly votes: Array<{
    agentId: string
    action: string
    confidence: number
  }>
}

// System events
export interface SystemStartEvent extends EventData {
  readonly mode: 'live' | 'paper' | 'backtest'
  readonly config: Record<string, unknown>
}

export interface SystemStopEvent extends EventData {
  readonly reason: string
  readonly graceful: boolean
}

export interface ErrorEvent extends EventData {
  readonly error: Error
  readonly context: string
  readonly severity: 'low' | 'medium' | 'high' | 'critical'
}

export interface SystemInfoEvent extends EventData {
  readonly message: string
  readonly context: string
  readonly details?: Record<string, unknown>
}

// Event type constants
export const EventTypes = {
  // Market data
  MARKET_TICK: 'market.tick',
  MARKET_CANDLE: 'market.candle',
  MARKET_ORDERBOOK: 'market.orderbook',

  // Orders
  ORDER_CREATED: 'order.created',
  ORDER_SUBMITTED: 'order.submitted',
  ORDER_FILLED: 'order.filled',
  ORDER_PARTIAL_FILL: 'order.partial.fill',
  ORDER_CANCELLED: 'order.cancelled',
  ORDER_REJECTED: 'order.rejected',
  ORDER_EXPIRED: 'order.expired',
  ORDER_STATE_CHANGED: 'order.state.changed',
  ORDER_CONSENSUS_REJECTED: 'order.consensus.rejected',
  ORDER_SIZE_TOO_SMALL: 'order.size.too.small',
  ORDER_VALIDATION_FAILED: 'order.validation.failed',
  ORDER_CANCEL_FAILED: 'order.cancel.failed',
  ORDER_MODIFIED: 'order.modified',
  ORDER_MODIFICATION_FAILED: 'order.modification.failed',
  ORDER_IMPROVED: 'order.improved',
  ORDER_IMPROVEMENT_FAILED: 'order.improvement.failed',
  ORDER_TIME_CONSTRAINT_ERROR: 'order.time.constraint.error',
  ORDER_FILL: 'order.fill',

  // Trades
  TRADE_EXECUTED: 'trade.executed',
  TRADE_CLOSED: 'trade.closed',

  // System
  SYSTEM_START: 'system.start',
  SYSTEM_STOP: 'system.stop',
  SYSTEM_ERROR: 'system.error',
  SYSTEM_WARNING: 'system.warning',
  SYSTEM_INFO: 'system.info',

  // Grid
  GRID_LEVEL_ACTIVATED: 'grid.level_activated',
  GRID_LEVEL_FILLED: 'grid.level_filled',
  GRID_REBALANCE: 'grid.rebalance',

  // Monitoring
  ORDER_EXECUTION_METRICS: 'order.execution.metrics',
  CIRCUIT_BREAKER_TRIPPED: 'circuit.breaker.tripped',
  CIRCUIT_BREAKER_RESET: 'circuit.breaker.reset',
  ORDER_EXECUTION_POOR: 'order.execution.poor',

  // Consensus
  CONSENSUS_STARTED: 'consensus.started',
  CONSENSUS_SIGNAL_RECEIVED: 'consensus.signal.received',
  CONSENSUS_TIMEOUT: 'consensus.timeout',
  CONSENSUS_COMPLETED: 'consensus.completed',
  CONSENSUS_FAILED: 'consensus.failed',
  SIGNAL_REQUEST: 'signal.request',
  
  // Market data events
  CANDLE: 'candle',
  TICK: 'tick',
  
  // Connection events
  CONNECTION_STATUS: 'connection.status',
  
  // Historical data events
  HISTORICAL_DATA_SAVED: 'historical.data.saved',
  HISTORICAL_DATA_VALIDATED: 'historical.data.validated',
  BACKFILL_COMPLETED: 'backfill.completed',
  BACKFILL_FAILED: 'backfill.failed',
  
  // Grid events
  GRID_CREATED: 'grid.created',
  GRID_UPDATED: 'grid.updated',
  GRID_CANCELLED: 'grid.cancelled',
  
  // Agent events
  AGENT_SIGNAL: 'agent.signal',
  AGENT_CONSENSUS: 'agent.consensus',
  AGENT_ERROR: 'agent.error',
  AGENT_REGISTERED: 'agent.registered',
  AGENT_UNREGISTERED: 'agent.unregistered',
  AGENT_LIFECYCLE_CHANGED: 'agent.lifecycle.changed',
  AGENT_REPLACED: 'agent.replaced',
  AGENT_RETIRED: 'agent.retired',
  AGENT_PERFORMANCE_UPDATED: 'agent.performance.updated',
  CONSENSUS_REACHED: 'consensus.reached',
} as const

export type EventTypeKeys = keyof typeof EventTypes
export type EventTypeValues = (typeof EventTypes)[EventTypeKeys]
