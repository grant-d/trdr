import type { EpochDate } from './dates'

export type StockSymbol = string // & { readonly __brand: 'StockSymbol' }

export function toStockSymbol(symbol: string | StockSymbol): StockSymbol {
  return symbol as StockSymbol
}

/** Order direction - buy (long) or sell (short) */
export type OrderSide = 'buy' | 'sell'

/** Order lifecycle states */
export type OrderStatus = 'pending' | 'open' | 'partial' | 'filled' | 'cancelled' | 'rejected'

/** Supported order types for execution */
export type OrderType = 'market' | 'limit' | 'stop' | 'trailing'

/**
 * Base order interface containing common properties.
 * Extended by specific order types.
 */
export interface OrderBase {
  /** Unique order identifier */
  readonly id: string
  /** Trading pair symbol (e.g., 'BTC-USD') */
  readonly symbol: StockSymbol
  /** Order direction */
  readonly side: OrderSide
  /** Order execution type */
  readonly type: OrderType
  /** Order size/quantity */
  readonly size: number
  /** Unix timestamp when order was created */
  readonly createdAt: EpochDate
  /** Unix timestamp of last update */
  updatedAt: EpochDate
  /** Current order status */
  status: OrderStatus
}

/**
 * Limit order - executes at specified price or better.
 */
export interface LimitOrder extends OrderBase {
  readonly type: 'limit'
  /** Limit price for execution */
  readonly price: number
}

/**
 * Market order - executes immediately at best available price.
 */
export interface MarketOrder extends OrderBase {
  readonly type: 'market'
}

/**
 * Stop order - triggers when price reaches stop level.
 * Can be stop-market or stop-limit.
 */
export interface StopOrder extends OrderBase {
  readonly type: 'stop'
  /** Trigger price for stop activation */
  readonly stopPrice: number
  /** Optional limit price after stop triggers */
  readonly limitPrice?: number
}

/**
 * Trailing order - follows price movements with dynamic trigger.
 * Core feature of the grid trading bot.
 */
export interface TrailingOrder extends OrderBase {
  readonly type: 'trailing'
  /** Trail distance as percentage (e.g., 2 for 2%) */
  readonly trailPercent: number
  /** Trail distance as fixed amount (alternative to percent) */
  readonly trailAmount?: number
  /** Optional limit price after trigger */
  readonly limitPrice?: number
  /** Price at which trailing starts (optional) */
  readonly activationPrice?: number
  /** Best price seen for sell orders */
  readonly highWaterMark?: number
  /** Best price seen for buy orders */
  readonly lowWaterMark?: number
}

/** Union type of all supported order types */
export type Order = LimitOrder | MarketOrder | StopOrder | TrailingOrder

/**
 * Represents a partial or complete order execution.
 * Multiple fills may occur for a single order.
 */
export interface OrderFill {
  /** Unique fill identifier */
  readonly id: string
  /** Parent order ID */
  readonly orderId: string
  /** Unix timestamp of execution */
  readonly timestamp: EpochDate
  /** Execution price */
  readonly price: number
  /** Fill size/quantity */
  readonly size: number
  /** Trading fee amount */
  readonly fee: number
  /** Currency of the fee */
  readonly feeCurrency: string
}

/**
 * Represents current position in a trading pair.
 * Tracks P&L and average entry price.
 */
export interface Position {
  /** Trading pair symbol */
  readonly symbol: StockSymbol
  /** Current position size (positive=long, negative=short) */
  readonly size: number
  /** Volume-weighted average entry price */
  readonly avgEntryPrice: number
  /** Profit/loss from closed trades */
  readonly realizedPnl: number
  /** Profit/loss from open position at current price */
  readonly unrealizedPnl: number
  /** Unix timestamp of last update */
  readonly updatedAt: number
}

/**
 * Request parameters for creating a new order.
 * Used by order management system.
 */
export interface OrderRequest {
  /** Trading pair symbol */
  readonly symbol: StockSymbol
  /** Order direction */
  readonly side: OrderSide
  /** Order execution type */
  readonly type: OrderType
  /** Order size/quantity */
  readonly size: number
  /** Limit price (for limit orders) */
  readonly price?: number
  /** Stop trigger price (for stop orders) */
  readonly stopPrice?: number
  /** Limit price after stop trigger */
  readonly limitPrice?: number
  /** Trail percentage (for trailing orders) */
  readonly trailPercent?: number
  /** Trail fixed amount (for trailing orders) */
  readonly trailAmount?: number
  /** Time in force - GTC (Good Till Cancel), IOC (Immediate or Cancel), FOK (Fill or Kill) */
  readonly timeInForce?: 'GTC' | 'IOC' | 'FOK'
  /** Maker-only order flag */
  readonly postOnly?: boolean
}

/**
 * Parameters for modifying an existing order.
 * Only specified fields will be updated.
 */
export interface OrderUpdate {
  /** Order ID to update */
  readonly orderId: string
  /** New limit price */
  readonly price?: number
  /** New order size */
  readonly size?: number
  /** New stop price */
  readonly stopPrice?: number
  /** New limit price (for stop-limit) */
  readonly limitPrice?: number
}

/**
 * Result of order submission attempt.
 * Contains either successful order or error details.
 */
export interface OrderResult {
  /** Whether order submission succeeded */
  readonly success: boolean
  /** Created order if successful */
  readonly order?: Order
  /** Error message if failed */
  readonly error?: string
  /** Unix timestamp of submission attempt */
  readonly timestamp: EpochDate
}

/**
 * Internal order state tracking.
 * Maintains fill history and execution status.
 */
export interface OrderState {
  /** The order being tracked */
  readonly order: Order
  /** All fills for this order */
  readonly fills: readonly OrderFill[]
  /** Total filled quantity */
  readonly filledSize: number
  /** Volume-weighted average fill price */
  readonly avgFillPrice: number
  /** Quantity remaining to fill */
  readonly remainingSize: number
  /** Whether order is still active */
  readonly isActive: boolean
}

/**
 * Discriminated union of order lifecycle events.
 * Used for event-driven order management.
 */
export type OrderEvent =
  | { type: 'created'; order: Order }
  | { type: 'submitted'; order: Order }
  | { type: 'partial_fill'; order: Order; fill: OrderFill }
  | { type: 'filled'; order: Order; fill: OrderFill }
  | { type: 'cancelled'; order: Order; reason?: string }
  | { type: 'rejected'; order: Order; reason: string }
  | { type: 'expired'; order: Order }

/**
 * Enhanced order states for state machine management
 */
export enum EnhancedOrderState {
  CREATED = 'CREATED',
  PENDING = 'PENDING',
  SUBMITTED = 'SUBMITTED',
  PARTIALLY_FILLED = 'PARTIALLY_FILLED',
  FILLED = 'FILLED',
  CANCELLED = 'CANCELLED',
  REJECTED = 'REJECTED',
  EXPIRED = 'EXPIRED'
}

/**
 * Time constraints for order execution
 */
export interface TimeConstraints {
  /** Maximum time to keep order open (milliseconds) */
  readonly maxDuration?: number
  /** Close before market close */
  readonly closeBeforeEOD?: boolean
  /** Don't trade during these periods */
  readonly blackoutPeriods?: readonly Period[]
  /** Absolute expiration time */
  readonly expiresAt?: EpochDate
}

/**
 * Time period definition
 */
export interface Period {
  readonly start: EpochDate
  readonly end: EpochDate
}

/**
 * Order-specific agent consensus data for order decisions
 */
export interface OrderAgentConsensus {
  readonly action: OrderSide
  readonly confidence: number
  readonly expectedWinRate: number
  readonly expectedRiskReward: number
  readonly trailDistance: number
  readonly leadAgentId: string
  readonly agentSignals: readonly OrderAgentSignal[]
  readonly timeConstraints?: TimeConstraints
  readonly symbol?: string
}

/**
 * Order-specific agent signal
 */
export interface OrderAgentSignal {
  readonly agentId: string
  readonly signal: OrderSide | 'hold'
  readonly confidence: number
  readonly weight: number
  readonly reason: string
  readonly timestamp: EpochDate
}

/**
 * Enhanced order metadata for lifecycle management
 */
export interface EnhancedOrderMetadata {
  readonly consensus?: OrderAgentConsensus
  readonly agentVotes?: readonly OrderAgentSignal[]
  readonly createdBy?: string
  readonly timeConstraints?: TimeConstraints
  readonly strategy?: string
  readonly gridLevel?: number
  readonly parentOrderId?: string
}

/**
 * Execution metrics for order performance tracking
 */
export interface OrderExecutionMetrics {
  /** Time from submission to first fill (milliseconds) */
  timeToFirstFill?: number
  /** Time from submission to complete fill (milliseconds) */
  timeToComplete?: number
  /** Slippage as percentage (negative = better price) */
  slippagePercent?: number
  /** Slippage in currency units */
  slippageAmount?: number
  /** Fill rate (filled size / total size) */
  fillRate: number
  /** Number of partial fills */
  fillCount: number
  /** Timestamp when order was submitted */
  submittedAt?: EpochDate
  /** Timestamp when order was completed */
  completedAt?: EpochDate
}

/**
 * Managed order with enhanced state tracking
 */
export interface ManagedOrder extends OrderBase {
  state: EnhancedOrderState
  filledSize: number
  averageFillPrice: number
  fees: number
  lastModified: EpochDate
  exchangeOrderId?: string
  fills: OrderFill[]
  rejectionReason?: string
  cancellationReason?: string
  metadata?: EnhancedOrderMetadata
  executionMetrics?: OrderExecutionMetrics
  // Specific order type properties
  price?: number
  trailPercent?: number
  trailAmount?: number
  stopPrice?: number
  limitPrice?: number
  activationPrice?: number
  highWaterMark?: number
  lowWaterMark?: number
}

/**
 * Position sizing parameters for dynamic calculations
 */
export interface PositionSizingParams {
  readonly availableCapital: number
  readonly riskLimit: number
  readonly baseSizePercent: number
  readonly confidence: number
  readonly volatility?: number
  readonly currentExposure?: number
}

/**
 * Order modification parameters
 */
export interface OrderModification {
  readonly price?: number
  readonly size?: number
  readonly trailPercent?: number
  readonly stopPrice?: number
  readonly timeInForce?: 'GTC' | 'IOC' | 'FOK' | 'GTD'
}

/**
 * Order validation result
 */
export interface OrderValidationResult {
  readonly valid: boolean
  readonly errors: readonly string[]
  readonly warnings: readonly string[]
}

/**
 * Circuit breaker configuration for risk management
 */
export interface CircuitBreakerConfig {
  /** Max consecutive failed orders before tripping */
  readonly maxConsecutiveFailures: number
  /** Max loss in USD before tripping */
  readonly maxLossThreshold: number
  /** Time window for loss calculation (milliseconds) */
  readonly lossWindowMs: number
  /** Cool-down period after trip (milliseconds) */
  readonly cooldownPeriodMs: number
  /** Max slippage percentage before tripping */
  readonly maxSlippagePercent: number
  /** Min fill rate before considering order failed */
  readonly minFillRate: number
}

/**
 * Order splitting configuration
 */
export interface OrderSplittingConfig {
  /** Enable order splitting for large orders */
  readonly enabled: boolean
  /** Max single order size (split if larger) */
  readonly maxSingleOrderSize: number
  /** Min order size after splitting */
  readonly minSplitSize: number
  /** Time delay between split orders (milliseconds) */
  readonly splitDelayMs: number
  /** Max number of splits allowed */
  readonly maxSplits: number
}

/**
 * Configuration for order lifecycle management
 */
export interface OrderLifecycleConfig {
  readonly minOrderSize: number
  readonly maxOrderSize: number
  readonly baseSizePercent: number
  readonly maxPositionSize: number
  readonly defaultTimeInForce: 'GTC' | 'IOC' | 'FOK' | 'GTD'
  readonly enableOrderImprovement: boolean
  readonly improvementThreshold: number
  readonly maxOrderDuration: number
  readonly minConfidenceThreshold: number
  readonly circuitBreaker?: CircuitBreakerConfig
  readonly orderSplitting?: OrderSplittingConfig
  readonly symbol?: string
}
