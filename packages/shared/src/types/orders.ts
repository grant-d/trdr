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
  readonly symbol: string
  /** Order direction */
  readonly side: OrderSide
  /** Order execution type */
  readonly type: OrderType
  /** Order size/quantity */
  readonly size: number
  /** Unix timestamp when order was created */
  readonly createdAt: number
  /** Unix timestamp of last update */
  readonly updatedAt: number
  /** Current order status */
  readonly status: OrderStatus
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
  readonly timestamp: number
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
  readonly symbol: string
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
  readonly symbol: string
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
  readonly timestamp: number
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
