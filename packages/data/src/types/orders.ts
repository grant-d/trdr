import type { OrderBase } from '@trdr/shared'

/**
 * Flattened order interface for database storage.
 * Combines all order type fields into a single interface.
 */
export interface Order extends Omit<OrderBase, 'createdAt' | 'updatedAt'> {
  /** Creation timestamp as Date object */
  readonly createdAt: Date
  /** Update timestamp as Date object */
  readonly updatedAt: Date
  /** Optional price for limit orders */
  readonly price?: number
  /** Stop trigger price for stop orders */
  readonly stopPrice?: number
  /** Optional limit price after stop triggers */
  readonly limitPrice?: number
  /** Trail percentage for trailing orders */
  readonly trailPercent?: number
  /** Trail fixed amount for trailing orders */
  readonly trailAmount?: number
  /** Price at which trailing starts */
  readonly activationPrice?: number
  /** Best price seen for sell orders */
  readonly highWaterMark?: number
  /** Best price seen for buy orders */
  readonly lowWaterMark?: number
  /** Amount filled so far */
  readonly filledSize?: number
  /** Average price of fills */
  readonly averageFillPrice?: number
  /** Agent that created this order */
  readonly agentId?: string
  /** Additional metadata */
  readonly metadata?: Record<string, unknown>
  /** When the order was submitted to exchange */
  readonly submittedAt?: Date
  /** When the order was completely filled */
  readonly filledAt?: Date
  /** When the order was cancelled */
  readonly cancelledAt?: Date
}