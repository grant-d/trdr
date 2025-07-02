import type { EpochDate } from '@trdr/shared'

/**
 * Base event data interface
 * Use for all event payloads. Extend for custom event types.
 */
export interface EventData {
  /** Event timestamp (epoch) */
  readonly timestamp: EpochDate
  [key: string]: unknown
}

/**
 * Event handler function type
 * Handle event data, sync or async
 */
export type EventHandler<T extends EventData = EventData> = (data: T) => void | Promise<void>

/**
 * Event subscription interface
 * Manage event subscription lifecycle
 */
export interface EventSubscription {
  /** Unique subscription ID */
  readonly id: number
  /** Event type string */
  readonly eventType: string
  /** Unsubscribe from event */
  unsubscribe(): void
}

/**
 * EventBus interface for event-driven communication
 * Implement to emit, subscribe, and manage events
 */
export interface EventBus {
  /**
   * Emit an event
   * @param event - Event type string
   * @param data - Event payload
   */
  emit(event: string, data: EventData): void
  /**
   * Subscribe to an event
   * @param event - Event type string
   * @param handler - Event handler function
   * @returns EventSubscription object
   */
  subscribe(event: string, handler: EventHandler): EventSubscription
  /**
   * Unsubscribe from an event
   * @param subscription - Subscription object
   */
  unsubscribe(subscription: EventSubscription): void
  /**
   * Register a new event type (optional)
   * @param eventType - Event type string
   */
  registerEvent?(eventType: string): void
  /**
   * Unregister an event type (optional)
   * @param eventType - Event type string
   */
  unregisterEvent?(eventType: string): void
  /**
   * Get all registered event types (optional)
   * @returns Array of event type strings
   */
  getRegisteredEvents?(): string[]
}

/**
 * Logger interface for consistent logging across packages
 * Implement for debug/info/warn/error logging
 */
export interface Logger {
  /** Log debug message */
  debug(message: string, context?: Record<string, any>): void
  /** Log info message */
  info(message: string, context?: Record<string, any>): void
  /** Log warning */
  warn(message: string, context?: Record<string, any>): void
  /** Log error */
  error(message: string, context?: Record<string, any>): void
}