import type { EventHandler, EventType, EventData, EventSubscription } from './types'
import { timeSourceManager } from './time-source'

/**
 * Handler wrapper for internal use
 */
interface HandlerWrapper<T extends EventData> {
  handler: EventHandler<T>
  priority: number
  isAsync: boolean
}

/**
 * Core event bus for system-wide communication.
 * Implements pub-sub pattern with type safety.
 */
export class EventBus {
  private static instance: EventBus
  private readonly handlers = new Map<EventType, HandlerWrapper<any>[]>()
  private readonly eventTypes = new Set<EventType>()
  private subscriptionId = 0
  private readonly pendingAsyncHandlers = new Set<Promise<void>>()

  private constructor() {
    // Empty constructor
  }

  /**
   * Get singleton instance of EventBus
   */
  static getInstance(): EventBus {
    if (!EventBus.instance) {
      EventBus.instance = new EventBus()
    }
    return EventBus.instance
  }

  /**
   * Register a new event type
   */
  registerEvent(eventType: EventType): void {
    this.eventTypes.add(eventType)
  }

  /**
   * Check if an event type is registered
   */
  isEventRegistered(eventType: EventType): boolean {
    return this.eventTypes.has(eventType)
  }

  /**
   * Subscribe to an event
   */
  subscribe<T extends EventData>(
    eventType: EventType,
    handler: EventHandler<T>,
    options: { priority?: number; isAsync?: boolean } = {},
  ): EventSubscription {
    if (!this.eventTypes.has(eventType)) {
      throw new Error(`Event type '${eventType}' is not registered`)
    }

    const { priority = 0, isAsync = false } = options

    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, [])
    }

    const handlers = this.handlers.get(eventType)!
    const wrapper: HandlerWrapper<T> = {
      handler,
      priority,
      isAsync,
    }

    // Insert handler in priority order (higher priority first)
    const insertIndex = handlers.findIndex(h => h.priority < priority)
    if (insertIndex === -1) {
      handlers.push(wrapper)
    } else {
      handlers.splice(insertIndex, 0, wrapper)
    }

    const subscriptionId = ++this.subscriptionId

    // Return subscription object with unsubscribe method
    return {
      id: subscriptionId,
      eventType,
      unsubscribe: () => {
        const index = handlers.findIndex(h => h.handler === handler)
        if (index !== -1) {
          handlers.splice(index, 1)
          if (handlers.length === 0) {
            this.handlers.delete(eventType)
          }
        }
      },
    }
  }

  /**
   * Emit an event to all subscribers
   */
  emit<T extends EventData>(eventType: EventType, data: T): void {
    if (!this.eventTypes.has(eventType)) {
      throw new Error(`Event type '${eventType}' is not registered`)
    }

    const handlers = this.handlers.get(eventType)
    if (!handlers || handlers.length === 0) {
      return
    }

    // Add timestamp if not present
    const eventData = {
      ...data,
      timestamp: data.timestamp || timeSourceManager.now(),
    }

    // Execute all handlers
    handlers.forEach(wrapper => {
      if (wrapper.isAsync) {
        // Handle async handlers
        const promise = this.executeAsyncHandler(wrapper.handler, eventData, eventType)
        this.pendingAsyncHandlers.add(promise)
        promise.finally(() => {
          this.pendingAsyncHandlers.delete(promise)
        })
      } else {
        // Handle sync handlers
        try {
          wrapper.handler(eventData)
        } catch (error) {
          this.handleError(error, eventType, 'sync')
        }
      }
    })
  }

  /**
   * Execute an async handler with error handling
   */
  private async executeAsyncHandler<T extends EventData>(
    handler: EventHandler<T>,
    data: T,
    eventType: EventType,
  ): Promise<void> {
    try {
      await handler(data)
    } catch (error) {
      this.handleError(error, eventType, 'async')
    }
  }

  /**
   * Handle errors from event handlers
   */
  private handleError(error: unknown, eventType: EventType, handlerType: 'sync' | 'async'): void {
    console.error(`Error in ${handlerType} event handler for '${eventType}':`, error)

    // Emit error event if it's registered and we're not already handling an error event
    if (eventType !== 'system.error' && this.eventTypes.has('system.error')) {
      this.emit('system.error', {
        error: error instanceof Error ? error : new Error(String(error)),
        context: `Event handler for '${eventType}'`,
        severity: 'medium',
        timestamp: timeSourceManager.now(),
      })
    }
  }

  /**
   * Wait for all pending async handlers to complete
   */
  async waitForAsyncHandlers(): Promise<void> {
    await Promise.all(Array.from(this.pendingAsyncHandlers))
  }

  /**
   * Remove all handlers for an event type
   */
  removeAllHandlers(eventType?: EventType): void {
    if (eventType) {
      this.handlers.delete(eventType)
    } else {
      this.handlers.clear()
    }
  }

  /**
   * Get registered event types
   */
  getRegisteredEvents(): EventType[] {
    return Array.from(this.eventTypes)
  }

  /**
   * Get handler count for an event type
   */
  getHandlerCount(eventType: EventType): number {
    const handlers = this.handlers.get(eventType)
    return handlers ? handlers.length : 0
  }

  /**
   * Reset the event bus (useful for testing)
   */
  reset(): void {
    this.handlers.clear()
    this.eventTypes.clear()
    this.subscriptionId = 0
  }
}

// Export singleton instance
export const eventBus = EventBus.getInstance()
