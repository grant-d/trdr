import type { EventHandler, EventType, EventData, EventSubscription } from './types'
import type { EventFilter, EventFilterConfig } from './event-filter'
import { EventBus } from './event-bus'

/**
 * Enhanced subscription with filtering capabilities
 */
export interface FilteredEventSubscription extends EventSubscription {
  /** Update the filter for this subscription */
  updateFilter(filter: EventFilter<EventData>): void
  /** Get current filter status */
  hasFilter(): boolean
}

/**
 * Enhanced handler wrapper with filtering
 */
interface EnhancedHandlerWrapper<T extends EventData> {
  handler: EventHandler<T>
  priority: number
  isAsync: boolean
  filter?: EventFilter<T>
  subscriptionId: number
}

/**
 * Global event filter configuration
 */
interface GlobalFilterConfig<T extends EventData> {
  filter: EventFilter<T>
  priority: number
  name?: string
}

/**
 * Enhanced EventBus with filtering capabilities and advanced features
 */
export class EnhancedEventBus {
  private static enhancedInstance: EnhancedEventBus
  private readonly baseEventBus: EventBus
  private readonly enhancedHandlers = new Map<EventType, EnhancedHandlerWrapper<EventData>[]>()
  private readonly globalFilters = new Map<EventType, GlobalFilterConfig<EventData>[]>()
  private readonly eventMetrics = new Map<EventType, {
    emitted: number
    filtered: number
    handled: number
    errors: number
    lastEmitted?: Date
  }>()
  private isDebugging = false
  private subscriptionId = 0
  private readonly asyncPromises = new Set<Promise<void>>()

  private constructor() {
    this.baseEventBus = EventBus.getInstance()
  }

  /**
   * Get enhanced singleton instance
   */
  static getEnhancedInstance(): EnhancedEventBus {
    if (!EnhancedEventBus.enhancedInstance) {
      EnhancedEventBus.enhancedInstance = new EnhancedEventBus()
    }
    return EnhancedEventBus.enhancedInstance
  }

  /**
   * Register an event type
   */
  registerEvent(eventType: EventType): void {
    this.baseEventBus.registerEvent(eventType)
  }

  /**
   * Check if event type is registered
   */
  isEventRegistered(eventType: EventType): boolean {
    return this.baseEventBus.isEventRegistered(eventType)
  }

  /**
   * Subscribe with optional filtering
   */
  subscribeWithFilter<T extends EventData>(
    eventType: EventType,
    handler: EventHandler<T>,
    options: {
      priority?: number
      isAsync?: boolean
      filter?: EventFilter<T>
    } = {},
  ): FilteredEventSubscription {
    if (!this.isEventRegistered(eventType)) {
      throw new Error(`Event type '${eventType}' is not registered`)
    }

    const { priority = 0, isAsync = false, filter } = options

    if (!this.enhancedHandlers.has(eventType)) {
      this.enhancedHandlers.set(eventType, [])
    }

    const handlers = this.enhancedHandlers.get(eventType)!
    const subscriptionId = ++this.subscriptionId

    const wrapper: EnhancedHandlerWrapper<T> = {
      handler,
      priority,
      isAsync,
      filter,
      subscriptionId,
    }

    // Insert handler in priority order (higher priority first)
    const insertIndex = handlers.findIndex(h => h.priority < priority)
    if (insertIndex === -1) {
      handlers.push(wrapper)
    } else {
      handlers.splice(insertIndex, 0, wrapper)
    }

    // Return enhanced subscription object
    return {
      id: subscriptionId,
      eventType,
      updateFilter: (newFilter: EventFilter<EventData>) => {
        wrapper.filter = newFilter
      },
      hasFilter: () => !!wrapper.filter,
      unsubscribe: () => {
        const index = handlers.findIndex(h => h.subscriptionId === subscriptionId)
        if (index !== -1) {
          handlers.splice(index, 1)
          if (handlers.length === 0) {
            this.enhancedHandlers.delete(eventType)
          }
        }
      },
    }
  }

  /**
   * Add a global filter for an event type
   */
  addGlobalFilter<T extends EventData>(config: EventFilterConfig<T>): void {
    if (!this.globalFilters.has(config.eventType)) {
      this.globalFilters.set(config.eventType, [])
    }

    const filters = this.globalFilters.get(config.eventType)!
    const filterConfig: GlobalFilterConfig<T> = {
      filter: config.filter,
      priority: config.priority || 0,
      name: config.name,
    }

    // Insert filter in priority order (higher priority first)
    const insertIndex = filters.findIndex(f => f.priority < filterConfig.priority)
    if (insertIndex === -1) {
      filters.push(filterConfig)
    } else {
      filters.splice(insertIndex, 0, filterConfig)
    }

    this.debug(`Added global filter for ${config.eventType}`, config.name)
  }

  /**
   * Remove global filter by name
   */
  removeGlobalFilter(eventType: EventType, filterName: string): void {
    const filters = this.globalFilters.get(eventType)
    if (!filters) return

    const index = filters.findIndex(f => f.name === filterName)
    if (index !== -1) {
      filters.splice(index, 1)
      if (filters.length === 0) {
        this.globalFilters.delete(eventType)
      }
      this.debug(`Removed global filter ${filterName} for ${eventType}`)
    }
  }

  /**
   * Subscribe using base EventBus for compatibility
   */
  subscribe<T extends EventData>(
    eventType: EventType,
    handler: EventHandler<T>,
    options: { priority?: number; isAsync?: boolean } = {},
  ): EventSubscription {
    return this.baseEventBus.subscribe(eventType, handler, options)
  }

  /**
   * Enhanced emit with filtering support
   */
  emitWithFiltering<T extends EventData>(eventType: EventType, data: T): void {
    if (!this.isEventRegistered(eventType)) {
      throw new Error(`Event type '${eventType}' is not registered`)
    }

    // Update metrics
    this.updateMetrics(eventType, 'emitted')

    // Apply global filters first
    if (!this.passesGlobalFilters(eventType, data)) {
      this.updateMetrics(eventType, 'filtered')
      this.debug(`Event filtered out by global filters: ${eventType}`)
      return
    }

    // Get enhanced handlers
    const enhancedHandlers = this.enhancedHandlers.get(eventType) || []

    // Add timestamp if not present
    const eventData = {
      ...data,
      timestamp: data.timestamp || new Date(),
    }

    // Execute enhanced handlers with filtering
    enhancedHandlers.forEach(wrapper => {
      // Apply individual handler filter
      if (wrapper.filter) {
        try {
          if (!wrapper.filter(eventData)) {
            this.updateMetrics(eventType, 'filtered')
            return
          }
        } catch (error) {
          this.debug(`Error in subscription filter: ${String(error)}`)
          // Emit error event for filter failures
          this.baseEventBus.emit('system.error', {
            timestamp: new Date(),
            error: error as Error,
            context: 'EnhancedEventBus',
            severity: 'medium',
          })
          this.updateMetrics(eventType, 'filtered')
          return // Skip handler on filter error
        }
      }

      this.updateMetrics(eventType, 'handled')

      if (wrapper.isAsync) {
        this.executeAsyncHandlerSafely(wrapper.handler, eventData, eventType)
      } else {
        this.executeSyncHandlerSafely(wrapper.handler, eventData, eventType)
      }
    })

    // Also emit to base EventBus for compatibility
    this.baseEventBus.emit(eventType, eventData)
  }

  /**
   * Get event metrics
   */
  getEventMetrics(eventType?: EventType): Map<EventType, { emitted: number; filtered: number; handled: number; errors: number; lastEmitted?: Date }> | { emitted: number; filtered: number; handled: number; errors: number; lastEmitted?: Date } {
    if (eventType) {
      return this.eventMetrics.get(eventType) || {
        emitted: 0,
        filtered: 0,
        handled: 0,
        errors: 0,
      }
    }
    return new Map(this.eventMetrics)
  }

  /**
   * Reset event metrics
   */
  resetMetrics(eventType?: EventType): void {
    if (eventType) {
      this.eventMetrics.delete(eventType)
    } else {
      this.eventMetrics.clear()
    }
  }

  /**
   * Enable/disable debug logging
   */
  setDebugMode(enabled: boolean): void {
    this.isDebugging = enabled
  }

  /**
   * Get all active global filters
   */
  getGlobalFilters(): Map<EventType, GlobalFilterConfig<EventData>[]> {
    return new Map(this.globalFilters)
  }

  /**
   * Get subscription count with filters
   */
  getSubscriptionStats(): {
    total: number
    withFilters: number
    byEventType: Map<EventType, { total: number; withFilters: number }>
  } {
    let total = 0
    let withFilters = 0
    const byEventType = new Map<EventType, { total: number; withFilters: number }>()

    for (const [eventType, handlers] of this.enhancedHandlers.entries()) {
      const eventTotal = handlers.length
      const eventWithFilters = handlers.filter(h => !!h.filter).length

      total += eventTotal
      withFilters += eventWithFilters

      byEventType.set(eventType, {
        total: eventTotal,
        withFilters: eventWithFilters,
      })
    }

    return { total, withFilters, byEventType }
  }

  /**
   * Override base emit to use enhanced version
   */
  emit<T extends EventData>(eventType: EventType, data: T): void {
    this.emitWithFiltering(eventType, data)
  }

  /**
   * Check if event passes global filters
   */
  private passesGlobalFilters<T extends EventData>(eventType: EventType, data: T): boolean {
    const filters = this.globalFilters.get(eventType)
    if (!filters || filters.length === 0) {
      return true
    }

    return filters.every(filterConfig => {
      try {
        return filterConfig.filter(data)
      } catch (error) {
        this.debug(`Error in global filter ${filterConfig.name}: ${String(error)}`)
        // Emit error event for filter failures
        this.baseEventBus.emit('system.error', {
          timestamp: new Date(),
          error: error as Error,
          context: 'EnhancedEventBus',
          severity: 'medium',
        })
        return false // Filter out on error to prevent bad data propagation
      }
    })
  }

  /**
   * Execute sync handler with error tracking
   */
  private executeSyncHandlerSafely<T extends EventData>(
    handler: EventHandler<T>,
    data: T,
    eventType: EventType,
  ): void {
    try {
      handler(data)
    } catch (error) {
      this.updateMetrics(eventType, 'errors')
      this.handleErrorSafely(error, eventType, 'sync')
    }
  }

  /**
   * Execute async handler with error tracking
   */
  private executeAsyncHandlerSafely<T extends EventData>(
    handler: EventHandler<T>,
    data: T,
    eventType: EventType,
  ): void {
    const promise = this.executeAsyncHandlerInternal(handler, data, eventType)
    this.asyncPromises.add(promise)

    promise
      .catch(() => {
        this.updateMetrics(eventType, 'errors')
      })
      .finally(() => {
        this.asyncPromises.delete(promise)
      })
  }

  /**
   * Internal async handler execution
   */
  private async executeAsyncHandlerInternal<T extends EventData>(
    handler: EventHandler<T>,
    data: T,
    eventType: EventType,
  ): Promise<void> {
    try {
      await handler(data)
    } catch (error) {
      this.handleErrorSafely(error, eventType, 'async')
    }
  }

  /**
   * Safe error handling
   */
  private handleErrorSafely(error: unknown, eventType: EventType, handlerType: 'sync' | 'async'): void {
    console.error(`Error in ${handlerType} event handler for '${eventType}':`, error)

    // Emit error event if it's registered and we're not already handling an error event
    if (eventType !== 'system.error' && this.isEventRegistered('system.error')) {
      this.baseEventBus.emit('system.error', {
        error: error instanceof Error ? error : new Error(String(error)),
        context: `Event handler for '${eventType}'`,
        severity: 'medium',
        timestamp: new Date(),
      })
    }
  }

  /**
   * Update event metrics
   */
  private updateMetrics(eventType: EventType, metric: 'emitted' | 'filtered' | 'handled' | 'errors'): void {
    // Always track metrics for testing and monitoring purposes
    if (!this.eventMetrics.has(eventType)) {
      this.eventMetrics.set(eventType, {
        emitted: 0,
        filtered: 0,
        handled: 0,
        errors: 0,
      })
    }

    const metrics = this.eventMetrics.get(eventType)!
    metrics[metric]++

    if (metric === 'emitted') {
      metrics.lastEmitted = new Date()
    }
  }

  /**
   * Debug logging
   */
  private debug(message: string, ...args: unknown[]): void {
    if (this.isDebugging) {
      // eslint-disable-next-line no-console
      console.debug(`[EnhancedEventBus] ${message}`, ...args)
    }
  }

  /**
   * Get registered event types
   */
  getRegisteredEvents(): EventType[] {
    return this.baseEventBus.getRegisteredEvents()
  }

  /**
   * Wait for async handlers
   */
  async waitForAsyncHandlers(): Promise<void> {
    // Wait for both base event bus and enhanced event bus async handlers
    await Promise.all([
      this.baseEventBus.waitForAsyncHandlers(),
      Promise.all(Array.from(this.asyncPromises)),
    ])
  }

  /**
   * Enhanced reset that clears all enhanced features
   */
  reset(): void {
    this.baseEventBus.reset()
    this.enhancedHandlers.clear()
    this.globalFilters.clear()
    this.eventMetrics.clear()
    this.asyncPromises.clear()
    this.subscriptionId = 0
  }
}

// Export enhanced singleton instance
export const enhancedEventBus = EnhancedEventBus.getEnhancedInstance()
