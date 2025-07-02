/**
 * Event system exports
 */

// Core event bus
export { EventBus, eventBus } from './event-bus'

// Enhanced event bus and filtering
export { EnhancedEventBus, enhancedEventBus } from './enhanced-event-bus'
export * from './event-filter'
export * from './market-data-events'

// Time source
export {
  TimeSource,
  RealTimeSource,
  SimulatedTimeSource,
  TimeSourceManager,
  timeSourceManager,
} from './time-source'

// Event logger
export { EventLogger, eventLogger, EventLogEntry, ReplayOptions } from './event-logger'

// Types
export * from './types'

// Import EventTypes and enhanced types
import { EventTypes } from './types'
import { EnhancedEventTypes } from './market-data-events'
import { eventBus } from './event-bus'
import { enhancedEventBus } from './enhanced-event-bus'

// Convenience function to register all standard events
export function registerStandardEvents(): void {
  // Register all event types in both buses
  Object.values(EventTypes).forEach(eventType => {
    eventBus.registerEvent(eventType as string)
    enhancedEventBus.registerEvent(eventType as string)
  })

  // Register enhanced event types
  Object.values(EnhancedEventTypes).forEach(eventType => {
    eventBus.registerEvent(eventType as string)
    enhancedEventBus.registerEvent(eventType as string)
  })
}

// Initialize standard events on import
registerStandardEvents()
