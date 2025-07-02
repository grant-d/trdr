/**
 * Event system exports
 */

// Core event bus
export { EventBus, eventBus } from './event-bus'

// Time source
export {
  TimeSource,
  RealTimeSource,
  SimulatedTimeSource,
  TimeSourceManager,
  timeSourceManager
} from './time-source'

// Event logger
export { EventLogger, eventLogger, EventLogEntry, ReplayOptions } from './event-logger'

// Types
export * from './types'

// Import EventTypes directly
import { EventTypes } from './types'
import { eventBus } from './event-bus'

// Convenience function to register all standard events
export function registerStandardEvents(): void {
  // Register all event types
  Object.values(EventTypes).forEach(eventType => {
    eventBus.registerEvent(eventType as string)
  })
}

// Initialize standard events on import
registerStandardEvents()