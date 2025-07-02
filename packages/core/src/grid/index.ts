/**
 * Grid Trading System
 * 
 * Implements grid trading functionality with dynamic spacing, 
 * trailing orders, and self-tuning parameters.
 */

export { GridManager } from './grid-manager'
export type {
  GridInitializationParams,
  GridInitializationResult,
  GridEvents
} from './grid-manager'

export { VolatilityGridSpacing } from './volatility-grid-spacing'
export type {
  VolatilitySpacingConfig,
  VolatilityMetrics,
  SpacingCalculationResult
} from './volatility-grid-spacing'

// Re-export grid-related types from shared
export type {
  GridConfig,
  GridLevel,
  GridState
} from '@trdr/shared'