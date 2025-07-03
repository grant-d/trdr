/**
 * Grid Trading System
 * 
 * Implements grid trading functionality with dynamic spacing, 
 * trailing orders, and self-tuning parameters.
 */

export { GridManager } from './grid-manager'
export type {
  GridEvents, GridInitializationParams,
  GridInitializationResult
} from './grid-manager'

export { FileGridStateRepository, GridStatePersistence } from './grid-state-persistence'
export type {
  GridManagerSnapshot, GridPersistenceConfig, GridStateRepository, PerformanceHistoryRecord, SerializableGridLevel, SerializableGridState, StateRecoveryInfo
} from './grid-state-persistence'
export { MockGridStateRepository } from './mock-grid-state-repository'
export { VolatilityGridSpacing } from './volatility-grid-spacing'
export type {
  SpacingCalculationResult, VolatilityMetrics, VolatilitySpacingConfig
} from './volatility-grid-spacing'
