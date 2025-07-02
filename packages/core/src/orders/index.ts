/**
 * Order Management System
 * 
 * Implements the complete order lifecycle management as specified in PRD section 3.3.
 * Includes agent consensus processing, dynamic position sizing, state management,
 * and order monitoring capabilities.
 */

export { OrderLifecycleManager } from './order-lifecycle-manager'
export { OrderManagementIntegration } from './order-management-integration'
export type { OrderManagementIntegrationConfig } from './order-management-integration'
export { OrderStateMachine } from './order-state-machine'
export { TrailingOrderManager } from './trailing-order-manager'
export type { TrailingOrderManagerConfig, TrailingOrderParams } from './trailing-order-manager'

// Re-export relevant types from shared
export type {
  AgentConsensus,
  AgentSignal, EnhancedOrderMetadata, EnhancedOrderState,
  ManagedOrder, OrderLifecycleConfig, OrderModification,
  OrderValidationResult, Period, PositionSizingParams, TimeConstraints
} from '@trdr/shared'
