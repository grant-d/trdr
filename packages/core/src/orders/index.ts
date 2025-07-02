/**
 * Order Management System
 * 
 * Implements the complete order lifecycle management as specified in PRD section 3.3.
 * Includes agent consensus processing, dynamic position sizing, state management,
 * and order monitoring capabilities.
 */

export { OrderStateMachine } from './order-state-machine'
export { OrderLifecycleManager } from './order-lifecycle-manager'

// Re-export relevant types from shared
export type {
  EnhancedOrderState,
  ManagedOrder,
  AgentConsensus,
  AgentSignal,
  TimeConstraints,
  Period,
  OrderLifecycleConfig,
  PositionSizingParams,
  OrderModification,
  OrderValidationResult,
  EnhancedOrderMetadata
} from '@trdr/shared'