import { EnhancedOrderState } from '@trdr/shared'
import type { ManagedOrder, Mutable } from '@trdr/shared'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'

/**
 * State machine for managing order lifecycle transitions.
 * 
 * Enforces valid state transitions and emits events for state changes.
 * Based on PRD section 3.3.2 - Order State Machine.
 */
export class OrderStateMachine {
  /**
   * Valid state transitions mapping.
   * Key: current state
   * Value: array of allowed next states
   * 
   * @remarks
   * State transition rules:
   * - CREATED → PENDING, CANCELLED
   * - PENDING → SUBMITTED, CANCELLED
   * - SUBMITTED → PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED
   * - PARTIALLY_FILLED → FILLED, CANCELLED
   * - Terminal states (FILLED, CANCELLED, REJECTED, EXPIRED) → no transitions
   */
  private readonly transitions: Record<EnhancedOrderState, EnhancedOrderState[]> = {
    [EnhancedOrderState.CREATED]: [
      EnhancedOrderState.PENDING,
      EnhancedOrderState.CANCELLED
    ],
    [EnhancedOrderState.PENDING]: [
      EnhancedOrderState.SUBMITTED,
      EnhancedOrderState.CANCELLED
    ],
    [EnhancedOrderState.SUBMITTED]: [
      EnhancedOrderState.PARTIALLY_FILLED,
      EnhancedOrderState.FILLED,
      EnhancedOrderState.CANCELLED,
      EnhancedOrderState.REJECTED
    ],
    [EnhancedOrderState.PARTIALLY_FILLED]: [
      EnhancedOrderState.FILLED,
      EnhancedOrderState.CANCELLED
    ],
    [EnhancedOrderState.FILLED]: [],
    [EnhancedOrderState.CANCELLED]: [],
    [EnhancedOrderState.REJECTED]: [],
    [EnhancedOrderState.EXPIRED]: []
  }

  private readonly eventBus: EventBus

  /**
   * Creates a new OrderStateMachine instance.
   * 
   * @param eventBus - Optional event bus instance (uses singleton if not provided)
   */
  constructor(eventBus?: EventBus) {
    this.eventBus = eventBus || EventBus.getInstance()
    this.registerEventTypes()
  }

  /**
   * Register all order-related event types
   */
  private registerEventTypes(): void {
    this.eventBus.registerEvent(EventTypes.ORDER_STATE_CHANGED)
    this.eventBus.registerEvent(EventTypes.ORDER_SUBMITTED)
    this.eventBus.registerEvent(EventTypes.ORDER_PARTIAL_FILL)
    this.eventBus.registerEvent(EventTypes.ORDER_FILLED)
    this.eventBus.registerEvent(EventTypes.ORDER_CANCELLED)
    this.eventBus.registerEvent(EventTypes.ORDER_REJECTED)
    this.eventBus.registerEvent(EventTypes.ORDER_EXPIRED)
  }

  /**
   * Check if a state transition is valid.
   * 
   * @param from - Current order state
   * @param to - Target order state
   * @returns true if transition is allowed, false otherwise
   * 
   * @example
   * ```typescript
   * if (stateMachine.canTransition(order.state, EnhancedOrderState.CANCELLED)) {
   *   await stateMachine.transition(order, EnhancedOrderState.CANCELLED)
   * }
   * ```
   */
  canTransition(from: EnhancedOrderState, to: EnhancedOrderState): boolean {
    const allowedTransitions = this.transitions[from]
    return allowedTransitions?.includes(to) ?? false
  }

  /**
   * Perform a state transition with validation and event emission.
   * 
   * @param order - Order to transition
   * @param newState - Target state
   * @param reason - Optional reason for terminal states (cancellation/rejection)
   * @throws Error if transition is invalid or pre-conditions not met
   * 
   * @remarks
   * - Validates transition before applying
   * - Checks pre-conditions for the transition
   * - Updates order.state and order.lastModified
   * - Stores reason for CANCELLED and REJECTED states
   * - Executes post-conditions after transition
   * - Emits state change events
   * 
   * @example
   * ```typescript
   * stateMachine.transition(order, EnhancedOrderState.SUBMITTED)
   * stateMachine.transition(order, EnhancedOrderState.CANCELLED, 'User requested')
   * ```
   */
  transition(
    order: ManagedOrder,
    newState: EnhancedOrderState,
    reason?: string
  ): void {
    // Validate transition is allowed
    if (!this.canTransition(order.state, newState)) {
      throw new Error(
        `Invalid state transition: ${order.state} → ${newState} for order ${order.id}`
      )
    }

    // Pre-condition checks
    this.checkPreConditions(order, newState, reason)

    const oldState = order.state
    order.state = newState
    order.lastModified = new Date()

    // Add reason for terminal states
    if (newState === EnhancedOrderState.CANCELLED && reason) {
      order.cancellationReason = reason
    } else if (newState === EnhancedOrderState.REJECTED && reason) {
      order.rejectionReason = reason
    }

    // Post-condition actions
    this.executePostConditions(order, oldState, newState)

    // Emit state change event
    this.eventBus.emit(EventTypes.ORDER_STATE_CHANGED, {
      orderId: order.id,
      oldState,
      newState,
      timestamp: new Date(),
      reason
    })

    // Emit specific events for important transitions
    switch (newState) {
      case EnhancedOrderState.SUBMITTED:
        this.eventBus.emit(EventTypes.ORDER_SUBMITTED, {
          order: { ...order },
          timestamp: new Date()
        })
        break

      case EnhancedOrderState.PARTIALLY_FILLED:
        this.eventBus.emit(EventTypes.ORDER_PARTIAL_FILL, {
          order: { ...order },
          timestamp: new Date()
        })
        break

      case EnhancedOrderState.FILLED:
        this.eventBus.emit(EventTypes.ORDER_FILLED, {
          order: { ...order },
          timestamp: new Date()
        })
        break

      case EnhancedOrderState.CANCELLED:
        this.eventBus.emit(EventTypes.ORDER_CANCELLED, {
          order: { ...order },
          reason,
          timestamp: new Date()
        })
        break

      case EnhancedOrderState.REJECTED:
        this.eventBus.emit(EventTypes.ORDER_REJECTED, {
          order: { ...order },
          reason: reason || 'Unknown rejection reason',
          timestamp: new Date()
        })
        break

      case EnhancedOrderState.EXPIRED:
        this.eventBus.emit(EventTypes.ORDER_EXPIRED, {
          order: { ...order },
          timestamp: new Date()
        })
        break
    }
  }

  /**
   * Get all valid transitions from a given state.
   * 
   * @param state - Current order state
   * @returns Array of states that can be transitioned to
   * 
   * @example
   * ```typescript
   * const transitions = stateMachine.getValidTransitions(EnhancedOrderState.PENDING)
   * // Returns: [EnhancedOrderState.SUBMITTED, EnhancedOrderState.CANCELLED]
   * ```
   */
  getValidTransitions(state: EnhancedOrderState): EnhancedOrderState[] {
    return [...(this.transitions[state] || [])]
  }

  /**
   * Check if an order is in a terminal state (no further transitions possible).
   * 
   * @param state - Order state to check
   * @returns true if state is terminal (FILLED, CANCELLED, REJECTED, EXPIRED)
   */
  isTerminalState(state: EnhancedOrderState): boolean {
    return this.transitions[state].length === 0
  }

  /**
   * Check if an order is in an active state (can still be executed).
   * 
   * @param state - Order state to check
   * @returns true if state is active (CREATED, PENDING, SUBMITTED, PARTIALLY_FILLED)
   * 
   * @remarks
   * Active states are those where the order can still be filled or modified.
   * Terminal states indicate the order has reached its final state.
   */
  isActiveState(state: EnhancedOrderState): boolean {
    return [
      EnhancedOrderState.CREATED,
      EnhancedOrderState.PENDING,
      EnhancedOrderState.SUBMITTED,
      EnhancedOrderState.PARTIALLY_FILLED
    ].includes(state)
  }

  /**
   * Get a human-readable description of the state.
   * 
   * @param state - Order state to describe
   * @returns Human-readable description of the state
   * 
   * @example
   * ```typescript
   * console.log(stateMachine.getStateDescription(EnhancedOrderState.PARTIAL))
   * // Output: "Order partially filled"
   * ```
   */
  getStateDescription(state: EnhancedOrderState): string {
    switch (state) {
      case EnhancedOrderState.CREATED:
        return 'Order created but not yet validated'
      case EnhancedOrderState.PENDING:
        return 'Order validated and ready for submission'
      case EnhancedOrderState.SUBMITTED:
        return 'Order submitted to exchange and awaiting execution'
      case EnhancedOrderState.PARTIALLY_FILLED:
        return 'Order partially filled'
      case EnhancedOrderState.FILLED:
        return 'Order completely filled'
      case EnhancedOrderState.CANCELLED:
        return 'Order cancelled before completion'
      case EnhancedOrderState.REJECTED:
        return 'Order rejected by exchange'
      case EnhancedOrderState.EXPIRED:
        return 'Order expired due to time constraints'
      default:
        return 'Unknown state'
    }
  }

  /**
   * Check pre-conditions before transitioning to a new state.
   *
   * @param order - Order being transitioned
   * @param newState - Target state
   * @param reason
   * @throws Error if pre-conditions are not met
   */
  private checkPreConditions(order: ManagedOrder, newState: EnhancedOrderState, reason?: string): void {
    switch (newState) {
      case EnhancedOrderState.PENDING:
        // Order must have basic required fields
        if (!order.symbol || !order.side || !order.size || order.size <= 0) {
          throw new Error('Order missing required fields for PENDING state')
        }
        break

      case EnhancedOrderState.SUBMITTED:
        // Order must be validated and have exchange connection
        if (order.size <= 0) {
          throw new Error('Cannot submit order with invalid size')
        }
        // In real implementation, check exchange connection
        break

      case EnhancedOrderState.PARTIALLY_FILLED:
        // Must have at least one fill
        if (!order.fills || order.fills.length === 0) {
          throw new Error('Cannot transition to PARTIALLY_FILLED without fills')
        }
        if (order.filledSize >= order.size) {
          throw new Error('Cannot be partially filled when fully filled')
        }
        break

      case EnhancedOrderState.FILLED:
        // Must be completely filled
        if (order.filledSize < order.size) {
          throw new Error('Cannot transition to FILLED when not completely filled')
        }
        break

      case EnhancedOrderState.CANCELLED:
        // Can only cancel active orders
        if (!this.isActiveState(order.state)) {
          throw new Error('Cannot cancel order in terminal state')
        }
        break

      case EnhancedOrderState.REJECTED:
        // Must have rejection reason
        if (!reason) {
          throw new Error('Rejection must include reason')
        }
        break
    }
  }

  /**
   * Execute post-conditions after a state transition.
   *
   * @param order - Order that was transitioned
   * @param _oldState
   * @param newState - New state
   */
  private executePostConditions(
    order: ManagedOrder, 
    _oldState: EnhancedOrderState, 
    newState: EnhancedOrderState
  ): void {
    // Update status field to match state (for compatibility)
    const mut = order as Mutable<ManagedOrder>
    switch (newState) {
      case EnhancedOrderState.CREATED:
      case EnhancedOrderState.PENDING:
        mut.status = 'pending'
        break
      case EnhancedOrderState.SUBMITTED:
        mut.status = 'open'
        break
      case EnhancedOrderState.PARTIALLY_FILLED:
        mut.status = 'partial'
        break
      case EnhancedOrderState.FILLED:
        mut.status = 'filled'
        break
      case EnhancedOrderState.CANCELLED:
        mut.status = 'cancelled'
        break
      case EnhancedOrderState.REJECTED:
        mut.status = 'rejected'
        break
    }
  }
}