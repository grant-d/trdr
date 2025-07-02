import { describe, it, beforeEach, mock } from 'node:test'
import assert from 'node:assert/strict'
import { OrderStateMachine } from './order-state-machine'
import { EnhancedOrderState } from '@trdr/shared'
import type { ManagedOrder } from '@trdr/shared'
import { EventBus } from '../events/event-bus'

describe('OrderStateMachine', () => {
  let stateMachine: OrderStateMachine
  let eventBus: EventBus
  let mockOrder: ManagedOrder

  beforeEach(() => {
    eventBus = EventBus.getInstance()
    stateMachine = new OrderStateMachine(eventBus)
    
    mockOrder = {
      id: 'test-order-1',
      symbol: 'BTC-USD',
      side: 'buy',
      type: 'trailing',
      size: 1000,
      trailPercent: 2,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      status: 'pending',
      state: EnhancedOrderState.PENDING,
      filledSize: 0,
      averageFillPrice: 0,
      fees: 0,
      lastModified: new Date(),
      fills: []
    }
  })

  describe('State Transitions', () => {
    it('should allow valid transitions from PENDING', () => {
      assert.ok(stateMachine.canTransition(EnhancedOrderState.PENDING, EnhancedOrderState.SUBMITTED))
      assert.ok(stateMachine.canTransition(EnhancedOrderState.PENDING, EnhancedOrderState.CANCELLED))
      assert.ok(!stateMachine.canTransition(EnhancedOrderState.PENDING, EnhancedOrderState.FILLED))
    })

    it('should allow valid transitions from SUBMITTED', () => {
      assert.ok(stateMachine.canTransition(EnhancedOrderState.SUBMITTED, EnhancedOrderState.PARTIALLY_FILLED))
      assert.ok(stateMachine.canTransition(EnhancedOrderState.SUBMITTED, EnhancedOrderState.FILLED))
      assert.ok(stateMachine.canTransition(EnhancedOrderState.SUBMITTED, EnhancedOrderState.CANCELLED))
      assert.ok(stateMachine.canTransition(EnhancedOrderState.SUBMITTED, EnhancedOrderState.REJECTED))
      assert.ok(!stateMachine.canTransition(EnhancedOrderState.SUBMITTED, EnhancedOrderState.PENDING))
    })

    it('should not allow transitions from terminal states', () => {
      assert.ok(!stateMachine.canTransition(EnhancedOrderState.FILLED, EnhancedOrderState.CANCELLED))
      assert.ok(!stateMachine.canTransition(EnhancedOrderState.CANCELLED, EnhancedOrderState.FILLED))
      assert.ok(!stateMachine.canTransition(EnhancedOrderState.REJECTED, EnhancedOrderState.SUBMITTED))
    })

    it('should perform valid state transition', async () => {
      const eventHandler = mock.fn()
      eventBus.subscribe('order.state.changed', eventHandler)

      await stateMachine.transition(mockOrder, EnhancedOrderState.SUBMITTED)

      assert.equal(mockOrder.state, EnhancedOrderState.SUBMITTED)
      assert.ok(mockOrder.lastModified instanceof Date)
      assert.equal(eventHandler.mock.calls.length, 1)
      
      const event = eventHandler.mock.calls[0]?.arguments[0]
      assert.equal(event.orderId, 'test-order-1')
      assert.equal(event.oldState, EnhancedOrderState.PENDING)
      assert.equal(event.newState, EnhancedOrderState.SUBMITTED)
    })

    it('should throw error for invalid transition', async () => {
      await assert.rejects(
        () => stateMachine.transition(mockOrder, EnhancedOrderState.FILLED),
        /Invalid state transition: PENDING → FILLED/
      )
    })

    it('should emit specific events for important transitions', async () => {
      const submittedHandler = mock.fn()
      const filledHandler = mock.fn()
      
      eventBus.subscribe('order.submitted', submittedHandler)
      eventBus.subscribe('order.filled', filledHandler)

      // Test submitted event
      await stateMachine.transition(mockOrder, EnhancedOrderState.SUBMITTED)
      assert.equal(submittedHandler.mock.calls.length, 1)

      // Set up order to be fully filled before transitioning to FILLED
      mockOrder.filledSize = mockOrder.size
      mockOrder.fills = [{
        id: 'fill-1',
        orderId: mockOrder.id,
        price: 50000,
        size: mockOrder.size,
        timestamp: Date.now(),
        fee: 0.1,
        feeCurrency: 'USD'
      }]

      // Test filled event
      await stateMachine.transition(mockOrder, EnhancedOrderState.FILLED)
      assert.equal(filledHandler.mock.calls.length, 1)
    })

    it('should store rejection reason', async () => {
      // First update order state to SUBMITTED since mockOrder starts at PENDING
      mockOrder.state = EnhancedOrderState.SUBMITTED
      await stateMachine.transition(mockOrder, EnhancedOrderState.REJECTED, 'Insufficient funds')

      assert.equal(mockOrder.rejectionReason, 'Insufficient funds')
    })

    it('should store cancellation reason', async () => {
      await stateMachine.transition(mockOrder, EnhancedOrderState.SUBMITTED)
      await stateMachine.transition(mockOrder, EnhancedOrderState.CANCELLED, 'User requested')

      assert.equal(mockOrder.cancellationReason, 'User requested')
    })
  })

  describe('State Queries', () => {
    it('should identify terminal states', () => {
      assert.ok(stateMachine.isTerminalState(EnhancedOrderState.FILLED))
      assert.ok(stateMachine.isTerminalState(EnhancedOrderState.CANCELLED))
      assert.ok(stateMachine.isTerminalState(EnhancedOrderState.REJECTED))
      assert.ok(stateMachine.isTerminalState(EnhancedOrderState.EXPIRED))
      assert.ok(!stateMachine.isTerminalState(EnhancedOrderState.PENDING))
      assert.ok(!stateMachine.isTerminalState(EnhancedOrderState.SUBMITTED))
    })

    it('should identify active states', () => {
      assert.ok(stateMachine.isActiveState(EnhancedOrderState.PENDING))
      assert.ok(stateMachine.isActiveState(EnhancedOrderState.SUBMITTED))
      assert.ok(stateMachine.isActiveState(EnhancedOrderState.PARTIALLY_FILLED))
      assert.ok(!stateMachine.isActiveState(EnhancedOrderState.FILLED))
      assert.ok(!stateMachine.isActiveState(EnhancedOrderState.CANCELLED))
    })

    it('should get valid transitions for a state', () => {
      const pendingTransitions = stateMachine.getValidTransitions(EnhancedOrderState.PENDING)
      assert.deepEqual(pendingTransitions, [EnhancedOrderState.SUBMITTED, EnhancedOrderState.CANCELLED])

      const filledTransitions = stateMachine.getValidTransitions(EnhancedOrderState.FILLED)
      assert.deepEqual(filledTransitions, [])
    })

    it('should provide state descriptions', () => {
      assert.equal(
        stateMachine.getStateDescription(EnhancedOrderState.PENDING),
        'Order validated and ready for submission'
      )
      assert.equal(
        stateMachine.getStateDescription(EnhancedOrderState.FILLED),
        'Order completely filled'
      )
    })
  })
})