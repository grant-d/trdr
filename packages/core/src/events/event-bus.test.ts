import { describe, it, beforeEach, mock } from 'node:test'
import assert from 'node:assert/strict'
import { epochDateNow, toEpochDate } from '@trdr/shared'
import { eventBus, EventTypes, eventLogger, timeSourceManager } from './index'
import type { MarketDataEvent, OrderCreatedEvent } from './types'

describe('EventBus', () => {
  beforeEach(() => {
    eventBus.reset()
    eventLogger.clear()
    timeSourceManager.useRealTime()
  })

  describe('Event Registration', () => {
    it('should register event types', () => {
      eventBus.registerEvent('test.event')
      assert.equal(eventBus.isEventRegistered('test.event'), true)
    })

    it('should throw error when subscribing to unregistered event', () => {
      assert.throws(
        () => eventBus.subscribe('unregistered', () => {
        }),
        /Event type 'unregistered' is not registered/,
      )
    })
  })

  describe('Event Subscription', () => {
    it('should subscribe to events', () => {
      eventBus.registerEvent('test.event')
      const handler = mock.fn()

      const subscription = eventBus.subscribe('test.event', handler)
      eventBus.emit('test.event', { timestamp: epochDateNow() })

      assert.equal(handler.mock.calls.length, 1)

      subscription.unsubscribe()
      eventBus.emit('test.event', { timestamp: epochDateNow() })

      assert.equal(handler.mock.calls.length, 1) // Still 1, not called again
    })

    it('should handle multiple subscribers', () => {
      eventBus.registerEvent('test.event')
      const handler1 = mock.fn()
      const handler2 = mock.fn()

      eventBus.subscribe('test.event', handler1)
      eventBus.subscribe('test.event', handler2)

      eventBus.emit('test.event', { timestamp: epochDateNow() })

      assert.equal(handler1.mock.calls.length, 1)
      assert.equal(handler2.mock.calls.length, 1)
    })

    it('should respect handler priority', () => {
      eventBus.registerEvent('test.event')
      const callOrder: number[] = []

      eventBus.subscribe('test.event', () => {
        callOrder.push(3)
      }, { priority: 0 })
      eventBus.subscribe('test.event', () => {
        callOrder.push(1)
      }, { priority: 10 })
      eventBus.subscribe('test.event', () => {
        callOrder.push(2)
      }, { priority: 5 })

      eventBus.emit('test.event', { timestamp: epochDateNow() })

      assert.deepEqual(callOrder, [1, 2, 3])
    })
  })

  describe('Async Handlers', () => {
    it('should support async handlers', async () => {
      eventBus.registerEvent('test.async')
      let resolved = false

      eventBus.subscribe('test.async', async () => {
        await new Promise(resolve => setTimeout(resolve, 10))
        resolved = true
      }, { isAsync: true })

      eventBus.emit('test.async', { timestamp: epochDateNow() })
      assert.equal(resolved, false) // Not resolved immediately

      await eventBus.waitForAsyncHandlers()
      assert.equal(resolved, true) // Resolved after waiting
    })

    it('should continue on async handler errors', async () => {
      eventBus.registerEvent('test.async.error')
      eventBus.registerEvent('system.error')

      const errorHandler = mock.fn()
      eventBus.subscribe('system.error', errorHandler)

      eventBus.subscribe('test.async.error', async () => {
        throw new Error('Async error')
      }, { isAsync: true })

      eventBus.emit('test.async.error', { timestamp: epochDateNow() })
      await eventBus.waitForAsyncHandlers()

      assert.equal(errorHandler.mock.calls.length, 1)
      if (errorHandler.mock.calls[0]) {
        assert.equal(errorHandler.mock.calls[0].arguments[0].context, 'Event handler for \'test.async.error\'')
      }
    })
  })

  describe('Error Handling', () => {
    it('should continue on sync handler errors', () => {
      eventBus.registerEvent('test.error')
      const handler1 = mock.fn(() => {
        throw new Error('Handler 1 error')
      })
      const handler2 = mock.fn()

      eventBus.subscribe('test.error', handler1)
      eventBus.subscribe('test.error', handler2)

      eventBus.emit('test.error', { timestamp: epochDateNow() })

      assert.equal(handler2.mock.calls.length, 1) // Handler 2 still called
    })
  })

  describe('Standard Events', () => {
    it('should handle market data events', () => {
      eventBus.registerEvent(EventTypes.MARKET_TICK)
      const handler = mock.fn()
      eventBus.subscribe(EventTypes.MARKET_TICK, handler)

      const marketData: MarketDataEvent = {
        symbol: 'BTC-USD',
        price: 50000,
        volume: 100,
        timestamp: epochDateNow(),
      }

      eventBus.emit(EventTypes.MARKET_TICK, marketData)

      assert.equal(handler.mock.calls.length, 1)
      if (handler.mock.calls[0]) {
        assert.equal(handler.mock.calls[0].arguments[0].symbol, 'BTC-USD')
      }
    })

    it('should handle order events', () => {
      eventBus.registerEvent(EventTypes.ORDER_CREATED)
      const handler = mock.fn()
      eventBus.subscribe(EventTypes.ORDER_CREATED, handler)

      const orderEvent: OrderCreatedEvent = {
        type: 'order.created',
        orderId: 'order123',
        symbol: 'BTC-USD',
        side: 'buy',
        price: 50000,
        size: 0.1,
        status: 'pending',
        timestamp: epochDateNow(),
      }

      eventBus.emit(EventTypes.ORDER_CREATED, orderEvent)

      assert.equal(handler.mock.calls.length, 1)
      if (handler.mock.calls[0]) {
        assert.equal(handler.mock.calls[0].arguments[0].orderId, 'order123')
      }
    })
  })
})

describe('EventLogger', () => {
  beforeEach(() => {
    eventBus.reset()
    eventLogger.clear()
    eventBus.registerEvent('test.log')
  })

  it('should record events', () => {
    eventLogger.startRecording(['test.log'])

    eventBus.emit('test.log', { data: 'test1', timestamp: epochDateNow() })
    eventBus.emit('test.log', { data: 'test2', timestamp: epochDateNow() })

    eventLogger.stopRecording()

    const events = eventLogger.getEvents()
    assert.equal(events.length, 2)
    assert.equal(events[0]?.data.data, 'test1')
    assert.equal(events[1]?.data.data, 'test2')
  })

  it('should replay events', async () => {
    const handler = mock.fn()

    // Record some events
    eventLogger.startRecording(['test.log'])
    eventBus.emit('test.log', { data: 'replay1', timestamp: epochDateNow() })
    eventBus.emit('test.log', { data: 'replay2', timestamp: epochDateNow() })
    eventLogger.stopRecording()

    // Subscribe and replay
    eventBus.subscribe('test.log', handler)
    await eventLogger.replay({ preserveTimestamps: true })

    assert.equal(handler.mock.calls.length, 2)
    if (handler.mock.calls[0]) {
      assert.equal(handler.mock.calls[0].arguments[0].data, 'replay1')
    }
    if (handler.mock.calls[1]) {
      assert.equal(handler.mock.calls[1].arguments[0].data, 'replay2')
    }
  })

  it('should export and import events', () => {
    eventLogger.startRecording(['test.log'])
    eventBus.emit('test.log', { data: 'export', timestamp: epochDateNow() })
    eventLogger.stopRecording()

    const json = eventLogger.exportToJSON()
    eventLogger.clear()

    assert.equal(eventLogger.getEvents().length, 0)

    eventLogger.importFromJSON(json)
    const events = eventLogger.getEvents()

    assert.equal(events.length, 1)
    assert.equal(events[0]?.data.data, 'export')
  })
})

describe('TimeSource', () => {
  it('should use real time by default', () => {
    const before = Date.now()
    const time = timeSourceManager.nowEpoch()
    const after = Date.now()

    assert.ok(time >= before && time <= after)
  })

  it('should support simulated time', () => {
    const startTime = toEpochDate(new Date('2024-01-01T00:00:00Z'))
    const simulated = timeSourceManager.useSimulatedTime(startTime)

    assert.equal(timeSourceManager.nowEpoch(), startTime)

    simulated.advance(1000) // Advance 1 second
    assert.equal(timeSourceManager.nowEpoch(), startTime + 1000)

    simulated.setSpeed(10) // 10x speed
    simulated.advance(1000) // Advance 10 seconds
    assert.equal(timeSourceManager.nowEpoch(), startTime + 11000)
  })
})
