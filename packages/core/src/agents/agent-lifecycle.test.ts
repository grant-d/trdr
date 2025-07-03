import { describe, it, beforeEach, afterEach, mock } from 'node:test'
import assert from 'node:assert/strict'
import { AgentLifecycleManager, AgentPhase } from './agent-lifecycle'
import { BaseAgent } from './base-agent'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import { epochDateNow } from '@trdr/shared'

// Mock agent for testing
class MockAgent extends BaseAgent {
  shutdownCalled = false
  resetCalled = false
  
  protected async onInitialize(): Promise<void> {
    // Mock initialization
  }
  
  protected async performAnalysis(): Promise<any> {
    return this.createSignal('hold', 0.5, 'Test signal')
  }
  
  async shutdown(): Promise<void> {
    this.shutdownCalled = true
  }
  
  async reset(): Promise<void> {
    this.resetCalled = true
  }
}

describe('AgentLifecycleManager', () => {
  let lifecycleManager: AgentLifecycleManager
  let eventBus: EventBus
  let mockAgent: MockAgent
  
  beforeEach(() => {
    eventBus = EventBus.getInstance()
    Object.values(EventTypes).forEach(type => {
      eventBus.registerEvent(type)
    })
    
    lifecycleManager = new AgentLifecycleManager(eventBus, {
      warmUpDuration: 100, // Reduce for testing
      coolDownDuration: 100,
      healthCheckInterval: 500
    })
    
    mockAgent = new MockAgent({
      id: 'test-agent',
      name: 'Test Agent',
      version: '1.0.0',
      description: 'Test agent',
      type: 'custom'
    })
  })
  
  afterEach(async () => {
    // Clean up lifecycle manager to prevent hanging timers
    await lifecycleManager.shutdown()
  })
  
  describe('agent registration', () => {
    it('should register agent successfully', () => {
      lifecycleManager.registerAgent(mockAgent)
      
      const state = lifecycleManager.getAgentState('test-agent')
      assert.ok(state)
      assert.equal(state.phase, AgentPhase.CREATED)
      assert.equal(state.errorCount, 0)
      assert.equal(state.restartCount, 0)
    })
    
    it('should prevent duplicate registration', () => {
      lifecycleManager.registerAgent(mockAgent)
      
      assert.throws(
        () => lifecycleManager.registerAgent(mockAgent),
        /already registered/
      )
    })
    
    it('should emit lifecycle event on registration', () => {
      const eventHandler = mock.fn()
      eventBus.subscribe(EventTypes.AGENT_LIFECYCLE_CHANGED, eventHandler)
      
      lifecycleManager.registerAgent(mockAgent)
      
      assert.equal(eventHandler.mock.calls.length, 1)
      const eventData = eventHandler.mock.calls[0]!.arguments[0]
      assert.equal(eventData.agentId, 'test-agent')
      assert.equal(eventData.phase, AgentPhase.CREATED)
      assert.equal(eventData.previousPhase, null)
    })
  })
  
  describe('agent initialization', () => {
    it('should initialize agent with warm-up phase', async () => {
      lifecycleManager.registerAgent(mockAgent)
      
      await lifecycleManager.initializeAgent(mockAgent)
      
      const state = lifecycleManager.getAgentState('test-agent')
      assert.equal(state?.phase, AgentPhase.READY)
    })
    
    it('should throw if agent not registered', async () => {
      await assert.rejects(
        lifecycleManager.initializeAgent(mockAgent),
        /not registered/
      )
    })
    
    it('should throw if agent already initialized', async () => {
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      
      await assert.rejects(
        lifecycleManager.initializeAgent(mockAgent),
        /already initialized/
      )
    })
    
    it('should restore configuration from persistence', async () => {
      const config = { threshold: 0.8, maxSize: 100 }
      lifecycleManager.registerAgent(mockAgent, config)
      
      await lifecycleManager.initializeAgent(mockAgent)
      
      const state = lifecycleManager.getAgentState('test-agent')
      assert.deepEqual(state?.configuration, config)
    })
  })
  
  describe('agent activation', () => {
    it('should activate ready agent', async () => {
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      
      await lifecycleManager.activateAgent('test-agent')
      
      const state = lifecycleManager.getAgentState('test-agent')
      assert.equal(state?.phase, AgentPhase.ACTIVE)
    })
    
    it('should activate paused agent', async () => {
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      await lifecycleManager.activateAgent('test-agent')
      await lifecycleManager.pauseAgent('test-agent')
      
      await lifecycleManager.activateAgent('test-agent')
      
      const state = lifecycleManager.getAgentState('test-agent')
      assert.equal(state?.phase, AgentPhase.ACTIVE)
    })
    
    it('should throw if agent in wrong phase', async () => {
      lifecycleManager.registerAgent(mockAgent)
      
      await assert.rejects(
        lifecycleManager.activateAgent('test-agent'),
        /cannot be activated/
      )
    })
  })
  
  describe('agent pause', () => {
    it('should pause active agent', async () => {
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      await lifecycleManager.activateAgent('test-agent')
      
      await lifecycleManager.pauseAgent('test-agent')
      
      const state = lifecycleManager.getAgentState('test-agent')
      assert.equal(state?.phase, AgentPhase.PAUSED)
    })
    
    it('should throw if agent not active', async () => {
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      
      await assert.rejects(
        lifecycleManager.pauseAgent('test-agent'),
        /cannot be paused/
      )
    })
  })
  
  describe('agent retirement', () => {
    it('should retire agent gracefully', async () => {
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      
      await lifecycleManager.retireAgent(mockAgent, 'Test retirement')
      
      const state = lifecycleManager.getAgentState('test-agent')
      assert.equal(state?.phase, AgentPhase.RETIRED)
      assert.equal(mockAgent.shutdownCalled, true)
    })
    
    it('should emit retirement event', async () => {
      const eventHandler = mock.fn()
      eventBus.subscribe(EventTypes.AGENT_RETIRED, eventHandler)
      
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      
      await lifecycleManager.retireAgent(mockAgent, 'Test retirement')
      
      assert.equal(eventHandler.mock.calls.length, 1)
      const eventData = eventHandler.mock.calls[0]!.arguments[0]
      assert.equal(eventData.agentId, 'test-agent')
      assert.equal(eventData.reason, 'Test retirement')
    })
    
    it('should handle already retired agent', async () => {
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      await lifecycleManager.retireAgent(mockAgent, 'First retirement')
      
      // Should not throw
      await lifecycleManager.retireAgent(mockAgent, 'Second retirement')
    })
  })
  
  describe('agent replacement', () => {
    it('should replace agent with new version', async () => {
      const oldAgent = mockAgent
      const newAgent = new MockAgent({
        id: 'test-agent-v2',
        name: 'Test Agent V2',
        version: '2.0.0',
        description: 'Test agent v2',
        type: 'custom'
      })
      
      lifecycleManager.registerAgent(oldAgent)
      await lifecycleManager.initializeAgent(oldAgent)
      await lifecycleManager.activateAgent('test-agent')
      
      await lifecycleManager.replaceAgent(oldAgent, newAgent, 'Version upgrade')
      
      const oldState = lifecycleManager.getAgentState('test-agent')
      const newState = lifecycleManager.getAgentState('test-agent-v2')
      
      assert.equal(oldState?.phase, AgentPhase.RETIRED)
      assert.equal(newState?.phase, AgentPhase.ACTIVE)
    })
    
    it('should emit replacement event', async () => {
      const eventHandler = mock.fn()
      eventBus.subscribe(EventTypes.AGENT_REPLACED, eventHandler)
      
      const oldAgent = mockAgent
      const newAgent = new MockAgent({
        id: 'test-agent-v2',
        name: 'Test Agent V2',
        version: '2.0.0',
        description: 'Test agent v2',
        type: 'custom'
      })
      
      lifecycleManager.registerAgent(oldAgent)
      await lifecycleManager.initializeAgent(oldAgent)
      
      await lifecycleManager.replaceAgent(oldAgent, newAgent, 'Version upgrade')
      
      assert.equal(eventHandler.mock.calls.length, 1)
      const eventData = eventHandler.mock.calls[0]!.arguments[0]
      assert.equal(eventData.replacement.oldAgentId, 'test-agent')
      assert.equal(eventData.replacement.newAgentId, 'test-agent-v2')
      assert.equal(eventData.replacement.reason, 'Version upgrade')
    })
  })
  
  describe('health checks', () => {
    it('should perform health check', async () => {
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      
      const health = await lifecycleManager.performHealthCheck(mockAgent)
      
      assert.equal(health.healthy, true)
      assert.ok(health.score > 0.5)
      // A fresh agent might have low win rate warning
      assert.ok(health.issues.length <= 1)
      if (health.issues.length > 0) {
        assert.ok(health.issues[0]?.includes('Low win rate'))
      }
    })
    
    it('should detect unhealthy agent', async () => {
      lifecycleManager.registerAgent(mockAgent)
      const state = lifecycleManager.getAgentState('test-agent')!
      
      // Simulate errors
      state.phase = AgentPhase.ERROR
      state.errorCount = 5
      
      const health = await lifecycleManager.performHealthCheck(mockAgent)
      
      assert.equal(health.healthy, false)
      assert.ok(health.score < 0.5)
      assert.ok(health.issues.length > 0)
    })
    
    it('should detect inactive agent', async () => {
      lifecycleManager.registerAgent(mockAgent)
      const state = lifecycleManager.getAgentState('test-agent')!
      
      // Simulate old activity
      state.lastActivityTime = epochDateNow() - 10 * 60 * 1000 // 10 minutes ago
      
      const health = await lifecycleManager.performHealthCheck(mockAgent)
      
      assert.ok(health.issues.includes('No recent activity'))
    })
  })
  
  describe('agent recovery', () => {
    it('should recover agent from error state', async () => {
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      
      // Force error state
      const state = lifecycleManager.getAgentState('test-agent')!
      state.phase = AgentPhase.ERROR
      state.errorCount = 5
      
      await lifecycleManager.recoverAgent(mockAgent)
      
      assert.equal(state.phase, AgentPhase.READY)
      assert.equal(state.errorCount, 0)
      assert.equal(state.restartCount, 1)
      assert.equal(mockAgent.resetCalled, true)
    })
    
    it('should throw if max restarts exceeded', async () => {
      lifecycleManager.registerAgent(mockAgent)
      const state = lifecycleManager.getAgentState('test-agent')!
      state.phase = AgentPhase.ERROR
      state.restartCount = 3 // Max is 3
      
      await assert.rejects(
        lifecycleManager.recoverAgent(mockAgent),
        /exceeded maximum restart attempts/
      )
    })
    
    it('should throw if agent not in error state', async () => {
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      
      await assert.rejects(
        lifecycleManager.recoverAgent(mockAgent),
        /not in error state/
      )
    })
  })
  
  describe('configuration management', () => {
    it('should update agent configuration', () => {
      lifecycleManager.registerAgent(mockAgent)
      
      const newConfig = { threshold: 0.9, maxSize: 200 }
      lifecycleManager.updateAgentConfiguration('test-agent', newConfig)
      
      const state = lifecycleManager.getAgentState('test-agent')
      assert.deepEqual(state?.configuration, newConfig)
    })
    
    it('should merge configuration updates', () => {
      const initialConfig = { threshold: 0.8, maxSize: 100 }
      lifecycleManager.registerAgent(mockAgent, initialConfig)
      
      const update = { maxSize: 200 }
      lifecycleManager.updateAgentConfiguration('test-agent', update)
      
      const state = lifecycleManager.getAgentState('test-agent')
      assert.deepEqual(state?.configuration, {
        threshold: 0.8,
        maxSize: 200
      })
    })
  })
  
  describe('lifecycle manager shutdown', () => {
    it('should shutdown cleanly', async () => {
      lifecycleManager.registerAgent(mockAgent)
      await lifecycleManager.initializeAgent(mockAgent)
      
      await lifecycleManager.shutdown()
      
      const states = lifecycleManager.getAllAgentStates()
      assert.equal(states.size, 0)
    })
  })
})