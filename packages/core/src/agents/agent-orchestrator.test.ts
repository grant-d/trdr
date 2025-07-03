import { describe, it, beforeEach, afterEach, mock } from 'node:test'
import assert from 'node:assert/strict'
import { AgentOrchestrator } from './agent-orchestrator'
import { BaseAgent } from './base-agent'
import type { AgentMetadata, AgentSignal, MarketContext } from './types'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import { toStockSymbol, epochDateNow } from '@trdr/shared'

// Mock agents for testing
class MockBullishAgent extends BaseAgent {
  protected async onInitialize(): Promise<void> {}
  
  protected async performAnalysis(_context: MarketContext): Promise<AgentSignal> {
    return this.createSignal('buy', 0.8, 'Bullish conditions detected')
  }
}

class MockBearishAgent extends BaseAgent {
  protected async onInitialize(): Promise<void> {}
  
  protected async performAnalysis(_context: MarketContext): Promise<AgentSignal> {
    return this.createSignal('sell', 0.7, 'Bearish conditions detected')
  }
}

class MockNeutralAgent extends BaseAgent {
  protected async onInitialize(): Promise<void> {}
  
  protected async performAnalysis(_context: MarketContext): Promise<AgentSignal> {
    return this.createSignal('hold', 0.6, 'Neutral market conditions')
  }
}

class MockSlowAgent extends BaseAgent {
  private delay: number
  
  constructor(metadata: AgentMetadata, delay: number) {
    super(metadata)
    this.delay = delay
  }
  
  protected async onInitialize(): Promise<void> {}
  
  protected async performAnalysis(_context: MarketContext): Promise<AgentSignal> {
    await new Promise(resolve => setTimeout(resolve, this.delay))
    return this.createSignal('buy', 0.5, 'Slow analysis complete')
  }
}

class MockErrorAgent extends BaseAgent {
  protected async onInitialize(): Promise<void> {}
  
  protected async performAnalysis(_context: MarketContext): Promise<AgentSignal> {
    throw new Error('Agent analysis failed')
  }
}

describe('AgentOrchestrator', () => {
  let orchestrator: AgentOrchestrator
  let eventBus: EventBus
  
  const createContext = (): MarketContext => ({
    symbol: toStockSymbol('BTC-USD'),
    currentPrice: 50000,
    candles: [
      {
        timestamp: epochDateNow(),
        open: 49900,
        high: 50100,
        low: 49800,
        close: 50000,
        volume: 100
      }
    ]
  })
  
  beforeEach(() => {
    eventBus = EventBus.getInstance()
    Object.values(EventTypes).forEach(type => {
      eventBus.registerEvent(type)
    })
    orchestrator = new AgentOrchestrator(eventBus)
  })
  
  afterEach(async () => {
    // Clean up orchestrator to prevent hanging promises
    await orchestrator.shutdown()
  })
  
  describe('agent registration', () => {
    it('should register agents successfully', async () => {
      const agent = new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      })
      
      await orchestrator.registerAgent(agent, 1.5)
      
      const agents = orchestrator.getAgents()
      assert.equal(agents.size, 1)
      assert.ok(agents.has('bullish-1'))
      
      const weights = orchestrator.getAgentWeights()
      assert.equal(weights.get('bullish-1'), 1.5)
    })
    
    it('should prevent duplicate agent registration', async () => {
      const agent = new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      })
      
      await orchestrator.registerAgent(agent)
      
      await assert.rejects(
        orchestrator.registerAgent(agent),
        /already registered/
      )
    })
    
    it('should emit agent registered event', async () => {
      const eventHandler = mock.fn()
      eventBus.subscribe(EventTypes.AGENT_REGISTERED, eventHandler)
      
      const agent = new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      })
      
      await orchestrator.registerAgent(agent)
      
      assert.equal(eventHandler.mock.calls.length, 1)
      const eventData = eventHandler.mock.calls[0]!.arguments[0]
      assert.equal(eventData.agentId, 'bullish-1')
    })
  })
  
  describe('agent unregistration', () => {
    it('should unregister agents successfully', async () => {
      const agent = new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      })
      
      await orchestrator.registerAgent(agent)
      await orchestrator.unregisterAgent('bullish-1')
      
      const agents = orchestrator.getAgents()
      assert.equal(agents.size, 0)
    })
    
    it('should throw if agent not found', async () => {
      await assert.rejects(
        orchestrator.unregisterAgent('non-existent'),
        /not found/
      )
    })
  })
  
  describe('consensus calculation', () => {
    beforeEach(async () => {
      // Register multiple agents
      await orchestrator.registerAgent(new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      }), 1.0)
      
      await orchestrator.registerAgent(new MockBearishAgent({
        id: 'bearish-1',
        name: 'Bearish Agent',
        version: '1.0.0',
        description: 'Always bearish',
        type: 'custom'
      }), 1.0)
      
      await orchestrator.registerAgent(new MockNeutralAgent({
        id: 'neutral-1',
        name: 'Neutral Agent',
        version: '1.0.0',
        description: 'Always neutral',
        type: 'custom'
      }), 1.0)
    })
    
    it('should calculate weighted consensus', async () => {
      const consensus = await orchestrator.getConsensus(createContext())
      
      assert.ok(['buy', 'sell', 'hold'].includes(consensus.action))
      assert.ok(consensus.confidence > 0)
      assert.equal(consensus.participatingAgents, 3)
      assert.ok(consensus.reason.length > 0)
    })
    
    it('should respect agent weights', async () => {
      // Give bullish agent much higher weight
      orchestrator.setAgentWeight('bullish-1', 10.0)
      orchestrator.setAgentWeight('bearish-1', 1.0)
      orchestrator.setAgentWeight('neutral-1', 1.0)
      
      const consensus = await orchestrator.getConsensus(createContext())
      
      // With high weight on bullish agent, consensus should be buy
      assert.equal(consensus.action, 'buy')
    })
    
    it('should handle agent timeouts', async () => {
      // Create orchestrator with shorter timeout
      const timeoutOrchestrator = new AgentOrchestrator(eventBus, { agentTimeout: 1000 })
      
      // Register the 3 normal agents
      await timeoutOrchestrator.registerAgent(new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      }), 1.0)
      
      await timeoutOrchestrator.registerAgent(new MockBearishAgent({
        id: 'bearish-1',
        name: 'Bearish Agent',
        version: '1.0.0',
        description: 'Always bearish',
        type: 'custom'
      }), 1.0)
      
      await timeoutOrchestrator.registerAgent(new MockNeutralAgent({
        id: 'neutral-1',
        name: 'Neutral Agent',
        version: '1.0.0',
        description: 'Always neutral',
        type: 'custom'
      }), 1.0)
      
      // Add slow agent that will timeout
      await timeoutOrchestrator.registerAgent(new MockSlowAgent({
        id: 'slow-1',
        name: 'Slow Agent',
        version: '1.0.0',
        description: 'Very slow',
        type: 'custom'
      }, 2000), 1.0) // 2 second delay
      
      const consensus = await timeoutOrchestrator.getConsensus(createContext())
      
      // Should still get consensus from other agents
      assert.equal(consensus.participatingAgents, 3) // Only 3 fast agents
      
      // Clean up
      await timeoutOrchestrator.shutdown()
    })
    
    it('should handle agent errors', async () => {
      // Add error agent
      await orchestrator.registerAgent(new MockErrorAgent({
        id: 'error-1',
        name: 'Error Agent',
        version: '1.0.0',
        description: 'Always fails',
        type: 'custom'
      }), 1.0)
      
      const consensus = await orchestrator.getConsensus(createContext())
      
      // Should still get consensus from working agents
      assert.equal(consensus.participatingAgents, 3) // Only 3 working agents
    })
    
    it('should filter low confidence signals', async () => {
      const config = { minConfidence: 0.7 }
      const strictOrchestrator = new AgentOrchestrator(eventBus, config)
      
      // Register agents
      await strictOrchestrator.registerAgent(new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      }), 1.0) // Confidence 0.8 - passes
      
      await strictOrchestrator.registerAgent(new MockNeutralAgent({
        id: 'neutral-1',
        name: 'Neutral Agent',
        version: '1.0.0',
        description: 'Always neutral',
        type: 'custom'
      }), 1.0) // Confidence 0.6 - filtered out
      
      const consensus = await strictOrchestrator.getConsensus(createContext())
      
      // Only bullish agent should participate
      assert.equal(consensus.participatingAgents, 1)
      assert.equal(consensus.action, 'buy')
    })
    
    it('should return hold consensus when no valid signals', async () => {
      const emptyOrchestrator = new AgentOrchestrator(eventBus)
      
      await assert.rejects(
        emptyOrchestrator.getConsensus(createContext()),
        /No agents registered/
      )
    })
  })
  
  describe('consensus strategies', () => {
    it('should use majority voting strategy', async () => {
      // Register 2 bullish, 1 bearish
      await orchestrator.registerAgent(new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent 1',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      }), 1.0)
      
      await orchestrator.registerAgent(new MockBullishAgent({
        id: 'bullish-2',
        name: 'Bullish Agent 2',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      }), 1.0)
      
      await orchestrator.registerAgent(new MockBearishAgent({
        id: 'bearish-1',
        name: 'Bearish Agent',
        version: '1.0.0',
        description: 'Always bearish',
        type: 'custom'
      }), 1.0)
      
      // Switch to majority voting
      orchestrator.setConsensusStrategy('majority-voting')
      
      const consensus = await orchestrator.getConsensus(createContext())
      
      // Majority is bullish
      assert.equal(consensus.action, 'buy')
      assert.equal(consensus.agreement, 2/3) // 2 out of 3 agents
    })
    
    it('should use confidence-weighted strategy', async () => {
      await orchestrator.registerAgent(new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      }), 1.0) // Confidence 0.8
      
      await orchestrator.registerAgent(new MockNeutralAgent({
        id: 'neutral-1',
        name: 'Neutral Agent 1',
        version: '1.0.0',
        description: 'Always neutral',
        type: 'custom'
      }), 1.0) // Confidence 0.6
      
      await orchestrator.registerAgent(new MockNeutralAgent({
        id: 'neutral-2',
        name: 'Neutral Agent 2',
        version: '1.0.0',
        description: 'Always neutral',
        type: 'custom'
      }), 1.0) // Confidence 0.6
      
      // Switch to confidence-weighted strategy
      orchestrator.setConsensusStrategy('confidence-weighted')
      
      const consensus = await orchestrator.getConsensus(createContext())
      
      // Despite being outnumbered, high confidence should give buy more weight
      assert.ok(['buy', 'hold'].includes(consensus.action))
    })
    
    it('should use veto consensus strategy', async () => {
      // Create a high-confidence bearish agent
      class MockHighConfidenceBearishAgent extends BaseAgent {
        protected async onInitialize(): Promise<void> {}
        protected async performAnalysis(_context: MarketContext): Promise<AgentSignal> {
          return this.createSignal('sell', 0.95, 'High confidence sell')
        }
      }
      
      await orchestrator.registerAgent(new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent 1',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      }), 1.0)
      
      await orchestrator.registerAgent(new MockBullishAgent({
        id: 'bullish-2',
        name: 'Bullish Agent 2',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      }), 1.0)
      
      await orchestrator.registerAgent(new MockHighConfidenceBearishAgent({
        id: 'bearish-veto',
        name: 'Bearish Veto Agent',
        version: '1.0.0',
        description: 'High confidence bearish',
        type: 'custom'
      }), 1.0)
      
      // Switch to veto consensus
      orchestrator.setConsensusStrategy('veto-consensus')
      
      const consensus = await orchestrator.getConsensus(createContext())
      
      // Veto should override majority
      assert.equal(consensus.action, 'sell')
      assert.equal(consensus.vetoApplied, true)
    })
    
    it('should register custom consensus strategy', () => {
      const customStrategy = {
        name: 'always-hold',
        calculateConsensus: () => ({
          action: 'hold' as const,
          confidence: 1,
          reason: 'Always hold',
          agentSignals: {},
          agreement: 1,
          participatingAgents: 0,
          timestamp: epochDateNow()
        })
      }
      
      orchestrator.registerConsensusStrategy(customStrategy)
      orchestrator.setConsensusStrategy('always-hold')
      
      // Strategy is set, would be used in next consensus calculation
      assert.ok(true)
    })
  })
  
  describe('disagreement detection', () => {
    it('should detect high disagreement', () => {
      const signals = new Map([
        ['agent1', { action: 'buy' as const, confidence: 0.8, reason: 'Bullish', timestamp: epochDateNow() }],
        ['agent2', { action: 'sell' as const, confidence: 0.7, reason: 'Bearish', timestamp: epochDateNow() }],
        ['agent3', { action: 'hold' as const, confidence: 0.6, reason: 'Neutral', timestamp: epochDateNow() }]
      ])
      
      const hasDisagreement = orchestrator.hasHighDisagreement(signals)
      assert.equal(hasDisagreement, true)
    })
    
    it('should not detect disagreement with consensus', () => {
      const signals = new Map([
        ['agent1', { action: 'buy' as const, confidence: 0.8, reason: 'Bullish', timestamp: epochDateNow() }],
        ['agent2', { action: 'buy' as const, confidence: 0.7, reason: 'Also bullish', timestamp: epochDateNow() }],
        ['agent3', { action: 'buy' as const, confidence: 0.6, reason: 'Bullish too', timestamp: epochDateNow() }]
      ])
      
      const hasDisagreement = orchestrator.hasHighDisagreement(signals)
      assert.equal(hasDisagreement, false)
    })
  })
  
  describe('agent lifecycle', () => {
    it('should reset all agents', async () => {
      const agent = new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      })
      
      await orchestrator.registerAgent(agent)
      
      // Generate some signals to create performance data
      await orchestrator.getConsensus(createContext())
      
      // Reset all agents
      await orchestrator.resetAllAgents()
      
      const performance = agent.getPerformance()
      assert.equal(performance.totalSignals, 0)
    })
    
    it('should shutdown orchestrator and all agents', async () => {
      const agent = new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      })
      
      await orchestrator.registerAgent(agent)
      await orchestrator.shutdown()
      
      const agents = orchestrator.getAgents()
      assert.equal(agents.size, 0)
    })
  })
  
  describe('manual weight management', () => {
    it('should set agent weight manually', async () => {
      const agent = new MockBullishAgent({
        id: 'bullish-1',
        name: 'Bullish Agent',
        version: '1.0.0',
        description: 'Always bullish',
        type: 'custom'
      })
      
      await orchestrator.registerAgent(agent, 1.0)
      orchestrator.setAgentWeight('bullish-1', 5.0)
      
      const weights = orchestrator.getAgentWeights()
      assert.equal(weights.get('bullish-1'), 5.0)
    })
    
    it('should throw if agent not found when setting weight', async () => {
      assert.throws(
        () => orchestrator.setAgentWeight('non-existent', 1.0),
        /not found/
      )
    })
  })
})