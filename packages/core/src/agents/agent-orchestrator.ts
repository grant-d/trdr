import type { 
  ITradeAgent, 
  AgentSignal, 
  MarketContext,
  ConsensusResult,
  AgentOrchestratorConfig,
  AgentExecutionResult,
  ConsensusStrategy
} from './types'
import { epochDateNow } from '@trdr/shared'
import type { Logger } from '@trdr/types'
import type { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import {
  performanceWeightedConsensus,
  exponentialWeightedConsensus,
  confidenceWeightedConsensus,
  bayesianConsensus,
  vetoConsensus
} from './consensus-algorithms'

/**
 * Default configuration for agent orchestrator
 */
const DEFAULT_CONFIG: AgentOrchestratorConfig = {
  agentTimeout: 5000, // 5 seconds
  minConfidence: 0.1,
  useAdaptiveWeights: true,
  weightUpdateFrequency: 100
}

/**
 * Orchestrates multiple trading agents and manages consensus
 */
export class AgentOrchestrator {
  private readonly agents = new Map<string, ITradeAgent>()
  private readonly agentWeights = new Map<string, number>()
  private readonly config: AgentOrchestratorConfig
  private readonly consensusStrategies = new Map<string, ConsensusStrategy>()
  private currentStrategy: ConsensusStrategy | null = null
  private tradeCount = 0
  
  constructor(
    private readonly eventBus: EventBus,
    config?: Partial<AgentOrchestratorConfig>,
    private readonly logger?: Logger
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config }
    
    // Register default consensus strategies
    this.registerDefaultStrategies()
    
    // Set default strategy
    this.currentStrategy = this.consensusStrategies.get('weighted-voting')!
    
    // Subscribe to relevant events
    this.subscribeToEvents()
  }
  
  /**
   * Register a trading agent
   */
  async registerAgent(agent: ITradeAgent, weight = 1.0): Promise<void> {
    const agentId = agent.metadata.id
    
    if (this.agents.has(agentId)) {
      throw new Error(`Agent ${agentId} already registered`)
    }
    
    // Initialize agent
    await agent.initialize()
    
    // Register agent and weight
    this.agents.set(agentId, agent)
    this.agentWeights.set(agentId, Math.max(0, weight))
    
    this.logger?.info(`Registered agent ${agentId}`, {
      name: agent.metadata.name,
      type: agent.metadata.type,
      weight
    })
    
    // Emit agent registered event
    this.eventBus.emit(EventTypes.AGENT_REGISTERED, {
      timestamp: epochDateNow(),
      agentId,
      metadata: agent.metadata,
      weight
    })
  }
  
  /**
   * Unregister a trading agent
   */
  async unregisterAgent(agentId: string): Promise<void> {
    const agent = this.agents.get(agentId)
    if (!agent) {
      throw new Error(`Agent ${agentId} not found`)
    }
    
    // Shutdown agent
    if (agent.shutdown) {
      await agent.shutdown()
    }
    
    // Remove agent
    this.agents.delete(agentId)
    this.agentWeights.delete(agentId)
    
    this.logger?.info(`Unregistered agent ${agentId}`)
    
    // Emit agent unregistered event
    this.eventBus.emit(EventTypes.AGENT_UNREGISTERED, {
      timestamp: epochDateNow(),
      agentId
    })
  }
  
  /**
   * Get consensus from all agents
   */
  async getConsensus(context: MarketContext): Promise<ConsensusResult> {
    if (this.agents.size === 0) {
      throw new Error('No agents registered')
    }
    
    const startTime = Date.now()
    
    // Execute all agents in parallel with timeout
    const agentSignals = await this.executeAgentsWithTimeout(context)
    
    // Filter out failed/timed out agents and low confidence signals
    const validSignals = new Map<string, AgentSignal>()
    for (const [agentId, result] of agentSignals) {
      if (!result.error && !result.timedOut && 
          result.signal.confidence >= this.config.minConfidence) {
        validSignals.set(agentId, result.signal)
      }
    }
    
    // Check if we have enough valid signals
    if (validSignals.size === 0) {
      this.logger?.warn('No valid signals from agents')
      return this.createHoldConsensus('No valid signals from agents')
    }
    
    // Get current weights
    const weights = new Map<string, number>()
    for (const agentId of validSignals.keys()) {
      weights.set(agentId, this.agentWeights.get(agentId) || 1.0)
    }
    
    // Calculate consensus using current strategy
    const consensus = this.currentStrategy!.calculateConsensus(validSignals, weights, this.agents)
    
    // Update trade count
    this.tradeCount++
    
    // Update weights if adaptive weights are enabled
    if (this.config.useAdaptiveWeights && 
        this.tradeCount % this.config.weightUpdateFrequency === 0) {
      await this.updateAgentWeights()
    }
    
    const executionTime = Date.now() - startTime
    
    this.logger?.info('Consensus calculated', {
      action: consensus.action,
      confidence: consensus.confidence,
      agreement: consensus.agreement,
      participatingAgents: consensus.participatingAgents,
      executionTime
    })
    
    // Emit consensus event
    this.eventBus.emit(EventTypes.CONSENSUS_REACHED, {
      timestamp: epochDateNow(),
      consensus,
      executionTime
    })
    
    return consensus
  }
  
  /**
   * Execute agents with timeout protection
   */
  private async executeAgentsWithTimeout(
    context: MarketContext
  ): Promise<Map<string, AgentExecutionResult>> {
    const results = new Map<string, AgentExecutionResult>()
    
    // Create promises for each agent
    const agentPromises: Promise<void>[] = []
    
    for (const [agentId, agent] of this.agents) {
      const promise = this.executeAgentWithTimeout(agentId, agent, context)
        .then(result => {
          results.set(agentId, result)
        })
        .catch(error => {
          results.set(agentId, {
            signal: {
              action: 'hold',
              confidence: 0,
              reason: 'Agent execution failed',
              timestamp: epochDateNow()
            },
            executionTime: 0,
            error: error.message
          })
        })
      
      agentPromises.push(promise)
    }
    
    // Wait for all agents to complete
    await Promise.all(agentPromises)
    
    return results
  }
  
  /**
   * Execute a single agent with timeout
   */
  private async executeAgentWithTimeout(
    agentId: string,
    agent: ITradeAgent,
    context: MarketContext
  ): Promise<AgentExecutionResult> {
    const startTime = Date.now()
    
    return new Promise<AgentExecutionResult>((resolve) => {
      // Set timeout
      const timeout = setTimeout(() => {
        resolve({
          signal: {
            action: 'hold',
            confidence: 0,
            reason: 'Agent timed out',
            timestamp: epochDateNow()
          },
          executionTime: this.config.agentTimeout,
          timedOut: true
        })
        
        this.logger?.warn(`Agent ${agentId} timed out`)
      }, this.config.agentTimeout)
      
      // Execute agent
      agent.analyze(context)
        .then(signal => {
          clearTimeout(timeout)
          const executionTime = Date.now() - startTime
          
          resolve({
            signal,
            executionTime
          })
          
          this.logger?.debug(`Agent ${agentId} executed`, {
            action: signal.action,
            confidence: signal.confidence,
            executionTime
          })
        })
        .catch(error => {
          clearTimeout(timeout)
          const executionTime = Date.now() - startTime
          
          resolve({
            signal: {
              action: 'hold',
              confidence: 0,
              reason: 'Agent error',
              timestamp: epochDateNow()
            },
            executionTime,
            error: error.message
          })
          
          this.logger?.error(`Agent ${agentId} failed`, { error })
        })
    })
  }
  
  /**
   * Calculate weighted consensus
   */
  calculateWeightedConsensus(
    signals: Map<string, AgentSignal>,
    weights?: Map<string, number>
  ): ConsensusResult {
    // Use provided weights or current weights
    const effectiveWeights = weights || this.agentWeights
    
    // Delegate to current strategy
    return this.currentStrategy!.calculateConsensus(signals, effectiveWeights, this.agents)
  }
  
  /**
   * Check if agents have high disagreement
   */
  hasHighDisagreement(signals: Map<string, AgentSignal>): boolean {
    if (signals.size < 2) return false
    
    // Count actions
    const actionCounts = new Map<string, number>()
    for (const signal of signals.values()) {
      const count = actionCounts.get(signal.action) || 0
      actionCounts.set(signal.action, count + 1)
    }
    
    // Calculate agreement as the ratio of most common action
    const maxCount = Math.max(...actionCounts.values())
    const agreement = maxCount / signals.size
    
    // High disagreement if agreement is below 0.6
    return agreement < 0.6
  }
  
  /**
   * Update agent weights based on performance
   */
  private async updateAgentWeights(): Promise<void> {
    if (!this.config.useAdaptiveWeights) return
    
    const performanceScores = new Map<string, number>()
    let totalScore = 0
    
    // Get performance scores from agents
    for (const [agentId, agent] of this.agents) {
      if (agent.getPerformance) {
        const performance = agent.getPerformance()
        
        // Calculate performance score (combination of win rate and Sharpe ratio)
        const score = performance.winRate * 0.5 + 
                     Math.max(0, Math.min(1, performance.sharpeRatio / 2)) * 0.5
        
        performanceScores.set(agentId, score)
        totalScore += score
      }
    }
    
    // Update weights based on performance
    if (totalScore > 0) {
      for (const [agentId, score] of performanceScores) {
        const newWeight = score / totalScore * this.agents.size
        this.agentWeights.set(agentId, newWeight)
        
        this.logger?.debug(`Updated weight for agent ${agentId}`, {
          oldWeight: this.agentWeights.get(agentId),
          newWeight,
          score
        })
      }
    }
  }
  
  /**
   * Register default consensus strategies
   */
  private registerDefaultStrategies(): void {
    // Weighted voting strategy
    this.consensusStrategies.set('weighted-voting', {
      name: 'weighted-voting',
      calculateConsensus: (signals, weights) => {
        const scores = new Map<string, number>()
        let totalWeight = 0
        
        // Calculate weighted scores for each action
        for (const [agentId, signal] of signals) {
          const weight = weights.get(agentId) || 1.0
          const score = scores.get(signal.action) || 0
          scores.set(signal.action, score + signal.confidence * weight)
          totalWeight += weight
        }
        
        // Find winning action
        let bestAction = 'hold'
        let bestScore = 0
        for (const [action, score] of scores) {
          if (score > bestScore) {
            bestAction = action
            bestScore = score
          }
        }
        
        // Calculate agreement
        const agreement = bestScore / totalWeight
        
        // Combine reasons
        const reasons: string[] = []
        const agentSignalsMap: Record<string, AgentSignal> = {}
        for (const [agentId, signal] of signals) {
          if (signal.action === bestAction) {
            reasons.push(`${agentId}: ${signal.reason}`)
          }
          agentSignalsMap[agentId] = signal
        }
        
        return {
          action: bestAction as AgentSignal['action'],
          confidence: agreement,
          reason: reasons.join('; '),
          agentSignals: agentSignalsMap,
          agreement,
          participatingAgents: signals.size,
          timestamp: epochDateNow()
        }
      }
    })
    
    // Majority voting strategy
    this.consensusStrategies.set('majority-voting', {
      name: 'majority-voting',
      calculateConsensus: (signals, _weights) => {
        const votes = new Map<string, number>()
        
        // Count votes
        for (const signal of signals.values()) {
          const count = votes.get(signal.action) || 0
          votes.set(signal.action, count + 1)
        }
        
        // Find majority action
        let bestAction = 'hold'
        let maxVotes = 0
        for (const [action, count] of votes) {
          if (count > maxVotes) {
            bestAction = action
            maxVotes = count
          }
        }
        
        // Calculate confidence as percentage of agents voting for winning action
        const confidence = maxVotes / signals.size
        
        // Combine reasons
        const reasons: string[] = []
        const agentSignalsMap: Record<string, AgentSignal> = {}
        for (const [agentId, signal] of signals) {
          if (signal.action === bestAction) {
            reasons.push(`${agentId}: ${signal.reason}`)
          }
          agentSignalsMap[agentId] = signal
        }
        
        return {
          action: bestAction as AgentSignal['action'],
          confidence,
          reason: reasons.join('; '),
          agentSignals: agentSignalsMap,
          agreement: confidence,
          participatingAgents: signals.size,
          timestamp: epochDateNow()
        }
      }
    })
    
    // Register new consensus strategies
    this.consensusStrategies.set('performance-weighted', performanceWeightedConsensus)
    this.consensusStrategies.set('exponential-weighted', exponentialWeightedConsensus)
    this.consensusStrategies.set('confidence-weighted', confidenceWeightedConsensus)
    this.consensusStrategies.set('bayesian', bayesianConsensus)
    this.consensusStrategies.set('veto-consensus', vetoConsensus)
  }
  
  /**
   * Create a hold consensus when no valid signals
   */
  private createHoldConsensus(reason: string): ConsensusResult {
    return {
      action: 'hold',
      confidence: 0,
      reason,
      agentSignals: {},
      agreement: 0,
      participatingAgents: 0,
      timestamp: epochDateNow()
    }
  }
  
  /**
   * Subscribe to relevant events
   */
  private subscribeToEvents(): void {
    // Subscribe to trade execution events to update agent performance
    this.eventBus.subscribe(EventTypes.ORDER_FILLED, async (_data) => {
      // const { order, fillPrice } = data
      
      // TODO: Notify agents of trade execution
      // This would need to be implemented with proper tracking of which agents
      // contributed to the decision
    })
  }
  
  /**
   * Set consensus strategy
   */
  setConsensusStrategy(strategyName: string): void {
    const strategy = this.consensusStrategies.get(strategyName)
    if (!strategy) {
      throw new Error(`Consensus strategy ${strategyName} not found`)
    }
    
    this.currentStrategy = strategy
    this.logger?.info(`Set consensus strategy to ${strategyName}`)
  }
  
  /**
   * Register custom consensus strategy
   */
  registerConsensusStrategy(strategy: ConsensusStrategy): void {
    this.consensusStrategies.set(strategy.name, strategy)
    this.logger?.info(`Registered consensus strategy ${strategy.name}`)
  }
  
  /**
   * Get registered agents
   */
  getAgents(): Map<string, ITradeAgent> {
    return new Map(this.agents)
  }
  
  /**
   * Get agent weights
   */
  getAgentWeights(): Map<string, number> {
    return new Map(this.agentWeights)
  }
  
  /**
   * Set agent weight manually
   */
  setAgentWeight(agentId: string, weight: number): void {
    if (!this.agents.has(agentId)) {
      throw new Error(`Agent ${agentId} not found`)
    }
    
    this.agentWeights.set(agentId, Math.max(0, weight))
    this.logger?.info(`Set weight for agent ${agentId} to ${weight}`)
  }
  
  /**
   * Reset all agents
   */
  async resetAllAgents(): Promise<void> {
    const resetPromises: Promise<void>[] = []
    
    for (const [agentId, agent] of this.agents) {
      if (agent.reset) {
        resetPromises.push(
          agent.reset()
            .catch(error => {
              this.logger?.error(`Failed to reset agent ${agentId}`, { error })
            })
        )
      }
    }
    
    await Promise.all(resetPromises)
    this.logger?.info('Reset all agents')
  }
  
  /**
   * Shutdown orchestrator
   */
  async shutdown(): Promise<void> {
    // Shutdown all agents
    const shutdownPromises: Promise<void>[] = []
    
    for (const [agentId, agent] of this.agents) {
      if (agent.shutdown) {
        shutdownPromises.push(
          agent.shutdown()
            .catch(error => {
              this.logger?.error(`Failed to shutdown agent ${agentId}`, { error })
            })
        )
      }
    }
    
    await Promise.all(shutdownPromises)
    
    // Clear all data
    this.agents.clear()
    this.agentWeights.clear()
    
    this.logger?.info('Agent orchestrator shutdown complete')
  }
}