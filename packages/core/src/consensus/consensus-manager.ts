import { epochDateNow } from '@trdr/shared'
import type {
  AgentSignal,
  ConsensusConfig,
  ConsensusResult,
  IConsensusStrategy,
  SignalRequest,
  AgentPerformance
} from './interfaces'
import { WeightedVotingStrategy } from './strategies/weighted-voting'
import { EventBus } from '../events/event-bus'
import { EventTypes } from '../events/types'
import type { OrderAgentConsensus } from '@trdr/shared'

/**
 * Manages consensus gathering from multiple trading agents.
 * 
 * Coordinates signal collection, applies timeouts, executes consensus
 * strategies, and tracks agent performance for adaptive weighting.
 */
export class ConsensusManager {
  private readonly config: ConsensusConfig
  private readonly eventBus: EventBus
  private readonly strategy: IConsensusStrategy
  private readonly agentPerformance = new Map<string, AgentPerformance>()
  private readonly activeRequests = new Map<string, {
    signals: AgentSignal[]
    expectedAgents: Set<string>
    timeout: NodeJS.Timeout
    resolve: (result: ConsensusResult) => void
    startTime: number
  }>()

  /**
   * Creates a new ConsensusManager instance
   * @param config - Consensus configuration
   * @param strategy - Consensus strategy to use (defaults to WeightedVoting)
   * @param eventBus - Event bus for communication
   */
  constructor(
    config: ConsensusConfig,
    strategy?: IConsensusStrategy,
    eventBus?: EventBus
  ) {
    this.config = config
    this.strategy = strategy || new WeightedVotingStrategy()
    this.eventBus = eventBus || EventBus.getInstance()
    
    this.registerEventTypes()
    this.setupEventHandlers()
  }

  /**
   * Register consensus-related event types
   */
  private registerEventTypes(): void {
    this.eventBus.registerEvent(EventTypes.CONSENSUS_STARTED)
    this.eventBus.registerEvent(EventTypes.CONSENSUS_SIGNAL_RECEIVED)
    this.eventBus.registerEvent(EventTypes.CONSENSUS_TIMEOUT)
    this.eventBus.registerEvent(EventTypes.CONSENSUS_COMPLETED)
    this.eventBus.registerEvent(EventTypes.CONSENSUS_FAILED)
    this.eventBus.registerEvent(EventTypes.AGENT_SIGNAL)
    this.eventBus.registerEvent(EventTypes.SIGNAL_REQUEST)
  }

  /**
   * Setup event handlers for agent signals
   */
  private setupEventHandlers(): void {
    // Listen for agent signals
    this.eventBus.subscribe(EventTypes.AGENT_SIGNAL, (data: unknown) => {
      const val = data as { requestId?: string, signal?: AgentSignal }
      if (typeof val.requestId === 'string' && val.signal && typeof val.signal === 'object') {
        this.addAgentSignal(val.requestId, val.signal)
      }
    })
  }

  /**
   * Request consensus from all registered agents
   * @param request - Signal request details
   * @param expectedAgents - List of agent IDs expected to respond
   * @returns Promise resolving to consensus result
   */
  async gatherConsensus(
    request: SignalRequest,
    expectedAgents: string[]
  ): Promise<ConsensusResult> {
    return new Promise((resolve) => {
      const startTime = Date.now()
      
      // Set up timeout
      const timeout = setTimeout(() => {
        this.handleTimeout(request.requestId)
      }, this.config.consensusTimeoutMs)

      // Store request
      this.activeRequests.set(request.requestId, {
        signals: [],
        expectedAgents: new Set(expectedAgents),
        timeout,
        resolve,
        startTime
      })

      // Emit consensus started event
      this.eventBus.emit(EventTypes.CONSENSUS_STARTED, {
        requestId: request.requestId,
        agentCount: expectedAgents.length,
        timestamp: epochDateNow()
      })

      // Broadcast signal request to agents
      this.eventBus.emit(EventTypes.SIGNAL_REQUEST, { ...request })
    })
  }

  /**
   * Add an agent signal to an active consensus request
   * @param requestId - Request identifier
   * @param signal - Agent signal
   */
  private addAgentSignal(requestId: string, signal: AgentSignal): void {
    const request = this.activeRequests.get(requestId)
    if (!request) return

    // Add signal with performance-adjusted weight
    const adjustedSignal = this.adjustSignalWeight(signal)
    request.signals.push(adjustedSignal)

    // Emit signal received event
    this.eventBus.emit(EventTypes.CONSENSUS_SIGNAL_RECEIVED, {
      requestId,
      agentId: signal.agentId,
      signal: adjustedSignal,
      timestamp: epochDateNow()
    })

    // Check if all expected agents have responded
    const receivedAgents = new Set(request.signals.map(s => s.agentId))
    const allReceived = Array.from(request.expectedAgents).every(id => receivedAgents.has(id))

    if (allReceived || request.signals.length >= this.config.minAgentsRequired) {
      this.evaluateConsensus(requestId)
    }
  }

  /**
   * Adjust signal weight based on agent performance
   */
  private adjustSignalWeight(signal: AgentSignal): AgentSignal {
    if (!this.config.enableAdaptiveWeights) {
      return signal
    }

    const performance = this.agentPerformance.get(signal.agentId)
    if (!performance) {
      return signal
    }

    // Apply performance-based weight adjustment
    const adjustedWeight = Math.min(
      performance.suggestedWeight,
      this.config.maxAgentWeight
    )

    return {
      ...signal,
      weight: adjustedWeight
    }
  }

  /**
   * Handle timeout for consensus gathering
   */
  private handleTimeout(requestId: string): void {
    const request = this.activeRequests.get(requestId)
    if (!request) return

    // Emit timeout event
    this.eventBus.emit(EventTypes.CONSENSUS_TIMEOUT, {
      requestId,
      receivedCount: request.signals.length,
      expectedCount: request.expectedAgents.size,
      timestamp: epochDateNow()
    })

    // Evaluate with whatever signals we have
    this.evaluateConsensus(requestId)
  }

  /**
   * Evaluate consensus from collected signals
   */
  private evaluateConsensus(requestId: string): void {
    const request = this.activeRequests.get(requestId)
    if (!request) return

    // Clear timeout
    clearTimeout(request.timeout)

    // Evaluate consensus using strategy
    const result = this.strategy.evaluate(request.signals, this.config)

    // Emit completion event
    this.eventBus.emit(EventTypes.CONSENSUS_COMPLETED, {
      requestId,
      result,
      timestamp: epochDateNow()
    })

    // Clean up and resolve
    this.activeRequests.delete(requestId)
    request.resolve(result)
  }

  /**
   * Convert consensus result to OrderAgentConsensus format
   */
  static toOrderAgentConsensus(
    result: ConsensusResult,
    symbol?: string
  ): OrderAgentConsensus | null {
    if (result.action === 'hold') {
      return null
    }

    return {
      action: result.action,
      confidence: result.confidence,
      expectedWinRate: result.expectedWinRate,
      expectedRiskReward: result.expectedRiskReward,
      trailDistance: result.trailDistance,
      leadAgentId: result.leadAgentId,
      agentSignals: result.agentSignals.map(s => ({
        agentId: s.agentId,
        signal: s.signal,
        confidence: s.confidence,
        weight: s.weight || 1,
        reason: s.reasoning,
        timestamp: s.timestamp
      })),
      symbol
    }
  }

  /**
   * Update agent performance metrics
   * @param agentId - Agent identifier
   * @param correct - Whether the agent's prediction was correct
   * @param confidence - Agent's confidence in the prediction
   * @param pnlContribution - P&L contribution from this prediction
   */
  updateAgentPerformance(
    agentId: string,
    correct: boolean,
    confidence: number,
    pnlContribution = 0
  ): void {
    const existing = this.agentPerformance.get(agentId) || {
      agentId,
      totalSignals: 0,
      correctPredictions: 0,
      avgConfidenceWhenCorrect: 0,
      avgConfidenceWhenWrong: 0,
      pnlContribution: 0,
      currentWeight: this.config.defaultAgentWeight,
      suggestedWeight: this.config.defaultAgentWeight,
      lastUpdated: epochDateNow()
    }

    // Update counts
    existing.totalSignals++
    if (correct) {
      existing.correctPredictions++
      existing.avgConfidenceWhenCorrect = 
        (existing.avgConfidenceWhenCorrect * (existing.correctPredictions - 1) + confidence) / 
        existing.correctPredictions
    } else {
      const wrongCount = existing.totalSignals - existing.correctPredictions
      existing.avgConfidenceWhenWrong = 
        (existing.avgConfidenceWhenWrong * (wrongCount - 1) + confidence) / wrongCount
    }

    // Update P&L
    existing.pnlContribution += pnlContribution

    // Calculate new suggested weight
    const accuracy = existing.correctPredictions / existing.totalSignals
    const calibration = correct 
      ? existing.avgConfidenceWhenCorrect 
      : (1 - existing.avgConfidenceWhenWrong)
    
    // Weight based on accuracy, calibration, and P&L contribution
    let newWeight = this.config.defaultAgentWeight
    newWeight *= (0.5 + accuracy * 0.5) // 50-100% based on accuracy
    newWeight *= (0.5 + calibration * 0.5) // 50-100% based on calibration
    
    // P&L adjustment (can increase or decrease weight)
    if (existing.pnlContribution > 0) {
      newWeight *= 1.2
    } else if (existing.pnlContribution < 0) {
      newWeight *= 0.8
    }

    existing.suggestedWeight = Math.min(
      Math.max(newWeight, 0.1), // Minimum weight of 0.1
      this.config.maxAgentWeight
    )
    existing.currentWeight = existing.suggestedWeight
    existing.lastUpdated = epochDateNow()

    this.agentPerformance.set(agentId, existing)
  }

  /**
   * Get performance metrics for all agents
   */
  getAgentPerformanceMetrics(): AgentPerformance[] {
    return Array.from(this.agentPerformance.values())
  }

  /**
   * Reset performance metrics for an agent
   */
  resetAgentPerformance(agentId: string): void {
    this.agentPerformance.delete(agentId)
  }

  /**
   * Get current configuration
   */
  getConfig(): Readonly<ConsensusConfig> {
    return this.config
  }
}