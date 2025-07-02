import type { OrderSide, EpochDate, StockSymbol } from '@trdr/shared'

/**
 * Individual agent signal for consensus
 */
export interface AgentSignal {
  /** Unique agent identifier */
  readonly agentId: string
  /** Agent type/name */
  readonly agentType: string
  /** Trading signal: buy, sell, or hold */
  readonly signal: OrderSide | 'hold'
  /** Confidence level (0-1) */
  readonly confidence: number
  /** Suggested trail distance percentage */
  readonly trailDistance?: number
  /** Expected win rate */
  readonly expectedWinRate?: number
  /** Expected risk/reward ratio */
  readonly expectedRiskReward?: number
  /** Reasoning for the signal */
  readonly reasoning: string
  /** Timestamp of signal generation */
  readonly timestamp: EpochDate
  /** Priority weight for this agent */
  readonly weight?: number
}

/**
 * Configuration for consensus mechanism
 */
export interface ConsensusConfig {
  /** Minimum confidence threshold for consensus (0-1) */
  readonly minConfidenceThreshold: number
  /** Minimum number of agents required for consensus */
  readonly minAgentsRequired: number
  /** Maximum time to wait for agent signals (milliseconds) */
  readonly consensusTimeoutMs: number
  /** Strategy to use when consensus cannot be reached */
  readonly fallbackStrategy: 'hold' | 'use-best' | 'use-majority'
  /** Whether to use weighted voting based on agent performance */
  readonly useWeightedVoting: boolean
  /** Minimum agreement percentage for consensus (0-1) */
  readonly minAgreementThreshold: number
  /** Enable adaptive weight adjustment based on performance */
  readonly enableAdaptiveWeights: boolean
  /** Default agent weight if not specified */
  readonly defaultAgentWeight: number
  /** Maximum weight any single agent can have */
  readonly maxAgentWeight: number
  /** Trading symbol */
  readonly symbol?: string
}

/**
 * Result of consensus evaluation
 */
export interface ConsensusResult {
  /** Whether consensus was reached */
  readonly consensusReached: boolean
  /** Final action decision */
  readonly action: OrderSide | 'hold'
  /** Aggregate confidence level */
  readonly confidence: number
  /** Average expected win rate */
  readonly expectedWinRate: number
  /** Average expected risk/reward */
  readonly expectedRiskReward: number
  /** Recommended trail distance */
  readonly trailDistance: number
  /** ID of the lead agent (highest confidence) */
  readonly leadAgentId: string
  /** All agent signals that contributed */
  readonly agentSignals: readonly AgentSignal[]
  /** Agreement percentage among agents */
  readonly agreementPercentage: number
  /** Dissent level (1 - agreement) */
  readonly dissentLevel: number
  /** Whether fallback strategy was used */
  readonly usedFallback: boolean
  /** Reason if consensus wasn't reached */
  readonly fallbackReason?: string
  /** Processing time in milliseconds */
  readonly processingTimeMs: number
  /** Timestamp of consensus */
  readonly timestamp: EpochDate
}

/**
 * Agent performance metrics for weight adjustment
 */
export interface AgentPerformance {
  /** Agent identifier */
  agentId: string
  /** Total number of signals */
  totalSignals: number
  /** Number of correct predictions */
  correctPredictions: number
  /** Average confidence when correct */
  avgConfidenceWhenCorrect: number
  /** Average confidence when wrong */
  avgConfidenceWhenWrong: number
  /** Profit/loss contribution */
  pnlContribution: number
  /** Current weight */
  currentWeight: number
  /** Suggested new weight */
  suggestedWeight: number
  /** Last updated timestamp */
  lastUpdated: EpochDate
}

/**
 * Interface for consensus strategies
 */
export interface IConsensusStrategy {
  /** Strategy name */
  readonly name: string
  
  /**
   * Evaluate signals and determine consensus
   * @param signals - Agent signals to evaluate
   * @param config - Consensus configuration
   * @returns Consensus result
   */
  evaluate(signals: AgentSignal[], config: ConsensusConfig): ConsensusResult
  
  /**
   * Calculate agreement level among signals
   * @param signals - Agent signals
   * @returns Agreement percentage (0-1)
   */
  calculateAgreement(signals: AgentSignal[]): number
}

/**
 * Events emitted by consensus system
 */
export interface ConsensusEvents {
  'consensus.started': {
    requestId: string
    agentCount: number
    timestamp: EpochDate
  }
  'consensus.signal.received': {
    requestId: string
    agentId: string
    signal: AgentSignal
    timestamp: EpochDate
  }
  'consensus.timeout': {
    requestId: string
    receivedCount: number
    expectedCount: number
    timestamp: EpochDate
  }
  'consensus.completed': {
    requestId: string
    result: ConsensusResult
    timestamp: EpochDate
  }
  'consensus.failed': {
    requestId: string
    reason: string
    timestamp: EpochDate
  }
}

/**
 * Request for agent signals
 */
export interface SignalRequest {
  /** Unique request identifier */
  readonly requestId: string
  /** Trading symbol */
  readonly symbol: StockSymbol
  /** Current market price */
  readonly currentPrice: number
  /** Request timestamp */
  readonly timestamp: EpochDate
  /** Additional context data */
  readonly context?: Record<string, unknown>
}