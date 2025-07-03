import type { Candle, OrderBook, StockSymbol, EpochDate } from '@trdr/shared'
import type { IndicatorResult } from '../indicators/types'

/**
 * Trading signal action types
 */
export type SignalAction = 'buy' | 'sell' | 'hold'

/**
 * Agent signal with recommendation and confidence
 */
export interface AgentSignal {
  /** The recommended trading action */
  readonly action: SignalAction
  
  /** Confidence level for the signal (0-1) */
  readonly confidence: number
  
  /** Reason for the signal */
  readonly reason: string
  
  /** Detailed analysis supporting the signal */
  readonly analysis?: string
  
  /** Specific price targets or levels */
  readonly priceTarget?: number
  
  /** Suggested stop loss level */
  readonly stopLoss?: number
  
  /** Suggested limit price for order execution */
  readonly limitPrice?: number
  
  /** Suggested stop trigger price (for stop-limit orders) */
  readonly stopPrice?: number
  
  /** Suggested position size (as percentage of capital) */
  readonly positionSize?: number
  
  /** Timestamp when signal was generated */
  readonly timestamp: EpochDate
}

/**
 * Market context provided to agents for analysis
 */
export interface MarketContext {
  /** Trading symbol */
  readonly symbol: StockSymbol
  
  /** Current price */
  readonly currentPrice: number
  
  /** Recent price candles */
  readonly candles: readonly Candle[]
  
  /** Current order book */
  readonly orderBook?: OrderBook
  
  /** Pre-calculated indicators */
  readonly indicators?: Record<string, IndicatorResult>
  
  /** Current account balance */
  readonly accountBalance?: number
  
  /** Current position size */
  readonly currentPosition?: number
  
  /** Recent trades */
  readonly recentTrades?: readonly Trade[]
}

/**
 * Agent metadata for identification and configuration
 */
export interface AgentMetadata {
  /** Unique agent identifier */
  readonly id: string
  
  /** Human-readable agent name */
  readonly name: string
  
  /** Agent version */
  readonly version: string
  
  /** Agent description */
  readonly description: string
  
  /** Agent type/category */
  readonly type: AgentType
  
  /** Required indicators for this agent */
  readonly requiredIndicators?: readonly string[]
  
  /** Default configuration */
  readonly defaultConfig?: Record<string, unknown>
}

/**
 * Agent types for categorization
 */
export type AgentType = 
  | 'volatility'
  | 'momentum' 
  | 'volume'
  | 'structure'
  | 'regime'
  | 'sentiment'
  | 'ai'
  | 'custom'

/**
 * Trade representation for agent context
 */
export interface Trade {
  readonly id: string
  readonly timestamp: EpochDate
  readonly side: 'buy' | 'sell'
  readonly price: number
  readonly size: number
  readonly fee: number
  readonly pnl?: number
}

/**
 * Agent execution result with timing information
 */
export interface AgentExecutionResult {
  /** The agent's signal */
  readonly signal: AgentSignal
  
  /** Execution time in milliseconds */
  readonly executionTime: number
  
  /** Any errors encountered */
  readonly error?: string
  
  /** Whether the agent timed out */
  readonly timedOut?: boolean
}

/**
 * Agent performance metrics
 */
export interface AgentPerformance {
  /** Total number of signals generated */
  readonly totalSignals: number
  
  /** Number of profitable signals */
  readonly profitableSignals: number
  
  /** Win rate (profitable/total) */
  readonly winRate: number
  
  /** Average return per signal */
  readonly averageReturn: number
  
  /** Sharpe ratio */
  readonly sharpeRatio: number
  
  /** Maximum drawdown */
  readonly maxDrawdown: number
  
  /** Average execution time */
  readonly avgExecutionTime: number
  
  /** Number of timeouts */
  readonly timeouts: number
  
  /** Last update timestamp */
  readonly lastUpdated: EpochDate
}

/**
 * Interface that all trading agents must implement
 */
export interface ITradeAgent {
  /** Agent metadata */
  readonly metadata: AgentMetadata
  
  /**
   * Initialize the agent with configuration
   * @param config Agent-specific configuration
   */
  initialize(config?: Record<string, unknown>): Promise<void>
  
  /**
   * Analyze market conditions and generate trading signal
   * @param context Current market context
   * @returns Trading signal with confidence and reasoning
   */
  analyze(context: MarketContext): Promise<AgentSignal>
  
  /**
   * Update agent state based on trade execution result
   * @param trade The executed trade
   * @param signal The signal that triggered the trade
   */
  updateOnTrade?(trade: Trade, signal: AgentSignal): Promise<void>
  
  /**
   * Get current agent performance metrics
   * @returns Performance metrics
   */
  getPerformance?(): AgentPerformance
  
  /**
   * Reset agent state
   */
  reset?(): Promise<void>
  
  /**
   * Cleanup resources when agent is stopped
   */
  shutdown?(): Promise<void>
}

/**
 * Configuration for agent orchestrator
 */
export interface AgentOrchestratorConfig {
  /** Maximum execution time per agent in milliseconds */
  readonly agentTimeout: number
  
  /** Minimum confidence threshold for signals */
  readonly minConfidence: number
  
  /** Whether to use performance-based weights */
  readonly useAdaptiveWeights: boolean
  
  /** Weight update frequency (number of trades) */
  readonly weightUpdateFrequency: number
  
  /** Initial agent weights (agent ID -> weight) */
  readonly initialWeights?: Record<string, number>
}

/**
 * Consensus result from multiple agents
 */
export interface ConsensusResult {
  /** The consensus action */
  readonly action: SignalAction
  
  /** Weighted confidence level */
  readonly confidence: number
  
  /** Combined reasoning from agents */
  readonly reason: string
  
  /** Individual agent signals */
  readonly agentSignals: Record<string, AgentSignal>
  
  /** Agreement level between agents (0-1) */
  readonly agreement: number
  
  /** Number of agents that contributed */
  readonly participatingAgents: number
  
  /** Timestamp of consensus */
  readonly timestamp: EpochDate
  
  /** Confidence interval for the consensus */
  readonly confidenceInterval?: { lower: number; upper: number }
  
  /** Posterior probabilities for Bayesian consensus */
  readonly posteriorProbabilities?: Record<string, number>
  
  /** Whether a veto was applied */
  readonly vetoApplied?: boolean
  
  /** Statistically calculated stop loss level */
  readonly stopLoss?: number
  
  /** Statistically calculated limit price */
  readonly limitPrice?: number
  
  /** Statistically calculated stop trigger price */
  readonly stopPrice?: number
  
  /** Consensus position size */
  readonly positionSize?: number
  
  /** Statistical confidence for price levels */
  readonly priceConfidence?: {
    stopLoss: { mean: number; stdDev: number; confidence: number }
    limitPrice: { mean: number; stdDev: number; confidence: number }
    stopPrice?: { mean: number; stdDev: number; confidence: number }
  }
}

/**
 * Strategy for combining agent signals into consensus
 */
export interface ConsensusStrategy {
  /** Strategy name */
  readonly name: string
  
  /**
   * Calculate consensus from multiple agent signals
   * @param signals Map of agent ID to signal
   * @param weights Map of agent ID to weight
   * @param agents Optional map of agents for performance-based strategies
   * @param priors Optional prior probabilities for Bayesian strategies
   * @param currentPrice Current market price for price level calculations
   * @returns Consensus result
   */
  calculateConsensus(
    signals: Map<string, AgentSignal>,
    weights: Map<string, number>,
    agents?: Map<string, ITradeAgent>,
    priors?: Record<string, number>,
    currentPrice?: number
  ): ConsensusResult
}