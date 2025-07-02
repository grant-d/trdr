export type AgentType = 
  | 'volatility'
  | 'momentum' 
  | 'volume'
  | 'market-structure'
  | 'regime'
  | 'time-decay'
  | 'microstructure'
  | 'correlation'
  | 'sentiment'
  | 'ai-pattern'
  | 'entropy'
  | 'swarm'
  | 'harmonic'
  | 'topology'
  | 'fractal'
  | 'quantum'

export type SignalStrength = 'strong-buy' | 'buy' | 'neutral' | 'sell' | 'strong-sell'

export interface AgentSignal {
  readonly agent: AgentType
  readonly timestamp: number
  readonly signal: SignalStrength
  readonly confidence: number
  readonly reasoning: string
  readonly metadata?: Record<string, unknown>
}

export interface AgentVote {
  readonly agent: AgentType
  readonly action: 'buy' | 'sell' | 'hold'
  readonly confidence: number
  readonly size?: number
  readonly price?: number
  readonly stopLoss?: number
  readonly takeProfit?: number
}

export interface AgentConsensus {
  readonly timestamp: number
  readonly votes: readonly AgentVote[]
  readonly decision: 'buy' | 'sell' | 'hold'
  readonly confidence: number
  readonly dissent: number
}

export interface AgentConfig {
  readonly type: AgentType
  readonly enabled: boolean
  readonly weight: number
  readonly parameters: Record<string, unknown>
}

export interface AgentPerformance {
  readonly agent: AgentType
  readonly signals: number
  readonly accuracy: number
  readonly profit: number
  readonly sharpe: number
  readonly maxDrawdown: number
  readonly updatedAt: number
}