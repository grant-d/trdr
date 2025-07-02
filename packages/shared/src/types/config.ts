import type { AgentConfig } from './agents'

export type RiskTolerance = 'conservative' | 'moderate' | 'aggressive'
export type TradingMode = 'live' | 'paper' | 'backtest'

export interface TradingConfig {
  readonly symbol: string
  readonly capital: number
  readonly riskTolerance: RiskTolerance
  readonly mode: TradingMode
}

export interface GridConfig {
  readonly gridSpacing: number
  readonly gridLevels: number
  readonly trailPercent: number
  readonly minOrderSize: number
  readonly maxOrderSize: number
  readonly rebalanceThreshold: number
}

export interface RiskConfig {
  readonly maxPositionSize: number
  readonly maxDrawdown: number
  readonly stopLossPercent: number
  readonly takeProfitPercent: number
  readonly maxLeverage: number
  readonly marginRequirement: number
}

export interface ExchangeConfig {
  readonly name: 'coinbase' | 'binance' | 'kraken'
  readonly apiKey?: string
  readonly apiSecret?: string
  readonly testnet: boolean
  readonly rateLimit: number
  readonly timeout: number
}

export interface BacktestConfig {
  readonly startDate: Date
  readonly endDate: Date
  readonly initialCapital: number
  readonly feeRate: number
  readonly slippage: number
  readonly dataSource: string
}

export interface SystemConfig {
  readonly trading: TradingConfig
  readonly grid: GridConfig
  readonly risk: RiskConfig
  readonly exchange: ExchangeConfig
  readonly backtest?: BacktestConfig
  readonly agents: readonly AgentConfig[]
}