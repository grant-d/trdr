import type { AgentConfig } from './agents'

/** Risk tolerance levels for position sizing and strategy */
export type RiskTolerance = 'conservative' | 'moderate' | 'aggressive'

/** Trading execution modes */
export type TradingMode = 'live' | 'paper' | 'backtest'

/**
 * Core trading configuration.
 * Defines basic trading parameters.
 */
export interface TradingConfig {
  /** Trading pair symbol (e.g., 'BTC-USD') */
  readonly symbol: string
  /** Total capital allocated for trading */
  readonly capital: number
  /** Risk tolerance level */
  readonly riskTolerance: RiskTolerance
  /** Trading execution mode */
  readonly mode: TradingMode
}

/**
 * Grid trading strategy configuration.
 * Controls grid placement and sizing.
 */
export interface GridConfig {
  /** Spacing between grid levels as percentage */
  readonly gridSpacing: number
  /** Number of grid levels to maintain */
  readonly gridLevels: number
  /** Trailing stop distance as percentage */
  readonly trailPercent: number
  /** Minimum order size allowed */
  readonly minOrderSize: number
  /** Maximum order size allowed */
  readonly maxOrderSize: number
  /** Threshold for grid rebalancing */
  readonly rebalanceThreshold: number
}

/**
 * Risk management configuration.
 * Defines position limits and risk controls.
 */
export interface RiskConfig {
  /** Maximum position size as percentage of capital */
  readonly maxPositionSize: number
  /** Maximum allowed drawdown percentage */
  readonly maxDrawdown: number
  /** Stop loss percentage */
  readonly stopLossPercent: number
  /** Take profit percentage */
  readonly takeProfitPercent: number
  /** Maximum leverage allowed */
  readonly maxLeverage: number
  /** Margin requirement percentage */
  readonly marginRequirement: number
}

/**
 * Exchange connection configuration.
 * Contains API credentials and connection settings.
 */
export interface ExchangeConfig {
  /** Exchange platform name */
  readonly name: 'coinbase' | 'binance' | 'kraken'
  /** API key (stored securely) */
  readonly apiKey?: string
  /** API secret (stored securely) */
  readonly apiSecret?: string
  /** Use testnet/sandbox environment */
  readonly testnet: boolean
  /** Rate limit in requests per second */
  readonly rateLimit: number
  /** Request timeout in milliseconds */
  readonly timeout: number
}

/**
 * Backtesting configuration.
 * Defines parameters for historical testing.
 */
export interface BacktestConfig {
  /** Backtest start date */
  readonly startDate: Date
  /** Backtest end date */
  readonly endDate: Date
  /** Starting capital for backtest */
  readonly initialCapital: number
  /** Trading fee rate (e.g., 0.001 for 0.1%) */
  readonly feeRate: number
  /** Simulated slippage percentage */
  readonly slippage: number
  /** Data source for historical data */
  readonly dataSource: string
}

/**
 * Complete system configuration.
 * Aggregates all configuration sections.
 */
export interface SystemConfig {
  /** Core trading parameters */
  readonly trading: TradingConfig
  /** Grid strategy settings */
  readonly grid: GridConfig
  /** Risk management rules */
  readonly risk: RiskConfig
  /** Exchange connection details */
  readonly exchange: ExchangeConfig
  /** Optional backtest configuration */
  readonly backtest?: BacktestConfig
  /** Agent configurations */
  readonly agents: readonly AgentConfig[]
}

/**
 * Minimal user configuration.
 * Self-tuning system fills in optimal values.
 * ONLY 3 user parameters - everything else self-tunes.
 */
export interface MinimalConfig {
  /** Trading pair symbol - what to trade */
  readonly symbol: string
  /** Total capital to trade - how much money */
  readonly capital: number
  /** Risk tolerance level - risk preference */
  readonly riskTolerance: RiskTolerance
}

/** Alias for MinimalConfig to match PRD naming */
export type MinimalUserConfig = MinimalConfig

/**
 * Market data feed configuration.
 * Controls real-time data connection.
 */
export interface DataFeedConfig {
  /** Data feed protocol type */
  readonly type: 'websocket' | 'rest' | 'hybrid'
  /** Maximum reconnection attempts */
  readonly reconnectAttempts: number
  /** Delay between reconnects in ms */
  readonly reconnectDelay: number
  /** Heartbeat interval in ms */
  readonly heartbeatInterval: number
}

/**
 * Logging system configuration.
 * Controls log output and retention.
 */
export interface LoggingConfig {
  /** Minimum log level to record */
  readonly level: 'debug' | 'info' | 'warn' | 'error'
  /** Where to send log output */
  readonly outputs: readonly ('console' | 'file' | 'remote')[]
  /** Optional file rotation settings */
  readonly fileRotation?: {
    /** Max file size (e.g., '10MB') */
    readonly maxSize: string
    /** Max number of log files to keep */
    readonly maxFiles: number
  }
}

/**
 * Performance monitoring configuration.
 * Controls metrics collection and alerting.
 */
export interface MonitoringConfig {
  /** Enable metrics collection */
  readonly metricsEnabled: boolean
  /** Metrics collection interval in ms */
  readonly metricsInterval: number
  /** Enable alert notifications */
  readonly alertsEnabled: boolean
  /** Alert trigger thresholds */
  readonly alertThresholds: {
    /** Max drawdown before alert (percentage) */
    readonly drawdown: number
    /** Max daily loss before alert (amount) */
    readonly lossPerDay: number
    /** Max error rate before alert (percentage) */
    readonly errorRate: number
  }
}
