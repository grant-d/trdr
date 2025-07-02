import type { AgentConfig } from './agents'
import type { EpochDate } from './dates'
import { StockSymbol, type OrderSide } from './orders'

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
  readonly symbol: StockSymbol
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
  /** Advanced trailing order configuration */
  readonly trailingOrderConfig?: GridTrailingOrderConfig
}

/**
 * Advanced trailing order configuration for grid levels.
 * Controls when and how trailing orders are activated.
 */
export interface GridTrailingOrderConfig {
  /** Distance from level price to activate trailing (percentage) */
  readonly activationThreshold: number
  /** Enable proximity-based activation */
  readonly enableProximityActivation: boolean
  /** Direct execution fallback if trailing fails */
  readonly enableDirectFallback: boolean
  /** Timeout for trailing order activation (ms) */
  readonly activationTimeoutMs: number
  /** Maximum trail adjustment frequency (ms) */
  readonly trailUpdateThrottleMs: number
  /** Advanced activation strategy */
  readonly activationStrategy: 'proximity' | 'price_approach' | 'volume_spike' | 'combined'
}

/**
 * Represents an individual grid level with its state
 */
export interface GridLevel {
  /** Unique identifier for this grid level */
  readonly id: string
  /** Target price for this grid level */
  readonly price: number
  /** Order side for this level (buy or sell) */
  readonly side: OrderSide
  /** Order size for this level */
  readonly size: number
  /** Whether this level is currently active */
  readonly isActive: boolean
  /** Order ID if an order is placed at this level */
  readonly orderId?: string
  /** Timestamp when this level was created */
  readonly createdAt: EpochDate
  /** Timestamp when this level was last updated */
  readonly updatedAt: EpochDate
  /** Number of times this level has been filled */
  readonly fillCount: number
  /** Total profit/loss from this level */
  readonly pnl: number
  /** Advanced trailing order state */
  readonly trailingState?: GridLevelTrailingState
}

/**
 * Trailing order state for a grid level
 */
export interface GridLevelTrailingState {
  /** Current activation status */
  readonly status: 'pending' | 'approaching' | 'active' | 'triggered' | 'failed'
  /** Price when level became active for trailing */
  readonly activationPrice?: number
  /** Timestamp when trailing was activated */
  readonly activatedAt?: EpochDate
  /** Last price that triggered an update */
  readonly lastUpdatePrice?: number
  /** Last update timestamp */
  readonly lastUpdatedAt?: EpochDate
  /** Number of trail adjustments made */
  readonly adjustmentCount: number
  /** Whether direct fallback has been attempted */
  readonly fallbackAttempted: boolean
  /** Reason for failure if status is 'failed' */
  readonly failureReason?: string
}

/**
 * Overall state of the grid trading system
 */
export interface GridState {
  /** Grid configuration being used */
  readonly config: GridConfig
  /** Symbol being traded */
  readonly symbol: StockSymbol
  /** All grid levels (active and inactive) */
  readonly levels: readonly GridLevel[]
  /** Center price around which grid is built */
  readonly centerPrice: number
  /** Current grid spacing being used */
  readonly currentSpacing: number
  /** Total capital allocated to grid */
  readonly allocatedCapital: number
  /** Available capital for new orders */
  readonly availableCapital: number
  /** Current position in the symbol */
  readonly currentPosition: number
  /** Total realized PnL from grid trading */
  readonly realizedPnl: number
  /** Total unrealized PnL from grid trading */
  readonly unrealizedPnl: number
  /** Timestamp when grid was initialized */
  readonly initializedAt: EpochDate
  /** Timestamp of last grid update */
  readonly lastUpdatedAt: EpochDate
  /** Whether grid is currently active */
  readonly isActive: boolean
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
  readonly name: 'coinbase' | 'alpaca'
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
  /** Unix timestamp of backtest start */
  readonly startDate: EpochDate
  /** Unix timestamp of backtest end */
  readonly endDate: EpochDate
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
  readonly symbol: StockSymbol
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
