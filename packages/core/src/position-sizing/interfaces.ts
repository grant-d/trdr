import type { OrderSide } from '@trdr/shared'

/**
 * Market conditions used for position sizing adjustments
 */
export interface MarketConditions {
  /** Current market volatility (0-1, where 1 is extremely volatile) */
  readonly volatility: number
  /** Bid-ask spread as percentage */
  readonly spread: number
  /** Market trend strength (-1 to 1, negative = downtrend, positive = uptrend) */
  readonly trendStrength: number
  /** Volume relative to average (1.0 = average volume) */
  readonly relativeVolume: number
  /** Market regime (trending, ranging, volatile) */
  readonly regime: 'trending' | 'ranging' | 'volatile'
  /** Time of day factor (0-1, where 1 is peak trading hours) */
  readonly timeOfDayFactor: number
}

/**
 * Risk parameters for position sizing
 */
export interface RiskParameters {
  /** Total account balance in USD */
  readonly accountBalance: number
  /** Maximum risk per trade as percentage (e.g., 0.02 = 2%) */
  readonly maxRiskPerTrade: number
  /** Maximum total portfolio risk as percentage */
  readonly maxPortfolioRisk: number
  /** Current total exposure in USD */
  readonly currentExposure: number
  /** Number of open positions */
  readonly openPositions: number
  /** Maximum allowed positions */
  readonly maxPositions: number
  /** Risk-free rate for calculations */
  readonly riskFreeRate: number
}

/**
 * Input parameters for position sizing calculations
 */
export interface PositionSizingInput {
  /** Trading side (buy/sell) */
  readonly side: OrderSide
  /** Entry price for the position */
  readonly entryPrice: number
  /** Stop loss price */
  readonly stopLoss: number
  /** Take profit price (optional) */
  readonly takeProfit?: number
  /** Expected win rate (0-1) */
  readonly winRate: number
  /** Risk/reward ratio */
  readonly riskRewardRatio: number
  /** Signal confidence (0-1) */
  readonly confidence: number
  /** Risk parameters */
  readonly riskParams: RiskParameters
  /** Current market conditions */
  readonly marketConditions: MarketConditions
  /** Historical performance metrics */
  readonly historicalMetrics?: HistoricalMetrics
}

/**
 * Historical performance metrics for adaptive sizing
 */
export interface HistoricalMetrics {
  /** Average win rate over last N trades */
  readonly avgWinRate: number
  /** Average risk/reward achieved */
  readonly avgRiskReward: number
  /** Maximum consecutive losses */
  readonly maxConsecutiveLosses: number
  /** Current consecutive losses */
  readonly currentConsecutiveLosses: number
  /** Sharpe ratio */
  readonly sharpeRatio: number
  /** Maximum drawdown percentage */
  readonly maxDrawdown: number
  /** Profit factor (gross profit / gross loss) */
  readonly profitFactor: number
}

/**
 * Output from position sizing calculation
 */
export interface PositionSizingOutput {
  /** Recommended position size in base units (e.g., BTC) */
  readonly positionSize: number
  /** Position value in USD */
  readonly positionValue: number
  /** Risk amount in USD */
  readonly riskAmount: number
  /** Risk as percentage of account */
  readonly riskPercentage: number
  /** Sizing method used */
  readonly method: string
  /** Confidence in the sizing (0-1) */
  readonly confidence: number
  /** Reasoning for the size */
  readonly reasoning: string
  /** Any warnings or constraints applied */
  readonly warnings: string[]
  /** Adjustments applied */
  readonly adjustments: SizingAdjustment[]
}

/**
 * Adjustments made to position size
 */
export interface SizingAdjustment {
  /** Type of adjustment */
  readonly type: 'volatility' | 'market_conditions' | 'risk_limit' | 'drawdown' | 'correlation'
  /** Adjustment factor applied (1.0 = no adjustment) */
  readonly factor: number
  /** Reason for adjustment */
  readonly reason: string
}

/**
 * Interface for position sizing strategies
 */
export interface IPositionSizingStrategy {
  /** Strategy name */
  readonly name: string
  
  /** Strategy description */
  readonly description: string
  
  /**
   * Calculate position size based on inputs
   * @param input - Position sizing parameters
   * @returns Calculated position size and metadata
   */
  calculateSize(input: PositionSizingInput): PositionSizingOutput
  
  /**
   * Validate if the strategy can be used with given inputs
   * @param input - Position sizing parameters
   * @returns Validation result with any errors
   */
  validate(input: PositionSizingInput): { valid: boolean; errors: string[] }
  
  /**
   * Get default parameters for the strategy
   * @returns Default configuration
   */
  getDefaultParams(): Record<string, any>
  
  /**
   * Backtest the strategy with historical data
   * @param trades - Historical trade data
   * @param initialBalance - Starting balance
   * @returns Backtest results
   */
  backtest?(trades: BacktestTrade[], initialBalance: number): BacktestResults
}

/**
 * Trade data for backtesting
 */
export interface BacktestTrade {
  readonly timestamp: Date
  readonly side: OrderSide
  readonly entryPrice: number
  readonly exitPrice: number
  readonly stopLoss: number
  readonly winRate: number
  readonly riskRewardRatio: number
  readonly confidence: number
  readonly marketConditions: MarketConditions
}

/**
 * Results from backtesting a position sizing strategy
 */
export interface BacktestResults {
  /** Total return percentage */
  readonly totalReturn: number
  /** Annualized return percentage */
  readonly annualizedReturn: number
  /** Maximum drawdown percentage */
  readonly maxDrawdown: number
  /** Sharpe ratio */
  readonly sharpeRatio: number
  /** Sortino ratio */
  readonly sortinoRatio: number
  /** Win rate */
  readonly winRate: number
  /** Average win/loss ratio */
  readonly avgWinLoss: number
  /** Total trades */
  readonly totalTrades: number
  /** Profit factor */
  readonly profitFactor: number
  /** Equity curve data points */
  readonly equityCurve: Array<{ timestamp: Date; equity: number }>
  /** Drawdown curve data points */
  readonly drawdownCurve: Array<{ timestamp: Date; drawdown: number }>
}

/**
 * Configuration for position sizing manager
 */
export interface PositionSizingConfig {
  /** Default strategy to use */
  readonly defaultStrategy: string
  /** Available strategies */
  readonly strategies: string[]
  /** Enable adaptive sizing based on performance */
  readonly enableAdaptive: boolean
  /** Enable market condition adjustments */
  readonly enableMarketAdjustments: boolean
  /** Minimum position size in base units */
  readonly minPositionSize: number
  /** Maximum position size in base units */
  readonly maxPositionSize: number
  /** Enable backtesting mode */
  readonly enableBacktesting: boolean
}