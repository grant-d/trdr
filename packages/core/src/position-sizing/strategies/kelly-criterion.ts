import type {
  IPositionSizingStrategy,
  PositionSizingInput,
  PositionSizingOutput,
  BacktestTrade,
  BacktestResults,
  SizingAdjustment
} from '../interfaces'

/**
 * Kelly Criterion position sizing strategy.
 * 
 * Calculates optimal position size using the Kelly formula:
 * f = (bp - q) / b
 * where:
 * - f = fraction of capital to wager
 * - b = odds received on the wager (risk/reward ratio)
 * - p = probability of winning
 * - q = probability of losing (1 - p)
 */
export class KellyCriterionStrategy implements IPositionSizingStrategy {
  readonly name = 'Kelly Criterion'
  readonly description = 'Calculates optimal position size using the Kelly formula with safety adjustments'
  
  private readonly kellySafetyFactor: number
  private readonly maxKellyFraction: number
  
  /**
   * Creates a new Kelly Criterion strategy
   * @param kellySafetyFactor - Safety factor to apply (default: 0.25 for 1/4 Kelly)
   * @param maxKellyFraction - Maximum allowed Kelly fraction (default: 0.25)
   */
  constructor(
    kellySafetyFactor = 0.25,
    maxKellyFraction = 0.25
  ) {
    this.kellySafetyFactor = kellySafetyFactor
    this.maxKellyFraction = maxKellyFraction
  }
  
  calculateSize(input: PositionSizingInput): PositionSizingOutput {
    const adjustments: SizingAdjustment[] = []
    const warnings: string[] = []
    
    // Calculate base Kelly fraction
    const p = input.winRate
    const q = 1 - p
    const b = input.riskRewardRatio
    
    let kellyFraction = (b * p - q) / b
    
    // Apply safety factor
    kellyFraction *= this.kellySafetyFactor
    
    // Cap at maximum fraction
    if (kellyFraction > this.maxKellyFraction) {
      warnings.push(`Kelly fraction capped at ${this.maxKellyFraction * 100}%`)
      kellyFraction = this.maxKellyFraction
    }
    
    // Ensure non-negative (don't bet if edge is negative)
    if (kellyFraction <= 0) {
      return {
        positionSize: 0,
        positionValue: 0,
        riskAmount: 0,
        riskPercentage: 0,
        method: this.name,
        confidence: 0,
        reasoning: 'Negative expectation - no position recommended',
        warnings: ['Negative Kelly fraction indicates unfavorable odds'],
        adjustments: []
      }
    }
    
    // Apply market condition adjustments
    let adjustedFraction = kellyFraction
    
    // Volatility adjustment
    const volatilityAdjustment = 1 - (input.marketConditions.volatility * 0.5)
    adjustedFraction *= volatilityAdjustment
    adjustments.push({
      type: 'volatility',
      factor: volatilityAdjustment,
      reason: `Volatility adjustment for ${(input.marketConditions.volatility * 100).toFixed(1)}% volatility`
    })
    
    // Confidence adjustment
    adjustedFraction *= input.confidence
    
    // Calculate risk per trade
    const riskPerTrade = Math.abs(input.entryPrice - input.stopLoss) / input.entryPrice
    
    // Calculate position value
    const positionValue = input.riskParams.accountBalance * adjustedFraction
    
    // Calculate position size in base units
    const positionSize = positionValue / input.entryPrice
    
    // Calculate actual risk amount
    const riskAmount = positionValue * riskPerTrade
    const riskPercentage = riskAmount / input.riskParams.accountBalance
    
    // Apply risk limits
    if (riskPercentage > input.riskParams.maxRiskPerTrade) {
      const limitFactor = input.riskParams.maxRiskPerTrade / riskPercentage
      adjustments.push({
        type: 'risk_limit',
        factor: limitFactor,
        reason: `Risk limited to ${(input.riskParams.maxRiskPerTrade * 100).toFixed(1)}% per trade`
      })
      
      return {
        positionSize: positionSize * limitFactor,
        positionValue: positionValue * limitFactor,
        riskAmount: riskAmount * limitFactor,
        riskPercentage: input.riskParams.maxRiskPerTrade,
        method: this.name,
        confidence: input.confidence * 0.9,
        reasoning: `Kelly sizing with safety factor ${this.kellySafetyFactor}, risk limited`,
        warnings: [...warnings, 'Position reduced to meet risk limits'],
        adjustments
      }
    }
    
    return {
      positionSize,
      positionValue,
      riskAmount,
      riskPercentage,
      method: this.name,
      confidence: input.confidence,
      reasoning: `Kelly fraction: ${(kellyFraction * 100).toFixed(2)}%, adjusted: ${(adjustedFraction * 100).toFixed(2)}%`,
      warnings,
      adjustments
    }
  }
  
  validate(input: PositionSizingInput): { valid: boolean; errors: string[] } {
    const errors: string[] = []
    
    if (input.winRate <= 0 || input.winRate >= 1) {
      errors.push('Win rate must be between 0 and 1')
    }
    
    if (input.riskRewardRatio <= 0) {
      errors.push('Risk/reward ratio must be positive')
    }
    
    if (input.entryPrice <= 0) {
      errors.push('Entry price must be positive')
    }
    
    if (input.stopLoss <= 0) {
      errors.push('Stop loss must be positive')
    }
    
    if (input.side === 'buy' && input.stopLoss >= input.entryPrice) {
      errors.push('Stop loss must be below entry price for buy orders')
    }
    
    if (input.side === 'sell' && input.stopLoss <= input.entryPrice) {
      errors.push('Stop loss must be above entry price for sell orders')
    }
    
    return {
      valid: errors.length === 0,
      errors
    }
  }
  
  getDefaultParams(): Record<string, unknown> {
    return {
      kellySafetyFactor: this.kellySafetyFactor,
      maxKellyFraction: this.maxKellyFraction
    }
  }
  
  backtest(trades: BacktestTrade[], initialBalance: number): BacktestResults {
    let balance = initialBalance
    let peakBalance = initialBalance
    let maxDrawdown = 0
    let wins = 0
    let losses = 0
    let totalProfit = 0
    let totalLoss = 0
    
    const equityCurve: Array<{ timestamp: Date; equity: number }> = []
    const drawdownCurve: Array<{ timestamp: Date; drawdown: number }> = []
    const returns: number[] = []
    
    // Add initial point
    equityCurve.push({ timestamp: trades[0]?.timestamp || new Date(), equity: balance })
    drawdownCurve.push({ timestamp: trades[0]?.timestamp || new Date(), drawdown: 0 })
    
    for (const trade of trades) {
      // Create mock input for position sizing
      const input: PositionSizingInput = {
        side: trade.side,
        entryPrice: trade.entryPrice,
        stopLoss: trade.stopLoss,
        winRate: trade.winRate,
        riskRewardRatio: trade.riskRewardRatio,
        confidence: trade.confidence,
        riskParams: {
          accountBalance: balance,
          maxRiskPerTrade: 0.02,
          maxPortfolioRisk: 0.06,
          currentExposure: 0,
          openPositions: 0,
          maxPositions: 5,
          riskFreeRate: 0.02
        },
        marketConditions: trade.marketConditions
      }
      
      // Calculate position size
      const sizing = this.calculateSize(input)
      
      if (sizing.positionSize > 0) {
        // Calculate P&L
        const priceDiff = trade.exitPrice - trade.entryPrice
        const pnl = trade.side === 'buy' ? priceDiff : -priceDiff
        const pnlAmount = pnl * sizing.positionSize
        
        // Update balance
        balance += pnlAmount
        
        // Track wins/losses
        if (pnlAmount > 0) {
          wins++
          totalProfit += pnlAmount
        } else {
          losses++
          totalLoss += Math.abs(pnlAmount)
        }
        
        // Calculate return
        const returnPct = pnlAmount / (balance - pnlAmount)
        returns.push(returnPct)
        
        // Update peak and drawdown
        if (balance > peakBalance) {
          peakBalance = balance
        }
        const currentDrawdown = (peakBalance - balance) / peakBalance
        maxDrawdown = Math.max(maxDrawdown, currentDrawdown)
        
        // Record equity and drawdown
        equityCurve.push({ timestamp: trade.timestamp, equity: balance })
        drawdownCurve.push({ timestamp: trade.timestamp, drawdown: currentDrawdown })
      }
    }
    
    // Calculate metrics
    const totalReturn = (balance - initialBalance) / initialBalance
    const lastTrade = trades[trades.length - 1]
    const firstTrade = trades[0]
    const years = trades.length > 1 && lastTrade && firstTrade
      ? (lastTrade.timestamp.getTime() - firstTrade.timestamp.getTime()) / (365 * 24 * 60 * 60 * 1000)
      : 1
    const annualizedReturn = years > 0 ? Math.pow(1 + totalReturn, 1 / years) - 1 : totalReturn
    
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length
    const stdDev = Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length)
    const sharpeRatio = (avgReturn - 0.02 / 252) / stdDev * Math.sqrt(252) // Assuming daily returns
    
    const negativeReturns = returns.filter(r => r < 0)
    const downStdDev = Math.sqrt(negativeReturns.reduce((sum, r) => sum + Math.pow(r, 2), 0) / negativeReturns.length)
    const sortinoRatio = (avgReturn - 0.02 / 252) / downStdDev * Math.sqrt(252)
    
    return {
      totalReturn,
      annualizedReturn,
      maxDrawdown,
      sharpeRatio,
      sortinoRatio,
      winRate: wins / (wins + losses),
      avgWinLoss: totalProfit / wins / (totalLoss / losses),
      totalTrades: wins + losses,
      profitFactor: totalProfit / totalLoss,
      equityCurve,
      drawdownCurve
    }
  }
}