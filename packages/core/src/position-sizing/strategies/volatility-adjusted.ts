import type {
  IPositionSizingStrategy,
  PositionSizingInput,
  PositionSizingOutput,
  BacktestTrade,
  BacktestResults,
  SizingAdjustment
} from '../interfaces'

/**
 * Volatility-Adjusted position sizing strategy.
 * 
 * Adjusts position size based on market volatility to maintain consistent risk.
 * Uses ATR (Average True Range) or standard deviation for volatility measurement.
 */
export class VolatilityAdjustedStrategy implements IPositionSizingStrategy {
  readonly name = 'Volatility Adjusted'
  readonly description = 'Adjusts position size inversely to market volatility for consistent risk'
  
  private readonly baseRiskPercentage: number
  private readonly volatilityLookback: number
  private readonly maxVolatilityMultiplier: number
  
  /**
   * Creates a new Volatility-Adjusted strategy
   * @param baseRiskPercentage - Base risk percentage when volatility is normal (default: 0.01 = 1%)
   * @param volatilityLookback - Lookback period for volatility calculation (default: 20)
   * @param maxVolatilityMultiplier - Maximum volatility adjustment multiplier (default: 3)
   */
  constructor(
    baseRiskPercentage: number = 0.01,
    volatilityLookback: number = 20,
    maxVolatilityMultiplier: number = 3
  ) {
    this.baseRiskPercentage = baseRiskPercentage
    this.volatilityLookback = volatilityLookback
    this.maxVolatilityMultiplier = maxVolatilityMultiplier
  }
  
  calculateSize(input: PositionSizingInput): PositionSizingOutput {
    const adjustments: SizingAdjustment[] = []
    const warnings: string[] = []
    
    // Base risk amount
    const baseRiskAmount = input.riskParams.accountBalance * this.baseRiskPercentage
    
    // Calculate volatility adjustment
    // Using market conditions volatility (0-1 scale)
    const normalizedVolatility = input.marketConditions.volatility
    
    // Inverse volatility scaling: lower size when volatility is high
    // When volatility = 0.5 (normal), multiplier = 1
    // When volatility = 1.0 (extreme), multiplier = 0.5
    // When volatility = 0.2 (low), multiplier = 1.6
    const volatilityMultiplier = Math.min(
      this.maxVolatilityMultiplier,
      Math.max(0.2, 1 / (0.5 + normalizedVolatility))
    )
    
    adjustments.push({
      type: 'volatility',
      factor: volatilityMultiplier,
      reason: `Volatility adjustment: ${(normalizedVolatility * 100).toFixed(1)}% volatility`
    })
    
    // Apply volatility adjustment
    let adjustedRiskAmount = baseRiskAmount * volatilityMultiplier
    let adjustedRiskPercentage = this.baseRiskPercentage * volatilityMultiplier
    
    // Apply confidence adjustment
    adjustedRiskAmount *= input.confidence
    adjustedRiskPercentage *= input.confidence
    
    // Market regime adjustments
    if (input.marketConditions.regime === 'ranging') {
      // Reduce size in ranging markets
      const regimeFactor = 0.7
      adjustedRiskAmount *= regimeFactor
      adjustedRiskPercentage *= regimeFactor
      adjustments.push({
        type: 'market_conditions',
        factor: regimeFactor,
        reason: 'Ranging market adjustment'
      })
    } else if (input.marketConditions.regime === 'trending' && Math.abs(input.marketConditions.trendStrength) > 0.7) {
      // Increase size in strong trends
      const trendFactor = 1.2
      adjustedRiskAmount *= trendFactor
      adjustedRiskPercentage *= trendFactor
      adjustments.push({
        type: 'market_conditions',
        factor: trendFactor,
        reason: 'Strong trend adjustment'
      })
    }
    
    // Time of day adjustment
    if (input.marketConditions.timeOfDayFactor < 0.5) {
      const timeFactor = 0.5 + input.marketConditions.timeOfDayFactor
      adjustedRiskAmount *= timeFactor
      adjustedRiskPercentage *= timeFactor
      adjustments.push({
        type: 'market_conditions',
        factor: timeFactor,
        reason: 'Off-peak hours adjustment'
      })
    }
    
    // Calculate position size based on stop loss distance
    const stopDistance = Math.abs(input.entryPrice - input.stopLoss)
    const positionSize = adjustedRiskAmount / stopDistance
    const positionValue = positionSize * input.entryPrice
    
    // Check against risk limits
    if (adjustedRiskPercentage > input.riskParams.maxRiskPerTrade) {
      const limitFactor = input.riskParams.maxRiskPerTrade / adjustedRiskPercentage
      adjustments.push({
        type: 'risk_limit',
        factor: limitFactor,
        reason: `Risk limited to ${(input.riskParams.maxRiskPerTrade * 100).toFixed(1)}% per trade`
      })
      
      return {
        positionSize: positionSize * limitFactor,
        positionValue: positionValue * limitFactor,
        riskAmount: adjustedRiskAmount * limitFactor,
        riskPercentage: input.riskParams.maxRiskPerTrade,
        method: this.name,
        confidence: input.confidence * 0.9,
        reasoning: `Volatility-adjusted sizing, risk limited`,
        warnings: [...warnings, 'Position reduced to meet risk limits'],
        adjustments
      }
    }
    
    return {
      positionSize,
      positionValue,
      riskAmount: adjustedRiskAmount,
      riskPercentage: adjustedRiskPercentage,
      method: this.name,
      confidence: input.confidence,
      reasoning: `Base risk ${(this.baseRiskPercentage * 100).toFixed(1)}%, volatility multiplier ${volatilityMultiplier.toFixed(2)}x`,
      warnings,
      adjustments
    }
  }
  
  validate(input: PositionSizingInput): { valid: boolean; errors: string[] } {
    const errors: string[] = []
    
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
    
    if (input.marketConditions.volatility < 0 || input.marketConditions.volatility > 1) {
      errors.push('Market volatility must be between 0 and 1')
    }
    
    return {
      valid: errors.length === 0,
      errors
    }
  }
  
  getDefaultParams(): Record<string, any> {
    return {
      baseRiskPercentage: this.baseRiskPercentage,
      volatilityLookback: this.volatilityLookback,
      maxVolatilityMultiplier: this.maxVolatilityMultiplier
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
    
    // Track volatility history for adaptive adjustments
    const volatilityHistory: number[] = []
    
    for (const trade of trades) {
      // Update volatility history
      volatilityHistory.push(trade.marketConditions.volatility)
      if (volatilityHistory.length > this.volatilityLookback) {
        volatilityHistory.shift()
      }
      
      // Create input for position sizing
      const input: PositionSizingInput = {
        side: trade.side,
        entryPrice: trade.entryPrice,
        stopLoss: trade.stopLoss,
        winRate: trade.winRate,
        riskRewardRatio: trade.riskRewardRatio,
        confidence: trade.confidence,
        riskParams: {
          accountBalance: balance,
          maxRiskPerTrade: this.baseRiskPercentage * 2,
          maxPortfolioRisk: this.baseRiskPercentage * 6,
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
    
    const avgReturn = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : 0
    const stdDev = returns.length > 0 ? Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length) : 0
    const sharpeRatio = stdDev > 0 ? (avgReturn - 0.02 / 252) / stdDev * Math.sqrt(252) : 0
    
    const negativeReturns = returns.filter(r => r < 0)
    const downStdDev = negativeReturns.length > 0 ? Math.sqrt(negativeReturns.reduce((sum, r) => sum + Math.pow(r, 2), 0) / negativeReturns.length) : 0
    const sortinoRatio = downStdDev > 0 ? (avgReturn - 0.02 / 252) / downStdDev * Math.sqrt(252) : 0
    
    return {
      totalReturn,
      annualizedReturn,
      maxDrawdown,
      sharpeRatio,
      sortinoRatio,
      winRate: (wins + losses) > 0 ? wins / (wins + losses) : 0,
      avgWinLoss: losses > 0 && wins > 0 ? (totalProfit / wins) / (totalLoss / losses) : 0,
      totalTrades: wins + losses,
      profitFactor: totalLoss > 0 ? totalProfit / totalLoss : 0,
      equityCurve,
      drawdownCurve
    }
  }
}